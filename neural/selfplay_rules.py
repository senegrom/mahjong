"""The last hand's rules in the learner's own self-play (`train_combined
--last-hand-rules`): the all-last dama rule (`all_last`) and decline-last
(`late_rules`), the two kept on 31 October, played by the learner in every
seat it holds at every table of a round. The seated others are opponents and
stay plain, as each rule's series measured it against a plain incumbent, and
so does the in-training measure (`selfplay.measure`), which only picks
best.pt.

Self-play draws its moves, so each rule reads the move actually drawn, not
the network's best: decline-last is asked about every row of the learner's
whose drawn move is a win (Mortal's 43), and the all-last dama rule about
every row whose move is still a riichi (37) once decline has spoken, at the
tile the reach really throws, drawn from the reach's preview in the same
forward (`preview_reach`, which the rules need: without it a reach is told to
the follower before its tile is drawn). Decline comes first because a
declined self-draw can leave a riichi as the move, and dama must judge that
riichi; the ruler seats one rule a player only because each series measured
one. A declined ron becomes the pass, as the ruler's does; a declined
self-draw becomes the argmax of the same forward's logits with the win
struck, as the ruler's does (`late_rules.py`, the turn's branch of
`decline_last`). It is not drawn from the remaining moves, because a move so
drawn could not be recorded for the learner to reproduce: a mask without the
win is an input to the head (Mortal's duelling mean, the fusion's
correction), so the learning forward would not give it the probability it
was drawn with. A dama firing throws the drawn tile undeclared. The engine
answers either rule only for a non-dealer in South 4, so a round that keeps
a record of its hands asks it about no other row (`last_hand`).

A move a rule played is recorded alone: its row's mask holds only that move,
its log-probability is exactly nought and it is flagged (`Records.ruled`).
The learning forward then gives it probability one, exactly, so it adds
nothing to the policy's gradient, its entropy, the drift guard or the clipped
fraction, while the critic and the reading of the hands, which never read the
mask, still learn its position, and its hand still earns its return. Where
dama fires, the reach's tile record is dropped: the position after the reach
never happens at the table. A reach that stands after a declined self-draw
keeps its tile record, an ordinary decision drawn from the preview of a
position that does happen.

Nothing of either rule reads anything hidden: dama reads the player's own
hand, the scores and what the player has seen, and decline reads a riichi
win at the most its ura dora could add (see `all_last` and `late_rules`).

The checks, every one raising `RuleCheckFailed`, which stops the trainer
without learning or saving the round (`train_combined`): at each decision,
codes a legal move can have and a fired move that is legal (`decline`,
`dama`), and records that match what is sent to the table, each reach's
tile record its row and its tile (`check_step`); at the round's end, the
flagged records, the account's records and the firings agreeing, and the
reaches the round sent to the table and the tile records it gathered,
counted apart from the account as the round steps the table and gathers
them, agreeing with each other and with the account (`reconcile`); before
learning, every flagged row read by the learner as a move of probability one
(`train_combined.check_ruled`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import riichi_py

from . import all_last, late_rules, scoreboard, zoo
from .cloud_requests import LAST_HAND_RULES

DAMA, DECLINE = LAST_HAND_RULES
KINDS = all_last.KINDS
#: South 4, the last hand, as the hands begun number their stages
#: (`scoreboard.HandStarts.stage`).
LAST_STAGE = scoreboard.STAGES - 1


class RuleCheckFailed(RuntimeError):
    """A check of the last hand's rules in self-play failed: the engine's
    codes and the legal moves disagree, or a move a rule played could reach
    the policy. The round is never learned: `train_combined` stops with
    `cloud_requests.RULE_CHECK_EXIT`, saying why."""


def canonical(rules) -> tuple[str, ...]:
    """`rules`, checked, in `LAST_HAND_RULES` order."""
    names = tuple(rules)
    if not names:
        raise ValueError("name at least one rule of the last hand")
    unknown = [name for name in names if name not in LAST_HAND_RULES]
    if unknown:
        raise ValueError(f"{', '.join(map(repr, unknown))}: no rule of the last hand the learner can be played "
                         f"by; they are {', '.join(LAST_HAND_RULES)}")
    if len(set(names)) != len(names):
        raise ValueError("a rule of the last hand is named twice")
    return tuple(name for name in LAST_HAND_RULES if name in names)


def require(rules) -> tuple[str, ...]:
    """The rules named, checked and in `LAST_HAND_RULES` order, once the
    engine is known to answer their codes as they are read here: from
    `ALL_LAST_API_VERSION` 3 (`late_rules.require_late_engine`), which covers
    the dama codes and decline's reading of a riichi win without its ura
    dora."""
    names = canonical(rules)
    late_rules.require_late_engine()
    return names


def engine_version() -> int:
    """The engine's `ALL_LAST_API_VERSION`, nought where it has none."""
    return int(getattr(riichi_py, "ALL_LAST_API_VERSION", 0))


@dataclass
class Played:
    """One round's account of the rules the learner is played by.

    `dama` and `decline` keep the ruler's vocabulary (`all_last.Account`,
    `late_rules.Account`): the decisions the engine judged, those a rule
    fired at, the tables and hands it fired at, why it let the move stand
    elsewhere. Besides: `in_window` and `at_turn`, decline's firings in a
    claim window and at the player's own turn; `to_riichi`, the declined
    self-draws whose best other move was a riichi; `both`, those of them
    whose reach dama then left undeclared; `records`, the decisions recorded
    flagged; `tile_records`, the reaches' tile records kept, and
    `reaches_sent`, the reaches played at the table, which must agree;
    `dama_calls` and `decline_calls`, the engine's questions; `seconds`, the
    time all of the rules' work took inside deciding, the records' flags
    included (see `mortal_learner.decide_in_mortal_space`), though not the
    round's own gathering of the flags and its counts of the reaches sent
    and the tile records gathered (`selfplay.play`), a few microseconds a
    step; and `events`, a few hundred a round, each (game, hands the game
    had finished, player by place, rule, the move the rule played in
    Mortal's moves), for the tests and never logged."""

    rules: tuple[str, ...]
    dama: all_last.Account = field(default_factory=all_last.Account)
    decline: late_rules.Account = field(default_factory=lambda: late_rules.Account(DECLINE))
    in_window: int = 0
    at_turn: int = 0
    to_riichi: int = 0
    both: int = 0
    records: int = 0
    tile_records: int = 0
    reaches_sent: int = 0
    dama_calls: int = 0
    decline_calls: int = 0
    seconds: float = 0.0
    events: list[tuple] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rules = canonical(self.rules)

    def report(self) -> dict:
        """The account as a generation's record carries it, with the tables
        and hands as counts; the record adds `ruled_begun_max` (see
        `recorded`)."""
        report: dict = {"rules": list(self.rules)}
        if DAMA in self.rules:
            report[DAMA] = {
                "judged": self.dama.riichi, "fired": self.dama.fired, "tables": len(self.dama.tables),
                "hands": len(self.dama.hands), "kept": dict(self.dama.kept),
                "riichi_after_firing": self.dama.late, "calls": self.dama_calls,
            }
        if DECLINE in self.rules:
            report[DECLINE] = {
                "judged": self.decline.judged, "fired": self.decline.fired, "in_window": self.in_window,
                "at_turn": self.at_turn, "to_riichi": self.to_riichi, "tables": len(self.decline.tables),
                "hands": len(self.decline.hands), "kept": dict(self.decline.kept),
                "kept_after_firing": self.decline.later, "calls": self.decline_calls,
            }
        report.update(ruled_records=self.records, both=self.both, tile_records=self.tile_records,
                      reaches_sent=self.reaches_sent, seconds=round(self.seconds, 3))
        return report


def recorded(report: dict, begun_max: float) -> dict:
    """A round's report (`Played.report`) as the generation's record carries
    it: with `ruled_begun_max`, the largest probability's distance from one,
    in nats, that the baseline pass read for a flagged decision (see
    `train_combined.check_ruled`), before the seconds."""
    record = {name: value for name, value in report.items() if name != "seconds"}
    record["ruled_begun_max"] = float(begun_max)
    record["seconds"] = report["seconds"]
    return record


def may_ask(views) -> bool:
    """Whether any row of a step may be asked about at all: not while no game
    of a round that keeps a record of its hands (`views.starts`) has begun
    South 4, which spares the rules any work for the first half of a round or
    so (see `last_hand`)."""
    starts = getattr(views, "starts", None)
    return starts is None or bool(getattr(starts, "reached_last", True))


def last_hand(views, games: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Which of a step's rows, each a game and a player by place, may be asked
    about at all: a non-dealer's decision in South 4, the only hand and seat
    the engine answers either rule for (`riichi_core::all_last`), as the
    round's record of the hands begun says (`views.starts`, a
    `scoreboard.HandStarts`, kept when the round keeps a ledger, as every
    run paid the placement reward does), read up to this step's decisions
    by `Views.advance`. Every row where there is no such record, or its game
    has begun no hand: the engine is then asked and answers nought for the
    rest. Asking only these spares the engine's questions, each of which
    wakes the engine's threads, on every step whose wins and riichi all
    fall in other hands."""
    starts = getattr(views, "starts", None)
    if starts is None or getattr(starts, "stage", None) is None:
        return np.ones(len(games), dtype=bool)
    stage = starts.stage[games]
    return (stage < 0) | ((stage == LAST_STAGE) & (starts.dealer[games] != players))


def decline(arena, who, rows: np.ndarray, allowed: np.ndarray, logits_of, played: Played) -> dict[int, int]:
    """Decline-last on the rows of a step whose drawn move is a win, those the
    engine can answer (see `last_hand`): the move each row the rule fires at
    plays instead, in Mortal's moves, by row.

    `who` names each row's game and player by place, `rows` those asked,
    `allowed` the step's moves in Mortal's space, and `logits_of(rows)` the
    host's copy of those rows' logits from the forward that drew the moves,
    asked only for a win declined at the player's own turn, a handful a
    round. The arena answers a player who owes a decision in South 4 as a
    non-dealer with a win open (`riichi_py.Arena.all_last_decline`). Code 1,
    the win would leave them alone in last: in a claim window, where the pass
    is open, the pass; at their own turn, the best other move of the same
    answer, the win struck. Code 2: the win stands. Code 0: not judged. Any
    other code, or a declined win that leaves no move, is refused."""
    codes = arena.all_last_decline(late_rules.named(who, rows))
    played.decline_calls += 1
    if len(codes) != len(rows):
        raise RuleCheckFailed(f"the arena answered {len(codes)} decline codes for {len(rows)} rows")
    # The arena's bytes as they come, a code a row: a handful of rows a call.
    if not codes.strip(b"\0"):
        return {}
    hands = arena.hands_done()
    moves: dict[int, int] = {}
    turns: list[int] = []
    for row, code in zip(np.asarray(rows).tolist(), codes):
        if not code:
            continue
        game, player = int(who[row][0]), int(who[row][1])
        hand = (game, int(hands[game]))
        played.decline.judged += 1
        if code == late_rules.LAST:
            if allowed[row, zoo.MORTAL_PASS]:
                # A claim window, where the pass is always open.
                moves[row] = zoo.MORTAL_PASS
                played.in_window += 1
            else:
                turns.append(row)
                played.at_turn += 1
            played.decline.fire(game, hand)
        elif code == late_rules.NOT_LAST:
            played.decline.keep("not_last", hand)
        else:
            raise RuleCheckFailed(f"row {row}: the arena's decline code {code} says nothing of a win")
    if turns:
        # The player's own turn, where a discard always is: the best other
        # move of the very answer the win was drawn from.
        logits = np.asarray(logits_of(np.asarray(turns, dtype=np.int64)), dtype=np.float32)
        for at, row in enumerate(turns):
            rest = np.where(allowed[row], logits[at], -np.inf)
            rest[zoo.MORTAL_AGARI] = -np.inf
            best = int(rest.argmax())
            if not np.isfinite(rest[best]):
                raise RuleCheckFailed(f"row {row}: the win declined at the player's turn leaves no other move open")
            moves[row] = best
    for row, move in moves.items():
        game, player = int(who[row][0]), int(who[row][1])
        played.events.append((game, int(hands[game]), player, DECLINE, int(move)))
    return moves


def dama(arena, who, rows: np.ndarray, tiles: np.ndarray, legal: np.ndarray, played: Played) -> list[int]:
    """The all-last dama rule on the rows of a step whose move is a riichi,
    drawn or left by a declined self-draw, that the engine can answer (see
    `last_hand`): which of them, by their place in `rows`, throw their tile
    undeclared.

    `tiles` are the tiles their reaches drew from their previews, `legal`
    the step's moves in our engine's space. The arena answers the game's
    turn player, a code a tile kind, in South 4 at a non-dealer's own move
    with riichi open (`riichi_py.Arena.all_last_dama`), and the code is read
    at the drawn tile, the one the reach would really throw: 1, it keeps the
    player first through every win, and its discard is played instead, which
    must be legal; 2, 3 and 4 (no yaku without riichi, short, dead), the
    reach stands, counted by reason. A row whose codes are all nought was
    not judged; any other code on a judged row is refused, as the ruler
    refuses it."""
    rows, tiles = np.asarray(rows).tolist(), np.asarray(tiles).tolist()
    codes = arena.all_last_dama([int(who[row][0]) for row in rows])
    played.dama_calls += 1
    if len(codes) != len(rows) * KINDS:
        raise RuleCheckFailed(f"the arena answered {len(codes)} dama codes for {len(rows)} rows")
    # The arena's bytes as they come, KINDS a row: a handful of rows a call.
    if not codes.strip(b"\0"):
        return []
    hands = arena.hands_done()
    fired: list[int] = []
    for slot, (row, tile) in enumerate(zip(rows, tiles)):
        row_codes = codes[slot * KINDS:(slot + 1) * KINDS]
        if not row_codes.strip(b"\0"):
            continue
        game, player = int(who[row][0]), int(who[row][1])
        hand = (game, int(hands[game]))
        code = row_codes[tile]
        played.dama.riichi += 1
        if code == all_last.KEEPS_FIRST:
            if not legal[row, zoo.DISCARD + tile]:
                raise RuleCheckFailed(f"row {row}: the discard of tile {tile}, which keeps first, is not open")
            played.dama.fired += 1
            played.dama.tables.add(game)
            played.dama.hands.add(hand)
            played.events.append((game, hand[1], player, DAMA, tile))
            fired.append(slot)
        elif code in all_last.KEPT:
            played.dama.kept[all_last.KEPT[code]] += 1
            if hand in played.dama.hands:
                played.dama.late += 1
        else:
            raise RuleCheckFailed(f"row {row}: the reach's tile {tile} has code {code}, which no legal riichi has: "
                                  "the arena's codes and its legal moves disagree")
    return fired


def check_step(choice: np.ndarray, picked: np.ndarray, legal: np.ndarray, decidable: np.ndarray,
               ruled: np.ndarray, tile_slots: np.ndarray, tile_actions: np.ndarray) -> int:
    """A step's records against what is sent to the table, in our engine's
    `choice` by row: the tile records (`tile_slots`, the rows they belong
    to, and `tile_actions`, the tiles they hold) are exactly the recorded
    rows that play a reach, each throwing the tile its record holds, so no
    tile record of a reach dama left undeclared survives, no reach goes
    unrecorded, and none is recorded with a tile the table is not sent; and
    each row a rule decided (`ruled`) sends the move recorded for it in
    Mortal's moves (`picked`), as `zoo.first_meaning` names it: the pass, a
    discard's tile, the best other move of a declined self-draw. A riichi
    left by a declined self-draw sends a reach with its drawn tile, which
    `first_meaning` cannot name, so it must be a legal reach, and it is a
    tile record's row, whose tile it throws (the first two checks). Answers
    how many reaches the step sends."""
    sending = (choice >= zoo.RIICHI_DISCARD) & (choice < zoo.TSUMO)
    tile_slots = np.asarray(tile_slots, dtype=np.int64)
    reaches = int(np.count_nonzero(sending))
    # As many tile records as reaches sent, each of a distinct recorded row
    # that sends one: so the rows are the same, without listing either.
    if (reaches != len(tile_slots) or not (sending[tile_slots] & decidable[tile_slots]).all()
            or (len(tile_slots) > 1 and not (np.diff(tile_slots) > 0).all())):
        raise RuleCheckFailed(f"the step's tile records belong to rows {tile_slots.tolist()} and its reaches are "
                              f"played at rows {np.flatnonzero(sending).tolist()}: a reach's tile record and the "
                              "table disagree")
    if len(tile_slots) or len(tile_actions):
        # And each of those reaches throws the tile its record holds, the
        # one the policy drew at the record's probability.
        thrown = choice[tile_slots] - zoo.RIICHI_DISCARD
        if not np.array_equal(thrown, np.asarray(tile_actions, dtype=np.int64)):
            raise RuleCheckFailed(f"the reaches sent at rows {tile_slots.tolist()} throw tiles {thrown.tolist()} "
                                  f"and their tile records hold {np.asarray(tile_actions).tolist()}: a reach's "
                                  "tile record and the table disagree")
    if not ruled.any():
        return reaches
    flagged = np.flatnonzero(ruled)
    sent = choice[flagged]
    reach = picked[flagged] == zoo.MORTAL_RIICHI
    meant = zoo.first_meaning(picked[flagged], legal[flagged])
    open_reach = (sent >= zoo.RIICHI_DISCARD) & (sent < zoo.TSUMO) & legal[flagged, sent]
    wrong = np.where(reach, ~open_reach, sent != meant)
    if wrong.any():
        at = flagged[wrong]
        raise RuleCheckFailed(f"rows {at.tolist()} were decided by a rule and send {choice[at].tolist()} to the "
                              f"table, which is not their recorded moves {picked[at].tolist()}")
    return reaches


def reconcile(played: Played, flagged: int, reaches: int, tiles: int) -> None:
    """The round's end: the decisions recorded flagged (`flagged`), the
    account's records and the rules' firings, dama's and decline's less the
    rows both fired at, agree; and every reach played at the table kept its
    tile record: the reaches the round sent to the table (`reaches`) and
    the tile records it gathered (`tiles`), each a second record of a row
    already recorded, both counted by the round itself as it steps the table
    and gathers the records (`selfplay.play`), apart from the account, agree
    with each other and with the account's own counts, which `check_step`
    keeps equal step by step. Decline's firings are the claim windows' and
    the turns', and only a declined self-draw that left a reach can be one
    dama then leaves undeclared."""
    fired = played.dama.fired + played.decline.fired - played.both
    problems = []
    if not flagged == played.records == fired:
        problems.append(f"{flagged} decisions were recorded as a rule's, the account counts {played.records}, and "
                        f"the rules fired {fired} times (dama {played.dama.fired}, decline {played.decline.fired}, "
                        f"both {played.both})")
    if not reaches == tiles == played.reaches_sent == played.tile_records:
        problems.append(f"{reaches} reaches were sent to the table and {tiles} tile records gathered, and the "
                        f"account counts {played.reaches_sent} reaches and {played.tile_records} tile records")
    if played.in_window + played.at_turn != played.decline.fired:
        problems.append(f"decline fired {played.decline.fired} times, {played.in_window} in a claim window and "
                        f"{played.at_turn} at a turn")
    if not played.both <= played.to_riichi <= played.at_turn:
        problems.append(f"{played.both} reaches left undeclared after a declined self-draw, {played.to_riichi} "
                        f"declined self-draws left a reach, of {played.at_turn} declined at a turn")
    if problems:
        raise RuleCheckFailed("the round's account of the last hand's rules does not add up: " + "; ".join(problems))
