"""Cases of the last hand's two rules for web/tests/last-hand-rules-parity.mjs,
which holds the page's Trained AI (web/src/lib/trained-move.js) to the Python
wrappers its rules are written from.

Each case is one decision: our engine's legal moves, the answers a network
gives its first question and the reach's preview, each in the game and in the
renamed game (see `neural.suits`), and the engine's codes for it. The
reference composes the real wrappers in the training's order, decline-last
around the average over the suits and the all-last dama rule around that,

    all_last.ruled(late_rules.decline_last(suits.averaged(answers, suits.Averaging()),
                                           late_rules.Account("decline-last")), all_last.Account())

and plays its move through `zoo.choose_in_mortal_space(..., preview_reach=True)`,
greedy, with `skip_forced` off, as the page's opponents, adviser and hand's
notes ask, and on, as the game's review asks. Each case keeps our move, the
page's moves in Mortal's space, the rules that fired, in order, or the rule
that refused codes no legal move can have, and the questions the page must
ask, the renamed game's only where the plain answer hesitates.

The cases: hand-written ones, one for each branch of the rules, the gate read
before a win is struck and a reach with one tile among them; random ones,
made as the learner's parity tests make theirs (`random_rows`,
`random_question`), and more with a sure win, the commonest decline; and
decisions with codes from two arenas the heuristic player plays, each paired
with made-up answers. A case is
passed over where a plain answer's top weight is within `MARGIN` of the gate,
or where a question's two best moves, the first question's with the win
struck too, are within `MARGIN` of each other in log space: the page works
the mean out in double precision from the worker's weights, and Python in
single precision from the logits, and may round a near tie apart. In log
space, a win above 0.999 struck from a row whose other moves hesitate is not
such a tie. Each kind of firing must be met often enough (`MINIMUM`), so that
a fixture that exercises nothing fails.

The learner is played by the rules too (the `rules` of
`mortal_learner.decide_in_mortal_space`, `neural.selfplay_rules`), so every
case is played a second way: on the plain answers, the learner never
averaging, by its greedy play with both rules, against the composed wrappers
with the average off; the moves, both rules' firings and the reaches dama
left undeclared after a declined self-draw must agree.

Run by CI's worker job, which publishing waits on:
`python -m neural.tests.export_rules_fixture <folder>`. Written for Python
3.12, CI's.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import riichi_py
import torch

from neural import all_last, late_rules, mortal_learner, selfplay_rules, suits, zoo
from neural.observe import Planes

#: The random cases' seed, and how many of them are kept.
SEED = 20261103
RANDOM_CASES = 300
#: How many random decisions with a win open are kept besides, each with its
#: win pushed `SURE` logits above the other moves, above 0.999 as most
#: decisions are, the others as drawn: a sure win struck from moves that
#: hesitate among themselves, the commonest decline of all.
SURE_WINS = 100
SURE = 12.0
#: How far a renamed game's answer strays from the game's answer renamed, a
#: logit's standard deviation: enough that the mean changes some moves.
RENAMED_NOISE = 0.8
#: The arenas whose decisions with codes are recorded: the seeds the rules'
#: own arena tests play (`neural.tests.test_ruler.SEED`, 9,100,000, plus 7
#: and plus 103), 48 games each.
ARENAS = (9_100_000 + 7, 9_100_000 + 103)
ARENA_GAMES = 48
#: The nearest a plain top weight may be to the gate, and a question's two
#: best moves to each other in log space, for a case to be kept.
MARGIN = 1e-3
#: The rules' names, as the page, the ruler and the learner give them.
DAMA_RULE, DECLINE_RULE = "all-last-dama", "decline-last"
#: The questions the page asks at a decision, in the order it asks them.
QUESTIONS = ("first", "first renamed", "preview", "preview renamed")
#: The fewest cases of each kind each setting must play, about half of what
#: the seeds give, and every hand-written chain and refusal.
MINIMUM = {
    "dama fired": 25,
    "decline in a claim window": 25,
    "decline at a turn": 25,
    "decline at a turn, the win above 0.999": 12,
    "decline, then dama": 3,
    "a reach with one tile": 50,
    "refused": 4,
    "renamed game asked": 200,
}

KINDS = all_last.KINDS
MOVES = zoo.MORTAL_ACTIONS
REACH, AGARI, PASS = zoo.MORTAL_RIICHI, zoo.MORTAL_AGARI, zoo.MORTAL_PASS
#: The renamed game's views: a question asked of them is answered by the
#: renamed game's logits.
RENAMED = SimpleNamespace(arena=None)


@dataclass
class Case:
    """One decision: our engine's legal moves (`riichi_py.ACTIONS` of them),
    the player deciding, the logits answered to the first question and to
    the reach's preview, each in the game's move order and in the renamed
    game's, and the codes: the turn player's dama codes, a byte a tile kind,
    and the deciding player's decline code. `mask` is the first question's
    moves in Mortal's space, `after` the preview's, the tiles a reach may
    throw. `want`, for a hand-written case, is the move it was written for:
    Mortal's moves, or the rule that refuses it."""

    name: str
    legal: np.ndarray
    player: int
    first: np.ndarray
    renamed_first: np.ndarray
    preview: np.ndarray
    renamed_preview: np.ndarray
    dama: np.ndarray
    decline: int
    want: object = None

    def __post_init__(self) -> None:
        self.legal = np.asarray(self.legal, dtype=bool)
        for name in ("first", "renamed_first", "preview", "renamed_preview"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        self.dama = np.asarray(self.dama, dtype=np.uint8)
        self.decline = int(self.decline)
        self.mask = zoo.translatable(self.legal[None])[0]
        assert self.mask.any(), f"{self.name}: no move our engine allows has a meaning in Mortal's space"
        self.after = np.zeros(MOVES, dtype=bool)
        self.after[:KINDS] = self.legal[zoo.RIICHI_DISCARD:zoo.TSUMO]


class Codes:
    """An arena as the rules read it, for game 0's decision: the turn
    player's dama codes and the deciding player's decline code, as the
    engine hands them over."""

    def __init__(self, dama: np.ndarray, decline: int) -> None:
        self.dama = bytes(np.asarray(dama, dtype=np.uint8))
        self.decline = int(decline)

    def all_last_dama(self, games):
        return self.dama * len(list(games))

    def all_last_decline(self, rows):
        return bytes([self.decline] * len(list(rows)))

    def hands_done(self):
        return [0]


def answers_of(case: Case):
    """The network's answers to `case`, as a player's question gives them
    (`zoo.choose_in_mortal_space`): the first question's logits for `who`,
    the preview's for `ahead`, the renamed game's when asked of its views, a
    move not allowed at minus infinity."""

    def answers(views, who, fresh, allowed, ahead=()):
        assert not fresh, "a reach is never told: its tile is previewed"
        renamed = views is RENAMED
        rows = [case.renamed_first if renamed else case.first for _ in who]
        rows += [case.renamed_preview if renamed else case.preview for _ in ahead]
        values = np.stack(rows) if rows else np.zeros((0, MOVES), dtype=np.float32)
        return np.where(allowed, values, -np.inf).astype(np.float32), np.zeros(values.shape, dtype=bool)

    return answers


def choose(case: Case, skip_forced: bool, gate: float = suits.GATE, dama_too: bool = True):
    """Our engine's move at `case` by the composed wrappers, as a ruler's
    player with `preview_reach` plays it, and the two rules' accounts; the
    average over the suits off at a gate of nought, and the all-last dama
    rule left out without `dama_too`."""
    views = SimpleNamespace(arena=Codes(case.dama, case.decline), renamed=RENAMED)
    dama, decline = all_last.Account(), late_rules.Account(DECLINE_RULE)
    ask = late_rules.decline_last(suits.averaged(answers_of(case), suits.Averaging(gate)), decline)
    if dama_too:
        ask = all_last.ruled(ask, dama)
    choice = zoo.choose_in_mortal_space(lambda who, fresh, allowed, ahead=(): ask(views, who, fresh, allowed, ahead),
                                        views, np.array([0]), np.array([case.player]), case.legal[None],
                                        skip_forced=skip_forced, preview_reach=True)
    return int(choice[0]), dama, decline


def mortal_moves(choice: int) -> list[int]:
    """Our engine's move as the page plays it in Mortal's moves: a discard's
    tile, a riichi and its tile, or the one move of Mortal's that means it."""
    if choice < zoo.RIICHI_DISCARD:
        return [choice - zoo.DISCARD]
    if choice < zoo.TSUMO:
        return [REACH, choice - zoo.RIICHI_DISCARD]
    return [int(np.flatnonzero(zoo.MEANS[:, choice])[0])]


def hesitates(logits: np.ndarray, allowed: np.ndarray) -> bool:
    """Whether a question's plain answer hesitates: its top weight below the
    gate, as `suits.averaged` reads it. A question with nothing open does
    not."""
    return bool(suits.top_weight(np.where(allowed, logits, -np.inf)[None])[0] < suits.GATE)


def played(case: Case, skip_forced: bool) -> dict:
    """What the page must do at `case` with `skip_forced` as given: our move,
    its moves in Mortal's space and the rules that fired, in order; or the
    rule that refused, which is decline's where its wrapper alone refuses
    the codes, else dama's; and the questions it asks."""
    try:
        choice, dama, decline = choose(case, skip_forced)
    except RuntimeError:
        try:
            choose(case, skip_forced, dama_too=False)
        except RuntimeError:
            refused = DECLINE_RULE
        else:
            refused = DAMA_RULE
        result = {"move": None, "moves": None, "rules": [], "refused": refused}
    else:
        rules = [DECLINE_RULE] * decline.fired + [DAMA_RULE] * dama.fired
        result = {"move": choice, "moves": mortal_moves(choice), "rules": rules, "refused": None}
    asks = []
    if not (skip_forced and case.mask.sum() == 1):
        asks.append("first")
        if hesitates(case.first, case.mask):
            asks.append("first renamed")
    # The reach's tile is asked where the move is a riichi once decline has
    # spoken: declared, thrown undeclared, or refused by dama.
    reached = (result["refused"] == DAMA_RULE or DAMA_RULE in result["rules"]
               or (result["moves"] is not None and result["moves"][0] == REACH))
    if reached and not (skip_forced and case.after.sum() == 1):
        asks.append("preview")
        if hesitates(case.preview, case.after):
            asks.append("preview renamed")
    result["asks"] = asks
    return result


def gap(values: np.ndarray) -> float:
    """How far apart a question's two best open moves are, in log space."""
    best = np.sort(values[np.isfinite(values)])[::-1]
    return float(best[0] - best[1]) if len(best) > 1 else np.inf


def near_tie(case: Case) -> str | None:
    """Why `case` is passed over, or None: a plain answer's top weight within
    `MARGIN` of the gate, or two best moves within `MARGIN` of each other in
    log space, of the plain logits or, where the plain answer hesitates, the
    mean of the two namings', at the first question, the first question with
    the win struck where decline may strike it (coded last, at a turn), and
    a preview with a choice of tile."""
    questions = [("first", case.first, case.renamed_first, case.mask)]
    if case.after.sum() > 1:
        questions.append(("preview", case.preview, case.renamed_preview, case.after))
    for name, logits, renamed, allowed in questions:
        plain = np.where(allowed, logits.astype(np.float64), -np.inf)
        top = float(suits.top_weight(plain[None])[0])
        if abs(top - suits.GATE) <= MARGIN:
            return f"the {name} question's top weight, {top}, is at the gate"
        values = plain
        if top < suits.GATE:
            other = np.where(suits.swap_moves(allowed), renamed.astype(np.float64), -np.inf)
            values = (plain + suits.swap_moves(other)) / 2
        if gap(values) < MARGIN:
            return f"the {name} question's two best moves are alike"
        if name == "first" and allowed[AGARI] and case.decline == late_rules.LAST and not allowed[PASS]:
            struck = values.copy()
            struck[AGARI] = -np.inf
            if gap(struck) < MARGIN:
                return "the first question's two best moves after the win are alike"
    return None


def logits(named: dict[int, float]) -> np.ndarray:
    """Logits over Mortal's moves: those named, by move, nought elsewhere."""
    row = np.zeros(MOVES, dtype=np.float32)
    for move, value in named.items():
        row[move] = value
    return row


def codes(named: dict[int, int]) -> np.ndarray:
    """Dama codes, a byte a tile kind: those named, nought elsewhere."""
    row = np.zeros(KINDS, dtype=np.uint8)
    for tile, code in named.items():
        row[tile] = code
    return row


def turn(discards, reaches=(), tsumo: bool = False, kan: bool = False) -> np.ndarray:
    """Our engine's moves open at a player's own turn: the discards of
    `discards`, riichi with `reaches`, and a self-draw and a concealed quad
    where asked."""
    legal = np.zeros(riichi_py.ACTIONS, dtype=bool)
    legal[zoo.DISCARD + np.asarray(discards, dtype=np.int64)] = True
    legal[zoo.RIICHI_DISCARD + np.asarray(reaches, dtype=np.int64)] = True
    legal[zoo.TSUMO] = tsumo
    legal[zoo.CONCEALED_KAN] = kan
    return legal


def claim(*more: int) -> np.ndarray:
    """Our engine's moves open in a claim window: the ron, the pass, and
    `more`."""
    legal = np.zeros(riichi_py.ACTIONS, dtype=bool)
    legal[[zoo.RON, zoo.PASS, *more]] = True
    return legal


def written(name, legal, first, want, preview=None, dama=None, decline=0, renamed=None, renamed_preview=None) -> Case:
    """A hand-written case: logits by Mortal's move, the renamed game's the
    game's renamed unless given, so that the mean changes nothing."""
    first, preview = logits(first), logits(preview or {})
    return Case(name, legal, 1, first, suits.swap_moves(first) if renamed is None else logits(renamed), preview,
                suits.swap_moves(preview) if renamed_preview is None else logits(renamed_preview), codes(dama or {}),
                decline, want)


def hand_written() -> list[Case]:
    """A case for each branch of the rules, as the page's own tests have them
    (web/tests/trained-move.test.js), each with the move it was written for."""
    nw, ks, ny, sh, dd = all_last.NOT_WAITING, all_last.KEEPS_FIRST, all_last.NO_YAKU, all_last.SHORT, all_last.DEAD
    last, not_last = late_rules.LAST, late_rules.NOT_LAST
    cases = [
        written("a discard, though riichi is open", turn([3, 5, 9], [5, 9]), {9: 6}, [9], {5: 3},
                {3: nw, 5: sh, 9: ks}),
        written("a reach whose codes are nought", turn([3, 5, 9], [5, 9]), {REACH: 6}, [REACH, 5], {5: 3}),
        written("dama at the reach's own tile, not the best other move", turn([3, 5, 9, 20], [3, 5, 9]),
                {REACH: 8, 9: 5}, [5], {5: 6, 3: 2}, {3: ks, 5: ks, 9: sh, 20: nw}),
        written("the reach's tile short, though others keep first", turn([3, 5, 9, 20], [3, 5, 9]),
                {REACH: 8, 9: 5}, [REACH, 9], {9: 6}, {3: ks, 5: ks, 9: sh, 20: nw}),
        *[written(f"the reach's tile coded {code}", turn([3, 5, 9], [5, 9]), {REACH: 6}, [REACH, 5], {5: 6},
                  {3: nw, 5: code, 9: ks}) for code in (ny, sh, dd)],
        written("refused: nought at the reach's tile among codes", turn([3, 5, 9], [5, 9]), {REACH: 6}, DAMA_RULE,
                {5: 6}, {3: nw, 9: ks}),
        written("refused: the reach's tile coded not waiting", turn([3, 5, 9], [5, 9]), {REACH: 6}, DAMA_RULE,
                {5: 6}, {3: nw, 5: nw, 9: ks}),
        written("refused: the tile that keeps first cannot be thrown plain", turn([3, 9], [5, 9]), {REACH: 6},
                DAMA_RULE, {5: 6}, {5: ks, 9: sh}),
        written("a ron declined, a pon second", claim(zoo.PON), {AGARI: 8, zoo.MORTAL_PON: 5}, [PASS],
                decline=last),
        written("a ron declined, a chi second", claim(zoo.CHII_HIGH), {AGARI: 8, zoo.MORTAL_CHI_LOW: 5}, [PASS],
                decline=last),
        written("a robbed quad's ron declined", claim(), {AGARI: 8}, [PASS], decline=last),
        written("a self-draw declined", turn([3, 9], tsumo=True), {AGARI: 8, 9: 3, 3: 1}, [9], decline=last),
        written("a self-draw declined for a quad", turn([3], tsumo=True, kan=True),
                {AGARI: 8, zoo.MORTAL_KAN: 3, 3: 1}, [zoo.MORTAL_KAN], decline=last),
        written("a self-draw above 0.999 declined, the rest hesitant", turn([3, 9, 20], tsumo=True),
                {AGARI: 14, 9: 2, 3: 1.7}, [9], decline=last),
        written("a self-draw kept, not last", turn([3, 9], tsumo=True), {AGARI: 8}, [AGARI], decline=not_last),
        written("a ron kept, not asked", claim(zoo.PON), {AGARI: 8}, [AGARI]),
        written("refused: a decline code that says nothing of a win", turn([3, 9], tsumo=True), {AGARI: 8},
                DECLINE_RULE, decline=3),
        # The win at 0.95 is sure, so nothing is renamed; in the renamed
        # game's mean the 3m would rank above the riichi after the strike.
        written("the gate read before the strike", turn([2], [2], tsumo=True),
                {AGARI: np.log(0.95), REACH: np.log(0.03), 2: np.log(0.02)}, [REACH, 2], {2: 0}, decline=last,
                renamed={AGARI: np.log(0.95), REACH: np.log(0.01), 11: np.log(0.04)}),
        written("decline, then dama", turn([3, 5, 9], [5, 9], tsumo=True), {AGARI: 30, REACH: 20, 9: 4}, [5],
                {5: 1}, {3: nw, 5: ks, 9: sh}, last),
        written("decline, then a reach that stands", turn([3, 5, 9], [5, 9], tsumo=True),
                {AGARI: 30, REACH: 20, 9: 4}, [REACH, 9], {9: 1}, {3: nw, 5: ks, 9: sh}, last),
        written("decline, then dama on a reach with one tile", turn([3, 5], [5], tsumo=True),
                {AGARI: 30, REACH: 20, 3: 4}, [5], {5: 0}, {3: nw, 5: ks}, last),
        written("decline, then dama at the tile the mean throws", turn([0, 9], [0, 9], tsumo=True),
                {AGARI: 30, REACH: 20}, [9], {0: 0.4}, {0: sh, 9: ks}, last, renamed_preview={0: 1.5}),
        written("a reach with one tile, which keeps first", turn([3, 5], [5]), {REACH: 6}, [5], {5: 0},
                {3: nw, 5: ks}),
        written("a reach with one tile, short", turn([3, 5], [5]), {REACH: 6}, [REACH, 5], {5: 0}, {3: nw, 5: sh}),
        written("a reach with one tile, codes nought", turn([3, 5], [5]), {REACH: 6}, [REACH, 5], {5: 0}),
        written("one move open", turn([7]), {}, [7]),
        written("the mean makes a win best", claim(zoo.PON), {zoo.MORTAL_PON: 1, AGARI: 0.8}, [PASS], decline=last,
                renamed={AGARI: 2}),
        written("the mean makes a riichi best", turn([0, 9, 20], [0, 9]), {20: 1, REACH: 0.8}, [9], {9: 4},
                {0: sh, 9: ks, 20: nw}, renamed={REACH: 2}),
        written("the mean changes the reach's tile", turn([0, 9], [0, 9]), {REACH: 6}, [REACH, 9], {0: 0.4},
                {0: ks, 9: sh}, renamed_preview={0: 1.5}),
        written("the mean changes the move after a declined self-draw", turn([0, 9], tsumo=True),
                {AGARI: 2, 0: 0.5}, [9], decline=last, renamed={AGARI: 2, 0: 1.5, 9: -1}),
    ]
    return cases


def random_rows(rng, n: int) -> np.ndarray:
    """`n` rows of our engine's moves, each a claim window with a ron or a
    turn, with discards, often reaches among them, often a self-draw and now
    and then a quad. As neural/tests/test_selfplay_rules.py makes them."""
    legal = np.zeros((n, riichi_py.ACTIONS), dtype=bool)
    for row in range(n):
        if rng.random() < 0.35:
            legal[row, [zoo.RON, zoo.PASS]] = True
            legal[row, zoo.PON] = rng.random() < 0.5
            continue
        discards = rng.choice(34, size=int(rng.integers(2, 8)), replace=False)
        legal[row, discards] = True
        if rng.random() < 0.7:
            reaches = rng.choice(discards, size=int(rng.integers(1, len(discards) + 1)), replace=False)
            legal[row, zoo.RIICHI_DISCARD + reaches] = True
        legal[row, zoo.TSUMO] = rng.random() < 0.5
        legal[row, zoo.CONCEALED_KAN] = rng.random() < 0.2
    return legal


def random_question(rng, legal: np.ndarray):
    """Values for each row and each possible preview, a win and a riichi
    often best, and codes for every reach's tiles and every win. As
    neural/tests/test_selfplay_rules.py makes them."""
    n = len(legal)
    first = rng.normal(size=(n, zoo.MORTAL_ACTIONS)).astype(np.float32)
    first[:, zoo.MORTAL_AGARI] += 2.0 * (rng.random(n) < 0.6)
    first[:, zoo.MORTAL_RIICHI] += 2.0 * (rng.random(n) < 0.6)
    previews = {row: rng.normal(size=zoo.MORTAL_ACTIONS).astype(np.float32) for row in range(n)}
    players = rng.integers(0, 4, size=n)
    dama = {row: {int(tile): int(rng.choice([all_last.KEEPS_FIRST, all_last.NO_YAKU, all_last.SHORT, all_last.DEAD]))
                  for tile in np.flatnonzero(legal[row, zoo.RIICHI_DISCARD:zoo.TSUMO])} for row in range(n)}
    decline = {(row, int(players[row])): int(rng.choice([0, late_rules.LAST, late_rules.NOT_LAST])) for row in range(n)}
    return first, previews, players, dama, decline


def renamed(rng, values: np.ndarray) -> np.ndarray:
    """The renamed game's answer: the game's renamed, astray by
    `RENAMED_NOISE`."""
    return (suits.swap_moves(values) + rng.normal(scale=RENAMED_NOISE, size=MOVES)).astype(np.float32)


def random_cases(rng, wanted: int, sure: bool = False) -> list[Case]:
    """`wanted` random decisions, near ties passed over; with `sure`, only
    those with a win open, the win pushed `SURE` logits above the rest."""
    found: list[Case] = []
    while len(found) < wanted:
        n = int(rng.integers(1, 7))
        legal = random_rows(rng, n)
        first, previews, players, dama, decline = random_question(rng, legal)
        for row in range(n):
            if sure:
                rest = np.where(zoo.translatable(legal[row][None])[0], first[row], -np.inf)
                if not np.isfinite(rest[AGARI]):
                    continue
                rest[AGARI] = -np.inf
                first[row, AGARI] = rest.max() + SURE
            player = int(players[row])
            case = Case(f"{'sure win' if sure else 'random'} {len(found)}", legal[row], player, first[row],
                        renamed(rng, first[row]), previews[row], renamed(rng, previews[row]), codes(dama[row]),
                        decline[(row, player)])
            if len(found) < wanted and near_tie(case) is None:
                found.append(case)
    return found


def recorded(seed: int) -> list[tuple[np.ndarray, int, np.ndarray, int]]:
    """The decisions with codes of `ARENA_GAMES` games the heuristic player
    plays at every seat: each one's legal moves, the player deciding, the
    turn player's dama codes and the deciding player's decline code."""
    arena = riichi_py.Arena(games=ARENA_GAMES, seed=seed)
    found = []
    while not arena.all_finished():
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(ARENA_GAMES, riichi_py.ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(ARENA_GAMES, 4)
        rows = [(int(game), int(players[game, seats[game]])) for game in np.flatnonzero(seats != 0xFF)]
        dama = np.frombuffer(arena.all_last_dama([game for game, _player in rows]),
                             dtype=np.uint8).reshape(len(rows), KINDS)
        decline = np.frombuffer(arena.all_last_decline(rows), dtype=np.uint8)
        for at, (game, player) in enumerate(rows):
            if dama[at].any() or decline[at]:
                found.append((legal[game].copy(), player, dama[at].copy(), int(decline[at])))
        moves = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
        moves[moves == 0xFF] = 0
        arena.step(moves.tolist())
    return found


def arena_cases(rng) -> list[Case]:
    """Each recorded decision with made-up answers, near ties passed over:
    riichi best; a win best; a win best with riichi next; and a win above
    0.999 whose other moves hesitate."""
    found: list[Case] = []
    for seed in ARENAS:
        for number, (legal, player, dama, decline) in enumerate(recorded(seed)):
            allowed = zoo.translatable(legal[None])[0]
            kinds = []
            if allowed[REACH]:
                kinds.append("riichi best")
            if allowed[AGARI]:
                kinds += ["a win best", "a win above 0.999"]
                if allowed[REACH]:
                    kinds.append("a win best with riichi next")
            for kind in kinds:
                first = rng.normal(size=MOVES).astype(np.float32)
                top = float(first[allowed].max())
                if kind == "riichi best":
                    first[REACH] = top + rng.uniform(0.5, 4.0)
                elif kind == "a win best":
                    first[AGARI] = top + rng.uniform(0.5, 4.0)
                elif kind == "a win best with riichi next":
                    first[REACH] = top + rng.uniform(0.5, 2.0)
                    first[AGARI] = first[REACH] + rng.uniform(0.5, 4.0)
                else:
                    rest = np.flatnonzero(allowed)
                    rest = rest[rest != AGARI]
                    if len(rest) < 2:
                        continue
                    ranked = rest[np.argsort(-first[rest], kind="stable")]
                    first[ranked[1]] = first[ranked[0]] - rng.uniform(0.05, 0.5)
                    first[AGARI] = first[ranked[0]] + 12.0
                preview = rng.normal(size=MOVES).astype(np.float32)
                case = Case(f"arena {seed}, decision {number}: {kind}", legal, player, first, renamed(rng, first),
                            preview, renamed(rng, preview), dama, decline)
                if near_tie(case) is None:
                    found.append(case)
    return found


def counted(cases: list[Case], results: list[dict]) -> dict[str, int]:
    """How many of `cases` played each kind `MINIMUM` names, in one setting."""
    counts = dict.fromkeys(MINIMUM, 0)
    for case, result in zip(cases, results):
        rules, moves = result["rules"], result["moves"] or [None]
        window, declined = bool(case.mask[PASS]), DECLINE_RULE in rules
        reached = result["refused"] == DAMA_RULE or DAMA_RULE in rules or moves[0] == REACH
        plain = np.where(case.mask, case.first.astype(np.float64), -np.inf)
        counts["dama fired"] += DAMA_RULE in rules
        counts["decline in a claim window"] += declined and window
        counts["decline at a turn"] += declined and not window
        counts["decline at a turn, the win above 0.999"] += (declined and not window
                                                              and float(suits.top_weight(plain[None])[0]) > 0.999)
        counts["decline, then dama"] += rules == [DECLINE_RULE, DAMA_RULE]
        counts["a reach with one tile"] += bool(reached and case.after.sum() == 1)
        counts["refused"] += result["refused"] is not None
        counts["renamed game asked"] += "first renamed" in result["asks"] or "preview renamed" in result["asks"]
    return {name: int(count) for name, count in counts.items()}


class Followed:
    """Views as the learner's deciding reads them, of `arena`: empty planes
    for whoever is asked, as they stand or with a reach previewed, and a
    follower that may be told nothing."""

    def __init__(self, arena) -> None:
        self.arena = arena
        self.observer = SimpleNamespace(follower=self)

    def tell(self, game, player, line):
        raise AssertionError("a reach is never told: its tile is previewed")

    def sparse_and_masks(self, rows, players, fresh=False, after_reach=False):
        n = len(rows)
        return (Planes(np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float16)),
                np.ones((n, MOVES), dtype=bool))


def learned(case: Case):
    """`case` played by the learner's greedy deciding with both rules, on the
    plain answers: our engine's move, or None where it refused, and the
    rules' account."""

    def score(_planes, mask):
        allowed = mask.numpy()
        rows = np.stack([case.first] + [case.preview] * (len(allowed) - 1))
        return torch.from_numpy(np.where(allowed, rows, -np.inf).astype(np.float32))

    account = selfplay_rules.Played((selfplay_rules.DAMA, selfplay_rules.DECLINE))
    try:
        choice, _records = mortal_learner.decide_in_mortal_space(
            score, Followed(Codes(case.dama, case.decline)), np.array([0]), np.array([case.player]),
            case.legal[None], greedy=True, device="cpu", timing={}, preview_reach=True, rules=account)
    except RuntimeError:
        return None, account
    return int(choice[0]), account


def compared(cases: list[Case]) -> dict[str, int]:
    """Every case played by the learner's greedy deciding and by the composed
    wrappers with the average off, on the same plain answers and codes: the
    moves, both rules' firings and the reaches dama left undeclared after a
    declined self-draw agree, and so do the refusals. What was compared."""
    assert (selfplay_rules.DAMA, selfplay_rules.DECLINE) == (DAMA_RULE, DECLINE_RULE)
    counts = {"cases": 0, "dama fired": 0, "decline fired": 0, "both": 0, "refused": 0}
    for case in cases:
        got, account = learned(case)
        try:
            want, dama, decline = choose(case, skip_forced=False, gate=0.0)
        except RuntimeError:
            assert got is None, f"{case.name}: the wrappers refuse the codes, and the learner plays {got}"
            counts["refused"] += 1
        else:
            assert got == want, f"{case.name}: the learner plays {got}, the wrappers {want}"
            fired = (account.dama.fired, account.decline.fired, account.both)
            assert fired == (dama.fired, decline.fired, int(dama.fired and decline.fired)), (
                f"{case.name}: the learner's firings (dama, decline, both) {fired}, the wrappers' "
                f"{(dama.fired, decline.fired)}")
            counts["dama fired"] += dama.fired
            counts["decline fired"] += decline.fired
            counts["both"] += account.both
        counts["cases"] += 1
    return counts


def floats(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def record(case: Case) -> dict:
    """A case as the page's runner reads it: Mortal's moves for each mask,
    each question's logits in its own game's move order, the codes, our move
    for each of Mortal's open moves and each riichi tile, and what each
    setting must play, `skip_forced` off then on."""
    meaning = np.full(MOVES, -1, dtype=np.int64)
    for move in np.flatnonzero(case.mask):
        if move != REACH:
            meaning[move] = zoo.first_meaning(np.array([move]), case.legal[None])[0]
    riichi = np.where(case.legal[zoo.RIICHI_DISCARD:zoo.TSUMO], zoo.RIICHI_DISCARD + np.arange(KINDS), -1)
    return {
        "name": case.name,
        "mask": case.mask.astype(int).tolist(), "after_mask": case.after.astype(int).tolist(),
        "logits": floats(case.first), "renamed_logits": floats(case.renamed_first),
        "preview": floats(case.preview), "renamed_preview": floats(case.renamed_preview),
        "hesitant": [hesitates(case.first, case.mask), hesitates(case.preview, case.after)],
        "dama": case.dama.astype(int).tolist(), "decline": case.decline,
        "meaning": meaning.tolist(), "riichi": riichi.astype(int).tolist(),
        "expected": [played(case, skip_forced) for skip_forced in (False, True)],
    }


def main(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    late_rules.require_late_engine()
    rng = np.random.default_rng(SEED)
    written_cases = hand_written()
    for case in written_cases:
        assert near_tie(case) is None, f"{case.name}: written as a near tie, {near_tie(case)}"
        for skip_forced in (False, True):
            result = played(case, skip_forced)
            got = result["refused"] or result["moves"]
            assert got == case.want, f"{case.name}, skip_forced={skip_forced}: plays {got}, written for {case.want}"
    cases = written_cases + random_cases(rng, RANDOM_CASES) + random_cases(rng, SURE_WINS, sure=True)
    cases += arena_cases(rng)
    records = [record(case) for case in cases]
    counts = [counted(cases, [entry["expected"][setting] for entry in records]) for setting in (0, 1)]
    for setting, count in zip(("skip_forced off", "skip_forced on"), counts):
        short = {name: count[name] for name, least in MINIMUM.items() if count[name] < least}
        assert not short, f"{setting}: too few cases of {short}, against {MINIMUM}; all: {count}"
    (folder / "rules.json").write_text(json.dumps({
        "codes": {
            "dama": {name: getattr(all_last, name) for name in
                     ("ABSENT", "KEEPS_FIRST", "NO_YAKU", "SHORT", "DEAD", "NOT_WAITING")},
            "decline": {"ABSENT": 0, "LAST": late_rules.LAST, "NOT_LAST": late_rules.NOT_LAST},
        },
        "gate": suits.GATE, "moves": suits.MOVES.tolist(),
        "mortal": {"reach": REACH, "agari": AGARI, "pass": PASS},
        "rules": {"dama": DAMA_RULE, "decline": DECLINE_RULE},
        "questions": list(QUESTIONS), "minimum": MINIMUM, "counts": counts,
        "cases": records,
    }, allow_nan=False))
    arenas = len(cases) - len(written_cases) - RANDOM_CASES - SURE_WINS
    print(f"Created {len(cases)} cases of the last hand's rules ({len(written_cases)} written by hand, "
          f"{RANDOM_CASES} random, {SURE_WINS} random with a sure win, {arenas} from arenas), each played by the "
          f"composed Python wrappers with skip_forced off and on: {counts[0]} and {counts[1]}")
    print(f"The learner's greedy play by both rules agrees with the composed wrappers, the average off: "
          f"{compared(cases)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
