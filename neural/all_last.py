"""All-last dama: the last hand's riichi left undeclared where its tile
keeps the player first through every win.

South 4 is the last hand of a game: there is no extension round and no
agari-yame, so a win by anyone but the dealer ends the game, and the places
are then the points. A non-dealer for whom riichi would add no win, and
whose every win without it leaves them alone in first, has nothing more to
win from declaring through any win of their own: no place is better than
first, and the reach costs a thousand points and tells the table the hand
is ready. The networks here declare there all the same: in the probe
of 29 October, a non-dealer in South 4 level with the leader or up to
9,000 behind, 810 and 1410 declared riichi at every gap tried. The rule:
where a player's best move in that hand is riichi, and the tile it would
throw with the reach, discarded without declaring, keeps the player first
through every win the hand could then make, that tile is discarded and
nothing is declared. Not declaring also changes how the others play and
leaves the hand free to change, which the rule does not weigh; a ruler
measures the net.

Which wins keep first is the engine's to say (`riichi_py.Arena.all_last_dama`,
from engine/riichi-core/src/all_last.rs): every live wait won by ron off each
of the others, unless the discard leaves the hand furiten, and by tsumo,
each scored by the engine's own scorer and settled by its own settlement,
so the rule cannot drift from the rules the game is played by. The dealer
is never asked: a dealer's win keeps the deal, and South 4 is played again.

Dama means the riichi tile played without the declaration: the tile the
reach itself would have thrown, the best of the legal riichi tiles in the
reach's own answer, asked with the first question (`preview_reach`); or,
where only one tile keeps the hand ready and `skip_forced` left the reach
unasked, that tile. Not the network's best move once riichi is put aside:
that would set a reach with one tile against silence with another, which
the first tile's code says nothing about.

The rule edits only the first question's answers, and only the rows where
it fires: the discard of that tile becomes the row's only finite answer, so
`zoo.choose_in_mortal_space` plays the discard and never enters its reach
branch, and no follower is told a reach the table never makes. It needs
the reach's tile asked with the reach, so that nothing has been told before
the rule decides, and fixed questions (`neural.ruler --questions fixed`), so
that what changes at a table where it fires changes no answer at any other:
it is refused without either. It is the player's outermost question, around
an average over the suits if there is one (see `suits`), whose renamed game
names no arena.

It is off unless asked for: only a ruler seats a player that follows it
(`neural.ruler --all-last-dama`). Self-play decides through
`Combined.decide`, never `_ask`, so the rule reaches training only through
`selfplay_rules`, which plays the learner's own drawn moves by it, at the
tile each reach drew, when the trainer is told (`train_combined
--last-hand-rules`). The site's Trained AI plays it too
(web/src/lib/trained-move.js). What it did is kept as the player's
`all_last` (`Account`): the decisions it was asked about and those it fired
on, the tables and hands it fired at, why it kept the reach elsewhere, and
how often a hand it had fired in declared later all the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import riichi_py

from . import zoo

#: Tile kinds, the width of a game's codes.
KINDS = 34
#: The engine's codes (`riichi_core::all_last::Dama`), one a tile kind: not
#: asked, keeps first, a live wait with no yaku without riichi, a win that
#: leaves another player level or ahead, no live wait, and a tile whose
#: riichi is not legal.
ABSENT, KEEPS_FIRST, NO_YAKU, SHORT, DEAD, NOT_WAITING = range(6)
#: Why the rule kept a reach, by the code of the reach's tile.
KEPT = {NO_YAKU: "no_yaku", SHORT: "short", DEAD: "dead"}


def require_all_last_engine() -> None:
    """The rule asks the engine for the last hand's codes, which it answers
    from `ALL_LAST_API_VERSION` 1."""
    if getattr(riichi_py, "ALL_LAST_API_VERSION", 0) < 1:
        raise RuntimeError("Rebuild and reinstall riichi_py: the all-last dama rule asks the arena which tiles "
                           "keep first, which needs riichi_py.Arena.all_last_dama")


@dataclass
class Account:
    """What a player that follows the rule has done since it was seated:
    `riichi`, the decisions at which its best move was riichi in the last
    hand, the engine's to judge; `fired`, those at which the rule threw the
    reach's tile undeclared; the arena's tables and the hands, by table and
    how many hands it had finished, it fired at; `kept`, why it let the
    reach be declared at the others, by the code of the reach's tile; and
    `late`, the reaches it let be declared in a hand it had fired in
    before, when a later turn's tile no longer kept first."""

    riichi: int = 0
    fired: int = 0
    tables: set[int] = field(default_factory=set)
    hands: set[tuple[int, int]] = field(default_factory=set)
    kept: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KEPT.values(), 0))
    late: int = 0

    def report(self) -> dict:
        """As a ruler's report says it: the tables it fired at by their
        index in the arena, and the hands as a count."""
        return {"riichi": self.riichi, "fired": self.fired, "tables": sorted(self.tables), "hands": len(self.hands),
                "kept": dict(self.kept), "riichi_after_firing": self.late}


def account_of(player) -> Account | None:
    """What `player` has done by the rule, or None if it does not follow it."""
    return getattr(player, "all_last", None)


def reach_tile(row: int, code: np.ndarray, values: np.ndarray, allowed: np.ndarray, preview: int | None) -> int:
    """The tile the reach of `row` would throw, as `zoo.choose_in_mortal_space`
    picks it: the best of the legal riichi tiles in the reach's own answer,
    row `preview` of the question, or, where the reach went unpreviewed
    because only one tile keeps it ready, that tile, the one tile whose
    code says its riichi is legal."""
    if preview is not None:
        return int(np.where(allowed[preview, :KINDS], values[preview, :KINDS], -np.inf).argmax())
    ready = np.flatnonzero((code >= KEEPS_FIRST) & (code <= DEAD))
    if len(ready) != 1:
        raise RuntimeError(f"row {row}'s reach was not previewed, and its codes name {len(ready)} riichi tiles, "
                           "not one: the all-last dama rule needs the reach's tile asked with the reach "
                           "(preview_reach)")
    return int(ready[0])


def ruled(ask, account: Account):
    """`ask(views, who, fresh, allowed, ahead=())`, a player's question in
    Mortal's moves (see `zoo.choose_in_mortal_space`), with its answers
    edited by the rule: where a row of the first question has riichi as its
    best move and the engine says the reach's tile keeps the player first
    (see the module's notes), the discard of that tile becomes the row's only
    finite answer. The rest is answered as `ask` answers it, the reaches'
    previews (the rows past `who`'s, in the order `ahead` names them)
    included, and a question asked afresh after a reach was told is left as
    it is. The arena is asked only about the rows whose best move is riichi,
    and `account` keeps what the rule did."""

    def dama_where_riichi_buys_nothing(views, who, fresh, allowed, ahead=()):
        values, masks = ask(views, who, fresh, allowed, ahead)
        if fresh:
            return values, masks
        n = len(who)
        allowed = np.asarray(allowed, dtype=bool)
        best = np.where(allowed[:n], values[:n], -np.inf).argmax(axis=1)
        rows = np.flatnonzero(best == zoo.MORTAL_RIICHI)
        if not len(rows):
            return values, masks
        arena = getattr(views, "arena", None)
        if arena is None:
            raise RuntimeError("these views name no arena: the all-last dama rule is a player's outermost "
                               "question, asked of a table's own views")
        games = [int(who[row][0]) for row in rows]
        codes = np.frombuffer(arena.all_last_dama(games), dtype=np.uint8).reshape(len(rows), KINDS)
        asked = codes.any(axis=1)
        if not asked.any():
            return values, masks
        previews = {(int(game), int(player)): n + at for at, (game, player) in enumerate(ahead)}
        hands = arena.hands_done()
        for row, code in zip(rows[asked], codes[asked]):
            game, player = int(who[row][0]), int(who[row][1])
            hand = (game, int(hands[game]))
            tile = reach_tile(int(row), code, values, allowed, previews.get((game, player)))
            account.riichi += 1
            if code[tile] == KEEPS_FIRST:
                if not allowed[row, tile]:
                    raise RuntimeError(f"row {row}: the discard of tile {tile}, which keeps first, is not open")
                values[row] = -np.inf
                values[row, tile] = 0.0
                account.fired += 1
                account.tables.add(game)
                account.hands.add(hand)
            elif int(code[tile]) in KEPT:
                account.kept[KEPT[int(code[tile])]] += 1
                if hand in account.hands:
                    account.late += 1
            else:
                raise RuntimeError(f"row {row}: the reach's tile {tile} has code {int(code[tile])}, which no legal "
                                   "riichi has: the arena's codes and its legal moves disagree")
        return values, masks

    return dama_where_riichi_buys_nothing


def dama(player, fixed: bool = False):
    """`player`, from now on leaving undeclared each riichi of the last
    hand whose tile keeps it first (see the module's notes); what it does
    is kept as `all_last`. Only a player that answers in Mortal's moves can,
    asked in fixed rows, which its seating says with `fixed`, with the
    reach's tile asked with the reach (see `neural.ruler.seat`), following
    no late rule of the last hand (see `neural.late_rules`), and the rule
    must be the last wrapped around its question. Told, not looked
    for: run as `python -m neural.ruler`, the ruler is a module twice over,
    and a forward cut into one copy's fixed rows is no instance of the
    other's."""
    from . import combined

    require_all_last_engine()
    if not isinstance(player, (combined.Combined, zoo.MortalSpacePlayer, zoo.MortalPlayer)):
        raise ValueError(f"{type(player).__name__} answers nothing in Mortal's moves: only a joined player, a Mortal "
                         "or one of ours in Mortal's moves can leave a riichi undeclared")
    if account_of(player) is not None:
        raise ValueError("the player already leaves the last hand's riichi undeclared")
    # Looked up by its name, as `late_rules.account_of` looks it up:
    # `late_rules` imports this module.
    if getattr(player, "late_rule", None) is not None:
        raise ValueError("the player already follows a late rule of the last hand, and follows one rule of the last "
                         "hand at most")
    if not fixed:
        raise ValueError("the all-last dama rule needs the questions fixed (neural.ruler --questions fixed): asked "
                         "by seating, a firing at one table changes the batches every other is asked in")
    if not getattr(player, "preview_reach", False):
        raise ValueError("the all-last dama rule needs the reach's tile asked with the reach (--preview-reach), "
                         "so that nothing is told before the rule decides")
    account = Account()
    player._ask = ruled(player._ask, account)
    player.all_last = account
    return player
