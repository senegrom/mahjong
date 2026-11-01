"""Late-game rules: exact rules for the last hand that a ruler can seat a
challenger with, against the same network plain (`neural.ruler --late-rule`).

The plateau diagnosis of 30 October found the policy settled: 81% of its
decisions with more than one move open sit above 0.999, and policy gradient
does not move a settled habit. An exact rule, measured by a paired ruler,
can. Three habits the probes found in the last hand, South 4, where there is
no extension round and no agari-yame, so a win by anyone but the dealer ends
the game and the places are then the points, were measured as rules on 31
October. The one kept is the rule here; the other two, wide dama and dealer
noten, were dropped and their code removed (docs/RULER.md keeps what they
were and what they measured):

- `decline-last`, a win that leaves the player last. Given the 14,269
  questions 810 was asked in its games against Club players and against
  published Mortal, which the screen of 15 October kept (see
  `neural.suits`), 2820 declined none of the 233 wins among them, 62 of
  which came in South 3 or 4; and in a round of 8,536 learner seat-games it
  took 219 South 4 wins that left it alone in last (2.6%). Where a
  non-dealer's best move in South 4 is a win, and the win, settled as the
  engine settles it, bets and counters included, leaves them strictly below
  every other player, it is declined: a ron is passed, and a self-draw gives
  way to the best other move of the same answer. A non-dealer's win ends the
  game, so where it is the only win, declining it can place them no lower.
  The rule asks only what the table shows. A riichi win's ura dora are
  turned only once it is declared, so the win is read at the most they could
  add (the most ura dora the winning hand could take), and declined only
  where even that leaves the player alone in last. And another player may
  win on the same discard, unseen, and then declining can cost (a dealer who
  wins too keeps the deal, and a second non-dealer has the discarder pay
  twice, which can lift the player off last). Rare, and not modelled.

Which wins qualify is the engine's to say (`riichi_py.Arena.all_last_decline`,
from engine/riichi-core/src/all_last.rs), each worked out on a copy of the
hand with the engine's own legal moves, scoring and settlement, so the rule
cannot drift from the rules the game is played by. The arena is asked only
about the rows whose best move is a win.

A rule edits only the first question's answers, and only the rows where it
fires: for a declined ron the pass becomes the row's only finite answer, and
for a declined self-draw the win is struck from the row and the rest stand,
so `zoo.choose_in_mortal_space` plays that move and no follower is told
anything the table does not do. As the all-last dama rule does, a rule needs
the reach's tile asked with the reach (`preview_reach`), so that a reach is
never told before the rule has decided, and fixed questions (`neural.ruler
--questions fixed`), so that what changes at a table where it fires changes
no answer at any other: it is refused without either. It is the player's
outermost question, around an average over the suits if there is one, and a
player follows one rule of the last hand: never the all-last dama rule and a
late rule.

It is off unless asked for: only a ruler seats a player that follows one.
Self-play decides through `Combined.decide`, never `_ask`, so a rule reaches
training only through `selfplay_rules`, which plays the learner's own drawn
wins by `decline-last` when the trainer is told (`train_combined
--last-hand-rules`); the other two never. What a rule did is kept as the
player's `late_rule` (`Account`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import riichi_py

from . import all_last, zoo

#: The rules, by the names `neural.ruler --late-rule` takes.
RULES = ("decline-last",)
#: The decline codes (`riichi_core::all_last::Decline`), one a row: the win
#: leaves the player alone in last, and it does not; 0 is not asked.
LAST, NOT_LAST = 1, 2
#: Why each rule let the move stand.
KEPT = {"decline-last": ("not_last",)}


def require_late_engine() -> None:
    """The rule asks the engine for its codes, which it answers from
    `ALL_LAST_API_VERSION` 2, and as they are read here, a riichi win's
    decline code read without its hidden ura dora, from 3."""
    if getattr(riichi_py, "ALL_LAST_API_VERSION", 0) < 3:
        raise RuntimeError("Rebuild and reinstall riichi_py: the late rule asks the arena for its codes, which needs "
                           "riichi_py.Arena.all_last_decline as ALL_LAST_API_VERSION 3 answers it: a riichi win's "
                           "decline code read without its hidden ura dora")


@dataclass
class Account:
    """What a player that follows a late rule has done since it was seated:
    `rule`, which; `judged`, the decisions the engine judged, those whose
    best move was the rule's kind (see the module's notes) where the rule's
    question is put; `fired`, those at which the rule
    changed the move; the arena's tables, and the hands by table and how
    many hands it had finished, it fired at; `kept`, why it let the move
    stand at the others; and `later`, the moves it let stand in a hand it
    had fired in before (for `decline-last`, a win taken)."""

    rule: str
    judged: int = 0
    fired: int = 0
    tables: set[int] = field(default_factory=set)
    hands: set[tuple[int, int]] = field(default_factory=set)
    kept: dict[str, int] = field(default_factory=dict)
    later: int = 0

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"{self.rule!r} is no late rule: the late rules are {', '.join(RULES)}")
        self.kept = {**dict.fromkeys(KEPT[self.rule], 0), **self.kept}

    def fire(self, game: int, hand: tuple[int, int]) -> None:
        self.fired += 1
        self.tables.add(game)
        self.hands.add(hand)

    def keep(self, why: str, hand: tuple[int, int]) -> None:
        self.kept[why] += 1
        if hand in self.hands:
            self.later += 1

    def report(self) -> dict:
        """As a ruler's report says it: the tables it fired at by their
        index in the arena, and the hands as a count."""
        return {"rule": self.rule, "judged": self.judged, "fired": self.fired, "tables": sorted(self.tables),
                "hands": len(self.hands), "kept": dict(self.kept), "kept_after_firing": self.later}


def account_of(player) -> Account | None:
    """What `player` has done by its late rule, or None if it follows none."""
    return getattr(player, "late_rule", None)


def named(who, rows) -> list[tuple[int, int]]:
    """The rows of a question the arena is asked about, each a game and a
    player by place, as plain integers."""
    return [(int(who[row][0]), int(who[row][1])) for row in rows]


def arena_of(views, rule: str):
    arena = getattr(views, "arena", None)
    if arena is None:
        raise RuntimeError(f"these views name no arena: the {rule} rule is a player's outermost question, asked of "
                           "a table's own views")
    return arena


def decline_last(ask, account: Account):
    """`ask(views, who, fresh, allowed, ahead=())`, a player's question in
    Mortal's moves (see `zoo.choose_in_mortal_space`), with its answers
    edited by the decline rule: where a row of the first question has a win
    as its best move and the engine says the win would leave the player
    alone in last (see the module's notes), the win is declined: in a claim
    window the pass becomes the row's only finite answer, and at the
    player's own turn the win is struck from the row, which leaves the best
    other move of the same answer. The rest is answered as `ask` answers it,
    and a question asked afresh is left as it is."""

    def decline_a_win_that_leaves_it_last(views, who, fresh, allowed, ahead=()):
        values, masks = ask(views, who, fresh, allowed, ahead)
        if fresh:
            return values, masks
        n = len(who)
        allowed = np.asarray(allowed, dtype=bool)
        best = np.where(allowed[:n], values[:n], -np.inf).argmax(axis=1)
        rows = np.flatnonzero(best == zoo.MORTAL_AGARI)
        if not len(rows):
            return values, masks
        arena = arena_of(views, account.rule)
        codes = np.frombuffer(arena.all_last_decline(named(who, rows)), dtype=np.uint8)
        asked = codes != 0
        if not asked.any():
            return values, masks
        hands = arena.hands_done()
        for row, code in zip(rows[asked], codes[asked]):
            game = int(who[row][0])
            hand = (game, int(hands[game]))
            account.judged += 1
            if code == LAST:
                if allowed[row, zoo.MORTAL_PASS]:
                    # A claim window, where the pass is always open.
                    values[row] = -np.inf
                    values[row, zoo.MORTAL_PASS] = 0.0
                else:
                    # The player's own turn, where a discard always is.
                    values[row, zoo.MORTAL_AGARI] = -np.inf
                account.fire(game, hand)
            elif code == NOT_LAST:
                account.keep("not_last", hand)
            else:
                raise RuntimeError(f"row {row}: the arena's decline code {int(code)} says nothing of a win")
        return values, masks

    return decline_a_win_that_leaves_it_last


#: Each rule's question, by its name.
WRAPPERS = {"decline-last": decline_last}


def follow(player, rule: str, fixed: bool = False):
    """`player`, from now on following the late rule `rule` (see the module's
    notes); what it does is kept as `late_rule`. Only a player that answers
    in Mortal's moves can, asked in fixed rows, which its seating says with
    `fixed`, with the reach's tile asked with the reach (see
    `neural.ruler.seat`), following no other rule of the last hand, and the
    rule must be the last wrapped around its question. Told, not looked for,
    as `all_last.dama` is told."""
    from . import combined

    require_late_engine()
    if rule not in RULES:
        raise ValueError(f"{rule!r} is no late rule: the late rules are {', '.join(RULES)}")
    if not isinstance(player, (combined.Combined, zoo.MortalSpacePlayer, zoo.MortalPlayer)):
        raise ValueError(f"{type(player).__name__} answers nothing in Mortal's moves: only a joined player, a Mortal "
                         "or one of ours in Mortal's moves can follow a late rule")
    if account_of(player) is not None or all_last.account_of(player) is not None:
        raise ValueError("the player already follows a rule of the last hand, and follows one at most")
    if not fixed:
        raise ValueError(f"the {rule} rule needs the questions fixed (neural.ruler --questions fixed): asked by "
                         "seating, a firing at one table changes the batches every other is asked in")
    if not getattr(player, "preview_reach", False):
        raise ValueError(f"the {rule} rule needs the reach's tile asked with the reach (--preview-reach), so that "
                         "nothing is told before the rule decides")
    account = Account(rule)
    player._ask = WRAPPERS[rule](player._ask, account)
    player.late_rule = account
    return player
