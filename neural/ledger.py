"""What each decision turned out to be worth.

A decision's value is not known when it is made. The hand it belongs to
pays it when the hand ends, and the game pays every decision of that player
when the game ends. Keeping that account is fiddly in exactly the way that
does not show: a hand whose ending goes unnoticed does not lose its points
loudly, it pays them to the decisions of the hand after, and every total
still adds up. That fault lived in the collector for a long time and was
found only by making the engine count hands too.

So the account is kept in one place, and the engine's own count is the
check on it. Anything that records decisions and wants to know what they
were worth uses this rather than writing the bookkeeping again.
"""

from __future__ import annotations

import numpy as np

import riichi_py

#: What a hand moved, brought to about the size of the placement term so
#: neither drowns the other. A big hand is worth a few tenths.
HAND_SCALE = 1.0 / 4000.0

#: What the game itself was worth, by the place it ended in.
PLACEMENT_VALUE = tuple(riichi_py.PLACEMENT_VALUE)

#: Which objective these rewards are. A round carries it so that a trainer
#: can refuse one built against a different one rather than mixing two
#: scales in a replay with nothing to say which is which.
#:
#: 1 — hand points over four thousand, plus the placement bonus.
REWARD_VERSION = 1


class Ledger:
    """The decisions waiting on a hand, and what they are owed.

    `open(game, person)` notes a decision and returns its row. `settle` is
    called after every step the arena takes — every one, including the
    steps on which only foreign opponents decided, because a hand can end
    on any of them and the engine forgets it did as soon as the next step
    is taken. `close` pays the placements and checks the account.
    """

    def __init__(self, games: int) -> None:
        self.games = games
        self.rewards: list[float] = []
        #: Decisions still waiting on the hand they were made in.
        self.waiting: list[list[list[int]]] = [
            [[] for _ in range(4)] for _ in range(games)
        ]
        #: Every decision a person made in a game, for the placement.
        self.everything: list[list[list[int]]] = [
            [[] for _ in range(4)] for _ in range(games)
        ]
        self.hands = 0

    def open(self, game: int, person: int) -> int:
        """Notes a decision and returns the row it will be paid into."""
        row = len(self.rewards)
        self.rewards.append(0.0)
        self.waiting[game][person].append(row)
        self.everything[game][person].append(row)
        return row

    def settle(self, arena) -> int:
        """Pays every hand that ended on the step just taken."""
        ended = np.frombuffer(arena.hand_ended(), dtype=np.uint8)
        if not ended.any():
            return 0
        results = np.frombuffer(arena.hand_result(), dtype=np.int32).reshape(self.games, 4)
        counted = 0
        for game in np.nonzero(ended)[0]:
            counted += 1
            for person in range(4):
                value = float(results[game][person]) * HAND_SCALE
                for row in self.waiting[game][person]:
                    self.rewards[row] += value
                self.waiting[game][person] = []
        self.hands += counted
        return counted

    def close(self, arena) -> np.ndarray:
        """Pays the placements, checks the account, and returns the rewards.

        Two checks, both of faults that are otherwise silent. Nothing may
        still be waiting: every game is over, so every hand ended, and an
        entry left here is a hand whose ending was missed. And the engine's
        own count of hands must match this one's, because a missed ending
        leaves nothing waiting either -- its decisions were paid, once
        each, by the hand after.
        """
        stranded = sum(
            len(self.waiting[game][person])
            for game in range(self.games)
            for person in range(4)
        )
        if stranded:
            raise RuntimeError(
                f"{stranded} decisions were still waiting on a hand that had already "
                "ended; a hand's end was missed and its points would have been "
                "credited to the wrong hand"
            )
        truly = int(np.asarray(arena.hands_done(), dtype=np.int64).sum())
        if truly != self.hands:
            raise RuntimeError(
                f"the engine finished {truly} hands and the collector noticed "
                f"{self.hands}; {truly - self.hands} endings went by unseen, so their "
                "points reached the decisions of the hand that followed"
            )

        # Tied players share the rewards of the places they occupy, the
        # convention `neural.outcomes` fixes for training and evaluation
        # alike and the engine's own search settles worlds by.
        from .outcomes import placement_rewards

        scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(self.games, 4)
        bonuses = placement_rewards(scores, PLACEMENT_VALUE)
        for game in range(self.games):
            for person in range(4):
                value = float(bonuses[game, person])
                for row in self.everything[game][person]:
                    self.rewards[row] += value
        return np.asarray(self.rewards, dtype=np.float32)


def refuse_if_unfinished(arena, games: int, steps: int, what: str) -> None:
    """Final scores mean nothing until the games are over.

    A loop bounded by a step limit can stop with tables frozen mid-hand,
    and their scores look exactly like results. The limit guards against a
    hand that will not end, so reaching it is a fault to report rather than
    a budget to spend.
    """
    if arena.all_finished():
        return
    unfinished = int((np.frombuffer(arena.seats(), dtype=np.uint8) != 0xFF).sum())
    raise RuntimeError(
        f"{what} stopped after {steps} steps with {unfinished} of {games} games "
        "unfinished; the result would not be a result, so it is refused"
    )
