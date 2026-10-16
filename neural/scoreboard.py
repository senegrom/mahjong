"""Where every hand of a round began, and which hand each decision was made in.

The placement reward (see `place_model`) credits a decision with what its
hand did to the player's expected finishing place: the expected final bonus
at the start of the next hand, read from the scoreboard alone. Under EMA's
rules nothing else carries from one hand to the next: the four scores, the
counters, the riichi sticks left on the table and who deals (the engine's
`Table`). So the scoreboard at each hand's start is the whole state at a
boundary, and cutting the return there is exact.

The scoreboard is read from the mjai events the arena already hands the
follower every step (`observe.Views.advance`): each hand opens with a
start_kyoku event carrying the round wind, the hand of the round, the
counters, the sticks, the dealer and the four scores, by player. They are
the raw points the table holds, so a score below zero, which EMA plays on
with, is kept; Mortal's planes clamp it to nought.

A round that keeps this record carries a `Ledger`: for every decision, its
game, the hand within the game and the player who made it; for every hand,
where it began; for every game, who sat where and how it ended. A missed
start_kyoku would credit a decision with the wrong hand's boundary, so the
ledger is checked against the engine's own count of hands before anything
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

#: How riichi-core writes a hand's opening event (`mjai::Event::to_json`):
#: the type first. Anything else is not a start_kyoku; a writer that
#: changed the order would leave every hand unread, which `Ledger.check`
#: refuses, so a change cannot pass unnoticed.
START = '{"type":"start_kyoku"'

#: The round winds a game of this engine plays: East, then South, and no
#: extension round (`riichi_core::table`). A hand of any other round has
#: no place in the scoreboard's stages and is refused rather than guessed.
WINDS = {"E": 0, "S": 1}

#: The hands of a game a scoreboard can stand at: East 1 to South 4.
STAGES = 8
STAGE_NAMES = tuple(f"{'East' if stage < 4 else 'South'} {stage % 4 + 1}" for stage in range(STAGES))

#: The columns of a hand's row in `Ledger.starts`: the game, the hand's
#: index within it (0 for the first dealt), the round wind (0 East, 1
#: South), the hand of the round (1 to 4), the counters, the riichi sticks
#: on the table, the dealer (a player, 0 to 3), and the four scores by
#: player.
GAME, HAND, ROUND, KYOKU, HONBA, STICKS, DEALER = range(7)
SCORES = slice(7, 11)
COLUMNS = 11


class HandStarts:
    """Every game's hands as they begin, read from the start_kyoku events
    on their way to the follower.

    `read` takes what `Arena.mjai_all` answered, one list of lines a game,
    and is called once a step, after the arena has moved and before anyone
    is asked for a decision (see `observe.Views.advance`); `count[game]` is
    then how many hands that game has begun, so the hand a decision is made
    in is `count[game] - 1`."""

    def __init__(self, games: int) -> None:
        self.games = games
        self.count = np.zeros(games, dtype=np.int64)
        self.rows: list[list[int]] = []

    def read(self, lines_by_game) -> None:
        for game, lines in enumerate(lines_by_game):
            for line in lines:
                if line.startswith(START):
                    self.begin(game, json.loads(line))

    def begin(self, game: int, event: dict) -> None:
        """A hand of `game` begins as `event`, a start_kyoku, says."""
        wind = WINDS.get(event["bakaze"])
        if wind is None:
            raise ValueError(f"game {game} began a hand in the round of {event['bakaze']!r}; "
                             "this engine plays East and South only")
        scores = [int(score) for score in event["scores"]]
        if len(scores) != 4:
            raise ValueError(f"game {game} began a hand with {len(scores)} scores")
        self.rows.append([game, int(self.count[game]), wind, int(event["kyoku"]), int(event["honba"]),
                          int(event["kyotaku"]), int(event["oya"]), *scores])
        self.count[game] += 1

    def current(self, games: np.ndarray) -> np.ndarray:
        """The hand each of `games` is playing now: the last one begun."""
        return self.count[np.asarray(games, dtype=np.int64)] - 1


@dataclass
class Ledger:
    """Which hand each decision of a round was made in, and the scoreboard
    every hand began at.

    One entry a decision, in the order the round recorded them: `games`, the
    game; `hands`, the hand's index within it, from 0; `players`, the player
    who decided, 0 to 3, fixed for the game while the seat winds move; and
    `points`, the hand's change in that player's score over 4,000, the term
    the default reward credits it with (see `selfplay.HAND_SCALE`).

    One row a hand in `starts` (see `COLUMNS`), ordered by game and then by
    hand, so a game's hand `h` is row `first[game] + h`; `count[game]` is
    how many hands it played. `owners[game, player]` is -1 for the learner
    and otherwise the index of the seated other in `members`, who are named
    there. `final_scores` are the engine's, by player, with the uma in.
    """

    games: np.ndarray
    hands: np.ndarray
    players: np.ndarray
    points: np.ndarray
    starts: np.ndarray
    first: np.ndarray
    count: np.ndarray
    owners: np.ndarray
    members: tuple[str, ...]
    final_scores: np.ndarray

    @classmethod
    def of_round(
        cls,
        begun: HandStarts,
        games: np.ndarray,
        hands: np.ndarray,
        players: np.ndarray,
        points: np.ndarray,
        owners: np.ndarray,
        members: tuple[str, ...],
        final_scores: np.ndarray,
    ) -> Ledger:
        rows = np.asarray(begun.rows, dtype=np.int64).reshape(-1, COLUMNS)
        # Read in the order the games reached them, which interleaves the
        # games; a game's own hands arrive in order.
        order = np.lexsort((rows[:, HAND], rows[:, GAME]))
        rows = rows[order]
        count = np.bincount(rows[:, GAME], minlength=begun.games).astype(np.int64)
        first = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
        return cls(
            games=np.asarray(games, dtype=np.int32),
            hands=np.asarray(hands, dtype=np.int32),
            players=np.asarray(players, dtype=np.int8),
            points=np.asarray(points, dtype=np.float64),
            starts=rows,
            first=first,
            count=count,
            owners=np.asarray(owners, dtype=np.int64),
            members=tuple(members),
            final_scores=np.asarray(final_scores, dtype=np.int32),
        )

    def __len__(self) -> int:
        return len(self.games)

    ARRAYS = ("games", "hands", "players", "points", "starts", "first", "count", "owners", "final_scores")

    def save(self, path) -> None:
        """To one `.npz`, small: a round of 4,096 games is a few megabytes,
        where its planes are gigabytes. U can then be fitted again later
        from rounds kept this way (`python -m neural.place_model fit`)."""
        np.savez_compressed(path, members=np.asarray(json.dumps(list(self.members))),
                            **{name: getattr(self, name) for name in self.ARRAYS})

    @classmethod
    def load(cls, path) -> Ledger:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in cls.ARRAYS}
            members = tuple(json.loads(str(saved["members"])))
        return cls(members=members, **arrays)

    @property
    def stages(self) -> np.ndarray:
        """Each hand's place in the game's schedule, East 1 (0) to South 4
        (7): a hand repeated on a counter keeps its stage."""
        return self.starts[:, ROUND] * 4 + self.starts[:, KYOKU] - 1

    def rows_of(self, games: np.ndarray, hands: np.ndarray) -> np.ndarray:
        """The rows of `starts` holding those games' hands."""
        return self.first[np.asarray(games, dtype=np.int64)] + np.asarray(hands, dtype=np.int64)

    def check(self, hands_done) -> None:
        """Refuses a ledger that cannot be right: a game whose hands begun
        are not the hands the engine finished, a hand out of order, or a
        decision outside its game's hands. Any of these would credit a
        decision with another hand's boundary."""
        done = np.asarray(hands_done, dtype=np.int64)
        if done.shape != self.count.shape or not np.array_equal(done, self.count):
            missing = np.flatnonzero(done != self.count)
            raise RuntimeError(
                f"the start_kyoku events name {int(self.count.sum())} hands and the engine finished "
                f"{int(done.sum())}; games {missing[:8].tolist()} disagree, so a hand's start went unread"
            )
        expected = np.arange(len(self.starts)) - np.repeat(self.first, self.count)
        if not np.array_equal(self.starts[:, HAND], expected):
            raise RuntimeError("a game's hands were not read in the order they were dealt")
        if len(self.games) and (
                self.hands.min() < 0 or np.any(self.hands >= self.count[self.games])):
            raise RuntimeError("a decision was recorded outside the hands its game played")
        if np.any((self.stages < 0) | (self.stages >= STAGES)):
            raise RuntimeError("a hand began outside East 1 to South 4")
