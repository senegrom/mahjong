"""Which of two networks is better, asked directly.

Measuring each network against the heuristic players and subtracting has a
floor: each figure carries an error near 0.017 over four thousand games,
so the difference carries 0.024, and separating two networks at two
standard errors needs a gap of about 0.05. Four times the games only
halves it. When two networks are close, that comparison never settles, and
the run reaches a point where every further generation is invisible to it.

Sitting them at the same table removes the problem. One network takes one
place and the other takes the remaining three, in the same games, so the
luck of the deal falls on both at once and cancels. What comes back is not
two placements to subtract but one: the challenger's, against the 2.50 it
would average if the two were the same player. Each deal contributes one
figure and the error comes from the deals, as in `arena.py`, and each set
of deals is played four times with the challenger in each seat.

    python -m neural.duel challenger.pt incumbent.pt --games 1000

`--timing` adds where the time went (see `Split`) and changes nothing
else; `neural.ruler` plays the same games in one arena, faster.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .outcomes import paired_summary, placements, require_finished, validate_budget, win_shares

import riichi_py

from .training_safety import TRAINING_API_VERSION, require_training_engine

from . import zoo
from .observe import Views

ACTIONS = riichi_py.ACTIONS
SEATS = 4


class TimedViews(Views):
    """`Views` that keep an account of the time spent encoding: `here`, on
    the thread that plays, waits for a worker's encoding included, and
    `aside`, on the worker (see `Views.prepare`), where it overlaps the
    playing thread's work rather than adding to it. What they encode is
    exactly what `Views` encodes."""

    def __init__(self, arena, games: int, kinds: set[str] | None = None, *, renamed=None) -> None:
        super().__init__(arena, games, kinds, renamed=renamed)
        self.here = 0.0
        self.aside = 0.0
        self._playing = threading.get_ident()
        self._lock = threading.Lock()

    def _count(self, seconds: float) -> None:
        with self._lock:
            if threading.get_ident() == self._playing:
                self.here += seconds
            else:
                self.aside += seconds

    def _encode(self, who, after_reach: bool = False, renamed: bool = False):
        began = time.perf_counter()
        try:
            return super()._encode(who, after_reach, renamed)
        finally:
            self._count(time.perf_counter() - began)

    def wait(self) -> None:
        began = time.perf_counter()
        try:
            super().wait()
        finally:
            self._count(time.perf_counter() - began)


class Split:
    """Where a table's wall time goes, when a caller asks (`timing`, a
    dict, else nothing is timed): `engine`, the arena's own calls;
    `follow`, the follower fed what the arena did; `encode`, the views
    encoded on the playing thread, waiting for a worker included;
    `challengers` and `incumbent`, the networks' asks less whatever of
    them was encoding; and `aside`, what a worker encoded meanwhile, which
    adds nothing to the wall. A part timed here hands the encoding done
    inside it to `encode`, so the parts add up to the time timed."""

    PARTS = ("engine", "follow", "encode", "challengers", "incumbent")

    def __init__(self, timing: dict | None) -> None:
        self.timing = timing
        self.views = None
        if timing is not None:
            for part in (*self.PARTS, "aside"):
                timing.setdefault(part, 0.0)
            timing.setdefault("decisions", {"challengers": 0, "incumbent": 0})

    @contextlib.contextmanager
    def __call__(self, part: str, rows: int = 0):
        if self.timing is None:
            yield
            return
        began = time.perf_counter()
        encoded = self.views.here
        aside = self.views.aside
        try:
            yield
        finally:
            spent = time.perf_counter() - began
            encoding = self.views.here - encoded
            self.timing["encode"] += encoding
            self.timing[part] += spent - encoding
            self.timing["aside"] += self.views.aside - aside
            if rows:
                self.timing["decisions"][part] += rows

    def views_for(self, arena, games: int, kinds: set[str], renamed=None):
        """The views a table timed this way is served from, and from now
        on the ones whose encoding it counts, the renamed game's included
        (see `Views`): `Views` itself when nothing is timed."""
        self.views = (TimedViews if self.timing is not None else Views)(arena, games, kinds, renamed=renamed)
        return self.views


@torch.no_grad()
def table(
    challenger,
    incumbent,
    games: int,
    seed: int,
    place: int,
    device: str = "cuda",
    max_steps: int = 4000,
    timing: dict | None = None,
) -> np.ndarray:
    """Plays `games` games with the challenger at `place` and the incumbent
    at the other three, and returns the final scores.

    Both networks play their best move rather than sampling, which is what
    the browser does and what the comparison is about. With `timing`, a
    dict, adds where the time went to it (see `Split`).
    """
    require_training_engine()
    validate_budget(games, max_steps)
    if type(place) is not int or not 0 <= place < 4:
        raise ValueError("place must be a player index from zero to three")
    challenger.eval()
    incumbent.eval()
    clock = Split(timing)
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    # Each network is served the planes it sees, so the two may be of
    # different lineages.
    views = clock.views_for(arena, games, {challenger.kind, incumbent.kind})
    steps = 0
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        with clock("engine"):
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            live = seats != 0xFF
        if not live.any():
            break
        with clock("follow"):
            views.advance()

        with clock("engine"):
            mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
            mask = mask.reshape(games, ACTIONS).astype(bool)
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)

        # Who owes each live game's decision, as a person rather than a
        # seat: the seats move between hands and the players do not.
        index = np.nonzero(live)[0]
        owner = np.array([players[game][seats[game]] for game in index])
        choice = np.zeros(games, dtype=np.int64)

        for player, wanted, part in ((challenger, owner == place, "challengers"),
                                     (incumbent, owner != place, "incumbent")):
            rows = index[wanted]
            if not len(rows):
                continue
            with clock(part, len(rows)):
                choice[rows] = zoo.choose(player, views, rows, owner[wanted], mask[rows], device)

        with clock("engine"):
            arena.step(choice.tolist())

    require_finished(arena, steps=steps, context="duel")
    if timing is not None:
        timing["steps"] = timing.get("steps", 0) + steps
    return np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()


def duel(challenger, incumbent, games: int, seed: int, device: str = "cuda",
         max_steps: int = 4000, timing: dict | None = None) -> dict:
    """The same deals four times, with the challenger in each seat. With
    `timing`, a dict, the report says where the time went, as `play`."""
    per_seat = []
    per_deal = []
    began = time.perf_counter()
    for place in range(SEATS):
        scores = table(challenger, incumbent, games, seed, place, device=device, max_steps=max_steps,
                       timing=timing)
        got = placements(scores)[:, place]
        per_seat.append(
            {
                "place": place,
                "placement": float(got.mean()),
                "score": float(scores[:, place].mean()),
                "wins": float(win_shares(scores)[:, place].mean()),
            }
        )
        per_deal.append(got.astype(float))

    overall = float(statistics.fmean(row["placement"] for row in per_seat))
    paired, error = paired_summary(per_deal)
    result = {
        "training_api_version": TRAINING_API_VERSION,
        "seed": seed,
        "games_per_seat": games,
        "games_total": games * SEATS,
        "placement": overall,
        "standard_error": error,
        "by_seat": per_seat,
        "by_deal": paired.tolist(),
    }
    if timing is not None:
        result["play"] = play_record(timing, time.perf_counter() - began, path="duel")
    return result


def play_record(timing: dict, seconds: float, **how) -> dict:
    """What a timed duel or ruler says of its play: how it played (`how`),
    its wall time, the parts of it (see `Split`) with what no part claimed
    as `other`, the steps and the decisions asked of each side, and the
    rates that compare one way of playing with another."""
    parts = {part: round(timing.get(part, 0.0), 3) for part in Split.PARTS}
    parts["other"] = round(seconds - sum(timing.get(part, 0.0) for part in Split.PARTS), 3)
    decisions = dict(timing.get("decisions", {}))
    asked = sum(decisions.values())
    return {
        **how,
        "seconds": round(seconds, 3),
        "parts": parts,
        "aside": round(timing.get("aside", 0.0), 3),
        "steps": timing.get("steps", 0),
        "decisions": decisions,
        "decisions_per_second": round(asked / seconds, 1) if seconds > 0 else None,
    }


def verdict(result: dict) -> str:
    """Two identical players average 2.50 at the same table, so that is the
    line, and the challenger is below it when it is the better network."""
    edge = 2.5 - result["placement"]
    error = result["standard_error"]
    if error == 0:
        return "not enough deals to say"
    sigmas = edge / error
    size = f"{edge:+.4f} placement, {sigmas:+.1f} standard errors"
    if sigmas > 2:
        return f"the challenger is stronger: {size}"
    if sigmas < -2:
        return f"the incumbent is stronger: {size}"
    return f"not settled either way, which needs more deals: {size}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenger", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--games", type=int, default=1000, help="deals per seating")
    parser.add_argument("--seed", type=int, default=555_000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--timing", action="store_true",
                        help="add where the time went to the report, as `play`; the games are the same")
    args = parser.parse_args()

    challenger = zoo.load_player(args.challenger, args.device)
    incumbent = zoo.load_player(args.incumbent, args.device)
    result = duel(challenger, incumbent, args.games, args.seed, device=args.device, max_steps=args.max_steps,
                  timing={} if args.timing else None)
    result["challenger"] = str(args.challenger)
    result["incumbent"] = str(args.incumbent)
    result["verdict"] = verdict(result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
