"""Can the critic tell one candidate discard from another?

This is the question the search turns on, and it is not the question a
value loss answers. A critic can predict returns respectably across
positions and still be useless at the only comparison a search makes: among
the moves available from *one* position, which is best. The differences
between siblings are far smaller than the differences between positions, so
an estimator can look good on the second while saying nothing about the
first.

`neural.counterfactual` measured what the critic knows about positions the
policy avoids, and found it loses only about three points of explained
spread out there — real, but far too little to explain a search that costs
0.19 placement. That left scale as the suspect: an error of about 1.33 in
the units of the return, used to resolve differences a fraction of that
size.

So this measures the differences themselves, by playing them.

Every seat is the network, playing its best move, so the games are
determined by the seed and two passes agree exactly until they are made to
differ. A root decision is chosen; one pass plays the policy's own choice
there and the others force a candidate; and each pass then runs to the end
of the game. What comes back is what that discard was really worth, on that
deal, against opponents that respond to it the way the ones it will meet
do — which is the part a rollout heuristic gets wrong, since a bot that
under-calls and under-rons makes every dangerous discard look safe.

Paired on the deal, so the luck cancels and what is left was caused by the
discard. Three things are reported:

- **the scale of the differences**, which says how fine an instrument the
  search needs;
- **how much the policy's own choice was really worth** over an
  alternative, which is the prize a search is competing for: it cannot win
  more than the policy is leaving on the table;
- **the noise on a single comparison**, which is what the search has to see
  that prize through.

The alternatives here are other legal moves, not the policy's second and
third preferences, so the prize measured is the prize against an arbitrary
move. A search only ever reorders the moves the policy already likes, and
those sit closer together than that — so the figure below is a ceiling on
the headroom, not an estimate of it.

    python -m neural.siblings <checkpoint> [games=128] [root=40] [candidates=4]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import riichi_py

from . import zoo
from .observe import Views
from .selfplay import HAND_SCALE, PLACEMENT_VALUE

SEATS = 4
ACTIONS = riichi_py.ACTIONS


@dataclass
class Root:
    """One decision that several moves were tried from."""

    game: int
    #: Whose decision it was, as a person rather than a seat.
    person: int
    #: The moves tried. The first is the policy's own choice, so every
    #: difference below is measured against what would have happened.
    candidates: list[int]


@torch.no_grad()
def pass_through(
    net,
    games: int,
    seed: int,
    device: str,
    root_step: int,
    force: dict[int, int] | None = None,
    watch: dict[int, int] | None = None,
) -> tuple[dict, dict]:
    """Plays every seat with the network, forcing one move where told.

    `force` names, per game, the action to take at `root_step` instead of
    the policy's. `watch` names the person whose earnings to follow. Both
    passes are greedy, so with the same seed they are the same game up to
    the step that is made to differ.
    """
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    arena.strict = True
    views = Views(arena, games, {net.kind})

    at_root: dict[int, dict] = {}
    earned = np.zeros(games, dtype=np.float64)
    banked = np.zeros(games, dtype=bool)
    steps = 0
    while not arena.all_finished() and steps < 4000:
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        views.advance()
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, ACTIONS)
        legal = legal.astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)
        deciding = np.where(live, players[np.arange(games), np.minimum(seats, 3)], 0)
        deciding = deciding.astype(np.int64)
        rows = np.nonzero(live)[0]

        choice = np.zeros(games, dtype=np.int64)
        choice[rows] = net.choose(views, rows, deciding[rows], legal[rows])

        if steps == root_step:
            for game in rows:
                at_root[int(game)] = {
                    "person": int(deciding[game]),
                    "legal": legal[game].copy(),
                    "chose": int(choice[game]),
                }
        if force and steps == root_step:
            for game, action in force.items():
                if game in at_root:
                    choice[game] = action

        arena.step(choice.tolist())

        # The hand containing the root is the one whose points that
        # decision moved; later hands are the game's own drift.
        if steps >= root_step:
            ended = np.frombuffer(arena.hand_ended(), dtype=np.uint8).astype(bool)
            if ended.any():
                results = np.frombuffer(arena.hand_result(), dtype=np.int32).reshape(games, 4)
                for game in np.nonzero(ended)[0]:
                    who = (watch or {}).get(int(game), at_root.get(int(game), {}).get("person"))
                    if who is not None and not banked[game]:
                        earned[game] = float(results[game][who]) * HAND_SCALE
                        banked[game] = True
        steps += 1

    if not arena.all_finished():
        raise RuntimeError("a pass stopped before its games were over")

    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS)
    places = (-scores).argsort(axis=1).argsort(axis=1)
    worth = {}
    for game in range(games):
        who = (watch or {}).get(game, at_root.get(game, {}).get("person"))
        if who is None:
            continue
        worth[game] = earned[game] + PLACEMENT_VALUE[int(places[game][who])]
    return at_root, worth


@torch.no_grad()
def measure(net, games: int, seed: int, root_step: int, candidates: int, device: str) -> dict:
    """Plays each candidate from the same root and scores the critic."""
    base_root, _base_worth = pass_through(net, games, seed, device, root_step)
    if not base_root:
        raise SystemExit(f"no game owed a decision at step {root_step}; try another root")

    watch = {game: row["person"] for game, row in base_root.items()}

    # The policy's own order over the legal moves at each root, and what
    # the critic makes of them, asked once before anything is played.
    roots: list[Root] = []
    for game, row in base_root.items():
        allowed = np.nonzero(row["legal"])[0]
        if allowed.size < 2:
            continue
        shortlist = [row["chose"]] + [int(a) for a in allowed if int(a) != row["chose"]]
        roots.append(
            Root(game=game, person=row["person"], candidates=shortlist[:candidates])
        )
    if not roots:
        raise SystemExit("no root offered a choice")

    paid: dict[int, list[float]] = {root.game: [] for root in roots}
    for slot in range(candidates):
        force = {
            root.game: root.candidates[slot]
            for root in roots
            if slot < len(root.candidates)
        }
        if not force:
            continue
        _seen, worth = pass_through(
            net, games, seed, device, root_step, force=force, watch=watch
        )
        for root in roots:
            if slot < len(root.candidates):
                paid[root.game].append(float(worth.get(root.game, 0.0)))

    # What separates the candidates, and how surely.
    #
    # The statistic has to be one chosen before the returns are seen. The
    # best of several realised returns is not: taking a maximum over noisy
    # samples is biased upwards by the noise alone, and would flatter a
    # rule that picked at random. So what is reported is the *paired*
    # difference between the policy's own choice, which is always the first
    # candidate, and each alternative, averaged over roots. That is
    # unbiased, and its error comes from the roots.
    differences: list[float] = []
    spreads: list[float] = []
    kept: list[float] = []
    for root in roots:
        got = paid[root.game]
        if len(got) < 2:
            continue
        kept.append(got[0])
        spreads.append(float(np.std(got)))
        differences.extend(got[0] - other for other in got[1:])

    differences = np.asarray(differences, dtype=np.float64)
    error = (
        float(differences.std(ddof=1) / np.sqrt(differences.size))
        if differences.size > 1
        else 0.0
    )
    advantage = float(differences.mean()) if differences.size else 0.0
    return {
        "games": games,
        "roots": len(roots),
        "root_step": root_step,
        "candidates": candidates,
        # How much the policy's own choice really beat an arbitrary other
        # legal move, one decision's worth. This is the prize a search is
        # competing for: it cannot win more than the policy is leaving on
        # the table, and it has to see the difference through the noise
        # below to win any of it.
        "the policy's choice over an alternative": {
            "mean": round(advantage, 4),
            "error": round(error, 4),
            "errors_from_zero": round(advantage / error, 2) if error else 0.0,
            "pairs": int(differences.size),
        },
        # The spread of a single realised difference. A search comparing
        # two candidates on one deal is reading a number with about this
        # much noise on it, which is what its margin is measuring.
        "noise on one deal": {
            "spread_of_differences": round(float(differences.std(ddof=1)), 4)
            if differences.size > 1
            else 0.0,
            "spread_within_a_root": round(float(np.mean(spreads)), 4) if spreads else 0.0,
        },
        "what the policy's choice paid": round(float(np.mean(kept)), 4) if kept else 0.0,
    }


def main() -> None:
    checkpoint = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    root_step = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    candidates = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 424_242
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = zoo.load_player(checkpoint, device)
    net.eval()
    report = measure(net, games, seed, root_step, candidates, device)
    report["checkpoint"] = str(checkpoint)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
