"""Does one ply of search beat the policy, with the belief we have now?

The last time this was asked the answer was no, and the diagnosis was that
the search was confidently wrong rather than noisy: the critic could tell
the candidates apart, it just told them apart wrongly. Both the belief and
the critic have been trained since, and the belief measurably so, so the
question is open again.

What it does, for one chair, at every decision where the network's first
choice is a plain discard: deal `worlds` worlds from the belief, make each
of the top `candidates` discards in each world, let the club heuristic move
the other seats to the next position this seat must decide from, value
every one of those positions with the critic, and keep the network's first
choice unless another beats it by `margin` standard errors of the
world-by-world difference. Every other decision, and every other seat,
plays the network's first choice, so the only thing that differs between
the two arms is whether that choice was checked.

The positions are valued through Mortal's planes, which is why this needs
the bridge: an imagined world has no event log, so a copy of the seat's
real state is advanced by the events the world invented.

    python search-test.py <checkpoint> [games=200] [worlds=32] [candidates=4]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

import riichi_py
from libriichi.follow import Imagined

from . import zoo
from .observe import Planes, Views

VERSION = 4
SEATS = 4
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
HANDS = riichi_py.HANDS
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES
MORTAL_ACTIONS = 46
MORTAL_RED = {34: 4, 35: 13, 36: 22}


def is_discard(action: int) -> bool:
    """Whether one of Mortal's moves is a plain discard of ours."""
    return action < 34 or action in MORTAL_RED


@torch.no_grad()
def value_leaves(net, lines, players, follower, device, chunk=4096):
    """The critic's worth of every leaf, through Mortal's planes.

    Each leaf is a copy of the seat's real state advanced by the events its
    world invented. A leaf with no events is one nothing can value this
    way — it settled, it broke, or its world dealt a new hand — and is
    given zero, which the engine ignores for the slots it does not want.
    """
    live = [at for at, events in enumerate(lines) if events]
    valued = np.zeros(len(lines), dtype=np.float32)
    if not live:
        return valued
    copies = Imagined.from_follower(follower, [(players[at][0], players[at][1]) for at in live], VERSION)
    copies.feed([lines[at] for at in live])
    indptr, indices, values, masks = copies.encode()
    for start in range(0, len(live), chunk):
        rows = slice(start, min(start + chunk, len(live)))
        piece = Planes.from_follower(
            indptr[rows.start : rows.stop + 1] - indptr[rows.start],
            indices[indptr[rows.start] : indptr[rows.stop]],
            values[indptr[rows.start] : indptr[rows.stop]],
        )
        mask = torch.from_numpy(np.asarray(masks[rows], dtype=bool)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            _logits, value, _guessed = net.everything(piece.dense(device), mask)
        valued[live[rows.start : rows.stop]] = value.float().cpu().numpy()
    return valued


@torch.no_grad()
def play(net, games, seed, searcher, worlds, candidates, margin, device):
    """Plays `games` out and returns the final scores. `searcher` is the
    place that thinks ahead, or None for nobody."""
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    arena.strict = True
    views = Views(arena, games, {net.kind})
    follower = views.observer.follower
    asked = changed = 0
    steps = 0
    while not arena.all_finished() and steps < 4000:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        views.advance()
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)
        deciding = np.where(live, players[np.arange(games), np.minimum(seats, 3)], 0).astype(np.int64)
        rows = np.nonzero(live)[0]
        views.prepare(rows, deciding[rows])

        planes, mortal_mask = views.sparse_and_masks(rows, deciding[rows])
        allowed = zoo.translatable(legal[rows])
        allowed[~allowed.any(axis=1), zoo.MORTAL_PASS] = True
        mask = torch.from_numpy(allowed).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits, _value, guessed = net.everything(planes.dense(device), mask)
        order = torch.argsort(logits.float(), dim=1, descending=True).cpu().numpy()
        belief = torch.softmax(guessed.float(), dim=2).reshape(len(rows), HANDS).cpu().numpy()

        # The network's first choice, in our engine's actions.
        first = zoo.first_meaning(order[:, 0], legal[rows])
        first = np.where(first >= 0, first, legal[rows].argmax(axis=1)).astype(np.int64)
        choice = np.zeros(games, dtype=np.int64)
        choice[rows] = first

        thinking = []
        if searcher is not None:
            for at, game in enumerate(rows):
                seat = int(seats[game])
                if int(players[game][seat]) != searcher:
                    continue
                if not is_discard(int(order[at, 0])):
                    continue
                thinking.append(at)

        if thinking:
            ranked = [[int(choice[game])] for game in range(games)]
            for at in thinking:
                game = int(rows[at])
                shortlist = []
                for rank in range(MORTAL_ACTIONS):
                    action = int(order[at, rank])
                    if not is_discard(action):
                        continue
                    ours = int(zoo.first_meaning(np.array([action]), legal[game : game + 1])[0])
                    if ours >= 0 and ours not in shortlist:
                        shortlist.append(ours)
                    if len(shortlist) >= candidates:
                        break
                if len(shortlist) > 1:
                    ranked[game] = shortlist
            flat = np.zeros((games, HANDS), dtype=np.float32)
            flat[rows] = belief
            _hands, counts = arena.imagine(flat.reshape(-1).tolist(), worlds=worlds)
            kept = [list(range(count)) for count in counts]
            weights = [[1.0 / count] * count if count else [] for count in counts]
            _planes, leaf_counts, _settled, _wanted = arena.leaves_from(
                ranked, kept, weights, candidates=candidates, hurried=True
            )
            total = sum(leaf_counts)
            if total:
                leaf_players, leaf_lines = arena.leaves_mjai()
                # Which game each leaf belongs to, so its copy is made from
                # the right table.
                game_of = np.repeat(np.arange(games), leaf_counts)
                pairs = [(int(game_of[at]), int(leaf_players[at])) for at in range(total)]
                valued = value_leaves(net, leaf_lines, pairs, follower, device)
                decided = arena.decide(valued.tolist(), margin, ranked)
                asked += len(thinking)
                changed += sum(
                    1 for at in thinking if int(decided[int(rows[at])]) != int(choice[int(rows[at])])
                )
                choice = np.asarray(decided, dtype=np.int64)

        arena.step(choice.tolist())

    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()
    return scores, (asked, changed)


def placements(scores, place):
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return order[:, place]


def main():
    checkpoint = Path(sys.argv[1])
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    worlds = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    candidates = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    margin = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = zoo.load_player(checkpoint, device)
    net.eval()
    print(f"{checkpoint.name} on {device}: {worlds} worlds, {candidates} candidates, margin {margin}")

    # The arm without search needs no games. All four seats are the same
    # network, so their placements sum to ten on every deal and average to
    # exactly 2.5 whatever happens; running it would spend an hour to
    # rediscover arithmetic. The searching arm is measured against that.
    per_deal = []
    asked = changed = 0
    began = time.perf_counter()
    for chair in range(SEATS):
        scores, tally = play(net, games, 90_210, chair, worlds, candidates, margin, device)
        asked += tally[0]
        changed += tally[1]
        per_deal.append(placements(scores, chair).astype(float))
        print(
            f"  one ply, chair {chair}: {per_deal[-1].mean():.4f} "
            f"({time.perf_counter() - began:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    paired = np.stack(per_deal).mean(axis=0)
    error = float(paired.std(ddof=1) / (len(paired) ** 0.5))
    edge = 2.5 - paired.mean()
    print(
        f"one ply: placement {paired.mean():.4f} +/- {error:.4f} against 2.5000 "
        f"for the same network not searching, so {edge:+.4f} "
        f"at {edge / error if error else 0:+.1f} standard errors. "
        f"Search changed {changed} of {asked} decisions. "
        f"{games} deals a chair, {time.perf_counter() - began:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
