"""Can a value head tell one candidate move from another?

The search compares positions that differ by which single tile was
discarded, and takes a move only when its average across imagined worlds
beats the incumbent by two standard errors of the world-by-world
difference. That comparison needs a head that separates near-identical
positions. It does not need a head that predicts the return well in
general, and the two are not the same thing: measured on this run, the
head with the lower error against the return made the search worse, and
the reason may be that it reads only the pooled position, a single vector
averaged across the thirty-four tile kinds, which is exactly the operation
that throws away which tile left the hand.

So this measures the quantity the search actually depends on. For each
decision it builds the same leaves the search would, values them with each
head, and reports two spreads:

  between candidates  how far apart the candidates' averages are
  within a candidate  the standard error of one candidate's average, from
                      the spread across the imagined worlds

Their ratio is the signal the margin rule is looking for, against the noise
it is looking through. Below one, the head cannot see the difference the
search is asking about, however well it predicts the return; the search is
then choosing among numbers that differ mostly by which worlds were drawn.

    python -m neural.discriminate checkpoint.pt --decisions 40
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import riichi_py

from .model import from_payload

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
HANDS = riichi_py.HANDS
HEADS = ("critic", "public", "mean")


@torch.no_grad()
def look(net, games: int, seed: int, warmup: int, worlds: int, candidates: int, device: str):
    """Plays a few turns, then builds one decision's leaves in every game."""
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    for _ in range(warmup):
        if arena.all_finished():
            break
        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        logits, _value = net(
            torch.from_numpy(planes).to(device), torch.from_numpy(mask).to(device)
        )
        arena.step(logits.argmax(dim=1).cpu().numpy().tolist())

    planes = np.frombuffer(arena.observations(), dtype=np.float32)
    planes = planes.reshape(games, PLANES, POSITIONS)
    mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
    mask = mask.reshape(games, ACTIONS).astype(bool)
    logits, _value, guessed = net.everything(
        torch.from_numpy(planes).to(device), torch.from_numpy(mask).to(device)
    )
    order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
    belief = torch.softmax(guessed, dim=2).reshape(games, HANDS).cpu().numpy()
    ranked = [[int(index) for index in order[game][:candidates]] for game in range(games)]
    leaves, counts, settled, wanted = arena.leaves(
        ranked, belief.reshape(-1).tolist(), worlds=worlds, candidates=candidates
    )
    return leaves, counts, np.asarray(settled, dtype=np.float32), worlds, candidates


def spreads(values: np.ndarray, worlds: int, candidates: int) -> tuple[float, float] | None:
    """The spread between candidates and the error within one, for a single
    decision's slots, which are candidate-major."""
    if values.size != worlds * candidates:
        return None
    table = values.reshape(candidates, worlds)
    means = table.mean(axis=1)
    if candidates < 2 or worlds < 2:
        return None
    between = float(means.std(ddof=1))
    # Paired against the first candidate, as the search compares them: the
    # worlds are shared, so their luck cancels and this is the error the
    # margin rule actually uses.
    differences = table[1:] - table[0]
    within = float(np.mean([row.std(ddof=1) / np.sqrt(worlds) for row in differences]))
    return between, within


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--decisions", type=int, default=40, help="games, one decision each")
    parser.add_argument("--worlds", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=24, help="turns played before looking")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    net = from_payload(state, args.device, args.channels, args.blocks)
    net.eval()

    leaves, counts, settled, worlds, candidates = look(
        net,
        games=args.decisions,
        seed=args.seed,
        warmup=args.warmup,
        worlds=args.worlds,
        candidates=args.candidates,
        device=args.device,
    )
    total = sum(counts)
    if total == 0:
        raise SystemExit("no decision produced leaves; try fewer warmup turns")
    positions = np.frombuffer(leaves, dtype=np.float32).reshape(total, PLANES, POSITIONS)

    report = {"checkpoint": str(args.checkpoint), "generation": state.get("generation"),
              "decisions": int(sum(1 for c in counts if c)), "worlds": worlds,
              "candidates": candidates}
    for head in HEADS:
        valued = np.empty(total, dtype=np.float32)
        for start in range(0, total, 4096):
            chunk = torch.from_numpy(positions[start : start + 4096]).to(args.device)
            with torch.no_grad():
                valued[start : start + 4096] = (
                    net.value_only(chunk, head=head).float().cpu().numpy()
                )
        valued = valued + settled
        offset = 0
        between_all, within_all = [], []
        for count in counts:
            if count:
                got = spreads(valued[offset : offset + count], worlds, candidates)
                if got:
                    between_all.append(got[0])
                    within_all.append(got[1])
            offset += count
        between = statistics.fmean(between_all) if between_all else 0.0
        within = statistics.fmean(within_all) if within_all else 0.0
        report[head] = {
            "between_candidates": round(between, 4),
            "within_candidate": round(within, 4),
            "signal_to_noise": round(between / within, 3) if within else None,
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
