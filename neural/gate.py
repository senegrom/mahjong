"""Conservative, bidirectional checkpoint promotion evidence.

    python -m neural.gate candidate.pt champion.pt --games 512 --seed 920001

Runs candidate-vs-three-champions AND champion-vs-three-candidates in all
four seats. One independent seed, not one correlated seating, is the unit
of evidence. This command never overwrites a checkpoint. `promote` is a gate
result, not an automatic claim of strength against arbitrary opponents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .checkpoints import copy_checkpoint
from .training_safety import TRAINING_API_VERSION, require_training_engine


def assess(forward: list[float], reverse: list[float], *, confidence: float = .95,
           minimum_edge: float = 0., attempt: int = 1, minimum_deals: int = 128) -> dict:
    """One-sided Hoeffding lower bound on the symmetric per-seed advantage.

    Per-deal improvement = (reverse placement - forward placement) / 2,
    in [-1.5, 1.5]. Radius = 3 sqrt(log(1/alpha_i) / (2 n)). Alpha spending
    alpha_i=(1-confidence)/(i(i+1)) sums to 1-confidence across attempts.
    Attempts require fresh held-out seeds and preselected settings. The caller
    owns that ledger; reusing seeds or retrying with attempt=1 voids this claim.
    """
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between zero and one')
    if not math.isfinite(minimum_edge) or not 0 <= minimum_edge < 1.5:
        raise ValueError('minimum_edge must be finite and in [0, 1.5)')
    if type(attempt) is not int or attempt <= 0:
        raise ValueError('attempt must be a positive integer')
    if type(minimum_deals) is not int or minimum_deals < 2:
        raise ValueError('minimum_deals must be an integer of at least two')
    a, b = np.asarray(forward, dtype=np.float64), np.asarray(reverse, dtype=np.float64)
    if (a.ndim != 1 or a.size == 0 or a.shape != b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any((a < 1) | (a > 4)) or np.any((b < 1) | (b > 4))):
        raise ValueError('paired placements must be finite matching vectors in [1, 4]')
    improvement = (b - a) / 2
    alpha = (1 - confidence) / (attempt * (attempt + 1))
    radius = 3 * math.sqrt(math.log(1 / alpha) / (2 * a.size))
    mean = float(improvement.mean())
    lower = max(-1.5, mean - radius)
    enough = a.size >= minimum_deals
    promote = bool(enough and lower > minimum_edge)
    return {'promote': promote, 'reason': 'passed' if promote else
            ('insufficient_deals' if not enough else 'insufficient_evidence'),
            'independent_deals': int(a.size), 'games_total': int(a.size) * 8,
            'mean_placement_improvement': mean, 'lower_bound': lower,
            'confidence_radius': radius, 'confidence': confidence,
            'attempt': attempt, 'attempt_alpha': alpha, 'minimum_edge': minimum_edge,
            'minimum_deals': minimum_deals,
            'by_deal_improvement': improvement.tolist(),
            'method': 'one-sided Hoeffding; paired by seed; summable alpha spending'}


def compare(candidate: Path, champion: Path, *, games: int, seed: int, device: str = 'cpu',
            max_steps: int = 4000, confidence: float = .95, minimum_edge: float = 0.,
            attempt: int = 1, minimum_deals: int = 128) -> dict:
    """Evaluate immutable copies, so a running trainer cannot change the inputs."""
    from . import duel, zoo
    from .outcomes import validate_budget

    require_training_engine()
    validate_budget(games, max_steps)
    # Validate the statistical settings before model loading or any game work.
    assess([2.5], [2.5], confidence=confidence, minimum_edge=minimum_edge,
           attempt=attempt, minimum_deals=minimum_deals)
    if type(seed) is not int or seed < 0 or seed + games > 2**64:
        raise ValueError('seed range must fit unsigned 64-bit game seeds')
    with tempfile.TemporaryDirectory(prefix='mahjong-gate-') as folder:
        snapshots = [Path(folder) / name for name in ('candidate.pt', 'champion.pt')]
        inputs = []
        for source, destination in zip((candidate, champion), snapshots):
            generation = copy_checkpoint(Path(source), destination)
            with destination.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            inputs.append({'path': str(source), 'sha256': digest, 'generation': generation})
        models = [zoo.load_player(path, device) for path in snapshots]
        forward = duel.duel(models[0], models[1], games, seed, device, max_steps=max_steps)
        reverse = duel.duel(models[1], models[0], games, seed, device, max_steps=max_steps)
    report = assess(forward['by_deal'], reverse['by_deal'], confidence=confidence,
                    minimum_edge=minimum_edge, attempt=attempt, minimum_deals=minimum_deals)
    report.update(candidate=inputs[0], champion=inputs[1], seed=seed,
                  training_api_version=TRAINING_API_VERSION,
                  candidate_one_vs_three=forward, champion_one_vs_three=reverse,
                  checkpoint_written=False)
    return report


def main() -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('champion', type=Path)
    parser.add_argument('--games', type=int, default=512, help='independent seeds, eight games each')
    parser.add_argument('--seed', type=int, required=True, help='fresh held-out seed range')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--minimum-edge', type=float, default=0.)
    parser.add_argument('--minimum-deals', type=int, default=128)
    parser.add_argument('--attempt', type=int, default=1, help='one-based attempt in the promotion series')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(compare(**vars(args)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
