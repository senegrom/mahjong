"""Descriptive value error after forced actions, not a causal coverage diagnosis.

The current action's exploration coin says nothing about how its *input* was
reached. We therefore group subsequent same-player information states within a
hand, using selfplay's after_exploration provenance. Different state/return
mixtures still confound this comparison: matched branch continuations are needed
to diagnose coverage or measure action-ranking quality. A diagnostic never turns
an MSE difference alone into `coverage_is_the_fault=True`.

    python -m neural.counterfactual checkpoint.pt --games 256 --share .25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import contract, selfplay, zoo
from .behavior import validate_exploration


def error_of(guess: np.ndarray, truth: np.ndarray) -> dict:
    guess, truth = np.asarray(guess), np.asarray(truth)
    if guess.shape != truth.shape or guess.ndim != 1:
        raise ValueError("value predictions and targets must be matching vectors")
    if not np.isfinite(guess).all() or not np.isfinite(truth).all():
        raise ValueError("value predictions and targets must be finite")
    if not guess.size:
        return {"rows": 0}
    mse = float(np.mean((truth - guess) ** 2))
    variance = float(truth.var())
    return {"rows": int(guess.size), "rmse": float(np.sqrt(mse)),
            "spread": float(np.sqrt(variance)),
            "explained": 1 - mse / variance if variance else None}


class _RecordedMortal:
    """Provide the existing two-stage recorder for a bare 46-action network."""
    kind = "mortal"

    def __init__(self, net, device):
        self.net, self.device = net, device

    def eval(self):
        self.net.eval()
        return self

    def decide(self, views, rows, players, legal, greedy=False, **kwargs):
        from .mortal_learner import decide_in_mortal_space
        return decide_in_mortal_space(lambda planes, mask: self.net(planes, mask)[0],
                                      views, rows, players, legal, greedy,
                                      self.device, **kwargs)


@torch.no_grad()
def measure(net, games: int, seed: int, share: float, device: str,
            valued_by: str = "critic") -> dict:
    validate_exploration(share)
    net = contract.unwrap(net)
    served = contract.serve(net)
    if served.contract.reads != "mortal":
        raise ValueError("This diagnostic requires recorded Mortal observations")
    actor = net
    if served.contract.answers == 46 and not hasattr(net, "decide"):
        actor = _RecordedMortal(net, device)
    source = net.ours if hasattr(net, "ours") else net
    has_oracle = hasattr(source, "with_oracle")
    batch = selfplay.play(actor, games=games, seed=seed, device=device,
                          explore_share=share, want_oracle=has_oracle)
    if batch.after_exploration is None:
        raise ValueError("Collector did not record successor-state provenance")
    follows = batch.after_exploration.cpu().numpy().astype(bool)
    returns = batch.returns.cpu().numpy()
    guessed = np.empty(batch.decisions, dtype=np.float32)
    oracle = np.empty_like(guessed) if has_oracle else None
    for start in range(0, batch.decisions, 256):
        rows = np.arange(start, min(start + 256, batch.decisions))
        planes = batch.observations.rows(rows).dense(device)
        guessed[rows] = served.value(planes, head=valued_by).float().cpu().numpy()
        if has_oracle:
            _, _, _, hidden_value, _ = source.with_oracle(
                planes, batch.legal[rows].to(device), batch.oracle[rows].to(device).float())
            oracle[rows] = hidden_value.float().cpu().numpy()
    def grouped(values):
        return {"after_forced_action": error_of(values[follows], returns[follows]),
                "other_decisions": error_of(values[~follows], returns[~follows])}
    report = {"games": games, "seed": seed, "decisions": batch.decisions,
              "exploration_share": share, "valued_by": valued_by,
              "successor_rows": int(follows.sum()), "value_error": grouped(guessed),
              "coverage_diagnosis": "not established; descriptive groups, not matched counterfactuals"}
    if has_oracle:
        report["oracle_error"] = grouped(oracle)
        report["oracle_note"] = "Checkpoint auxiliary head; no accuracy assumption or actor access"
    else:
        report["oracle_note"] = "Checkpoint has no oracle head"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--share", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=606_060)
    parser.add_argument("--valued-by", choices=("critic", "public", "mean"), default="critic")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = zoo.load_player(args.checkpoint, device)
    report = measure(net, args.games, args.seed, args.share, device, args.valued_by)
    report["checkpoint"] = str(args.checkpoint)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
