"""Placement-aligned PPO experiments on the shared engine and self-play collector.

    python -m neural.train_league --initial joined.pt --opponents mortal.pt old.pt \
        --objective placement --critic privileged --out runs/placement --rounds 20

The legacy trainers remain controls. This command never promotes or exports a
network, launches cloud compute, or enables search. See docs/SELFPLAY_EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import tempfile

import numpy as np
import torch
from torch import nn

from . import combined, model, mortal_learner, selfplay, zoo, rl_critics, rl_policy
from .checkpoints import atomic_save, copy_checkpoint
from .league_state import pin_population, digest
from .observe import PLANES
from .rl_objective import Objective, advantages, normalize_actor_advantages
from .seed_ledger import SeedLedger, atomic_json, exclusive
from .training_safety import require_training_engine, TRAINING_API_VERSION
from .training_state import capture_random_state, restore_random_state

VERSION = 1


class RolloutActor:
    """Only the public policy is callable during collection; no critic is attached."""
    kind = "mortal"
    actions = 46

    def __init__(self, net, device):
        self.net, self.device = net, device
        self.timing = {"encode": 0.0, "network": 0.0, "translate": 0.0}

    def eval(self):
        self.net.eval()
        return self

    def decide(self, views, rows, players, legal, greedy=False, **kwargs):
        return mortal_learner.decide_in_mortal_space(
            lambda planes, mask: rl_policy.logits(self.net, planes, mask),
            views, rows, players, legal, greedy=greedy, device=self.device, timing=self.timing, **kwargs)


def load_actor(path, device, temperature=1.0):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if combined.is_combined(payload):
        net, _ = combined.load(path, device)
        kind = "combined"
    elif "model" in payload:
        net = model.from_payload(payload, device)
        kind = "native"
    elif "mortal" in payload and "current_dqn" in payload:
        net, _ = mortal_learner.load(path, device, payload.get("temperature", temperature))
        kind = "mortal"
    else:
        raise ValueError("unsupported actor checkpoint")
    if getattr(net, "actions", None) != 46:
        raise ValueError("league training requires the 46-action policy; re-head legacy policies first")
    return net, payload, kind


def actor_state(net, kind, origin):
    if kind == "native":
        return {"model": net.state_dict(), **net.payload_fields()}
    if kind == "mortal":
        return {**net.state(), "config": origin["config"], "temperature": net.temperature}
    return net.state()


def parameter_groups(net, kind, args):
    """Separate optimizers/clipping make auxiliary magnitude irrelevant to actor LR."""
    if kind == "combined":
        groups = [{"params": net.head_trained(), "lr": args.lr},
                  {"params": net.ours_trained(), "lr": args.lr_ours},
                  {"params": net.mortal_trained(), "lr": args.lr_mortal}]
        auxiliary = net.always_trained()
    elif kind == "mortal":
        groups = [{"params": list(net.mortal.parameters()), "lr": args.lr_mortal}]
        auxiliary = list(net.value_head.parameters())
    else:
        names = ("stem", "tower", "tail", "policy_tiles", "policy_pooled")
        actor = [p for name in names for p in getattr(net, name).parameters()]
        groups = [{"params": actor, "lr": args.lr_ours}]
        names = ("value", "belief_stem", "belief_tower", "belief_tail", "hands")
        auxiliary = [p for name in names for p in getattr(net, name).parameters()]
    actor_ids = {id(p) for group in groups for p in group["params"]}
    if actor_ids & {id(p) for p in auxiliary}:
        raise ValueError("actor and auxiliary optimizers must have disjoint parameters")
    return groups, auxiliary


def fit_auxiliary(net, kind, optimizer, batch, args, device):
    """Keep existing browser heads useful without changing their HYBRID units.

    Public actor features and Mortal's vector are read under no_grad; only the
    value/hand heads and the independent belief tower receive these gradients.
    """
    total, steps = 0.0, 0
    net.eval()  # Mortal BN statistics remain frozen.
    for _ in range(args.aux_epochs):
        order = torch.randperm(batch.decisions)
        for start in range(0, len(order), args.batch):
            picks = order[start:start + args.batch]
            planes = batch.observations.rows(picks.numpy()).dense(device)
            target = batch.returns[picks].to(device)
            if kind == "mortal":
                with torch.no_grad():
                    phi = net.mortal.features(planes)
                value = net.value_head(phi.detach()).squeeze(1)
                loss = args.legacy_value_weight * nn.functional.mse_loss(value, target)
            else:
                ours = net.ours if kind == "combined" else net
                with torch.no_grad():
                    features = ours.tail(ours.tower(ours.stem(planes)))
                    phi = net.mortal.features(planes) if kind == "combined" else None
                value = ours.value(features.mean(dim=2).detach()).squeeze(1)
                guessed = ours.hands_from(planes, features.detach())
                if kind == "combined":
                    value = net.fuse.judge(phi, value)
                    guessed = net.fuse.read_hands(phi, guessed)
                held = batch.held[picks].to(device)
                holding = held.sum(dim=2) > 0
                ce = -(held * guessed.log_softmax(dim=2)).sum(dim=2)
                hands_loss = (ce * holding).sum() / holding.sum().clamp_min(1)
                loss = (args.legacy_value_weight * nn.functional.mse_loss(value, target)
                        + args.hands_weight * hands_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
            nn.utils.clip_grad_norm_(params, args.aux_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
    return {"loss": total / max(steps, 1), "updates": steps, "value_objective": "hybrid-v1"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial", type=Path, help="warm-start actor only; fresh critics and optimizer")
    source.add_argument("--resume", type=Path, help="resume a checkpoint produced by THIS trainer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--objective", choices=("hybrid", "placement"), default="placement")
    p.add_argument("--critic", choices=("public", "privileged"), default="privileged")
    p.add_argument("--advantage", choices=("mc", "gae"), default="mc")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--critic-width", type=int, default=64)
    p.add_argument("--critic-blocks", type=int, default=2)
    p.add_argument("--critic-epochs", type=int, default=1)
    p.add_argument("--critic-warmup", type=int, default=1, help="initial critic-only generations")
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--reset-critics", action="store_true", help="explicitly allow a new objective/critic contract")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-ours", type=float, default=4e-5)
    p.add_argument("--lr-mortal", type=float, default=3e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--entropy", type=float, default=0.0005)
    p.add_argument("--actor-clip", type=float, default=1.0)
    p.add_argument("--aux-clip", type=float, default=1.0)
    p.add_argument("--aux-epochs", type=int, default=1)
    p.add_argument("--hands-weight", type=float, default=1.0)
    p.add_argument("--legacy-value-weight", type=float, default=0.5)
    p.add_argument("--schedule", choices=("staged", "random", "joint"), default="staged")
    p.add_argument("--freeze-generations", type=int, default=20)
    p.add_argument("--reference", type=Path, help="KL anchor; initially defaults to the starting actor")
    p.add_argument("--refresh-reference", action="store_true", help="explicitly replace anchor with --reference")
    p.add_argument("--leash", type=float, default=0.0)
    p.add_argument("--opponents", type=Path, nargs="*", default=[], help="fixed reference players")
    p.add_argument("--champion", type=Path)
    p.add_argument("--recent", type=Path, nargs="*", default=[])
    p.add_argument("--older", type=Path, nargs="*", default=[])
    p.add_argument("--refresh-opponents", action="store_true")
    p.add_argument("--table-mix", type=float, nargs=3, default=None,
                   metavar=("SELF", "ONE_LEARNER", "THREE_LEARNERS"))
    p.add_argument("--nominate-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--seed-ledger", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args(argv)
    for name in ("rounds", "games", "batch", "epochs", "critic_epochs", "nominate_every"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    for name in ("critic_warmup", "aux_epochs", "freeze_generations", "seed"):
        if getattr(args, name) < 0:
            p.error(f"{name} must be nonnegative")
    for name in ("lr", "lr_ours", "lr_mortal", "critic_lr", "temperature", "actor_clip", "aux_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    for name in ("target_kl", "entropy", "leash", "hands_weight", "legacy_value_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if not 0 < args.clip < 1 or not 0 <= args.gae_lambda <= 1:
        p.error("clip must be in (0,1), and gae-lambda in [0,1]")
    if args.advantage == "gae" and args.objective != "placement":
        p.error("GAE is defined here only for the terminal-placement objective")
    if args.refresh_reference and args.reference is None:
        p.error("--refresh-reference requires an explicitly selected --reference")
    if args.amp and args.device != "cuda":
        p.error("--amp requires CUDA")
    return args


def main(argv=None):
    args = parse_args(argv)
    original = args.resume or args.initial
    if not original.is_file():
        raise FileNotFoundError(original)
    args.out.mkdir(parents=True, exist_ok=True)
    # Protect the whole run, not just individual atomic checkpoint renames.
    with exclusive(args.out / "run-state"):
        with tempfile.TemporaryDirectory(prefix=".actor-input-", dir=args.out) as where:
            source = Path(where) / "actor.pt"
            copy_checkpoint(original, source)
            return run(args, source)


def run(args, source):
    require_training_engine()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    objective = Objective(args.objective)
    # Validate before creating output or replacing recovery state.
    config = rl_critics.CriticConfig(PLANES, 0, args.critic_width, args.critic_blocks)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.initial and (args.out / "latest.pt").exists():
        raise FileExistsError("run already has latest.pt; use --resume or an unused --out")
    net, origin, kind = load_actor(source, args.device, args.temperature)
    if args.resume and origin.get("league_version") != VERSION:
        raise ValueError("legacy checkpoints are warm starts: use --initial, not --resume")
    start = int(origin["generation"]) if args.resume else 0
    if args.resume and origin.get("run_options", {}).get("seed", args.seed) != args.seed:
        raise ValueError("--seed must match the resumed sampling stream")
    contract = {"objective": objective.contract(), "critic": args.critic,
                "width": args.critic_width, "blocks": args.critic_blocks}
    old_contract = origin.get("league_contract") if args.resume else None
    if old_contract is not None and old_contract != contract and not args.reset_critics:
        raise ValueError("critic/objective contract changed; refit with --reset-critics")
    restore_critics = bool(args.resume and not args.reset_critics)
    critics = {"public": rl_critics.TrainingCritic(config).to(args.device)}
    if args.critic == "privileged":
        critics["privileged"] = rl_critics.TrainingCritic(rl_critics.CriticConfig(
            PLANES, selfplay.ORACLE_PLANES, args.critic_width, args.critic_blocks)).to(args.device)
    optimizers = {name: torch.optim.AdamW(c.parameters(), lr=args.critic_lr, weight_decay=1e-4)
                  for name, c in critics.items()}
    if restore_critics:
        for name, c in critics.items():
            c.load_state_dict(origin["training_critics"][name]["state"])
            optimizers[name].load_state_dict(origin["critic_optimizers"][name])
            for group in optimizers[name].param_groups:
                group['lr'] = args.critic_lr
    groups, aux_params = parameter_groups(net, kind, args)
    actor_optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    aux_optimizer = torch.optim.AdamW(aux_params, lr=args.lr, weight_decay=1e-4)
    if args.resume:
        wanted_rates = [g['lr'] for g in actor_optimizer.param_groups]
        actor_optimizer.load_state_dict(origin["actor_optimizer"])
        aux_optimizer.load_state_dict(origin["aux_optimizer"])
        for g, rate in zip(actor_optimizer.param_groups, wanted_rates):
            g['lr'] = rate
        for g in aux_optimizer.param_groups:
            g['lr'] = args.lr
    requested = [(path, "reference") for path in args.opponents]
    requested += [(path, "recent") for path in args.recent]
    requested += [(path, "older") for path in args.older]
    if args.champion:
        requested.insert(0, (args.champion, "champion"))
    roster, paths, manifest = pin_population(requested, args.out / "snapshots",
        saved=origin.get("population") if args.resume else None, refresh=args.refresh_opponents)
    mix = args.table_mix
    if mix is None:
        mix = origin.get("table_mix") if args.resume else None
    if mix is None:
        mix = [0.4, 0.4, 0.2] if paths else [1.0, 0.0, 0.0]
    from .population import table_seats
    table_seats(1, roster.members, mix, np.random.default_rng(0))  # validate before collecting
    opponents = [zoo.load_player(path, args.device) for path in paths]
    for opponent in opponents:
        opponent.eval()
        for parameter in getattr(opponent, "parameters", list)():
            parameter.requires_grad_(False)
    anchor = args.out / "reference.pt"
    if args.resume and not args.refresh_reference:
        if not anchor.is_file() or digest(anchor) != origin["reference_sha256"]:
            raise ValueError("missing or changed reference.pt; copy the run or explicitly refresh the anchor")
    else:
        if args.resume and args.reference is None:
            raise ValueError("refreshing an anchor requires a reference")
        copy_checkpoint(args.reference or source, anchor)
    reference_sha = digest(anchor)
    reference = None
    if args.leash:
        reference, _, reference_kind = load_actor(anchor, args.device, args.temperature)
        reference.eval().requires_grad_(False)
    if args.compile:
        # Same policy-only method is used during acting, likelihood replay and PPO.
        if hasattr(net, "policy_only"):
            net.policy_only = torch.compile(net.policy_only, dynamic=True)
        else:
            net.forward = torch.compile(net.forward, dynamic=True)
    drawer = restore_random_state(origin.get("random_state") if args.resume else None,
                                  seed=args.seed, generation=start, modes=list(combined.Combined.MODES))
    # Shared ledgers use a stable root; --seed controls model/action randomness.
    seeds = SeedLedger(args.seed_ledger or args.out / "seeds.json", seed=0,
                       saved=origin.get("seed_state") if args.resume else None)
    atomic_json(args.out / "opponents.json", {"population": manifest, "table_mix": list(mix)})
    def portable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [portable(item) for item in value]
        return value
    options = {key: portable(value) for key, value in vars(args).items()}
    atomic_json(args.out / "experiment.json", {**options, "contract": contract,
                                             "actor_kind": kind, "source_sha256": digest(source)})
    warmup_start = start if args.reset_critics else (origin.get("critic_started_generation", 0) if args.resume else 0)
    for generation in range(start, start + args.rounds):
        began = time.perf_counter()
        if kind == "combined":
            mode = (str(drawer.choice(combined.Combined.MODES)) if args.schedule == "random" else
                    "mortal" if args.schedule == "staged" and generation < args.freeze_generations else "none")
            net.set_mode(mode)
        else:
            mode = "none"
        reservation = seeds.reserve("train", args.games, f"league-generation-{generation}")
        batch = selfplay.play(RolloutActor(net, args.device), games=args.games,
                              seed=reservation["seed"], device=args.device, amp=args.amp,
                              opponents=opponents, population=roster, table_mix=tuple(mix),
                              want_oracle=args.critic == "privileged", explore_share=0.0)
        play_seconds = time.perf_counter() - began
        targets = objective.targets(batch)
        if batch.next_index is None or batch.player_of is None or batch.terminal is None:
            raise ValueError("collector did not retain complete same-player trajectories")
        from .rl_objective import trajectory_links
        expected, terminals = trajectory_links(batch.game_of.numpy(), batch.player_of.numpy())
        if not np.array_equal(expected, batch.next_index.numpy()) or not np.array_equal(terminals, batch.terminal.numpy()):
            raise ValueError("invalid same-player trajectory links")
        old = rl_policy.old_distributions(net, batch, batch_size=args.batch, device=args.device, amp=args.amp)
        predictions = {name: rl_critics.predict(c, batch.observations, batch.oracle, args.batch, args.device)
                       for name, c in critics.items()}
        raw, _lambda_targets = advantages(targets, predictions[args.critic], method=args.advantage,
                                          next_index=batch.next_index.numpy(), gae_lambda=args.gae_lambda)
        actor_rows = batch.legal.sum(dim=1) > 1
        normalized = normalize_actor_advantages(raw, actor_rows)
        warmup = generation < warmup_start + args.critic_warmup
        actor_metrics = {"actor_updates": 0, "critic_warmup": True}
        if not warmup:
            actor_metrics = rl_policy.update(net, actor_optimizer, batch, normalized, old,
                epochs=args.epochs, batch_size=args.batch, ratio_clip=args.clip, target_kl=args.target_kl,
                entropy_weight=args.entropy, grad_clip=args.actor_clip, device=args.device, amp=args.amp,
                reference=reference, reference_weight=args.leash)
        # These passes continue even when PPO stops on its KL limit.
        critic_metrics = {}
        variance = float(targets.var(unbiased=False))
        for name, critic in critics.items():
            metrics = rl_critics.fit(critic, optimizers[name], batch.observations, batch.oracle, targets,
                        epochs=args.critic_epochs, batch_size=args.batch, clip_norm=args.aux_clip, device=args.device)
            mse = float((targets - predictions[name]).square().mean())
            metrics.update(pre_fit_mse=mse, pre_fit_explained=1 - mse / variance if variance else None)
            critic_metrics[name] = metrics
        auxiliary_metrics = fit_auxiliary(net, kind, aux_optimizer, batch, args, args.device)
        record = {"generation": generation, "checkpoint_generation": generation + 1,
                  "objective": objective.contract(), "fixed": mode, "actor_kind": kind,
                  "games": args.games, "hands": batch.hands, "decisions": batch.decisions,
                  "play_seconds": play_seconds, "seconds": time.perf_counter() - began,
                  "seed_reservation": reservation, "matchups": batch.matchups, "table_mix": list(mix),
                  "play_split": batch.timing, "critics": critic_metrics, "auxiliary": auxiliary_metrics,
                  "return_variance": variance, "advantage_std": float(raw.std(unbiased=False)),
                  "training_options": options, **actor_metrics}
        payload = {**actor_state(net, kind, origin), "league_version": VERSION,
                   "generation": generation + 1, "training_api_version": TRAINING_API_VERSION,
                   "league_contract": contract, "learner": kind, "actor_optimizer": actor_optimizer.state_dict(),
                   "aux_optimizer": aux_optimizer.state_dict(),
                   "training_critics": {name: c.payload() for name, c in critics.items()},
                   "critic_optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
                   "random_state": capture_random_state(drawer), "seed_state": seeds.snapshot(),
                   "population": manifest, "table_mix": list(mix), "reference_sha256": reference_sha,
                   "last_metrics": record, "auxiliary_value_objective": "hybrid-v1",
                   "critic_started_generation": warmup_start,
                   "run_options": options}
        atomic_save(payload, args.out / "latest.pt")
        if (generation + 1) % args.nominate_every == 0:
            # Nominations follow a predeclared schedule, never lucky heuristic scores.
            atomic_save(payload, args.out / "candidate.pt")
            atomic_save(payload, args.out / "history" / f"gen-{generation + 1:05d}.pt", keep_previous=False)
        with (args.out / "log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        del batch, old, targets, predictions, raw, normalized
    return payload


if __name__ == "__main__":
    main()
