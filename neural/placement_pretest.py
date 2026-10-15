"""The placement reward's pre-test: before a fork is paid for, does paying
by placement, hand by hand, give a generation's step more signal than
today's return, and can U be trusted to pay it?

    python -m neural.placement_pretest checkpoint.pt --opponents A.pt B.pt club \\
        --seat-share 0.5 --games 4096 --probes 20000 --out pretest/

It plays one round of the checkpoint as its run plays one, at the same
tables, keeping the round's ledger, and then:

1. Fits U on the round, every game held out once (`--folds`), and reads its
   calibration by stage and by tenth of its answers beside the boundary
   judge's curve. The first kill rule: U is refused when its mean answer
   misses the mean bonus by more than 0.03 in some tenth at some stage, by
   more than that cell's noise (see `place_model.calibration`), or when a
   member of the roster never sat, whose seats U would then read through
   weights no game taught. A seed whose round leaves a member out of its
   pool is refused before anything is loaded.
2. Pays the round three ways: today's return, and the placement reward at
   each of `--alphas`, each game paid by a U that never saw it.
3. Teaches a copy of the critic each return as the fork's warm-up
   generation would, on one half of the games, and measures the other
   half's advantages by it, and the other way round, so no decision's
   baseline has seen its own return. The advantages' spread by stage.
4. Asks whether one update agrees with itself: the games are split into
   halves three ways at random; each half's PPO gradient over the fusion
   head (the first step of a generation, where every ratio is one) is
   compared with the other's, as a cosine; and each half's gradient, taken
   as a whole generation's step preconditioned by the checkpoint's own
   AdamW moments, is applied to the head and read on positions the
   checkpoint hesitates over (top move under `--top`), taken from a greedy
   round against three published Mortals: which of their best moves flip,
   and how far the two halves' flips coincide. Both are measured again with
   the advantages shuffled among the decisions, which is what chance
   agreement reads. The second kill rule: the placement reward at alpha 0
   must agree with itself, above chance, at least 1.5 times as well as
   today's return does.

Everything goes to `--out`: report.json, the round's ledger.npz, and
place-model.pt, U fitted on every game of the round, which a fork starts
from (`train_combined --reward placement --place-model`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import combined, duel, place_model, policy_inference, ppo_loop, selfplay, zoo
from .observe import Planes
from .population import Population
from .scoreboard import STAGE_NAMES, STAGES

#: The pre-registered rule: alpha 0's agreement above chance, as a multiple
#: of today's return's (the roadmap, 14 October 2026).
NEEDED = 1.5
#: Fewer flips than this in either half leaves the flips unread: their
#: overlap is then a handful of positions and says nothing.
FEW_FLIPS = 20


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="the joined player to test, with its optimiser's state")
    parser.add_argument("--opponents", type=Path, nargs="*", default=[],
                        help="the others its run seats, as the cloud stages them (see neural.population)")
    parser.add_argument("--seat-share", type=float, default=0.0)
    parser.add_argument("--opponent-share", type=float, default=0.0)
    parser.add_argument("--games", type=int, default=4096, help="the round's games")
    parser.add_argument("--seed", type=int, default=9_700_000, help="the round's first deal")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05],
                        help="the placement reward's arms; the first is the one the kill rule reads")
    parser.add_argument("--folds", type=int, default=8, help="U is fitted this many times, a fold of games held out each")
    parser.add_argument("--splits", type=int, default=3, help="random splits of the games into halves")
    parser.add_argument("--batch", type=int, default=2048, help="the run's minibatch")
    parser.add_argument("--chunk", type=int, default=1024,
                        help="rows a pass over the round or the probes reads at once; the gradients' "
                        "products hold one a vector, so a small card wants a small chunk")
    parser.add_argument("--epochs", type=int, default=2, help="the run's epochs a generation")
    parser.add_argument("--lr", type=float, default=2e-6,
                        help="the head's rate: fusion-long's before generation 873, which the fork trains at")
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--value-warm-up-lr", type=float, default=1e-4,
                        help="the critic's rate as the fork's warm-up teaches it")
    parser.add_argument("--probes", type=int, default=20000, help="hesitant positions to read flips on")
    parser.add_argument("--probe-games", type=int, default=1024,
                        help="games a greedy round of the probes plays at once: a step's cost hardly "
                        "grows with its rows, so large tables are what make the probes cheap")
    parser.add_argument("--probe-rounds", type=int, default=16, help="the most greedy rounds played for probes")
    parser.add_argument("--probe-opponent", type=Path, default=None,
                        help="the published Mortal the probes' greedy rounds seat three of; the first "
                        "of --opponents that is not the Club by default")
    parser.add_argument("--top", type=float, default=0.8, help="a position is hesitant below this top probability")
    parser.add_argument("--flip-scale", type=float, default=1.0,
                        help="the generation's step is multiplied by this before the flips are read")
    parser.add_argument("--amp", action="store_true", help="play and read in bfloat16 on the card, as the run does")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("games", "folds", "splits", "batch", "chunk", "epochs", "probes", "probe_games", "probe_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least two")
    if any(not math.isfinite(alpha) or alpha < 0 for alpha in args.alphas):
        parser.error("--alphas must be finite and nonnegative")
    if not 0 < args.top <= 1:
        parser.error("--top must be above nought and at most one")
    if args.seat_share and args.opponent_share:
        parser.error("seat others by game or by player, not both")
    return args


def arm_name(alpha: float) -> str:
    return f"alpha {alpha:g}"


# ---------------------------------------------------------------- the round


def seat_others(args, device: str) -> tuple[Population, list]:
    """The others, loaded as a run's trainer loads them, and their roster."""
    roster = Population.from_paths(args.opponents)
    seated = []
    for path in args.opponents:
        other = zoo.load_player(path, device)
        other.eval()
        for parameter in getattr(other, "parameters", list)():
            parameter.requires_grad_(False)
        seated.append(zoo.play_in_bfloat16(other, device))
    return roster, seated


def play_round(net, args, roster, seated, device: str) -> selfplay.Batch:
    """One round of the checkpoint as its run plays one: sampled moves,
    the others seated as the run seats them, the ledger kept."""
    amp = args.amp and device == "cuda"
    player = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False) if amp else net
    player.keep_phi = False
    batch = selfplay.play(
        player, games=args.games, seed=args.seed, device=device, amp=amp, opponents=seated,
        opponent_share=args.opponent_share, seat_share=args.seat_share, population=roster,
        want_held=False, ledger=True,
    )
    del player
    return batch


# ---------------------------------------------------------------------- U


def absent_from(roster: Population, seed: int) -> list[str]:
    """The members the round of `seed` leaves out of its pool (see
    `Population.for_round`): U fitted on that round would never see them
    seated, and a fork that seats them refuses such a U."""
    return [member.name for member in roster.for_round(seed).members if member.weight <= 0]


def fit_u(ledger, members, args, device: str) -> tuple[dict, np.ndarray, place_model.PlaceModel]:
    """U's calibration on games it never saw, U at every hand's start from
    every player's view by a model that never saw that hand's game, and U
    fitted on the whole round, which a fork starts from. The first kill
    rule fails when U misses beyond noise, or when some member never sat,
    since U then reads that member's seats through weights no game taught."""
    table = place_model.Boundaries.of(ledger, members)
    answers, _fold = place_model.cross_fitted(table, members, folds=args.folds, seed=args.seed % 997,
                                              device=device)
    report = place_model.calibration(answers, table.bonus, table.stages, table.games)
    report["unseen"] = place_model.unseen(table, members)
    report["kill"] = not report["passed"] or bool(report["unseen"])
    whole = place_model.fit(table, members, seed=args.seed % 997, device=device)
    whole.fitted["calibration"] = {key: report[key] for key in (
        "passed", "rows", "games", "cells_missing_beyond_noise", "largest_miss")}
    return report, answers.reshape(len(ledger.starts), 4), whole


# ----------------------------------------------------------------- the critic


@torch.no_grad()
def critic_inputs(net, rollout, args, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the critic reads of every decision, which nothing it learns can
    change: our network's pooled features and Mortal's vector, both held
    back from it (`Combined.backbones`, `Fuse.judge`); and the critic's own
    value as the checkpoint has it."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    pooled, phi, own = [], [], []
    for start in range(0, rollout.decisions, args.chunk):
        stop = min(start + args.chunk, rollout.decisions)
        planes = rollout.observations.slice(start, stop).dense(device, dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            vector, _q, features, _a1, value, _guessed = net.backbones(planes, rollout.legal[start:stop], hands=False)
            judged = net.fuse.judge(vector.float(), value.float())
        pooled.append(features.mean(dim=2).to(dtype))
        phi.append(vector.to(dtype))
        own.append(judged.float())
    return torch.cat(pooled), torch.cat(phi), torch.cat(own)


def fit_critic(net, pooled, phi, returns, rows: torch.Tensor, args, seed: int):
    """A copy of the critic taught `returns` on decisions `rows`, as the
    fork's warm-up generation teaches it (`train_combined.value_warm_up`):
    the same parts, the same loss and clip, `--epochs` passes at the warm-up
    rate. Answers the copy's two parts."""
    value, fix = copy.deepcopy(net.ours.value).float(), copy.deepcopy(net.fuse.value_fix).float()
    for part in (value, fix):
        part.train()
        part.requires_grad_(True)
    parts = [*value.parameters(), *fix.parameters()]
    optimiser = torch.optim.AdamW(parts, lr=args.value_warm_up_lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(args.epochs):
        order = rows[torch.randperm(len(rows), generator=generator).to(rows.device)]
        for start in range(0, len(order) - args.batch + 1, args.batch):
            picks = order[start:start + args.batch]
            said = value(pooled[picks].float()).squeeze(1) + fix(phi[picks].float()).squeeze(1)
            loss = args.value_weight * nn.functional.mse_loss(said, returns[picks])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parts, 1.0)
            optimiser.step()
    return value.eval(), fix.eval()


@torch.no_grad()
def critic_values(value, fix, pooled, phi, rows: torch.Tensor) -> torch.Tensor:
    out = []
    for start in range(0, len(rows), 65536):
        picks = rows[start:start + 65536]
        out.append(value(pooled[picks].float()).squeeze(1) + fix(phi[picks].float()).squeeze(1))
    return torch.cat(out) if out else torch.zeros(0, device=pooled.device)


def cross_fitted_baseline(net, pooled, phi, returns, halves: torch.Tensor, args, seed: int) -> torch.Tensor:
    """Every decision's baseline from a critic taught the other half of the
    games' returns, so none has seen its own."""
    baseline = torch.empty_like(returns)
    for side in (False, True):
        taught = torch.nonzero(halves == side).squeeze(1)
        asked = torch.nonzero(halves != side).squeeze(1)
        value, fix = fit_critic(net, pooled, phi, returns, taught, args, seed + int(side))
        baseline[asked] = critic_values(value, fix, pooled, phi, asked)
    return baseline


def by_stage(values: torch.Tensor, stages: np.ndarray) -> dict:
    """The spread of `values` over each stage's decisions, and overall."""
    values = values.double().cpu().numpy()
    out = {"all": round(float(values.var()), 5)}
    for stage in range(STAGES):
        at = stages == stage
        if at.sum() > 1:
            out[STAGE_NAMES[stage]] = round(float(values[at].var()), 5)
    return out


# ------------------------------------------------------------- the gradients


def head_gradients(net, rollout, weights: torch.Tensor, args, device: str) -> tuple[torch.Tensor, str]:
    """Σ_i w_ki ∇ log π(a_i | s_i) over the fusion head's parameters, for
    every row k of `weights` (k, decisions): one pass over the round, the
    networks beneath the head read once a minibatch and held still.
    Answers (k, parameters) and how the products were taken: batched over k
    where the graph allows, one at a time otherwise."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    net.set_mode("mortal+ours")
    net.eval()
    params = net.head_trained()
    total = sum(p.numel() for p in params)
    out = torch.zeros(len(weights), total, device=device)
    how = "batched"
    for start in range(0, rollout.decisions, args.chunk):
        stop = min(start + args.chunk, rollout.decisions)
        legal = rollout.legal[start:stop]
        planes = rollout.observations.slice(start, stop).dense(device, dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            phi, q, features, a1, _value, _guessed = net.backbones(planes, legal, hands=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = net.fuse(phi.float(), q.float(), features.float(), a1.float(), legal)
        log_prob = torch.distributions.Categorical(logits=logits.float(), validate_args=False).log_prob(
            rollout.actions[start:stop])
        asked = weights[:, start:stop]
        grads = None
        if how == "batched":
            try:
                grads = torch.autograd.grad(log_prob, params, grad_outputs=asked, is_grads_batched=True,
                                            retain_graph=True, allow_unused=True)
            except (RuntimeError, NotImplementedError):
                how = "one at a time"
        if grads is not None:
            out += torch.cat([g.reshape(len(weights), -1) if g is not None
                              else torch.zeros(len(weights), p.numel(), device=device)
                              for g, p in zip(grads, params)], dim=1)
            continue
        for k in range(len(weights)):
            one = torch.autograd.grad((asked[k] * log_prob).sum(), params, retain_graph=k < len(weights) - 1,
                                      allow_unused=True)
            out[k] += torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                                 for g, p in zip(one, params)])
    return out, how


def preconditioner(net, saved: dict | None, args, device: str) -> tuple[list[torch.Tensor] | None, str]:
    """√v̂ + ε of every head parameter as the checkpoint's AdamW holds it:
    the denominator its next steps divide by. None, with the reason, when
    the checkpoint carries no state that fits."""
    if saved is None:
        return None, "the checkpoint carries no optimiser state"
    from .train_combined import optimiser_groups

    rates = argparse.Namespace(lr=args.lr, lr_ours=args.lr, lr_mortal=args.lr)
    optimiser = torch.optim.AdamW(optimiser_groups(net, rates), lr=args.lr, weight_decay=1e-4)
    ppo_loop.restore_optimiser(optimiser, saved,
                               migrate=lambda state: combined.migrate_belief_optimizer(state, optimiser, net))
    beta2, eps = optimiser.param_groups[0]["betas"][1], optimiser.param_groups[0]["eps"]
    out = []
    for parameter in net.head_trained():
        state = optimiser.state.get(parameter)
        if not state or "exp_avg_sq" not in state:
            return None, "the checkpoint's optimiser state does not fit this head"
        step = float(state["step"])
        correction = 1.0 - beta2 ** step if step > 0 else 1.0
        out.append((state["exp_avg_sq"].float().to(device) / correction).sqrt().reshape(-1) + eps)
    return out, "the checkpoint's AdamW second moments"


# ------------------------------------------------------------------ the probes


class Hesitation:
    """The checkpoint at a duel's table, playing its best move as the ruler
    does, writing down every position whose first question it hesitates
    over: its top move under `top`, with more than one move open."""

    kind = "mortal"
    actions = zoo.MORTAL_ACTIONS

    def __init__(self, net, device: str, amp: bool, top: float) -> None:
        self.net = net
        self.device = device
        self.amp = amp
        self.top = top
        self.planes: list[Planes] = []
        self.masks: list[np.ndarray] = []
        self.count = 0

    def eval(self) -> Hesitation:
        return self

    def choose(self, views, rows, players, legal):
        return zoo.choose_in_mortal_space(
            lambda who, fresh, allowed, ahead=(): self._ask(views, who, fresh, allowed, ahead),
            views, rows, players, legal)

    @torch.no_grad()
    def _ask(self, views, who, fresh, allowed, ahead=()):
        rows = np.array([game for game, _player in who], dtype=np.int64)
        players = np.array([player for _game, player in who], dtype=np.int64)
        sparse, masks = views.sparse_and_masks(rows, players, fresh=fresh)
        mask = torch.from_numpy(allowed).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits, _value, _phi = self.net.decision(sparse.dense(self.device), mask)
        logits = logits.float()
        if not fresh:
            top = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
            hesitant = np.flatnonzero((top < self.top) & (allowed.sum(axis=1) > 1))
            if len(hesitant):
                self.planes.append(sparse.rows(hesitant))
                self.masks.append(allowed[hesitant].copy())
                self.count += len(hesitant)
        return logits.cpu().numpy(), masks


def probes(net, args, device: str) -> tuple[Planes, torch.Tensor, int]:
    """Up to `--probes` positions the checkpoint hesitates over, from greedy
    rounds of it in one seat against three published Mortals, and the games
    played for them."""
    opponent = args.probe_opponent
    if opponent is None:
        opponent = next((path for path in args.opponents if str(path) != "club"), None)
    if opponent is None:
        raise SystemExit("name the published Mortal the probes are played against (--probe-opponent)")
    mortal = zoo.play_in_bfloat16(zoo.load_player(opponent, device), device)
    amp = args.amp and device == "cuda"
    # Asked as the trainer's player is asked, its weights cast to bfloat16
    # once rather than at every step (see `policy_inference.precast`).
    player = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False) if amp else net
    learner = Hesitation(player, device, amp, args.top)
    played = 0
    for index in range(args.probe_rounds):
        if learner.count >= args.probes:
            break
        duel.table(learner, mortal, args.probe_games, args.seed + 10_000_000 + index * args.probe_games,
                   place=index % 4, device=device)
        played += args.probe_games
    planes = Planes.cat(learner.planes)
    masks = torch.from_numpy(np.concatenate(learner.masks)) if learner.masks else torch.zeros(0, 46, dtype=torch.bool)
    keep = min(len(planes), args.probes)
    del player, learner, mortal
    return planes.slice(0, keep), masks[:keep], played


@torch.no_grad()
def probe_parts(net, planes: Planes, masks: torch.Tensor, args, device: str) -> list[tuple]:
    """What the head reads of every probe, from the networks beneath it,
    kept so that each step's reading is the head's forward alone."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    parts = []
    for start in range(0, len(planes), args.chunk):
        stop = min(start + args.chunk, len(planes))
        legal = masks[start:stop].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            phi, q, features, a1, _value, _guessed = net.backbones(
                planes.slice(start, stop).dense(device, dtype), legal, hands=False)
        parts.append((phi.to(dtype), q.float(), features.to(dtype), a1.float(), legal))
    return parts


@torch.no_grad()
def probe_logits(net, parts, args, device: str) -> torch.Tensor:
    amp = args.amp and device == "cuda"
    out = []
    for phi, q, features, a1, legal in parts:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(net.fuse(phi.float(), q, features.float(), a1, legal).float())
    return torch.cat(out)


@torch.no_grad()
def stepped_logits(net, step: torch.Tensor, parts, args, device: str) -> torch.Tensor:
    """The probes' logits with the head moved by `step`, a flat vector over
    `head_trained`, and the head put back exactly as it was after."""
    params = net.head_trained()
    kept = [parameter.detach().clone() for parameter in params]
    at = 0
    for parameter in params:
        parameter.add_(step[at:at + parameter.numel()].reshape(parameter.shape).to(parameter.dtype))
        at += parameter.numel()
    try:
        return probe_logits(net, parts, args, device)
    finally:
        for parameter, original in zip(params, kept):
            parameter.copy_(original)


# --------------------------------------------------------------- agreement


def cosine(one: torch.Tensor, two: torch.Tensor) -> float:
    """In double precision: two halves' gradients agree by a few hundredths
    at most, and over a million single-precision products the sum's own
    rounding is not far below that."""
    one, two = one.double(), two.double()
    norm = float(one.norm() * two.norm())
    return float(one @ two) / norm if norm > 0 else 0.0


def ochiai(one: np.ndarray, two: np.ndarray) -> float | None:
    """How far two sets of flipped positions coincide: their overlap over
    the geometric mean of their sizes, one for the same set and about the
    rate of flips for two unrelated ones. None when either is empty."""
    sizes = one.sum() * two.sum()
    return float((one & two).sum() / math.sqrt(sizes)) if sizes else None


def correlation(one: np.ndarray, two: np.ndarray) -> float | None:
    if one.std() == 0 or two.std() == 0:
        return None
    return float(np.corrcoef(one, two)[0, 1])


def mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def agreement_verdict(arms: dict, today: str, chosen: str, needed: float = NEEDED) -> dict:
    """Whether `chosen`'s step agrees with itself, above chance, at least
    `needed` times as well as `today`'s, on each of the two measures."""
    out = {"needed": needed, "today": today, "arm": chosen}
    passes = []
    for measure in ("cosine", "flips"):
        mine, theirs = arms[chosen]["agreement"][measure], arms[today]["agreement"][measure]
        signal = None if mine["above_chance"] is None else mine["above_chance"]
        baseline = None if theirs["above_chance"] is None else theirs["above_chance"]
        if signal is None or baseline is None:
            out[measure] = {"ratio": None, "passed": None, "why": "unread: too few flips or no spread"}
            continue
        if baseline <= 0:
            ratio = math.inf if signal > 0 else None
            passed = signal > 0
            why = "today's return agrees no better than chance"
        else:
            ratio = signal / baseline
            passed = ratio >= needed
            why = f"{ratio:.2f} times today's"
        out[measure] = {"ratio": None if ratio is None or math.isinf(ratio) else round(ratio, 3),
                        "passed": passed, "why": why}
        passes.append(passed)
    if not passes:
        out["verdict"] = "unread"
    elif all(passes):
        out["verdict"] = "go"
    elif not any(passes):
        out["verdict"] = "kill"
    else:
        out["verdict"] = "split: the two measures disagree; the owner decides"
    return out


# ------------------------------------------------------------------- the run


def main(argv=None) -> dict:
    args = parse_args(argv)
    # Refused before anything is loaded or played: a round whose seed sits
    # a member out cannot fit the U a fork that seats it starts from.
    absent = absent_from(Population.from_paths(args.opponents), args.seed)
    if absent:
        raise SystemExit(f"the round of seed {args.seed} leaves {absent} out of its pool, so U would "
                         "never see them seated; choose a --seed whose round seats every member")
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    clock = {"began": time.time()}
    seconds = {}

    def lap(name: str) -> None:
        now = time.time()
        seconds[name] = round(now - clock["began"], 1)
        clock["began"] = now
        print(json.dumps({"phase": name, "seconds": seconds[name]}), flush=True)

    net, payload = combined.load(args.checkpoint, device)
    generation = payload.get("generation")
    saved = payload.get("optimizer_state")
    payload = None
    net.eval()
    roster, seated = seat_others(args, device)
    members = tuple(member.name for member in roster.members)
    lap("load")

    batch = play_round(net, args, roster, seated, device)
    ledger = batch.ledger
    ledger.save(args.out / "ledger.npz")
    seated = None
    lap("play")

    u_report, at_starts, whole = fit_u(ledger, members, args, device)
    whole.save(args.out / "place-model.pt")
    lap("u")

    returns = {"today": batch.returns.numpy().astype(np.float32)}
    for alpha in args.alphas:
        returns[arm_name(alpha)] = place_model.placement_returns(ledger, at_starts, alpha)
    rollout = ppo_loop.on_device(batch, device, card=False)
    stages = ledger.stages[ledger.rows_of(ledger.games, ledger.hands)]
    rng = np.random.default_rng(args.seed)
    games = torch.from_numpy(ledger.games.astype(np.int64)).to(device)
    halves_of_games = torch.from_numpy(rng.permutation(args.games) < args.games // 2).to(device)
    critic_halves = halves_of_games[games]
    pooled, phi, own = critic_inputs(net, rollout, args, device)
    lap("critic inputs")

    arms: dict[str, dict] = {}
    advantages = {}
    for index, (name, paid) in enumerate(returns.items()):
        paid = torch.from_numpy(paid).to(device)
        baseline = cross_fitted_baseline(net, pooled, phi, paid, critic_halves, args, args.seed + 7 * index)
        advantage, spread = ppo_loop.standardised(paid, baseline)
        advantages[name] = advantage
        arms[name] = {
            "return_mean": round(float(paid.mean()), 5),
            "return_variance": by_stage(paid, stages),
            "advantage_variance": by_stage(paid - baseline, stages),
            "critic_explains": round(1.0 - float(((paid - baseline) ** 2).mean() / paid.var(unbiased=False)), 4),
        }
    arms["today"]["as_trained"] = {
        "advantage_variance": by_stage(torch.from_numpy(returns["today"]).to(device) - own, stages),
        "note": "today's return against the checkpoint's own critic, the baseline its run uses",
    }
    pooled = phi = own = None
    lap("critics")

    # Three random splits of the games into halves; the shuffled advantages
    # the null reads, one shuffle an arm over the whole round. Each arm's
    # gradients come as sums over the whole round and over each split's
    # first half, the second half being the whole less the first.
    splits = [torch.from_numpy(rng.permutation(args.games) < args.games // 2).to(device)[games]
              for _ in range(args.splits)]
    sizes = [(int(split.sum()), int((~split).sum())) for split in splits]
    vectors, layout = [], []
    for name, advantage in advantages.items():
        shuffled = advantage[torch.from_numpy(rng.permutation(len(advantage))).to(device)]
        for kind, values in (("real", advantage), ("null", shuffled)):
            vectors.append(values)
            layout.append((name, kind, None))
            for which, split in enumerate(splits):
                vectors.append(values * split)
                layout.append((name, kind, which))
    sums, how = head_gradients(net, rollout, torch.stack(vectors), args, device)
    vectors = None
    lap("gradients")

    root_v, preconditioned_by = preconditioner(net, saved, args, device)
    saved = None
    probe_planes, probe_masks, probe_games = probes(net, args, device)
    parts = probe_parts(net, probe_planes, probe_masks, args, device)
    before = probe_logits(net, parts, args, device)
    order = before.topk(2, dim=1).indices
    gap_before = before.gather(1, order).diff(dim=1).squeeze(1)
    best_before = order[:, 0]
    steps_per_generation = args.epochs * (batch.decisions // args.batch)
    scale = args.lr * steps_per_generation * args.flip_scale
    denominator = torch.cat(root_v) if root_v is not None else None
    lap("probes")

    def halves(name: str, kind: str, which: int) -> tuple[torch.Tensor, torch.Tensor]:
        whole_sum = sums[layout.index((name, kind, None))]
        first = sums[layout.index((name, kind, which))]
        return first / sizes[which][0], (whole_sum - first) / sizes[which][1]

    def read(step: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        moved = step * scale / denominator if denominator is not None else step * scale
        after = stepped_logits(net, moved, parts, args, device)
        flipped = (after.argmax(dim=1) != best_before).cpu().numpy()
        gap = after.gather(1, order).diff(dim=1).squeeze(1) - gap_before
        return flipped, gap.cpu().numpy()

    for name in advantages:
        measures = {"cosine": {}, "flips": {}, "gap_correlation": {}}
        flip_counts = []
        for kind in ("real", "null"):
            cosines, overlaps, gaps = [], [], []
            for which in range(args.splits):
                one, two = halves(name, kind, which)
                cosines.append(cosine(one, two))
                flips_one, gap_one = read(one)
                flips_two, gap_two = read(two)
                flip_counts.append((kind, int(flips_one.sum()), int(flips_two.sum())))
                enough = min(flips_one.sum(), flips_two.sum()) >= FEW_FLIPS
                overlaps.append(ochiai(flips_one, flips_two) if enough else None)
                gaps.append(correlation(gap_one, gap_two))
            measures["cosine"][kind] = [round(value, 5) for value in cosines]
            measures["flips"][kind] = [None if value is None else round(value, 5) for value in overlaps]
            measures["gap_correlation"][kind] = [None if value is None else round(value, 5) for value in gaps]
        for measure in measures.values():
            real, null = mean(measure["real"]), mean(measure["null"])
            measure["mean"] = None if real is None else round(real, 5)
            measure["null_mean"] = None if null is None else round(null, 5)
            measure["above_chance"] = None if real is None or null is None else round(real - null, 5)
        measures["flip_counts"] = flip_counts
        arms[name]["agreement"] = measures
    lap("agreement")

    verdict = agreement_verdict(arms, "today", arm_name(args.alphas[0]))
    if u_report["kill"]:
        verdict["verdict"] = "kill"
        verdict["why"] = ("U misses the bonus beyond noise in some tenth at some stage"
                          if not u_report["passed"] else
                          f"U never saw {u_report['unseen']} seated, whom a fork would seat")
    report = {
        "checkpoint": str(args.checkpoint),
        "generation": generation,
        "games": args.games,
        "seed": args.seed,
        "decisions": int(batch.decisions),
        "hands": int(batch.hands),
        "members": list(members),
        "u": {key: value for key, value in u_report.items() if key != "by_stage_decile"},
        "u_cells": u_report["by_stage_decile"],
        "arms": arms,
        "agreement_method": {
            "splits": args.splits, "gradient": how, "preconditioner": preconditioned_by,
            "steps_per_generation": steps_per_generation, "lr": args.lr, "flip_scale": args.flip_scale,
            "probes": len(probe_planes), "probe_games": probe_games, "top": args.top,
            "few_flips": FEW_FLIPS,
        },
        "verdict": verdict,
        "seconds": seconds,
        "saved": {"ledger": str(args.out / "ledger.npz"), "place_model": str(args.out / "place-model.pt")},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    # One line, last, for whoever reads the call's answer.
    # The literal reading of U's rule beside the one applied: tenths whose
    # gap is over 0.03 whatever their noise (see `place_model.calibration`).
    print("VERDICT " + json.dumps({"verdict": verdict, "u_passed": u_report["passed"],
                                   "u_largest_miss": u_report["largest_miss"],
                                   "u_cells_missing_literally": u_report["cells_missing_literally"],
                                   "u_unseen": u_report["unseen"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
