"""The noise pre-test: before an oracle critic or twin deals are paid for,
how much of the noise in a generation's step could either take away?

    python -m neural.noise_pretest checkpoint.pt --opponents A.pt B.pt club \\
        --seat-share 0.5 --games 4096 --out noise/

It plays one round of the checkpoint as its run plays one, at the same
tables and paid the reward its checkpoint carries (the placement reward,
settled by the checkpoint's own U; a checkpoint paid anything else is
refused), and keeps, for every decision the learner records, what the
deciding seat could not see: the oracle's planes, the three opponents'
concealed tiles, the next sixteen draws and the ura indicators
(`selfplay.play`'s `keep_oracle`). Then:

1. Every decision's return is measured against four baselines, none of
   them taught the deal it is asked about. `own` is the checkpoint's
   judge, the run's own baseline, which never saw the round. `warm` is a
   copy of it taught one half of the deals as the reward's warm-up teaches
   it and asked about the other half, and the other way round
   (`placement_pretest.cross_fitted_baseline`). `oracle` is the
   checkpoint's judge held still plus a term that starts at nought and
   reads the oracle's planes beside the position's own, through a tower
   shaped as our network's untrained oracle critic (`Correction`), taught
   on three quarters of a half's deals, kept at the epoch that read the
   fourth quarter best, and asked about the other half. `control` is the
   same term fed noughts for the oracle's planes: the same capacity and
   nothing hidden. The residual's spread is read plain and weighted by
   (1 - π(a))², overall, by stage and by tiles left, each with a jackknife
   error over groups of deals. What the oracle's planes add is what they
   take off the weighted residual of the better of the two baselines that
   see nothing hidden, the control and own; what the term's capacity adds,
   what the control takes off own's.
2. One gradient pass over the fusion head
   (`placement_pretest.head_gradients`) sums each arm's advantage, the
   return less own, the control or the oracle, centred and not
   standardised, over each group of deals: sixty vectors at twenty groups,
   of which only their products are kept, the Gram matrices, plain and
   with every sum divided by the checkpoint's AdamW √v̂. The gradient's
   noise N is the spread of an arm's group sums; the oracle's is read
   against own's and against the control's, each with a jackknife error
   over the groups.
3. From each decision's chance that two draws from the policy differ, it
   reads how much of the round's gradient twin copies of each deal could
   pair, game by game: the ceiling of duplicate deals, which needs no twin
   round.

Before anything is played, the gradient pass is tried on one chunk of
made-up rows and its time projected; a run whose projection, with what
the other phases are known to cost, is over `--max-minutes` stops there.
The registered readings (docs/TRAINING_SAFETY.md) end the output as one
line, VERDICT {...}. Everything goes to `--out`, which must be empty or
absent, every file written under a hidden name and renamed into place:
report.json, written once before the gradient pass and again at the end;
ledger.npz, the round's; baselines.npz, every decision's return,
baselines, log-probability, chance of a differing draw, stage, tiles
left, group and half, from which every table can be read again; and
gram.npz, the Gram matrices.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from . import combined, place_model, placement_pretest, ppo_loop, selfplay
from .checkpoints import staging_file
from .model import GROUPS, ORACLE_BLOCKS, ORACLE_CHANNELS, ORACLE_PLANES, Residual
from .observe import POSITIONS, Planes
from .population import Population
from .prefetch import Prefetcher
from .scoreboard import STAGE_NAMES

#: The registered readings (docs/TRAINING_SAFETY.md, 30 October 2026). The
#: oracle's planes go ahead only if they take at least this share off the
#: residual, weighted by (1 - π(a))², of the better of the two baselines
#: that see nothing hidden, the control and own,
INFORMATION_CUT = 0.30
#: and the gradient's noise under the oracle is at most this share of its
#: noise under the checkpoint's own judge,
NOISE_RATIO = 0.8
#: with that ratio plus two of its errors below this.
NOISE_BOUND = 0.9
#: The fit worked when the control's weighted residual is at most this many
#: times own's: a term that cannot even keep up with the judge it is added
#: to did not fit. A little over one, since the control is kept at the
#: epoch that read its own half's check deals best and is then read on the
#: other half, where it can come out a little above own by chance though it
#: fit; and the cut, read against the better of the two, gains nothing by
#: the slack.
CONTROL_SLACK = 1.02
#: Twin deals are built next only if they could pair at least this share of
#: the round's gradient-carrying draws, read game by game (`twin_ceiling`).
TWIN_CEILING = 0.20

#: The arms of the gradient pass, in the order its vectors are laid out:
#: an arm's groups one after another.
ARMS = ("own", "control", "oracle")
#: Every baseline the residual is read against.
BASELINES = ("own", "warm", "oracle", "control")

#: The plane of Mortal's thousand and twelve that holds the tiles left in
#: the live wall, over 69, at every position: libriichi's `encode_obs` at
#: version 4 writes it after the hand (4), the red fives (3), the scores
#: (8), the ranks (4), the hand of the round (4), the counters (2), the
#: winds (2), the hand of the game (1), the dora indicators (7), the seat's
#: own river (97) and the others' (3 x 195). A row whose plane is not one
#: number is read as unknown, and where fewer than sixteen tiles are left
#: the oracle's draws must count the same (`tiles_left_check`).
TILES_LEFT_PLANE = 717
TILES_LEFT_SCALE = 69
#: The oracle's planes of the next sixteen draws, one lit position for each
#: draw still to come, after the three opponents' four planes each
#: (`riichi-core`'s `encoding::oracle`).
DRAW_PLANES = slice(3 * 4, 3 * 4 + 16)
#: Tiles left are read in bins of ten: 0-9, 10-19, ..., 60-69.
LEFT_BIN = 10
LEFT_BINS = 7
LEFT_NAMES = tuple(f"{LEFT_BIN * index}-{LEFT_BIN * index + LEFT_BIN - 1}" for index in range(LEFT_BINS))

#: Epochs without a better reading of the check deals before a fit stops.
PATIENCE = 2
#: Where the fits' rate ends, as a share of where it starts.
FINAL_RATE = 0.01
#: The decisions a round of a005's holds a game, for projecting the
#: gradient pass before the round is played: 1.53 to 1.57 million a round
#: of 4,096 games at generations 2710 to 2756.
ROWS_A_GAME = 380
#: The fewest rows the preflight halves the gradient's chunk down to.
FEWEST_ROWS = 32
#: The most of the card the preflight's chunk may hold at its peak, the
#: network and the seated others included: the pass holds more than the
#: preflight did (every vector's weights over the whole round), and a
#: cuDNN failure may leave the process unable to go on.
CARD_SHARE = 0.8
#: What a failure to find room on the card says when it is not PyTorch's
#: own out-of-memory error: cuBLAS's and cuDNN's allocations fail as plain
#: RuntimeErrors, and a convolution's backward without room has been seen
#: to fail as CUDNN_STATUS_EXECUTION_FAILED_CUDART.
MEMORY_SIGNS = ("out of memory", "CUDNN_STATUS_ALLOC_FAILED", "CUDNN_STATUS_EXECUTION_FAILED",
                "CUBLAS_STATUS_ALLOC_FAILED")
#: What every phase after the preflight but the gradient pass is known to
#: cost, in minutes, at a round of `PHASE_GAMES` games on the L40S, at the
#: top of each estimate: play with the oracle's planes, the critics'
#: inputs, the warm-up, the oracle's term and its control at ten epochs,
#: and the readings. Scaled by the round's games, they are added to the
#: minutes spent and the preflight's projection of the gradient pass
#: before anything is played (`--max-minutes`).
PHASE_MINUTES = {"play": 14, "critic inputs": 3, "warm": 1, "oracle": 8, "readings": 1}
PHASE_GAMES = 4096


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path,
                        help="the joined player to test: one a placement run wrote, with its U and its "
                        "optimiser's state")
    parser.add_argument("--opponents", type=Path, nargs="*", default=[],
                        help="the others its run seats, as the cloud stages them (see neural.population)")
    parser.add_argument("--seat-share", type=float, default=0.0)
    parser.add_argument("--opponent-share", type=float, default=0.0,
                        help="nought: the others are seated by player, as the run seats them; taken because "
                        "the cloud's staging names it")
    parser.add_argument("--games", type=int, default=4096, help="the round's games, each a deal of its own")
    parser.add_argument("--seed", type=int, default=9_700_000, help="the round's first deal")
    parser.add_argument("--groups", type=int, default=20,
                        help="groups of deals the errors are read over and the gradient summed over")
    parser.add_argument("--batch", type=int, default=2048, help="the critics' minibatch")
    parser.add_argument("--chunk", type=int, default=1024, help="rows a pass over the round reads at once")
    parser.add_argument("--gradient-chunk", type=int, default=512,
                        help="rows the gradient pass reads at once; its products hold one a vector")
    parser.add_argument("--epochs", type=int, default=2, help="the warm-up's epochs, for `warm`")
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--value-warm-up-lr", type=float, default=1e-4, help="the warm-up's rate, for `warm`")
    parser.add_argument("--critic-epochs", type=int, default=10,
                        help="the most epochs the oracle's term and its control are taught")
    parser.add_argument("--critic-lr", type=float, default=1e-3,
                        help="their rate, falling on a cosine to a hundredth of it")
    parser.add_argument("--check-share", type=float, default=0.25,
                        help="the share of a half's deals its fit is checked on rather than taught")
    parser.add_argument("--still-falling", type=float, default=0.002,
                        help="a fit whose best epoch is its last, still this share below the one before, "
                        "has not finished, and the oracle's reading is then unread")
    parser.add_argument("--tower-channels", type=int, default=ORACLE_CHANNELS,
                        help="the term's tower, as wide as our network's oracle critic's")
    parser.add_argument("--tower-blocks", type=int, default=ORACLE_BLOCKS)
    parser.add_argument("--lr", type=float, default=2e-6,
                        help="the head's rate; read only by the optimiser rebuilt for its second moments")
    parser.add_argument("--no-preview-reach", dest="preview_reach", action="store_false",
                        help="ask a reach's tile in a forward of its own, as a run without --preview-reach does")
    parser.add_argument("--amp", action="store_true", help="play and read in bfloat16 on the card, as the run does")
    parser.add_argument("--max-minutes", type=float, default=75.0,
                        help="the most the run may take: once the preflight has timed the gradient's chunk, a "
                        "run expected to take longer stops before anything is played")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True,
                        help="an empty folder, or one not there yet: every run writes a folder of its own")
    args = parser.parse_args(argv)
    for name in ("games", "groups", "batch", "chunk", "gradient_chunk", "epochs", "critic_epochs",
                 "tower_channels", "tower_blocks", "max_minutes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.groups < 3:
        parser.error("--groups must be at least three: an error is read with each group left out")
    if args.games < max(4, args.groups):
        parser.error("--games must be at least four and at least --groups: every group needs a deal, and "
                     "each half one deal to teach and one to check")
    if not 0 < args.check_share < 1:
        parser.error("--check-share must be above nought and below one")
    if args.still_falling < 0:
        parser.error("--still-falling must be nonnegative")
    if args.tower_channels % GROUPS:
        parser.error(f"--tower-channels must be a multiple of {GROUPS}, the tower's normalisation groups")
    if args.opponent_share:
        parser.error("the noise pre-test seats its others by player, as the run does: --opponent-share must "
                     "be nought")
    return args


def free_card(device: str) -> None:
    """The card's cached blocks handed back before a pass, so that a pass
    needing more than the last one left free finds it there."""
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def number(value, digits: int = 6):
    """A float for the report, to `digits` significant figures, or None
    where it is not a number. Figures rather than places: N, preconditioned,
    runs to millions and an error of a cut to thousandths."""
    value = float(value)
    return float(f"{value:.{digits}g}") if math.isfinite(value) else None


def refuse_kept(out: Path) -> None:
    """Refuses an `--out` that already holds anything: one folder a run, so
    that no file an earlier run left there is read as this one's."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"{out} is not an empty folder: the noise pre-test writes a folder of its own, so that "
                         "nothing an earlier run left there can be read as this run's")


def kept(path: Path, write) -> None:
    """`write` handed a stream to a hidden file beside `path`, which is
    renamed onto `path` once whole (`checkpoints.staging_file`): a run
    stopped while it writes leaves the file as it was, never a part of
    one."""
    with staging_file(path) as staged:
        with staged.open("wb") as stream:
            write(stream)
        os.replace(staged, path)


# ---------------------------------------------------------------- the round


def carried_reward(payload: dict) -> place_model.PlacementReward:
    """The reward the checkpoint's run is paid, as the checkpoint carries
    it: the placement reward at its alpha, settled by its U (see
    `place_model.PlacementRun.fields`). Anything else is refused: the
    pre-test reads the noise of the run's own return, and only a placement
    run's checkpoint says what that return is."""
    record = payload.get("reward")
    if not isinstance(record, dict) or record.get("kind") != "placement":
        raise SystemExit("the checkpoint carries no placement reward: the noise pre-test reads the return a "
                         "placement run is paid, settled by the U its checkpoint carries")
    if payload.get("place_model") is None:
        raise SystemExit("the checkpoint says it is paid the placement reward but carries no U to settle it by")
    return place_model.PlacementReward(place_model.PlaceModel.from_state(payload["place_model"]),
                                       float(record["alpha"]))


def deal_draws(games: int, groups: int, check_share: float, seed: int) -> tuple[np.ndarray, ...]:
    """For every deal, which here is a game, each dealt from a seed of its
    own: the group its decisions are read and summed in, the half whose
    critics are taught it (0 or 1), and whether it is one of the deals that
    half's fit is checked on. Groups are balanced and halves even; each half
    checks on `check_share` of its deals, at least one, and teaches on the
    rest, at least one. A deal's decisions never land on two sides."""
    rng = np.random.default_rng(seed)
    group = rng.permutation(games) % groups
    half = (rng.permutation(games) >= games // 2).astype(np.int64)
    check = np.zeros(games, dtype=bool)
    for side in (0, 1):
        mine = np.flatnonzero(half == side)
        kept = min(len(mine) - 1, max(1, int(round(check_share * len(mine)))))
        check[rng.permutation(mine)[:kept]] = True
    return group, half, check


def disagreement(logits: torch.Tensor) -> torch.Tensor:
    """1 - Σπ², the chance that two draws from the policy differ, in double
    precision, as Σπ(1 - π): for the likeliest move 1 - π is the sum of all
    the others' chances, and for the rest, each at most a half, 1 - π
    itself. So a row the policy is all but sure of keeps its last digits
    instead of losing them to 1 less a number all but one."""
    probs = torch.softmax(logits.double(), dim=1)
    top = probs.argmax(dim=1, keepdim=True)
    unlike = 1.0 - probs
    unlike.scatter_(1, top, probs.scatter(1, top, 0.0).sum(dim=1, keepdim=True))
    return (probs * unlike).sum(dim=1)


@torch.no_grad()
def round_inputs(net, rollout, args, device: str) -> dict[str, torch.Tensor]:
    """What the critics read of every decision, as
    `placement_pretest.critic_inputs` reads it: our network's pooled
    features and Mortal's vector, both held back from the critic, and the
    checkpoint's own judge. From the same pass, two things more of each
    decision: the chance two draws from its policy differ (`disagreement`),
    and the tiles left in the wall, read from Mortal's plane of them (-1
    where the plane was not one number at every position, which would mean
    it is not the plane counted, see `TILES_LEFT_PLANE`).

    Every row is checked in full on its way to the card here, so the round
    is trusted from then on, as after the pass that values a trainer's
    round (see `ppo_loop.baseline`)."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    pooled, phi, own, differ, left = [], [], [], [], []
    for start in range(0, rollout.decisions, args.chunk):
        stop = min(start + args.chunk, rollout.decisions)
        legal = rollout.legal[start:stop]
        planes = rollout.observations.slice(start, stop).dense(device, dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            vector, q, features, a1, value, _guessed = net.backbones(planes, legal, hands=False)
            judged = net.fuse.judge(vector.float(), value.float())
            logits = net.fuse(vector.float(), q.float(), features.float(), a1.float(), legal)
        pooled.append(features.mean(dim=2).to(dtype))
        phi.append(vector.to(dtype))
        own.append(judged.float())
        differ.append(disagreement(logits))
        wall = planes[:, TILES_LEFT_PLANE, :].float()
        tiles = torch.round(wall[:, 0] * TILES_LEFT_SCALE).long()
        left.append(torch.where(wall.amax(dim=1) == wall.amin(dim=1), tiles, torch.full_like(tiles, -1)))
    observations = rollout.observations
    if isinstance(observations, Planes) and not observations.trusted:
        rollout.observations = Planes(observations.indptr, observations.indices, observations.values, trusted=True)
    return {"pooled": torch.cat(pooled), "phi": torch.cat(phi), "own": torch.cat(own),
            "differ": torch.cat(differ), "left": torch.cat(left)}


def tiles_left_check(left: np.ndarray, oracle: np.ndarray) -> dict:
    """Mortal's plane of the tiles left read against the oracle's draws:
    where fewer than sixteen tiles are left, the oracle lights one plane for
    each draw still to come, so the two must say the same. How many rows
    could be checked, the share that agreed, and how many rows' plane was
    not one number."""
    draws = np.empty(len(left), dtype=np.int64)
    for start in range(0, len(left), 65536):
        draws[start:start + 65536] = oracle[start:start + 65536, DRAW_PLANES].any(axis=2).sum(axis=1)
    checked = (draws < DRAW_PLANES.stop - DRAW_PLANES.start) | ((left >= 0) & (left < 16))
    rows = int(checked.sum())
    return {"rows": rows, "agreeing": number((draws == left)[checked].mean()) if rows else None,
            "unread_rows": int((left < 0).sum())}


# ------------------------------------------------- the oracle's term and its control


class Correction(nn.Module):
    """What a critic adds to the checkpoint's judge, read from the
    position's planes and the oracle's together: a tower shaped as our
    network's oracle critic (`oracle_stem`, `oracle_tower`, `oracle_tail`
    and `oracle_value` of `model.PolicyValueNet`), whose value reads our
    pooled features beside the tower's own, pooled over the line. Its last
    layer starts at nought, so it adds nothing until it has learned
    something.

    `blind` feeds it noughts for the oracle's planes, whatever it is
    handed: the control, the same capacity seeing nothing hidden."""

    def __init__(self, planes: int, channels: int, width: int = ORACLE_CHANNELS, blocks: int = ORACLE_BLOCKS,
                 attention: bool = True, blind: bool = False) -> None:
        super().__init__()
        self.blind = blind
        self.stem = nn.Sequential(
            nn.Conv1d(planes + ORACLE_PLANES, width, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, width),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[Residual(width, attention) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.GroupNorm(GROUPS, width), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(channels + width, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, pooled: torch.Tensor, planes: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
        if self.blind:
            oracle = torch.zeros_like(oracle)
        together = torch.cat([planes, oracle.to(planes.dtype)], dim=1)
        hidden = self.tail(self.tower(self.stem(together))).mean(dim=2)
        return self.value(torch.cat([pooled.float(), hidden.float()], dim=1)).squeeze(1)


def correction_pair(net, args, seed: int, device: str) -> dict[str, Correction]:
    """The oracle's term and its control, made at one seed, so the two
    start the same in every weight; the control is blind (see
    `Correction`)."""
    torch.manual_seed(seed)
    seeing = Correction(net.planes, net.channels, args.tower_channels, args.tower_blocks,
                        net.ours.attention).to(device)
    blind = copy.deepcopy(seeing)
    blind.blind = True
    return {"oracle": seeing, "control": blind}


def staged(rollout, oracle: np.ndarray, batches, device: str, dtype: torch.dtype):
    """Each array of rows in `batches` as (rows, planes, oracle's planes)
    on the device. The host's share, the rows' sparse planes and their
    oracle's planes, both kept on the host since the round's planes dense
    would be tens of gigabytes, is gathered a few batches ahead in a thread
    of its own (`prefetch.Prefetcher`)."""
    def prepare(rows: np.ndarray):
        return rows, rollout.observations.rows(rows), np.ascontiguousarray(oracle[rows])

    with Prefetcher(batches, prepare) as ready:
        for rows, sparse, hidden in ready:
            yield (torch.from_numpy(rows).to(device), sparse.dense(device, dtype),
                   torch.from_numpy(hidden).to(device).to(dtype))


@torch.no_grad()
def corrections_of(heads: dict, rollout, pooled, oracle: np.ndarray, rows: np.ndarray, args,
                   device: str) -> dict[str, torch.Tensor]:
    """What each term of `heads` adds for decisions `rows`, a chunk at a time."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    out = {name: torch.empty(len(rows), device=device) for name in heads}
    for head in heads.values():
        head.eval()
    batches = [rows[at:at + args.chunk] for at in range(0, len(rows), args.chunk)]
    at = 0
    for picks, planes, hidden in staged(rollout, oracle, batches, device, dtype):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for name, head in heads.items():
                out[name][at:at + len(picks)] = head(pooled[picks], planes, hidden).float()
        at += len(picks)
    return out


def check_errors(heads: dict, rollout, pooled, oracle: np.ndarray, target: torch.Tensor, checked: np.ndarray,
                 args, device: str) -> dict[str, float]:
    """Each term's mean squared miss of `target` over the check rows."""
    added = corrections_of(heads, rollout, pooled, oracle, checked, args, device)
    wanted = target[torch.from_numpy(checked).to(device)]
    return {name: float(((wanted - said) ** 2).mean()) for name, said in added.items()}


def still_falling(curve: list[float], planned: int, tolerance: float) -> bool:
    """Whether a fit had not finished. `curve` is its error on the check
    deals before the first epoch and after each: it had not finished when
    it ran every one of its `planned` epochs, its best was its last, and
    that last was still more than `tolerance` of the one before below it,
    so that more epochs might yet have read the check deals better."""
    if len(curve) - 1 < planned or min(curve) < curve[-1]:
        return False
    return curve[-1] < curve[-2] * (1.0 - tolerance)


def fit_corrections(net, rollout, pooled, oracle: np.ndarray, target: torch.Tensor, taught: np.ndarray,
                    checked: np.ndarray, args, seed: int, device: str) -> tuple[dict, dict]:
    """The oracle's term and its control taught `target`, the return less
    the checkpoint's judge, on decisions `taught`: the same minibatches in
    the same order for both, AdamW at `--critic-lr` falling on a cosine to
    a hundredth of it over `--critic-epochs`, the loss and the clip the
    warm-up teaches the critic with. After every epoch each reads the
    decisions `checked`, and each is kept at its best epoch, the start
    counted as epoch nought: a term no epoch taught to read them better
    than nought does stays nought, the checkpoint's judge. The fit stops
    once neither has read them better for `PATIENCE` epochs. Answers both
    terms and how the fit went."""
    amp = args.amp and device == "cuda"
    dtype = torch.bfloat16 if amp else torch.float32
    heads = correction_pair(net, args, seed, device)
    optimisers = {name: torch.optim.AdamW(head.parameters(), lr=args.critic_lr, weight_decay=1e-4)
                  for name, head in heads.items()}
    rng = np.random.default_rng(seed)
    total = max(1, args.critic_epochs * math.ceil(len(taught) / args.batch))
    lowest = args.critic_lr * FINAL_RATE

    def rate(step: int) -> float:
        return lowest + (args.critic_lr - lowest) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))

    # Both start at nought, so their first reading is the judge's own.
    before = check_errors(heads, rollout, pooled, oracle, target, checked, args, device)["oracle"]
    curves: dict[str, list[float]] = {name: [before] for name in heads}
    best = {name: (before, copy.deepcopy(head.state_dict()), 0) for name, head in heads.items()}
    step = 0
    for epoch in range(1, args.critic_epochs + 1):
        for head in heads.values():
            head.train()
        order = rng.permutation(taught)
        batches = [order[at:at + args.batch] for at in range(0, len(order), args.batch)]
        for rows, planes, hidden in staged(rollout, oracle, batches, device, dtype):
            wanted, seen = target[rows], pooled[rows]
            for name, head in heads.items():
                for group in optimisers[name].param_groups:
                    group["lr"] = rate(step)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    said = head(seen, planes, hidden)
                loss = args.value_weight * nn.functional.mse_loss(said.float(), wanted)
                optimisers[name].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimisers[name].step()
            step += 1
        errors = check_errors(heads, rollout, pooled, oracle, target, checked, args, device)
        for name in heads:
            curves[name].append(errors[name])
            if errors[name] < best[name][0]:
                best[name] = (errors[name], copy.deepcopy(heads[name].state_dict()), epoch)
        if all(epoch - best[name][2] >= PATIENCE for name in heads):
            break
    account = {"steps": step}
    for name, head in heads.items():
        head.load_state_dict(best[name][1])
        head.eval()
        account[name] = {
            # Before the first epoch, and after each.
            "check_errors": [number(error) for error in curves[name]],
            "best_epoch": best[name][2],
            "still_falling": still_falling(curves[name], args.critic_epochs, args.still_falling),
        }
    return heads, account


def cross_fitted_corrections(net, rollout, inputs: dict, oracle: np.ndarray, returns: torch.Tensor,
                             half: np.ndarray, check: np.ndarray, args, device: str) -> tuple[dict, list]:
    """The oracle's term and its control, a pair taught on each half of the
    deals (`fit_corrections`, checked on that half's own check deals) and
    asked about the other half: every decision's two baselines, the
    checkpoint's judge plus each term, from terms that never saw its deal.
    Answers both baselines and how each half's fit went."""
    own = inputs["own"]
    target = returns - own
    baselines = {name: own.clone() for name in ("oracle", "control")}
    fits = []
    for side in (0, 1):
        taught = np.flatnonzero((half == side) & ~check)
        checked = np.flatnonzero((half == side) & check)
        asked = np.flatnonzero(half != side)
        heads, account = fit_corrections(net, rollout, inputs["pooled"], oracle, target, taught, checked, args,
                                         args.seed + 11 + side, device)
        added = corrections_of(heads, rollout, inputs["pooled"], oracle, asked, args, device)
        index = torch.from_numpy(asked).to(device)
        for name, said in added.items():
            baselines[name][index] = own[index] + said
        fits.append({"half": side, "taught_rows": len(taught), "checked_rows": len(checked), **account})
        heads = added = None
        free_card(device)
    return baselines, fits


# --------------------------------------------------------------- the readings


def cells_of(stages: np.ndarray, left: np.ndarray) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """How the readings split the decisions, each into labelled cells: all
    of them together, by stage, and by tiles left in tens. A row whose tiles
    left were not read is in no cell of that split (-1)."""
    bins = np.where(left >= 0, np.minimum(left // LEFT_BIN, LEFT_BINS - 1), -1)
    return {
        "all": (np.zeros(len(stages), dtype=np.int64), ("all",)),
        "by_stage": (np.asarray(stages, dtype=np.int64), STAGE_NAMES),
        "by_tiles_left": (bins.astype(np.int64), LEFT_NAMES),
    }


def group_sums(values: np.ndarray, weights: np.ndarray, cells: np.ndarray, group: np.ndarray, groups: int,
               count: int) -> np.ndarray:
    """Σw, Σw·v and Σw·v² over each group's decisions in each cell, (groups,
    count, 3), in double precision: what a weighted variance is read from,
    with every group or with any one left out."""
    inside = cells >= 0
    at = group[inside] * count + cells[inside]
    w = np.asarray(weights, dtype=np.float64)[inside]
    v = np.asarray(values, dtype=np.float64)[inside]
    size = groups * count
    return np.stack([np.bincount(at, w, size), np.bincount(at, w * v, size), np.bincount(at, w * v * v, size)],
                    axis=-1).reshape(groups, count, 3)


def variance(sums: np.ndarray) -> np.ndarray:
    """The weighted variance read from (Σw, Σw·v, Σw·v²): not a number
    where a cell holds no weight."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums[..., 1] / sums[..., 0]
        return sums[..., 2] / sums[..., 0] - mean * mean


def jackknifed(statistic, *sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`statistic` of the variances of `sums` over every group, cell by
    cell, and its delete-a-group jackknife error: the statistic read again
    with each group left out in turn. Nothing is taught again without the
    group, so this is the error of reading the critics on these deals and
    not of teaching them there: a lower bound."""
    groups = sums[0].shape[0]
    whole = statistic(*(variance(part.sum(axis=0)) for part in sums))
    left = np.stack([statistic(*(variance(part.sum(axis=0) - part[g]) for part in sums)) for g in range(groups)])
    error = np.sqrt((groups - 1) / groups * ((left - left.mean(axis=0)) ** 2).sum(axis=0))
    return whole, error


def labelled(values: np.ndarray, errors: np.ndarray, names: tuple[str, ...]) -> dict:
    return {name: {"value": number(values[index]), "se": number(errors[index])} for index, name in enumerate(names)}


def residual_readings(returns: np.ndarray, baselines: dict[str, np.ndarray], weights: np.ndarray,
                      cells: dict, group: np.ndarray, groups: int) -> tuple[dict, dict]:
    """V = Var(G - b) for each baseline, plain and weighted, in every cell
    of every split, with jackknife errors; and the sums each was read from,
    which the cuts read again."""
    plain = np.ones(len(returns))
    readings, sums = {}, {}
    for name, baseline in baselines.items():
        residual = returns - baseline
        readings[name] = {}
        for kind, weight in (("plain", plain), ("weighted", weights)):
            readings[name][kind] = {}
            for split, (labels, names) in cells.items():
                summed = group_sums(residual, weight, labels, group, groups, len(names))
                sums[name, kind, split] = summed
                readings[name][kind][split] = labelled(*jackknifed(lambda v: v, summed), names)
    return readings, sums


def cut(sums: dict, top: str, bottom: str | tuple[str, ...], cells: dict) -> dict:
    """1 - V(top) / V(bottom), plain and weighted, in every cell of every
    split, with jackknife errors: the share of the residual `top` takes off
    `bottom`'s, read on the same decisions. Given more than one `bottom`,
    the cut is read against whichever of them has the least V, cell by cell
    and again with each group left out."""
    bottoms = (bottom,) if isinstance(bottom, str) else tuple(bottom)
    out = {}
    for kind in ("plain", "weighted"):
        out[kind] = {}
        for split, (_labels, names) in cells.items():
            value, error = jackknifed(lambda a, *b: 1.0 - a / np.minimum.reduce(b), sums[top, kind, split],
                                      *(sums[name, kind, split] for name in bottoms))
            out[kind][split] = labelled(value, error, names)
    return out


def overall(sums: dict, name: str, kind: str = "weighted") -> float:
    """V over every decision, as read, unrounded."""
    return float(variance(sums[name, kind, "all"].sum(axis=0))[0])


def twin_ceiling(differ: np.ndarray, games: np.ndarray, players: np.ndarray) -> dict:
    """How much of the round's gradient twin copies of each deal could pair.

    `differ[i]` is 1 - Σπ² at decision i, the chance two independent draws
    there differ; it is also what a draw carries of the gradient on
    average, Σ_a π_a |e_a - π|² over the logits. Two copies of a deal are
    one game until a draw of any of its learner seats first differs: over
    the game's learner decisions, every seat's in one sequence, in the
    order they were made, they part at decision i with chance
    Π_{j<i} (1 - differ_j) · differ_i. Only that parting pairs two draws
    from one state; every draw after it, any seat's, is made in two
    different games. The ceiling is the expected partings over Σ differ,
    the round's differing draws, read game by game, as registered. Even
    that is a ceiling: a differing answer to a call that a ron or a pon
    overrides does not part the copies. The same read seat by seat, each
    seat's decisions a sequence of their own as though the other seats'
    draws could not part the game, is reported beside it and not read: a
    game parts once, so it is never below the reading game by game, and
    where each seat's own draws would all but surely part the copies, it
    counts a game's parting about once for each of its learner seats."""
    differ = np.asarray(differ, dtype=np.float64)
    carried = float(differ.sum())

    def partings(*keys: np.ndarray) -> tuple[float, int]:
        # Each sequence's decisions together, in the order they were made.
        order = np.lexsort((np.arange(len(differ)), *keys[::-1]))
        chance = differ[order]
        same = np.log1p(-np.clip(chance, 0.0, 1.0 - 1e-12))
        # The chance the copies are still one game before each decision:
        # the sum of the logarithms before it, less those of the
        # sequences before its own.
        before = np.cumsum(same) - same
        sorted_keys = [np.asarray(key)[order] for key in keys]
        new = np.ones(len(chance), dtype=bool)
        new[1:] = np.any([key[1:] != key[:-1] for key in sorted_keys], axis=0)
        starts = np.flatnonzero(new)
        base = before[starts][np.cumsum(new) - 1]
        return float((np.exp(before - base) * chance).sum()), len(starts)

    by_seat, seats = partings(np.asarray(games, dtype=np.int64), np.asarray(players, dtype=np.int64))
    by_game, played = partings(np.asarray(games, dtype=np.int64))
    return {
        "decisions": int(len(differ)),
        "differing_draws": number(carried),
        "by_seat": {"sequences": seats, "expected_partings": number(by_seat),
                    "share": number(by_seat / carried) if carried > 0 else None},
        "by_game": {"sequences": played, "expected_partings": number(by_game),
                    "share": number(by_game / carried) if carried > 0 else None},
    }


# ------------------------------------------------------------- the gradients


def made_up_rows(rows: int, planes: int, rng, device: str) -> SimpleNamespace:
    """`rows` decisions made up for the preflight: planes about as sparse
    as a round's, one value in eighteen lit, every move legal, a move drawn
    at random. The pass's cost is the dense forward and the products, which
    do not depend on what the planes say."""
    width = planes * POSITIONS
    lit = width // 18
    indices = np.sort(rng.choice(width, size=(rows, lit)), axis=1).astype(np.uint16)
    sparse = Planes(np.arange(rows + 1, dtype=np.int64) * lit, indices.reshape(-1),
                    rng.random(rows * lit).astype(np.float16))
    legal = torch.ones(rows, combined.ACTIONS, dtype=torch.bool, device=device)
    actions = torch.from_numpy(rng.integers(0, combined.ACTIONS, size=rows)).to(device)
    return SimpleNamespace(observations=sparse, legal=legal, actions=actions, decisions=rows)


def synchronised(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def out_of_memory(error: BaseException) -> bool:
    """Whether `error` is the card running out of room: PyTorch's own
    error, or a RuntimeError of cuBLAS's or cuDNN's that says so
    (`MEMORY_SIGNS`)."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and any(sign in str(error) for sign in MEMORY_SIGNS))


def card_peak(device: str) -> tuple[int, int] | None:
    """The most this process has held on the card since its peak was last
    reset (`torch.cuda.max_memory_allocated`), and all the card holds, in
    bytes; None off the card."""
    if not str(device).startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(device), torch.cuda.get_device_properties(device).total_memory


def preflight(net, args, device: str) -> dict:
    """The gradient pass tried before the round is paid for, on one chunk of
    `--gradient-chunk` made-up rows with as many vectors as the pass will
    take. A chunk the card has no room for (`out_of_memory`), or one whose
    peak holds more than `CARD_SHARE` of the card, is halved, down to
    `FEWEST_ROWS`, and the pass then reads that many rows at once. On the
    card, products that cannot be taken batched for any other reason end
    the run here: one at a time they would outlast the call. The second of
    two tries is timed, the first having warmed the card, and the time
    projected over a round of a005's size (`ROWS_A_GAME`)."""
    cuda = str(device).startswith("cuda")
    vectors = len(ARMS) * args.groups
    rng = np.random.default_rng(args.seed ^ 0x5EED)
    while True:
        rows = args.gradient_chunk
        # Why the chunk failed, if it did, and whether for want of room.
        failed, roomy, peak = None, True, None
        free_card(device)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        try:
            made = made_up_rows(rows, net.planes, rng, device)
            weights = torch.from_numpy(rng.standard_normal((vectors, rows)).astype(np.float32)).to(device)
            times = []
            for _try in range(2):
                began = time.perf_counter()
                _, how = placement_pretest.head_gradients(
                    net, made, weights, SimpleNamespace(amp=args.amp, chunk=rows), device, fallback=not cuda)
                synchronised(device)
                times.append(time.perf_counter() - began)
            _ = made = weights = None
            peak = card_peak(device)
        except (RuntimeError, NotImplementedError) as error:
            failed, roomy = repr(error), out_of_memory(error)
        made = weights = _ = None
        if failed is None and peak is not None and peak[0] > CARD_SHARE * peak[1]:
            failed = f"its peak held {peak[0] / peak[1]:.0%} of the card"
        if failed is None:
            break
        if not roomy:
            raise SystemExit(f"the gradient's {vectors} products cannot be taken at once here ({failed}); "
                             "one at a time they would outlast the call")
        if rows // 2 < FEWEST_ROWS:
            raise SystemExit(f"the card has no room for the gradient's {vectors} products even at {rows} rows "
                             f"({failed[:300]})")
        free_card(device)
        args.gradient_chunk = rows // 2
        print(json.dumps({"preflight": f"no room at {rows} rows a chunk ({failed[:300]}); "
                                       f"trying {args.gradient_chunk}"}), flush=True)
    free_card(device)
    expected = args.games * ROWS_A_GAME
    return {"rows_a_chunk": args.gradient_chunk, "vectors": vectors, "how": how,
            "seconds_a_chunk": round(times[-1], 3), "first_try_seconds": round(times[0], 3),
            "peak_bytes": None if peak is None else peak[0], "card_bytes": None if peak is None else peak[1],
            "projected_rows": expected,
            "projected_minutes": round(math.ceil(expected / args.gradient_chunk) * times[-1] / 60, 1)}


def budget(flight: dict, spent: float, args) -> dict:
    """What the whole run is expected to take, in minutes, once the
    preflight has timed the gradient's chunk: the minutes spent so far, the
    other phases' known costs (`PHASE_MINUTES`) scaled to the round's games,
    and the gradient pass as the preflight projects it; and the most it may
    take (`--max-minutes`)."""
    others = sum(PHASE_MINUTES.values()) * args.games / PHASE_GAMES
    return {"spent": round(spent, 1), "other_phases": round(others, 1), "gradient": flight["projected_minutes"],
            "minutes": round(spent + others + flight["projected_minutes"], 1), "most": args.max_minutes}


def gradient_sums(net, rollout, weights: torch.Tensor, args, device: str) -> tuple[torch.Tensor, str, int]:
    """`placement_pretest.head_gradients` over the round, `--gradient-chunk`
    rows at a time. A chunk the card has no room for (`out_of_memory`) is
    taken again at half its rows, as often as it takes, and the rest of the
    pass is read at the size that went through, `args.gradient_chunk` being
    left there: what failed once would fail again, at a chunk's cost each
    time. On the card a batched product that fails for any other reason
    raises; on the processor it falls back to one at a time, as it always
    did. Answers the sums, how the products were taken, and how many times
    the chunk was halved."""
    cuda = str(device).startswith("cuda")
    total, how, halved = None, "batched", 0
    start = 0
    while start < rollout.decisions:
        stop = min(start + args.gradient_chunk, rollout.decisions)
        part = SimpleNamespace(observations=rollout.observations.slice(start, stop),
                               legal=rollout.legal[start:stop], actions=rollout.actions[start:stop],
                               decisions=stop - start)
        sums = None
        try:
            sums, taken = placement_pretest.head_gradients(
                net, part, weights[:, start:stop], SimpleNamespace(amp=args.amp, chunk=stop - start), device,
                fallback=not cuda)
        except RuntimeError as error:
            if not out_of_memory(error) or stop - start < 2:
                raise
        if sums is None:
            # Out of the exception's reach, which holds the failed chunk's
            # tensors until it is let go of.
            part = None
            free_card(device)
            args.gradient_chunk = (stop - start) // 2
            halved += 1
            continue
        total = sums if total is None else total.add_(sums)
        if taken != "batched":
            how = taken
        start = stop
    return total, how, halved


def advantage_weights(returns: torch.Tensor, baselines: dict[str, torch.Tensor], group: torch.Tensor,
                      groups: int) -> torch.Tensor:
    """The rows of weights the gradient pass takes, arm after arm and group
    after group (`ARMS`): each arm's advantage, the return less its
    baseline, centred over the round as a generation centres it but not
    standardised, since N is read in the advantages' own units and dividing
    each arm by its own spread would hide what its baseline took away; on
    the group's decisions, and nought on the others."""
    rows = []
    for arm in ARMS:
        advantage = returns - baselines[arm]
        advantage = advantage - advantage.mean()
        for index in range(groups):
            rows.append(torch.where(group == index, advantage, torch.zeros_like(advantage)))
    return torch.stack(rows)


def grams(sums: torch.Tensor, denominator: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray | None]:
    """The products of every pair of the pass's sums, in double precision:
    as they are, and with each sum divided coordinate by coordinate by
    √v̂ + ε, as AdamW's next steps divide the gradient (None when the
    checkpoint carries no moments that fit)."""
    wide = sums.double()
    plain = (wide @ wide.T).cpu().numpy()
    if denominator is None:
        return plain, None
    scaled = wide / denominator.double().unsqueeze(0)
    return plain, (scaled @ scaled.T).cpu().numpy()


def noise(gram: np.ndarray) -> float:
    """N: the spread of the group sums whose products `gram` holds, summed
    over the parameters, (tr K - 1ᵀK1/G) / (G - 1). Every arm's baseline
    leaves the gradient's expectation where it was, being taught none of
    the deals it is asked about; it can only spread the sums less."""
    groups = len(gram)
    return float((np.trace(gram) - gram.sum() / groups) / (groups - 1))


def noise_readings(gram: np.ndarray, groups: int) -> dict:
    """N for each arm, and the oracle's against own's and against the
    control's (and the control's against own's), each with its
    delete-a-group jackknife error over the groups."""
    blocks = {arm: np.arange(index * groups, (index + 1) * groups) for index, arm in enumerate(ARMS)}
    everyone = np.arange(groups)

    def of(arm: str, kept: np.ndarray) -> float:
        rows = blocks[arm][kept]
        return noise(gram[np.ix_(rows, rows)])

    def reading(statistic) -> dict:
        whole = statistic(everyone)
        left = np.array([statistic(np.delete(everyone, index)) for index in range(groups)])
        error = math.sqrt((groups - 1) / groups * float(((left - left.mean()) ** 2).sum()))
        return {"value": number(whole), "se": number(error)}

    return {
        "N": {arm: reading(lambda kept, arm=arm: of(arm, kept)) for arm in ARMS},
        "oracle_over_own": reading(lambda kept: of("oracle", kept) / of("own", kept)),
        "oracle_over_control": reading(lambda kept: of("oracle", kept) / of("control", kept)),
        "control_over_own": reading(lambda kept: of("control", kept) / of("own", kept)),
    }


# ----------------------------------------------------------------- the verdict


def oracle_verdict(information: dict, gate: dict, unfinished: list[str], ratio: dict | None) -> dict:
    """The registered reading of the oracle's planes (see the module's
    constants): unread when the fit did not work or had not finished, or
    when the gradient's noise was not read; go when both the weighted
    information cut, read against the better of the control and own, and
    the noise ratio clear their bars; kill otherwise."""
    out = {"information_cut": information, "noise_ratio": ratio, "gate": gate, "unfinished": unfinished,
           "needed": {"information_cut": INFORMATION_CUT, "noise_ratio": NOISE_RATIO, "noise_bound": NOISE_BOUND}}
    if not gate["passed"]:
        return {**out, "verdict": "unread",
                "why": f"the control's weighted residual is over {CONTROL_SLACK} times own's: the fit did not work"}
    if unfinished:
        return {**out, "verdict": "unread", "why": f"{', '.join(unfinished)} still falling at the last epoch"}
    if ratio is None or ratio["value"] is None or ratio["se"] is None or information["value"] is None:
        return {**out, "verdict": "unread", "why": "the gradient's noise was not read"}
    failed = []
    if information["value"] < INFORMATION_CUT:
        failed.append(f"the information cut is {information['value']:.3f}, under {INFORMATION_CUT}")
    if ratio["value"] > NOISE_RATIO:
        failed.append(f"N(oracle)/N(own) is {ratio['value']:.3f}, over {NOISE_RATIO}")
    if not ratio["value"] + 2 * ratio["se"] < NOISE_BOUND:
        failed.append(f"N(oracle)/N(own) plus two errors is {ratio['value'] + 2 * ratio['se']:.3f}, "
                      f"not under {NOISE_BOUND}")
    if failed:
        return {**out, "verdict": "kill", "why": "; ".join(failed)}
    return {**out, "verdict": "go", "why": "both the information cut and the noise ratio clear their bars"}


def duplicates_verdict(ceiling: dict) -> dict:
    """The registered reading of twin deals: the ceiling read game by game
    (see `twin_ceiling`). The reading seat by seat is said beside it and
    not read."""
    share = ceiling["by_game"]["share"]
    said = {"ceiling": share, "by_seat": ceiling["by_seat"]["share"], "needed": TWIN_CEILING}
    if share is not None and share >= TWIN_CEILING:
        return {"verdict": "build a twin round next", **said}
    return {"verdict": "stop", **said}


# ------------------------------------------------------------------- the run


def main(argv=None) -> dict:
    args = parse_args(argv)
    refuse_kept(args.out)
    # Refused before anything is loaded or played, as the placement
    # pre-test refuses it: the checkpoint's U must read every member seated.
    absent = placement_pretest.absent_from(Population.from_paths(args.opponents), args.seed)
    if absent:
        raise SystemExit(f"the round of seed {args.seed} leaves {absent} out of its pool; choose a --seed "
                         "whose round seats every member, as the run's rounds do")
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.time()
    clock = {"began": started}
    seconds = {}

    def lap(name: str) -> None:
        now = time.time()
        seconds[name] = round(now - clock["began"], 1)
        clock["began"] = now
        print(json.dumps({"phase": name, "seconds": seconds[name]}), flush=True)

    # The network first: the gradient's preflight needs it, and a
    # checkpoint paid any other reward is refused before anything is played.
    net, payload = combined.load(args.checkpoint, device)
    generation = payload.get("generation")
    saved = payload.get("optimizer_state")
    reward = carried_reward(payload)
    payload = None
    net.eval()
    roster, seated = placement_pretest.seat_others(args, device)
    members = tuple(member.name for member in roster.members)
    strangers = [name for name in members if name not in reward.model.members]
    if strangers:
        raise SystemExit(f"the checkpoint's U was fitted without {strangers}, whom the round seats")
    lap("load")

    flight = preflight(net, args, device)
    flight["budget"] = budget(flight, (time.time() - started) / 60, args)
    print(json.dumps({"preflight": flight}), flush=True)
    lap("preflight")
    if flight["budget"]["minutes"] > args.max_minutes:
        raise SystemExit(f"the run would take about {flight['budget']['minutes']} minutes, over --max-minutes "
                         f"{args.max_minutes} (see the preflight's budget): stopped before anything was played")

    batch = placement_pretest.play_round(net, args, roster, seated, device, keep_oracle=True,
                                         preview_reach=args.preview_reach)
    seated = None
    selfplay.settle_placement(batch, reward)
    ledger = batch.ledger
    kept(args.out / "ledger.npz", ledger.save)
    oracle = batch.oracle.numpy()
    if oracle.shape != (batch.decisions, ORACLE_PLANES, POSITIONS):
        raise RuntimeError(f"the round kept {oracle.shape[0]} rows of the oracle's planes for "
                           f"{batch.decisions} decisions")
    played = {
        "decisions": int(batch.decisions), "hands": int(batch.hands), "steps": int(batch.steps),
        "oracle_seconds": round(batch.oracle_seconds, 3),
        "oracle_seconds_a_step": round(batch.oracle_seconds / max(batch.steps, 1), 5),
        "gradient_minutes": round(math.ceil(batch.decisions / args.gradient_chunk) * flight["seconds_a_chunk"]
                                  / 60, 1),
    }
    print(json.dumps({"round": played}), flush=True)
    rollout = ppo_loop.on_device(batch, device, card=False)
    batch = None
    lap("play")

    returns = rollout.returns.float()
    deals = ledger.games.astype(np.int64)
    group_of, half_of, check_of = deal_draws(args.games, args.groups, args.check_share, args.seed)
    group, half, check = group_of[deals], half_of[deals], check_of[deals]
    stages = ledger.stages[ledger.rows_of(ledger.games, ledger.hands)]
    free_card(device)
    inputs = round_inputs(net, rollout, args, device)
    left = inputs["left"].cpu().numpy()
    differ = inputs["differ"].cpu().numpy()
    checked_left = tiles_left_check(left, oracle)
    lap("critic inputs")

    free_card(device)
    halves = torch.from_numpy(half.astype(bool)).to(device)
    warm = placement_pretest.cross_fitted_baseline(net, inputs["pooled"], inputs["phi"], returns, halves, args,
                                                   args.seed + 7)
    lap("warm")

    free_card(device)
    fitted, fits = cross_fitted_corrections(net, rollout, inputs, oracle, returns, half, check, args, device)
    baselines = {"own": inputs["own"], "warm": warm, **fitted}
    inputs = oracle = fitted = warm = None
    free_card(device)
    lap("oracle")

    paid = returns.double().cpu().numpy()
    log_probs = rollout.old_log_probs.double().cpu().numpy()
    weights = np.expm1(log_probs) ** 2
    cells = cells_of(stages, left)
    answers = {name: baselines[name].double().cpu().numpy() for name in BASELINES}
    residual, sums = residual_readings(paid, answers, weights, cells, group, args.groups)
    # The information the oracle's planes add, read against the better of
    # the two baselines that see nothing hidden; the capacity the term
    # adds; the oracle against each of the two alone; and the warm-up's;
    # each on the same rows.
    cuts = {"information": cut(sums, "oracle", ("control", "own"), cells),
            "capacity": cut(sums, "control", "own", cells),
            "oracle_against_control": cut(sums, "oracle", "control", cells),
            "oracle_against_own": cut(sums, "oracle", "own", cells),
            "warm_against_own": cut(sums, "warm", "own", cells)}
    rows = {split: dict(zip(names, (int(n) for n in sums["own", "plain", split][..., 0].sum(axis=0))))
            for split, (_labels, names) in cells.items()}
    gate = {"control": number(overall(sums, "control")), "own": number(overall(sums, "own")),
            "slack": CONTROL_SLACK, "passed": overall(sums, "control") <= CONTROL_SLACK * overall(sums, "own")}
    unfinished = [f"the {name} fit of half {fit['half']}" for fit in fits for name in ("oracle", "control")
                  if fit[name]["still_falling"]]
    information = {**cuts["information"]["weighted"]["all"]["all"],
                   "against": "control" if overall(sums, "control") <= overall(sums, "own") else "own"}
    ceiling = twin_ceiling(differ, ledger.games, ledger.players)
    kept(args.out / "baselines.npz", lambda stream: np.savez(
        stream, returns=paid.astype(np.float32),
        **{name: answer.astype(np.float32) for name, answer in answers.items()},
        log_probs=log_probs.astype(np.float32), differ=differ.astype(np.float32),
        stage=np.asarray(stages, dtype=np.int8), tiles_left=left.astype(np.int8), game=ledger.games,
        player=ledger.players, group=group.astype(np.int16), half=half.astype(np.int8), check=check))

    report = {
        "checkpoint": str(args.checkpoint),
        "generation": generation,
        "games": args.games,
        "seed": args.seed,
        "groups": args.groups,
        "members": list(members),
        "reward": {"kind": "placement", "alpha": reward.alpha},
        "preview_reach": args.preview_reach,
        "round": played,
        "preflight": flight,
        "own_mse": number(float(((paid - answers["own"]) ** 2).mean())),
        "rows": rows,
        "tiles_left_check": checked_left,
        "residual": residual,
        "cuts": cuts,
        "fits": fits,
        "twins": ceiling,
        "gradient": None,
        "verdict": {"oracle": {"verdict": "pending", "why": "written before the gradient pass",
                               "information_cut": information, "gate": gate, "unfinished": unfinished},
                    "duplicates": duplicates_verdict(ceiling)},
        "seconds": seconds,
        "saved": {name: str(args.out / name) for name in ("ledger.npz", "baselines.npz")},
    }
    kept(args.out / "report.json", lambda stream: stream.write(json.dumps(report, indent=1).encode("utf-8")))
    lap("readings")

    # The gradient pass: the oracle's planes, our features and Mortal's
    # vectors are let go of first, and only the products of its sums kept.
    root_v, preconditioned_by = placement_pretest.preconditioner(net, saved, args, device)
    saved = None
    weighed = advantage_weights(returns, {arm: baselines[arm] for arm in ARMS},
                                torch.from_numpy(group).to(device), args.groups)
    baselines = None
    free_card(device)
    summed, how, halved = gradient_sums(net, rollout, weighed, args, device)
    weighed = None
    plain, preconditioned = grams(summed, torch.cat(root_v) if root_v is not None else None)
    summed = root_v = None
    kept(args.out / "gram.npz", lambda stream: np.savez(
        stream, plain=plain, arms=np.asarray(ARMS), groups=np.asarray(args.groups),
        **({} if preconditioned is None else {"preconditioned": preconditioned})))
    report["gradient"] = {
        "how": how, "halved_chunks": halved, "rows_a_chunk": args.gradient_chunk,
        "vectors": len(ARMS) * args.groups, "preconditioner": preconditioned_by,
        "plain": noise_readings(plain, args.groups),
        "preconditioned": None if preconditioned is None else noise_readings(preconditioned, args.groups),
    }
    lap("gradients")

    verdict = {"oracle": oracle_verdict(information, gate, unfinished,
                                        report["gradient"]["plain"]["oracle_over_own"]),
               "duplicates": duplicates_verdict(ceiling)}
    report["verdict"] = verdict
    report["saved"]["gram.npz"] = str(args.out / "gram.npz")
    kept(args.out / "report.json", lambda stream: stream.write(json.dumps(report, indent=1).encode("utf-8")))
    # One line, last, for whoever reads the call's answer.
    print("VERDICT " + json.dumps(verdict), flush=True)
    return report


if __name__ == "__main__":
    main()
