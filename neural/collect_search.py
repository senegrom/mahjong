"""Collect current-lineage search supervision from complete all-seat self-play.

    python -m neural.collect_search combined.pt --out runs/search/round-0001 \
        --source-revision <40-character-code-commit> --games 2 --worlds 8

No checkpoint is trained or promoted here. The result is an immutable shard
accepted by SearchReplay.load/loss, not the older diagnostic Recording format.
The existing reward and shallow-search improvement rule are explicit; this does
not claim MCTS, a stronger teacher, or a fully automated AlphaZero loop.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import shutil
from pathlib import Path
import tempfile

import numpy as np
import torch

from .training_safety import SEARCH_API_VERSION
from . import inference
from .teacher_actions import representable_moves
from .search_replay import (
    ACTIONS, ENGINE_ACTIONS, PLANES, POSITIONS, REWARD, VERSION, TEACHER_VERSION, DENSE,
    SearchReplay, action_contract, improvement_policy, validate_metadata,
)


@dataclass(frozen=True)
class SearchSettings:
    worlds: int = 8
    candidates: int = 4
    pool: int = 1
    margin: float = 2.0
    depth: int = 0
    temperature: float = 0.0
    played_by: str = "network"
    valued_by: str = "critic"
    hurried: bool = False
    improve: float = .5
    max_steps: int = 4000
    objective: str = "hybrid"
    search_calls: bool = False
    confirm_worlds: int = 0
    extra_candidates: int = 0
    audit_share: float = 0.0
    sure: float = 1.0
    rollout_batch: int = inference.DEFAULT_ROLLOUT_BATCH


def metadata(*, games: int, seed: int, settings: SearchSettings,
             actor_sha256: str, source_revision: str, training_api_version: int,
             head_provenance: dict | None = None, device: str = "cpu") -> dict:
    m = {
        "version": TEACHER_VERSION, "complete": True, "rows": 1,
        "reward": dict(REWARD),
        "observation": {"planes": PLANES, "positions": POSITIONS, "encoder_version": 4},
        "actions": {"policy": ACTIONS, "engine": ENGINE_ACTIONS},
        "collection": "all_seats_search",
        "target_kind": "frozen_actor_plus_search_action",
        "actor_sha256": actor_sha256,
        "opponent_sha256": [actor_sha256] * 4,
        "source_revision": source_revision,
        "training_api_version": training_api_version,
        "games": games, "seed": seed, "search": asdict(settings),
        "teacher": {"version": 1, "policy_inference": inference.describe("mortal", device), "search_api_version": SEARCH_API_VERSION, "objective": settings.objective,
                    "placement_head": head_provenance,
                    "student_value_head": "critic" if settings.valued_by == "placement" else settings.valued_by},
    }
    validate_metadata(m)
    return m


class _Inputs:
    """Capture the actual adapter inputs without changing search's decisions."""
    def __init__(self, served):
        self.served = served
        self.root_inputs = None
        self.reach_inputs = None
        self.reach_rows = None

    def __getattr__(self, name):
        return getattr(self.served, name)

    def root(self, *args, **kwargs):
        self.root_inputs = self.served.root(*args, **kwargs)
        return self.root_inputs

    def after_reach(self, arena, rows, *args, **kwargs):
        self.reach_rows = np.asarray(rows).copy()
        self.reach_inputs = self.served.after_reach(arena, rows, *args, **kwargs)
        return self.reach_inputs


def _sparse(planes):
    """Preserve exactly the float16-derived inputs the actor was served."""
    from .observe import Planes
    if planes.ndim != 3 or tuple(planes.shape[1:]) != (PLANES, POSITIONS):
        raise ValueError("Actor inputs do not match Mortal-v4 observations")
    flat = planes.detach().reshape(len(planes), -1)
    coordinates = flat.nonzero()
    counts = torch.bincount(coordinates[:, 0], minlength=len(planes))
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).cpu().numpy()
    columns = coordinates[:, 1].cpu().numpy().astype(np.uint16)
    values = flat[coordinates[:, 0], coordinates[:, 1]].float().cpu().numpy().astype(np.float16)
    return Planes(ptr, columns, values)


@torch.no_grad()
def collect(net, *, games: int, seed: int, settings: SearchSettings,
            actor_sha256: str, source_revision: str, device: str = "cpu",
            placement_head: Path | None = None) -> SearchReplay:
    """Freeze one actor for a collection; every seat searches and every game ends.

    Each actual engine transition is recorded once at its ordinary decision,
    and once more for its conditional tile decision ONLY when riichi is actually
    executed. Both ledger entries receive the same hand and final-game payoff.
    Failed collections return no training data.
    """
    from . import contract, ledger, searched
    from .observe import Planes, Views
    from .outcomes import require_finished, validate_budget
    from .training_safety import TRAINING_API_VERSION, require_training_engine
    import riichi_py

    from .teacher_options import validate_controls, candidate_set
    validate_controls(**{name: getattr(settings, name) for name in (
        "objective", "valued_by", "played_by", "depth", "search_calls", "confirm_worlds",
        "extra_candidates", "audit_share", "rollout_batch")})
    head = provenance = None
    if settings.valued_by == "placement":
        if placement_head is None:
            raise ValueError("placement teacher requires --placement-head")
        from . import placement
        # Hash and load the SAME private snapshot even if the original is replaced.
        with tempfile.TemporaryDirectory(prefix="mahjong-teacher-head-") as folder:
            snapshot = Path(folder) / "head.pt"
            with Path(placement_head).open("rb") as source, snapshot.open("xb") as target:
                shutil.copyfileobj(source, target)
            with snapshot.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            head, head_meta = placement.load(snapshot, device)
            placement.require_head_for(head_meta, contract.unwrap(net))
        provenance = {"sha256": digest, "features": head_meta["features"],
                      "feature_version": head.feature_version}
    elif placement_head is not None:
        raise ValueError("a placement head was supplied to a different teacher")
    m = metadata(games=games, seed=seed, settings=settings, actor_sha256=actor_sha256,
                 source_revision=source_revision, training_api_version=TRAINING_API_VERSION,
                 head_provenance=provenance, device=device)
    validate_budget(games, settings.max_steps)
    require_training_engine()
    if ledger.REWARD_VERSION != REWARD["version"] or ledger.HAND_SCALE != 1 / 4000:
        raise ValueError("The engine reward no longer matches search replay version 1")
    served = contract.serve(net) if head is None else contract.serve(net, placement_head=head)
    if (served.contract.reads != "mortal" or served.contract.planes != PLANES
            or served.contract.answers != ACTIONS):
        raise contract.UnsupportedSearchLayout("Search replay v1 requires Mortal-v4 observations and 46 actions")
    net = served.net
    net.eval()
    torch.manual_seed(seed)
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    views = Views(arena, games, {"mortal"})
    contract.remember_follower(arena, views.observer.follower)
    account = ledger.Ledger(games)
    blocks = []
    dense = {name: [] for name in DENSE if name != "returns"}
    candidate_rng = [np.random.default_rng(seed + game) for game in range(games)]
    steps = 0

    def record(planes, legal, actor, actions, rows, players, stage, engine_legal):
        stages = np.full(len(rows), stage, dtype=np.int64)
        expected, aliases = action_contract(engine_legal, actions, stages)
        if not np.array_equal(legal, expected):
            raise ValueError("Served masks disagree with the versioned action contract")
        target = improvement_policy(actor, legal, aliases, settings.improve)
        blocks.append(_sparse(planes))
        values = {
            "legal": legal, "actor_policy": actor, "policy_target": target,
            "engine_legal": engine_legal, "engine_action": actions,
            "game": rows, "player": players, "stage": stages,
            "step": np.full(len(rows), steps, dtype=np.int64),
        }
        for name, value in values.items():
            dense[name].append(np.asarray(value, dtype=DENSE[name][0]).copy())
        for game, player in zip(rows, players):
            account.open(int(game), int(player))

    try:
        while not arena.all_finished() and steps < settings.max_steps:
            steps += 1
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            rows = np.nonzero(seats != 0xFF)[0].astype(np.int64)
            if not len(rows):
                break
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
            deciding = players[rows, seats[rows]].astype(np.int64)
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, ENGINE_ACTIONS).astype(bool)
            views.advance()
            views.prepare(rows, deciding)
            captured = _Inputs(served)
            order, logits, _value, guessed = contract.root_order(
                captured, arena, views, rows, deciding, legal, device
            )
            ranked = [[int(order[game, 0])] for game in range(games)]
            top = torch.softmax(logits.float(), dim=1).max(dim=1).values.cpu().numpy()
            searchable = representable_moves(legal)
            for at, game in enumerate(rows):
                if top[at] < settings.sure or candidate_rng[game].random() < settings.audit_share:
                    ranked[game] = candidate_set(order[game], searchable[game], settings.candidates,
                                                 settings.extra_candidates, candidate_rng[game])
            belief = np.zeros((games, riichi_py.HANDS), dtype=np.float32)
            belief[rows] = torch.softmax(guessed.float(), dim=2).reshape(len(rows), -1).cpu().numpy()
            choices = np.asarray(searched.search_with_value_head(
                net, arena, ranked, belief.reshape(-1).tolist(), worlds=settings.worlds,
                candidates=min(ENGINE_ACTIONS, settings.candidates + settings.extra_candidates), margin=settings.margin, hurried=settings.hurried,
                device=device, pool=settings.pool, played_by=settings.played_by, depth=settings.depth,
                temperature=settings.temperature, valued_by=settings.valued_by, served=served,
                objective=settings.objective, search_calls=settings.search_calls,
                confirm_worlds=settings.confirm_worlds, rollout_batch=settings.rollout_batch,
            ))
            if (choices.shape != (games,) or choices.dtype.kind not in "iu"
                    or np.any((choices < 0) | (choices >= ENGINE_ACTIONS))):
                raise ValueError("Search must return one valid integer action per game")
            choices = choices.astype(np.int64)
            planes, own_legal = captured.root_inputs
            actor = torch.softmax(logits.float(), dim=1).cpu().numpy()
            record(planes, own_legal.cpu().numpy(), actor, choices[rows], rows, deciding, 0, legal[rows])
            # root_order already asked the hypothetical declaration from a copy,
            # so using its captured inputs does not mutate the real follower.
            second = np.nonzero((choices[rows] >= 34) & (choices[rows] < 68))[0]
            if len(second):
                if captured.reach_inputs is None:
                    raise ValueError("A searched riichi has no conditional actor input")
                lookup = {int(game): i for i, game in enumerate(captured.reach_rows)}
                indices = torch.tensor([lookup[int(game)] for game in rows[second]], device=device)
                after, after_legal = (tensor[indices] for tensor in captured.reach_inputs)
                after_logits, _v, _hands = inference.everything(net, after, after_legal)
                actor_after = torch.softmax(after_logits.float(), dim=1).cpu().numpy()
                record(after, after_legal.cpu().numpy(), actor_after, choices[rows[second]],
                       rows[second], deciding[second], 1, legal[rows[second]])
            arena.step(choices.tolist())
            account.settle(arena)
        require_finished(arena, steps=steps, context="search training collection")
        returns = account.close(arena)
    finally:
        # The follower is remembered by the arena; forgotten, a finished or
        # failed collection is not held for good.
        contract.forget_follower(arena)
    if not len(returns):
        raise ValueError("Search collection produced no decisions")
    observations = Planes.cat(blocks)
    a = {f"root_{name}": value for name, value in observations.arrays().items()}
    a.update({name: np.concatenate(values) for name, values in dense.items()})
    a["returns"] = returns
    m["rows"] = len(returns)
    replay = SearchReplay(m, a)
    replay.validate()
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new immutable shard directory")
    parser.add_argument("--source-revision", required=True, help="40-character Git revision of the collecting code")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for name in ("worlds", "candidates", "pool", "depth", "max_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(SearchSettings(), name))
    for name in ("margin", "temperature", "improve"):
        parser.add_argument("--" + name, type=float, default=getattr(SearchSettings(), name))
    parser.add_argument("--played-by", choices=("network", "club"), default="network")
    parser.add_argument("--valued-by", choices=("critic", "public", "mean", "placement"), default="critic")
    parser.add_argument("--hurried", action="store_true")
    from .teacher_options import add_arguments, validate_controls
    add_arguments(parser)
    parser.add_argument("--placement-head", type=Path)
    parser.add_argument("--sure", type=float, default=1.0)
    args = parser.parse_args()
    settings = SearchSettings(**{name: getattr(args, name) for name in SearchSettings.__dataclass_fields__})
    # Validate cheap settings before loading checkpoints or creating output.
    validate_controls(**{name: getattr(settings, name) for name in (
        "objective", "valued_by", "played_by", "depth", "search_calls", "confirm_worlds",
        "extra_candidates", "audit_share", "rollout_batch")})
    if (settings.valued_by == "placement") != (args.placement_head is not None):
        raise ValueError("--placement-head must be supplied exactly for --valued-by placement")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite search replay: {args.out}")
    from .checkpoints import copy_checkpoint
    from . import zoo
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="mahjong-search-actor-") as temporary:
        snapshot = Path(temporary) / "actor.pt"
        copy_checkpoint(args.checkpoint, snapshot)
        with snapshot.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        net = zoo.load_player(snapshot, args.device)
        replay = collect(net, games=args.games, seed=args.seed, settings=settings,
                         actor_sha256=digest, source_revision=args.source_revision, device=args.device,
                         placement_head=args.placement_head)
        replay.save(args.out)
    print(f"Wrote {replay.metadata['rows']} supervised decisions from {args.games} complete games to {args.out}")


if __name__ == "__main__":
    main()
