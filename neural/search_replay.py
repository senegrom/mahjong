"""Versioned, outcome-grounded supervision for Mortal-space search.

This is supervised search replay, NOT PPO experience or MCTS visit counts.
Version 1 supports the current 1012-plane/46-action player. Engine actions
remain explicit, including riichi's separate declaration and tile decisions.
Only completed collections may be published. Legacy diagnostic recordings
lack these fields and cannot be silently upgraded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

VERSION = 1  # Original supervised format remains resumable without relabelling.
TEACHER_VERSION = 2
PLANES, POSITIONS, ACTIONS, ENGINE_ACTIONS = 1012, 34, 46, 78
REWARD = {"version": 1, "name": "hand_points_over_4000_plus_placement"}
# Schema-v1 meanings, in exactly the priority order used by neural.zoo.
# The native integration test compares this table to zoo, rather than allowing
# future action changes to reinterpret existing shards silently.
MEANINGS = tuple((i,) for i in range(34)) + (
    (4,), (13,), (22,), tuple(range(34, 68)),
    (73,), (72,), (71,), (74,), (75, 76, 77), (68, 69), (), (70,),
)
DENSE = {
    "legal": (np.bool_, (ACTIONS,)),
    "actor_policy": (np.float32, (ACTIONS,)),
    "policy_target": (np.float32, (ACTIONS,)),
    "engine_legal": (np.bool_, (ENGINE_ACTIONS,)),
    "engine_action": (np.int64, ()),
    "returns": (np.float32, ()),
    "game": (np.int64, ()), "player": (np.int64, ()),
    "step": (np.int64, ()), "stage": (np.int64, ()),
}
SPARSE = ("root_indptr", "root_indices", "root_values")
FIELDS = (*SPARSE, *DENSE)


def action_contract(engine_legal: np.ndarray, actions: np.ndarray,
                    stages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return policy legality and the aliases of each selected engine move.

    Stage 0 is the ordinary decision; stage 1 is the discard after declaring
    riichi. For non-riichi moves an alias must actually execute the selected
    move under first-legal-meaning translation, not merely name it somewhere
    in a list. An unrepresentable search move is rejected, never relabelled.
    """
    n = len(actions)
    if (engine_legal.shape != (n, ENGINE_ACTIONS) or engine_legal.dtype != np.bool_
            or actions.shape != (n,) or actions.dtype != np.int64
            or stages.shape != (n,) or stages.dtype != np.int64
            or np.any((actions < 0) | (actions >= ENGINE_ACTIONS))
            or np.any((stages < 0) | (stages > 1))):
        raise ValueError("Invalid search action dimensions, dtypes or stages")
    if not engine_legal[np.arange(n), actions].all():
        raise ValueError("Search selected an illegal engine action")
    legal = np.zeros((n, ACTIONS), dtype=bool)
    aliases = np.zeros_like(legal)
    root = stages == 0
    for action, meanings in enumerate(MEANINGS):
        if not meanings:
            continue
        found = engine_legal[:, meanings]
        permitted = found.any(axis=1)
        legal[:, action] = root & permitted
        first = np.asarray(meanings)[found.argmax(axis=1)]
        matches = ((actions >= 34) & (actions < 68) if action == 37
                   else actions == first)
        aliases[:, action] = legal[:, action] & matches
    second = stages == 1
    if np.any(second & ((actions < 34) | (actions >= 68))):
        raise ValueError("A second-stage row must execute a riichi discard")
    legal[second, :34] = engine_legal[second, 34:68]
    aliases[np.nonzero(second)[0], actions[second] - 34] = True
    if not aliases.any(axis=1).all():
        raise ValueError("Selected engine action cannot be expressed by this policy")
    return legal, aliases


def improvement_policy(actor: np.ndarray, legal: np.ndarray,
                       aliases: np.ndarray, improve: float) -> np.ndarray:
    """Frozen actor mass plus extra mass on search's action-equivalence class.

    Red/plain aliases share the extra mass in the actor's existing proportions
    (uniformly when their total probability is zero). They must not each receive
    a full copy of the same engine action's target probability.
    """
    if not math.isfinite(improve) or not 0 <= improve <= 1:
        raise ValueError("improve must be finite and in [0, 1]")
    if (actor.ndim != 2 or actor.shape != legal.shape or aliases.shape != legal.shape
            or legal.dtype != np.bool_ or aliases.dtype != np.bool_
            or not legal.any(axis=1).all() or not aliases.any(axis=1).all()
            or np.any(aliases & ~legal) or not np.isfinite(actor).all()
            or np.any(actor < 0) or np.any(actor[~legal] != 0)
            or not np.allclose(actor.sum(axis=1), 1, atol=1e-5, rtol=1e-5)):
        raise ValueError("Targets need normalized legal actor probabilities and legal aliases")
    share = np.where(aliases, actor, 0).astype(np.float64)
    mass = share.sum(axis=1, keepdims=True)
    share = np.divide(share, mass, out=np.zeros_like(share), where=mass > 0)
    empty = mass[:, 0] == 0
    share[empty] = aliases[empty] / aliases[empty].sum(axis=1, keepdims=True)
    return np.asarray((1 - improve) * actor + improve * share, dtype=np.float32)


def _integer(value, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_dir(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def validate_metadata(m: dict) -> None:
    """Validate settings and provenance before allocating or loading a collection."""
    if not isinstance(m, dict):
        raise ValueError("Search replay metadata must be an object")
    if (type(m.get("version")) is not int or m.get("version") not in (VERSION, TEACHER_VERSION) or m.get("complete") is not True
            or m.get("reward") != REWARD
            or m.get("observation") != {"planes": PLANES, "positions": POSITIONS, "encoder_version": 4}
            or m.get("actions") != {"policy": ACTIONS, "engine": ENGINE_ACTIONS}
            or m.get("collection") != "all_seats_search"
            or m.get("target_kind") != "frozen_actor_plus_search_action"):
        raise ValueError("Incomplete or incompatible search replay contract")
    for key, size in (("actor_sha256", 64), ("source_revision", 40)):
        if not isinstance(m.get(key), str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", m[key]):
            raise ValueError(f"Search replay needs an immutable {key}")
    for key, minimum in (("rows", 1), ("games", 1), ("seed", 0), ("training_api_version", 1)):
        _integer(m.get(key), key, minimum)
    if m["seed"] + m["games"] > 2**64:
        raise ValueError("Game seed range overflows u64")
    if m.get("opponent_sha256") != [m["actor_sha256"]] * 4:
        raise ValueError("All-seat search must identify the frozen actor in all four seats")
    s = m.get("search", {})
    if not isinstance(s, dict):
        raise ValueError("Search settings must be an object")
    for key in ("worlds", "candidates", "pool", "max_steps"):
        _integer(s.get(key), key, 1)
    if type(s.get("depth")) is not int or s["depth"] < -1:
        raise ValueError("depth must be -1 or a nonnegative integer")
    for key in ("margin", "temperature", "improve"):
        value = s.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if (s["improve"] > 1 or s["candidates"] > ENGINE_ACTIONS
            or s.get("played_by") not in ("network", "club")
            or s.get("valued_by") not in (("critic", "public", "mean", "placement") if m["version"] == TEACHER_VERSION else ("critic", "public", "mean"))
            or type(s.get("hurried")) is not bool):
        raise ValueError("Invalid search settings")

    if m["version"] == VERSION and ("teacher" in m or any(key in s for key in (
            "objective", "search_calls", "confirm_worlds", "extra_candidates", "audit_share", "sure"))):
        raise ValueError("new teacher semantics cannot be relabelled as legacy replay")

    if m["version"] == TEACHER_VERSION:
        from .teacher_options import validate_controls
        validate_controls(**{name: s.get(name) for name in (
            "objective", "valued_by", "played_by", "depth", "search_calls", "confirm_worlds",
            "extra_candidates", "audit_share")})
        sure = s.get("sure")
        if type(sure) not in (int, float) or not math.isfinite(sure) or not 0 <= sure <= 1:
            raise ValueError("invalid teacher confidence shortcut")
        teacher = m.get("teacher")
        if (not isinstance(teacher, dict) or type(teacher.get("version")) is not int
                or teacher["version"] != 1 or type(teacher.get("search_api_version")) is not int
                or teacher["search_api_version"] != 4 or teacher.get("objective") != s["objective"]
                or teacher.get("student_value_head") != ("critic" if s["valued_by"] == "placement" else s["valued_by"])):
            raise ValueError("teacher provenance/value-target contract is incomplete")
        head = teacher.get("placement_head")
        if s["valued_by"] == "placement":
            if (not isinstance(head, dict)
                    or re.fullmatch(r"[0-9a-f]{64}", str(head.get("sha256", ""))) is None
                    or re.fullmatch(r"[0-9a-f]{16}", str(head.get("features", ""))) is None
                    or type(head.get("feature_version")) is not int or head["feature_version"] not in (1, 2)):
                raise ValueError("placement teacher needs immutable head provenance")
        elif head is not None:
            raise ValueError("non-placement teacher must not name an unused placement head")


def student_value_head(metadata: dict) -> str:
    """The learner's value label is explicitly separate from the teacher's utility.

    V2 still supervises the student's hybrid critic with completed hybrid returns.
    A placement-only teacher changes policy labels, not the meaning of that head.
    """
    if metadata["version"] == TEACHER_VERSION:
        return metadata["teacher"]["student_value_head"]
    return metadata["search"]["valued_by"]


@dataclass
class SearchReplay:
    metadata: dict
    arrays: dict[str, np.ndarray]

    def validate(self) -> None:
        """Check semantic versions, frozen targets and trajectory linkage."""
        m, a = self.metadata, self.arrays
        validate_metadata(m)
        n = m["rows"]
        s = m["search"]
        if set(a) != set(FIELDS):
            raise ValueError("Search replay fields do not match the schema")
        for name, (dtype, tail) in DENSE.items():
            if not isinstance(a[name], np.ndarray) or a[name].dtype != np.dtype(dtype) or a[name].shape != (n, *tail):
                raise ValueError(f"Invalid shape or dtype for {name}")
        ptr, columns, values = (a[name] for name in SPARSE)
        if (not all(isinstance(v, np.ndarray) and v.ndim == 1 for v in (ptr, columns, values))
                or ptr.dtype != np.int64 or columns.dtype != np.uint16 or values.dtype != np.float16
                or ptr.shape != (n + 1,) or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1])
                or ptr[-1] != len(columns) or len(columns) != len(values)):
            raise ValueError("Invalid sparse observations")
        for start in range(0, len(columns), 1_000_000):
            if (np.any(columns[start:start + 1_000_000] >= PLANES * POSITIONS)
                    or not np.isfinite(values[start:start + 1_000_000]).all()):
                raise ValueError("Invalid sparse observation values or indices")
            # The collector emits sorted, unique columns. Duplicate indices
            # would make a GPU scatter's result dependent on write order.
            lo, hi = max(1, start), min(len(columns), start + 1_000_000)
            resets = np.nonzero(columns[lo:hi] <= columns[lo - 1:hi - 1])[0] + lo
            if len(resets) and np.any(ptr[np.searchsorted(ptr, resets)] != resets):
                raise ValueError("Sparse columns must be strictly increasing within each row")
        # Every root has at most one after-reach row. Both represent the same
        # actual engine step and must receive the same completed-game return.
        roots, seconds = {}, {}
        for start in range(0, n, 4096):
            rows = slice(start, start + 4096)
            legal, aliases = action_contract(a["engine_legal"][rows], a["engine_action"][rows], a["stage"][rows])
            if not np.array_equal(legal, a["legal"][rows]):
                raise ValueError("Policy masks disagree with engine legality or decision stage")
            target = improvement_policy(a["actor_policy"][rows], legal, aliases, s["improve"])
            if (not np.isfinite(a["policy_target"][rows]).all()
                    or np.any(a["policy_target"][rows] < 0)
                    or np.any(a["policy_target"][rows][~legal] != 0)
                    or not np.allclose(target, a["policy_target"][rows], atol=1e-6, rtol=1e-5)):
                raise ValueError("Search targets disagree with the frozen actor and action")
            if (not np.isfinite(a["returns"][rows]).all()
                    or np.any((a["game"][rows] < 0) | (a["game"][rows] >= m["games"]))
                    or np.any((a["player"][rows] < 0) | (a["player"][rows] > 3))
                    or np.any((a["step"][rows] < 1) | (a["step"][rows] > s["max_steps"]))):
                raise ValueError("Invalid outcomes or trajectory identifiers")
        for i in range(n):
            key = (int(a["game"][i]), int(a["step"][i]))
            stage = int(a["stage"][i])
            group = seconds if stage else roots
            if key in group:
                raise ValueError("Duplicate decision stage in a game step")
            group[key] = i
        for key, i in roots.items():
            reach = 34 <= a["engine_action"][i] < 68
            if reach != (key in seconds):
                raise ValueError("Riichi needs exactly one linked second-stage row")
            if reach:
                j = seconds[key]
                if (any(a[name][i] != a[name][j] for name in ("player", "engine_action", "returns"))
                        or not np.array_equal(a["engine_legal"][i], a["engine_legal"][j])):
                    raise ValueError("Riichi stages disagree about the executed transition")
        if not seconds.keys() <= roots.keys():
            raise ValueError("Second-stage row has no parent root")

    def save(self, folder: Path) -> None:
        """Publish into a NEW directory; the complete manifest is committed last.

        mkdir is exclusive, so concurrent writers cannot overwrite a shard.
        A failure may leave an incomplete directory; load refuses it. Existing
        diagnostic recordings and already published shards are never replaced.
        """
        self.validate()
        folder = Path(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        folder.mkdir()
        hashes = {}
        for name in FIELDS:
            path = folder / f"{name}.npy"
            with path.open("xb") as stream:
                np.save(stream, self.arrays[name], allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            hashes[path.name] = _digest(path)
        _sync_dir(folder)
        temporary = folder / "manifest.tmp"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({**self.metadata, "sha256": hashes}, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / "manifest.json")
        _sync_dir(folder)
        _sync_dir(folder.parent)

    @classmethod
    def load(cls, folder: Path) -> SearchReplay:
        folder = Path(folder)
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(m, dict):
            raise ValueError("Search replay manifest must be an object")
        validate_metadata(m)
        hashes = m.pop("sha256", None)
        if not isinstance(hashes, dict) or set(hashes) != {f"{name}.npy" for name in FIELDS}:
            raise ValueError("Search replay has no complete file manifest")
        for name, digest in hashes.items():
            if digest != _digest(folder / name):
                raise ValueError(f"Search replay checksum mismatch: {name}")
        a = {name: np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FIELDS}
        replay = cls(m, a)
        replay.validate()
        return replay

    def observations(self):
        from .observe import Planes
        return Planes(*(self.arrays[name] for name in SPARSE))

    def loss(self, net, rows: np.ndarray, device: str = "cpu", value_weight: float = .5):
        """A supervised policy/value loss using the explicit student value contract.

        Call validate/load once before training; this gathers one minibatch.
        It never uses a PPO ratio or invents a search behavior log-probability.
        The caller owns the optimizer, train mode, drift control and checkpoints.
        """
        import torch
        if not math.isfinite(value_weight) or value_weight < 0:
            raise ValueError("value_weight must be finite and nonnegative")
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype != np.int64 or not len(rows) or np.any((rows < 0) | (rows >= self.metadata["rows"])):
            raise ValueError("A minibatch needs valid int64 row indices")
        x = self.observations().rows(rows).dense(device)
        tensor = lambda name: torch.from_numpy(self.arrays[name][rows].copy()).to(device)
        legal, target, returns = (tensor(name) for name in ("legal", "policy_target", "returns"))
        logits, value, _ = net.everything(x, legal)
        if hasattr(net, "value_only"):
            value = net.value_only(x, head=student_value_head(self.metadata))
        if logits.shape != target.shape or value.shape != returns.shape:
            raise ValueError("Learner outputs do not match the replay action/value contract")
        logp = torch.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=1)
        policy_loss = -(target * torch.where(legal, logp, torch.zeros_like(logp))).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value.float(), returns)
        total = policy_loss + value_weight * value_loss
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite search-supervision loss")
        return total
