"""Fit completed search-replay shards and checkpoint at whole-epoch boundaries.

    python -m neural.train_search --checkpoint combined.pt --replay round-0001 \
        --out runs/search-fit-1 --epochs 2
    python -m neural.train_search --resume runs/search-fit-1/latest.pt \
        --replay round-0001 --out runs/search-fit-2 --epochs 2

This is supervised policy/value learning, not PPO or automatic promotion.
Resume requires the same data bytes, semantic contracts and training options.
Every invocation writes a NEW directory, so a failed restart cannot destroy its
input. Repeat epochs on one dataset are not new generations of self-play.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from .checkpoints import atomic_save, copy_checkpoint
from .training_safety import TRAINING_API_VERSION

STATE_VERSION = 1


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Options:
    batch: int = 256
    lr: float = 1e-5
    weight_decay: float = 1e-4
    value_weight: float = .5
    max_grad_norm: float = 1.0
    validation_every: int = 5
    seed: int = 20260913

    def validate(self):
        _integer(self.batch, "batch", 2)
        _integer(self.validation_every, "validation_every")
        if self.validation_every >= 2**64:
            raise ValueError("validation_every must fit u64")
        if self.validation_every == 1:
            raise ValueError("validation_every must be zero (explicitly disabled) or >= 2")
        _integer(self.seed, "seed")
        if self.seed >= 2**64:
            raise ValueError("seed must fit u64")
        for key in ("lr", "weight_decay", "value_weight", "max_grad_norm"):
            value = getattr(self, key)
            if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
                    or key in ("lr", "max_grad_norm") and value == 0):
                raise ValueError(f"Invalid {key}")


def fingerprint(replay) -> str:
    """Hash the actual validated arrays, not a mutable path or only its metadata."""
    digest = hashlib.sha256(_json(replay.metadata))
    for name, values in sorted(replay.arrays.items()):
        digest.update(_json([name, values.dtype.str, list(values.shape)]))
        # Bounded temporary copies, including for non-C-contiguous input arrays.
        for start in range(0, len(values), 4096):
            digest.update(values[start:start + 4096].tobytes(order="C"))
    return digest.hexdigest()


class Dataset:
    """Fixed shards, sorted by content identity, split by WHOLE environment seed.

    All players, decisions and riichi stages from a deal stay in the same split.
    A deal collected under another actor also stays there; paths and input order
    cannot move it between training and validation. Validation is a fitting
    diagnostic, not an independent playing-strength or promotion test.
    """
    def __init__(self, replays, validation_every: int):
        Options(validation_every=validation_every).validate()
        if not replays:
            raise ValueError("At least one completed replay shard is required")
        keyed = []
        contract = None
        for replay in replays:
            replay.validate()
            m = replay.metadata
            if (type(m.get("training_api_version")) is not int
                    or m["training_api_version"] != TRAINING_API_VERSION):
                raise ValueError("Replay training API is incompatible; regenerate under the current engine")
            expected = {"version": 1,
                        "reward": {"version": 1, "name": "hand_points_over_4000_plus_placement"},
                        "observation": {"planes": 1012, "positions": 34, "encoder_version": 4},
                        "actions": {"policy": 46, "engine": 78},
                        "target_kind": "frozen_actor_plus_search_action"}
            if any(_json(m.get(key)) != _json(value) for key, value in expected.items()):
                raise ValueError("Unsupported replay training semantics")
            semantics = {key: m[key] for key in (
                "version", "reward", "observation", "actions", "target_kind", "source_revision",
            )}
            semantics["valued_by"] = m["search"]["valued_by"]
            if contract is not None and _json(semantics) != _json(contract):
                raise ValueError("Do not mix replay objectives, evaluators, encoders or source revisions")
            contract = semantics
            keyed.append((fingerprint(replay), replay))
        keyed.sort(key=lambda item: item[0])
        self.identities = [key for key, _ in keyed]
        if len(set(self.identities)) != len(keyed):
            raise ValueError("Duplicate replay content would silently reweight training")
        self.replays = [replay for _, replay in keyed]
        self.contract = deepcopy(contract)
        self.training, self.validation = [], []
        for replay in self.replays:
            game_seeds = np.uint64(replay.metadata["seed"]) + replay.arrays["game"].astype(np.uint64)
            held = (game_seeds % np.uint64(validation_every) == 0 if validation_every
                    else np.zeros(len(game_seeds), dtype=bool))
            self.training.append(np.nonzero(~held)[0].astype(np.int64))
            self.validation.append(np.nonzero(held)[0].astype(np.int64))
        self.offsets = np.cumsum([0, *(len(rows) for rows in self.training)], dtype=np.int64)
        self.train_rows = int(self.offsets[-1])
        self.validation_rows = sum(len(rows) for rows in self.validation)
        if self.train_rows < 2:
            raise ValueError("Need at least two training rows after the whole-game split")
        if validation_every and not self.validation_rows:
            raise ValueError("No validation games: collect more deals or explicitly use --validation-every 0")

    @classmethod
    def load(cls, paths, validation_every):
        from .search_replay import SearchReplay
        return cls([SearchReplay.load(path) for path in paths], validation_every)

    def groups(self, indices):
        """Gather one global minibatch across shards, without densifying the replay."""
        which = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard in np.unique(which):
            local = indices[which == shard] - self.offsets[shard]
            yield self.replays[shard], self.training[shard][local]


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite model or optimizer state; no checkpoint may be published")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Nonfinite checkpoint scalar")


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return deepcopy(value)


def _model_payload(net):
    if hasattr(net, "state"):
        return net.state()
    return {"model": net.state_dict(), **net.payload_fields()}


def _load_network(path, device):
    # copy_checkpoint has already safely validated these private snapshot bytes.
    from . import combined, model
    from .training_safety import require_training_engine
    require_training_engine()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "combined" in payload:
        net, _ = combined.load(path, device)
    elif "model" in payload:
        net = model.from_payload(payload, device)
    else:
        raise ValueError("Search training supports combined or standalone policy/value checkpoints")
    return net, payload


class Learner:
    def __init__(self, net, dataset: Dataset, options: Options, *, device="cpu",
                 initial_sha256: str, initial_generation: int = 0, resume=None):
        options.validate()
        _integer(initial_generation, "initial_generation")
        if (not isinstance(initial_sha256, str) or len(initial_sha256) != 64
                or any(c not in "0123456789abcdef" for c in initial_sha256)):
            raise ValueError("The initial checkpoint needs a SHA-256 identity")
        if (getattr(net, "kind", None) != "mortal" or getattr(net, "planes", None) != 1012
                or getattr(net, "actions", None) != 46 or not hasattr(net, "everything")):
            raise ValueError("Search learner requires the current 1012-plane/46-action contract")
        self.net, self.data, self.options = net, dataset, options
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA training are supported")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.runtime = {"torch": str(torch.__version__), "device": str(self.device),
                        "numpy": str(np.__version__), "threads": torch.get_num_threads(),
                        "deterministic": torch.are_deterministic_algorithms_enabled()}
        self.initial_sha256, self.initial_generation = initial_sha256, initial_generation
        self.epochs, self.updates, self.ready = 0, 0, True
        self.metrics = None
        self.shuffle = np.random.default_rng(options.seed)
        if hasattr(net, "set_mode"):
            net.set_mode("none")
        net.to(self.device)
        # Frozen auxiliary parameters remain frozen unless the model's own mode
        # explicitly enables them. No optimizer state is borrowed from PPO.
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=options.lr,
                                          weight_decay=options.weight_decay)
        self.parameter_signature = [[name, list(p.shape), str(p.dtype), p.requires_grad]
                                    for name, p in net.named_parameters()]
        _finite_tree(net.state_dict())
        if resume is None:
            torch.manual_seed(options.seed)
        else:
            self._restore(resume)

    def _restore(self, saved):
        if not isinstance(saved, dict) or type(saved.get("version")) is not int or saved["version"] != STATE_VERSION:
            raise ValueError("Not a supported search-training resume state")
        for key, expected in (("data", self.data.identities), ("contract", self.data.contract),
                              ("options", asdict(self.options)), ("runtime", self.runtime),
                              ("parameters", self.parameter_signature),
                              ("initial_sha256", self.initial_sha256),
                              ("initial_generation", self.initial_generation)):
            if _json(saved.get(key)) != _json(expected):
                raise ValueError(f"Resume {key} changed; use --checkpoint and a new run for a new experiment")
        _integer(saved.get("epochs"), "saved epochs", 1)
        _integer(saved.get("updates"), "saved updates", 1)
        expected_steps = ((self.data.train_rows + self.options.batch - 1) // self.options.batch) * saved["epochs"]
        if saved["updates"] != expected_steps:
            raise ValueError("Resume update count does not describe complete epochs")
        # Restore only after model and optimizer construction. Never silently
        # reset missing/incompatible Adam moments or a missing shuffle stream.
        groups = [{k: v for k, v in group.items() if k != "params"}
                  for group in self.optimizer.param_groups]
        self.optimizer.load_state_dict(saved["optimizer"])
        if groups != [{k: v for k, v in group.items() if k != "params"}
                      for group in self.optimizer.param_groups]:
            raise ValueError("Resume optimizer settings disagree with the saved options")
        steps = []
        for parameter, state in self.optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError("Resume Adam moments are missing or have the wrong shape")
            step = state.get("step")
            if (not isinstance(step, torch.Tensor) or step.ndim != 0
                    or not torch.isfinite(step) or step.item() % 1 or not 0 < step.item() <= saved["updates"]):
                raise ValueError("Invalid Adam update counter")
            steps.append(int(step.item()))
        if not steps or max(steps) != saved["updates"]:
            raise ValueError("Resume has missing/reset optimizer history")
        _finite_tree(self.optimizer.state_dict())
        self.shuffle.bit_generator.state = deepcopy(saved["shuffle"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        cuda = saved["cuda_rng"]
        if self.device.type == "cuda":
            if cuda is None:
                raise ValueError("CUDA resume has no generator state")
            torch.cuda.set_rng_state(cuda.cpu(), self.device)
        elif cuda is not None:
            raise ValueError("CPU resume unexpectedly carries CUDA state")
        self.epochs, self.updates = saved["epochs"], saved["updates"]
        self.metrics = deepcopy(saved["metrics"])

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        total = 0.0
        for replay, rows in zip(self.data.replays, self.data.validation):
            for start in range(0, len(rows), self.options.batch):
                picks = rows[start:start + self.options.batch]
                loss = replay.loss(self.net, picks, str(self.device), self.options.value_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                total += float(loss) * len(picks)
        return total / self.data.validation_rows if self.data.validation_rows else None

    def epoch(self):
        if not self.ready:
            raise RuntimeError("Previous epoch failed; reload a completed checkpoint before continuing")
        self.ready = False
        self.net.train()
        order = self.shuffle.permutation(self.data.train_rows)
        total = 0.0
        for start in range(0, len(order), self.options.batch):
            indices = order[start:start + self.options.batch]
            self.optimizer.zero_grad(set_to_none=True)
            for replay, rows in self.data.groups(indices):
                loss = replay.loss(self.net, rows, str(self.device), self.options.value_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite search loss; optimizer step refused")
                # Weight per row, not per shard. Tiny shards must not count as
                # much as large groups within the same minibatch.
                (loss * (len(rows) / len(indices))).backward()
                total += float(loss.detach()) * len(rows)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.options.max_grad_norm,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
        _finite_tree(self.net.state_dict())
        _finite_tree(self.optimizer.state_dict())
        validation = self.evaluate()
        self.epochs += 1
        self.metrics = {"epoch": self.epochs, "optimizer_updates": self.updates,
                        "training_rows": self.data.train_rows, "validation_rows": self.data.validation_rows,
                        "optimization_loss": total / self.data.train_rows, "validation_loss": validation}
        self.ready = True
        return dict(self.metrics)

    def checkpoint(self):
        if not self.ready or not self.epochs:
            raise RuntimeError("Only a completed, nonempty training epoch can be checkpointed")
        state = {"version": STATE_VERSION, "data": self.data.identities, "contract": self.data.contract,
                 "options": asdict(self.options), "runtime": self.runtime,
                 "parameters": self.parameter_signature,
                 "initial_sha256": self.initial_sha256, "initial_generation": self.initial_generation,
                 "epochs": self.epochs, "updates": self.updates, "metrics": self.metrics,
                 "optimizer": self.optimizer.state_dict(), "shuffle": self.shuffle.bit_generator.state,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        result = {**_model_payload(self.net), "learner": "search_supervised",
                  "generation": self.initial_generation + self.epochs,
                  "training_api_version": TRAINING_API_VERSION, "search_training": state}
        _finite_tree(result)
        return _cpu_copy(result)


def run(source: Path, replays, out: Path, *, epochs: int = 2, resume: bool = False,
        device: str = "cpu", overrides: dict | None = None):
    _integer(epochs, "epochs", 1)
    out, source = Path(out), Path(source)
    if out.exists():
        raise FileExistsError("Use a NEW output directory; existing runs are never overwritten")
    with tempfile.TemporaryDirectory(prefix="mahjong-search-learner-") as temporary:
        snapshot = Path(temporary) / "input.pt"
        copy_checkpoint(source, snapshot)
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        saved = payload.get("search_training") if resume else None
        if resume and (payload.get("learner") != "search_supervised" or not isinstance(saved, dict)):
            raise ValueError("--resume needs this trainer's complete checkpoint; use --checkpoint to initialize")
        settings = dict(saved["options"]) if saved is not None else asdict(Options())
        settings.update(overrides or {})
        options = Options(**settings)
        options.validate()
        dataset = Dataset.load(replays, options.validation_every)
        net, _ = _load_network(snapshot, device)
        with snapshot.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        learner = Learner(net, dataset, options, device=device,
                          initial_sha256=saved["initial_sha256"] if saved else source_hash,
                          initial_generation=saved["initial_generation"] if saved else payload.get("generation", 0),
                          resume=saved)
        # Claim the run only after validating data, model, settings and resume.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir()
        for _ in range(epochs):
            metrics = learner.epoch()
            atomic_save(learner.checkpoint(), out / "latest.pt")
            # The checkpoint, not stdout or a separate log, owns completed state.
            print(json.dumps(metrics, allow_nan=False), flush=True)
    return out / "latest.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--replay", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2, help="additional complete fitting epochs, not self-play rounds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for key in Options.__dataclass_fields__:
        parser.add_argument("--" + key.replace("_", "-"),
                            type=int if key in ("batch", "validation_every", "seed") else float,
                            default=None, help="inherits the saved value on resume")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.resume or args.checkpoint, args.replay, args.out, epochs=args.epochs,
        resume=args.resume is not None, device=args.device,
        overrides={key: getattr(args, key) for key in Options.__dataclass_fields__ if getattr(args, key) is not None})


if __name__ == "__main__":
    main()
