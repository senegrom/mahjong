"""Dense replay schema, checked on publication and reopening, in bounded chunks."""
from __future__ import annotations

import numpy as np


def validate_dense_replay(maps: dict, n: int) -> None:
    """The standalone PPO ring stores native actions and auxiliary byte planes.

    Exact vector shape prevents target broadcasting. Canonical dtypes prevent
    lossy casts, and value checks reject data before it reaches any loss.
    Chunking bounds temporary memory even for multi-gigabyte memory maps.
    """
    import riichi_py

    if type(n) is not int or n <= 0:
        raise ValueError("Replay needs a positive integer number of decisions")
    shapes = {
        "legal": ((n, riichi_py.ACTIONS), np.dtype(np.bool_)),
        "held": ((n, riichi_py.OPPONENTS, riichi_py.POSITIONS), np.dtype(np.float32)),
        "oracle": ((n, riichi_py.ORACLE_PLANES, riichi_py.POSITIONS), np.dtype(np.uint8)),
        "imagined": ((n, riichi_py.HIDDEN_HANDS_PLANES, riichi_py.POSITIONS), np.dtype(np.uint8)),
        "returns": ((n,), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in shapes.items():
        values = maps.get(name)
        if not isinstance(values, np.ndarray) or values.shape != shape or values.dtype != dtype:
            raise ValueError(f"Replay {name} must have shape {shape} and dtype {dtype}")
    for start in range(0, n, 4096):
        section = slice(start, start + 4096)
        if not maps["legal"][section].any(axis=1).all():
            raise ValueError("Replay legal masks need at least one legal action per row")
        if not np.isfinite(maps["returns"][section]).all():
            raise ValueError("Replay returns must be finite")
        held = maps["held"][section]
        if not np.isfinite(held).all() or np.any(held < 0) or np.any(held > 1):
            raise ValueError("Replay held targets must be finite probabilities")
        mass = held.sum(axis=2)
        if not np.all((mass == 0) | np.isclose(mass, 1, rtol=1e-5, atol=1e-6)):
            raise ValueError("Replay held targets must sum to one or describe an empty hand")
        for name in ("oracle", "imagined"):
            if np.any(maps[name][section] > 1):
                raise ValueError(f"Replay {name} must contain binary byte planes")
