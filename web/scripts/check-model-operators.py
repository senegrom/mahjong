"""Check explicitly supplied ONNX exports against the reduced runtime.

Usage: python3 web/scripts/check-model-operators.py [model.onnx ...]
No model is shipped in web/public. With no arguments this validates the
operator configuration; export-time checks and real browser inference verify
the remote model. No empty checksum file is created or treated as evidence.
"""
import sys
from pathlib import Path
import onnx

CONFIG = Path(__file__).resolve().parent.parent / "runtime" / "reduced-ops.config"


def allowed_operators(config: Path) -> set[str]:
    for line in reversed(config.read_text(encoding="utf-8").strip().splitlines()):
        if not line.startswith("#") and ";" in line:
            return {name.strip() for name in line.split(";")[-1].split(",") if name.strip()}
    raise SystemExit(f"{config} names no operators")


def main() -> None:
    allowed = allowed_operators(CONFIG)
    models = [Path(name) for name in sys.argv[1:]]
    if not models:
        print(f"configuration contains {len(allowed)} operators; no local model checked. Remote exports are checked by neural.export and browser tests.")
        return
    used, trouble = set(), []
    for model in models:
        needed = {node.op_type for node in onnx.load(str(model)).graph.node}
        used.update(needed)
        missing = sorted(needed - allowed)
        if missing:
            trouble.append(f"{model.name} needs {', '.join(missing)}")
        print(f"{model.name}: {len(needed)} operators; " + (f"MISSING {', '.join(missing)}" if missing else "ok"))
    if trouble:
        raise SystemExit("; ".join(trouble) + f". Widen {CONFIG.name} and rebuild the runtime.")
    if allowed - used:
        print(f"built for but unused: {', '.join(sorted(allowed - used))}")


if __name__ == "__main__":
    main()
