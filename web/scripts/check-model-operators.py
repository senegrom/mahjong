"""Every shipped network against the operator list the runtime is built from.

The runtime in `web/runtime` carries only the operators named in
`reduced-ops.config`. A network that asks for one more does not degrade in
the browser, it fails to load at all, and the failure surfaces as an
opponent that never moves. So the list is checked here, before a build,
rather than in someone's tab.

    python3 web/scripts/check-model-operators.py [model.onnx ...]

With no arguments it checks every `.onnx` in `web/public`.
"""

import sys
from hashlib import sha256
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent.parent
CONFIG = HERE / "runtime" / "reduced-ops.config"
PUBLIC = HERE / "public"


def allowed_operators(config: Path) -> set[str]:
    """The operators the runtime was built with."""
    for line in reversed(config.read_text(encoding="utf-8").strip().splitlines()):
        if line.startswith("#") or ";" not in line:
            continue
        return {name.strip() for name in line.split(";")[-1].split(",") if name.strip()}
    raise SystemExit(f"{config} names no operators")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit(f"no operator list at {CONFIG}")
    allowed = allowed_operators(CONFIG)
    named = [Path(name) for name in sys.argv[1:]]
    models = named or sorted(PUBLIC.glob("*.onnx"))
    if not models:
        raise SystemExit(f"no networks to check in {PUBLIC}")

    trouble = []
    for model in models:
        needed = {node.op_type for node in onnx.load(str(model)).graph.node}
        missing = sorted(needed - allowed)
        size = model.stat().st_size / 1e6
        if missing:
            trouble.append(f"{model.name} needs {', '.join(missing)}")
            print(f"  {model.name:<24} {size:6.1f} MB  MISSING {', '.join(missing)}")
        else:
            print(f"  {model.name:<24} {size:6.1f} MB  ok, {len(needed)} operators")

    spare = sorted(allowed - {
        node.op_type
        for model in models
        for node in onnx.load(str(model)).graph.node
    })
    if spare:
        print(f"built for but unused: {', '.join(spare)}")
    if trouble:
        raise SystemExit(
            "the runtime is not built for these: "
            + "; ".join(trouble)
            + f". Widen {CONFIG.name} and rebuild the runtime."
        )
    print(f"every network fits the {len(allowed)} operators the runtime carries")

    # What was checked, so the site's own build can refuse to publish a
    # network nobody has checked. Written only over the whole folder, since
    # a partial list would say the rest is fine.
    if not named:
        record = HERE / "runtime" / "models.sha256"
        lines = [
            f"{sha256(model.read_bytes()).hexdigest()}  {model.name}" for model in models
        ]
        record.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {record.name} for {len(lines)} networks")


if __name__ == "__main__":
    main()
