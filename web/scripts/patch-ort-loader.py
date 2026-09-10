"""Bounds the ONNX Runtime loader's memory, on a freshly built loader.

The loader Emscripten emits reserves a heap of four gigabytes and grows it
by its own rule. Phones refuse that reservation outright, and the tab dies
before the network has loaded. Three edits hand those three decisions to
`memory-budget.mjs`, which asks for what the device will give and grows
only when a run actually needs it.

The edits used to be made by hand, which meant a rebuild silently threw
them away. This makes them again, by pattern rather than by line, and
stops if the loader has changed shape rather than guessing.

    python3 web/scripts/patch-ort-loader.py <emitted.mjs> <patched.mjs>
"""

import re
import sys
from pathlib import Path

IMPORT = "import { memoryBudget as mahjongMemoryBudget } from './memory-budget.mjs';\n"

# The heap ceiling: a lone function returning 0xFFFF0000.
CEILING = re.compile(r"function (\w+)\(\)\{return 4294901760\}")

# The grower: takes a byte count, checks the current heap, and calls grow on
# the memory before telling the module its views moved.
GROWER = re.compile(r"function (\w+)\(a\)\{a>>>=0;var b=\(B\(\),I\)\.length;")

# Which name holds the memory, and which redoes the views after it grows.
INSIDE = re.compile(r"(\w+)\.grow\(c\);(\w+)\(\);")

# The reservation itself. The loader makes two; the one with a real size is
# the module's own heap, the empty one belongs to a worker.
RESERVATION = "new WebAssembly.Memory({initial:256,maximum:65536,shared:!0})"


def body_of(text: str, start: int) -> tuple[int, int]:
    """Where the function whose body opens at `start` ends."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise SystemExit("a function in the loader never closes")


def only(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    found = pattern.findall(text)
    if len(found) != 1:
        raise SystemExit(
            f"expected one {what} in the loader and found {len(found)}. "
            f"The build has emitted a loader of a different shape; read it and "
            f"teach this script the new one rather than shipping it unbounded."
        )
    match = pattern.search(text)
    assert match is not None
    return match


def patch(text: str) -> str:
    if "mahjongMemoryBudget" in text:
        raise SystemExit("this loader is already bounded; patch the emitted one")

    if text.count(RESERVATION) != 1:
        raise SystemExit(
            f"expected one heap reservation and found {text.count(RESERVATION)}"
        )
    text = text.replace(RESERVATION, "mahjongMemoryBudget.create()", 1)

    ceiling = only(CEILING, text, "heap ceiling")
    text = text.replace(
        ceiling.group(0),
        f"function {ceiling.group(1)}(){{return mahjongMemoryBudget.maximumBytes}}",
        1,
    )

    grower = only(GROWER, text, "heap grower")
    start, stop = body_of(text, text.index("{", grower.start()))
    inside = INSIDE.search(text[start:stop])
    if inside is None:
        raise SystemExit("the grower does not grow a memory the way this script reads")
    memory, refresh = inside.group(1), inside.group(2)
    text = (
        text[: grower.start()]
        + f"function {grower.group(1)}(a){{return mahjongMemoryBudget.grow({memory},a>>>0,{refresh})}}"
        + text[stop:]
    )

    return IMPORT + text


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch-ort-loader.py <emitted.mjs> <patched.mjs>")
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    patched = patch(source.read_text(encoding="utf-8"))
    destination.write_text(patched, encoding="utf-8")
    print(
        f"bounded {source.name} -> {destination.name} "
        f"({len(patched)} bytes, {patched.count('mahjongMemoryBudget')} hooks)"
    )


if __name__ == "__main__":
    main()
