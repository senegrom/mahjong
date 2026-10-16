"""Check that the Python engines action installed nothing it did not pin.

    python check_pins.py snapshot BEFORE.json
    python check_pins.py check BEFORE.json PINS.txt [PINS.txt ...] [--allow NAME ...]

`snapshot` records every distribution the job's Python holds before the
action installs anything. `check` then fails on every pin that is not
installed at its version, and on every distribution that is new or changed
since the snapshot and that no pin file pins. A name given with --allow is
pinned elsewhere: PyTorch, on its own install line, since it comes from
PyTorch's index rather than PyPI. A package that slips in unpinned, such as
a new dependency of a pinned release, would otherwise float with every
release on its index, which is what the pin files exist to prevent. Only the
standard library is used, since the snapshot is taken before anything is
installed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re


def canonical(name: str) -> str:
    """A distribution's name as package indexes compare it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed() -> dict[str, str]:
    """Every distribution on the path, by canonical name, at the version an
    import would find: the first of that name on the path."""
    found: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            found.setdefault(canonical(name), distribution.version)
    return found


def pinned(files: list[Path]) -> dict[str, str]:
    """The pins of requirement files that hold only `name==version` lines
    and comments. Anything else is refused: a range, a marker or a pin
    written twice would not say which release is installed."""
    pins: dict[str, str] = {}
    for file in files:
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            name, _, version = (part.strip() for part in text.partition("=="))
            if not (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                    and re.fullmatch(r"[0-9][0-9A-Za-z.!+-]*", version)):
                raise SystemExit(f"{file}:{number}: {text!r} is not an exact name==version pin")
            if canonical(name) in pins:
                raise SystemExit(f"{file}:{number}: {name} is pinned twice")
            pins[canonical(name)] = version
    return pins


def same(pin: str, version: str) -> bool:
    """Whether an installed version is the pinned release, as `==` reads
    it: trailing zero components and a local label such as +cpu aside."""
    def release(text: str) -> str:
        return re.sub(r"(\.0+)+$", "", text.split("+", 1)[0])
    return release(pin) == release(version)


def problems(earlier: dict[str, str], pins: dict[str, str], now: dict[str, str],
             allowed: set[str]) -> list[str]:
    """What breaks the promise that the pin files name every package the
    action installed, at the version it installed."""
    found = []
    for name, version in sorted(pins.items()):
        if name not in now:
            found.append(f"{name}=={version} is pinned, and not installed")
        elif not same(version, now[name]):
            found.append(f"{name} {now[name]} is installed, where {version} is pinned")
    for name, version in sorted(now.items()):
        if name not in pins and name not in allowed and earlier.get(name) != version:
            found.append(f"{name} {version} was installed, and no pin file pins it")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="record what is installed now")
    snapshot.add_argument("before", type=Path)
    check = commands.add_parser("check", help="compare what was installed since with the pins")
    check.add_argument("before", type=Path)
    check.add_argument("pins", type=Path, nargs="+")
    check.add_argument("--allow", action="append", default=[],
                       help="a package pinned elsewhere, such as torch")
    arguments = parser.parse_args()

    if arguments.command == "snapshot":
        arguments.before.write_text(json.dumps(installed(), indent=1, sort_keys=True),
                                    encoding="utf-8")
        return 0
    earlier = json.loads(arguments.before.read_text(encoding="utf-8"))
    pins = pinned(arguments.pins)
    found = problems(earlier, pins, installed(), {canonical(name) for name in arguments.allow})
    # A workflow command, so that GitHub shows each one on the run's summary.
    prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") == "true" else ""
    for problem in found:
        print(f"{prefix}{problem}")
    if found:
        return 1
    print(f"{len(pins)} pinned packages are installed at their pins, and nothing unpinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
