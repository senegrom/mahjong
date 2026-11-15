"""Check that each crate of the workspace declares a Rust release its locked
dependencies build with.

Cargo refuses to build a package whose rust-version is newer than the
compiler, so a crate whose declared rust-version is older than one of the
packages it builds is refused on exactly the releases its manifest says are
enough. A Dependabot bump that raises a dependency's rust-version does that
without touching the crate's manifest; this check fails its pull request.

It reads the locked graph with `cargo metadata --locked` and walks each
member's normal and build dependencies, on every platform, so it needs no
other toolchain. Development dependencies are left out: the declared release
promises a build, not the tests.

    python3 engine/rust-versions.py
"""

import json
import os
import subprocess
import sys


def version(text):
    """A rust-version as a tuple of numbers: "1.85" is (1, 85, 0)."""
    parts = [int(part) for part in text.split(".")]
    return tuple(parts + [0] * (3 - len(parts)))


def problems(metadata):
    """One line for each workspace member that declares a rust-version older
    than a package it builds; empty when every one holds."""
    packages = {package["id"]: package for package in metadata["packages"]}
    edges = {
        node["id"]: [
            dep["pkg"] for dep in node["deps"]
            if any(kind["kind"] in (None, "build") for kind in dep["dep_kinds"])
        ]
        for node in metadata["resolve"]["nodes"]
    }
    found = []
    for member in metadata["workspace_members"]:
        declared = packages[member]["rust_version"]
        if declared is None:
            continue
        seen, stack = set(), list(edges[member])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edges[node])
            needs = packages[node]["rust_version"]
            if needs is not None and version(needs) > version(declared):
                found.append(
                    f"{packages[member]['name']} declares Rust {declared}, but the locked "
                    f"{packages[node]['name']} {packages[node]['version']} requires {needs}"
                )
    return sorted(found)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    found = problems(json.loads(output))
    for line in found:
        print(line, file=sys.stderr)
    if found:
        return 1
    print("every crate's declared Rust release builds its locked dependencies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
