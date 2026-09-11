"""Invocation-owned training scratch: reused containers are never resume sources."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import tempfile
from typing import Iterator


def validate_run(run: str) -> str:
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run):
        raise ValueError("run must be one nonempty directory name, without path separators")
    return run


@contextmanager
def workspace(run: str, root: Path = Path("/scratch")) -> Iterator[Path]:
    """No checkpoint, replay, log or reference can leak into a later invocation.

    Only this newly created directory is removed. Compiler caches and all
    existing user data (including old scratch directories) remain untouched.
    """
    validate_run(run)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{run}-", dir=root) as folder:
        where = Path(folder)
        (where / "run.json").write_text(json.dumps({"run": run, "workspace": where.name}),
                                       encoding="utf-8")
        yield where


@contextmanager
def managed_process(process):
    """Stop an owned trainer if publication fails or its controller is cancelled."""
    import subprocess
    try:
        yield process
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
