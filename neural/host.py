"""What a trainer runs on and what holds it back, for the generation's record.

A generation is bound by whichever of its two halves takes longer: the
round of self-play, on the processors, or the learning, on the card, and a
round played beside the learning (`--play-ahead`) takes some of each from
the other. Telling which bounds it, and why one container is faster than
another, needs what the record could not say before: which card and which
processors the generation had, the container's share of them, and whether
the container was held back. This reads them from the platform and nothing
else; nothing here changes anything played or learned.

Whatever the platform does not show is left out rather than guessed: a
Windows desktop shows none of these files, and a sandboxed container only
some of them (Modal's has printed "quota unknown" for `cpu.max`).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path("/")


def card(device: str | torch.device) -> str | None:
    """The card's name as its driver gives it, or None on the processor."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(device)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _quota(root: Path) -> float | str | None:
    """The container's share of the processors, in processors, from the
    cgroup's quota and period (version 2's `cpu.max`, else version 1's CFS
    files); "unlimited" where it has none, None where neither is shown."""
    text = _read(root / "sys/fs/cgroup/cpu.max")
    try:
        if text is not None:
            quota, period = text.split()[:2]
            return "unlimited" if quota == "max" else round(int(quota) / int(period), 2)
        quota_text = _read(root / "sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_text = _read(root / "sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_text is not None and period_text is not None:
            quota = int(quota_text)
            return "unlimited" if quota < 0 else round(quota / int(period_text), 2)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def processors(root: Path = ROOT) -> dict:
    """The processors: their model as `/proc/cpuinfo` names it, how many the
    system shows, how many this process may run on, the container's quota
    of them, and how many threads the encoder spreads a batch over
    (`RAYON_NUM_THREADS`, which the cloud sets to the processors asked
    for, not to how many it shows)."""
    described: dict = {"cpus": os.cpu_count()}
    info = _read(root / "proc/cpuinfo")
    for line in (info or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            described["cpu_model"] = value.strip()
            break
    if hasattr(os, "sched_getaffinity"):
        described["cpus_allowed"] = len(os.sched_getaffinity(0))
    quota = _quota(root)
    if quota is not None:
        described["cpu_quota"] = quota
    threads = os.environ.get("RAYON_NUM_THREADS", "")
    if threads.isdigit():
        described["encoder_threads"] = int(threads)
    return described


def _ticks_per_second() -> int:
    try:
        return os.sysconf("SC_CLK_TCK")
    except (AttributeError, ValueError, OSError):
        return 100


def counters(root: Path = ROOT) -> dict[str, float]:
    """Running totals the platform keeps of time this container was held
    back, where it shows them: the periods in which the cgroup's quota
    throttled it and for how long (`cpu.stat`), the processors' steal time,
    taken by the hypervisor for others (`/proc/stat`), and the time some
    task here waited for a processor (`/proc/pressure/cpu`). Seconds, but
    for the periods; a generation's record has what each grew by (see
    `change`)."""
    found: dict[str, float] = {}
    stat = _read(root / "sys/fs/cgroup/cpu.stat")
    ticks = 1e6  # version 2 counts microseconds
    if stat is None:
        stat = _read(root / "sys/fs/cgroup/cpu/cpu.stat")
        ticks = 1e9  # version 1 counts nanoseconds
    for line in (stat or "").splitlines():
        name, _, value = line.partition(" ")
        try:
            if name == "nr_throttled":
                found["throttled_periods"] = int(value)
            elif name in ("throttled_usec", "throttled_time"):
                found["throttled_seconds"] = int(value) / ticks
        except ValueError:
            pass
    proc_stat = _read(root / "proc/stat")
    for line in (proc_stat or "").splitlines():
        fields = line.split()
        # cpu user nice system idle iowait irq softirq steal ...
        if fields and fields[0] == "cpu" and len(fields) > 8:
            try:
                found["steal_seconds"] = int(fields[8]) / _ticks_per_second()
            except ValueError:
                pass
            break
    pressure = _read(root / "proc/pressure/cpu")
    for line in (pressure or "").splitlines():
        if line.startswith("some "):
            for field in line.split():
                if field.startswith("total="):
                    try:
                        found["cpu_pressure_seconds"] = int(field[len("total="):]) / 1e6
                    except ValueError:
                        pass
    return found


def change(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """What each counter grew by between two readings, for the counters
    both readings show."""
    return {name: round(after[name] - before[name], 3) for name in after if name in before}


def cpu_seconds() -> float:
    """The processor time this process has used, its every thread's, user
    and system."""
    times = os.times()
    return times.user + times.system
