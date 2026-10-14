"""Reading a ruler a few seeds at a time, by a rule fixed before the first look.

A ruler is a series of calls, one a seed, each a thousand deals played four
times (`neural.duel`, `neural.ruler`). A clear difference shows long before
the twentieth seed and a level one never does, so the series is read as it
comes in. Reading it after every seed and stopping at the first reading past
two standard errors would find a difference that is not there about one time
in four over twenty looks rather than one in twenty, so the looks and the
bound are fixed before the first seed is spawned (`Design`, kept beside the
results as `design.json`):

- The seeds are read in their registered order, seed `k` being
  `first_seed + k * step` (9,100,000 + k x 100,003 for the published rulers);
  a look covers the first seeds only, and waits until every one of them is
  in. A failed call is run again with its own seed, never replaced by a
  later one.
- A look is taken after every `per_look` seeds, `looks` of them in all: five
  seeds a look, four looks, twenty seeds; or ten seeds a look, forty seeds,
  when the effect looked for is 0.006 placement or less (`design`).
- At a look, `z` is the mean per-deal difference over every deal so far over
  its standard error, from the spread across deals: thousands of deals by
  the first look, so the spread is known well enough to treat as known.
- The comparison stops at the first look where |z| passes the bound, 2.361
  for four looks: Pocock's bound, which holds the chance that any look
  crosses it with no difference at all to 5% two-sided, as one look at 1.96
  would. The side `z` points to is then the stronger. No crossing by the
  last look reads level.

The difference read is the first player's edge less the second's, positive
when the first is the stronger: a paired call's two challengers (see
`neural.ruler`), two series of single rulers on the same seeds read deal by
deal, or one ruler's challenger against its incumbent, 2.5 less its
placement.

    python -m neural.looks results/<tag>
    python -m neural.looks results/g910-vs-mortal --against results/g810-vs-mortal

Standard library only: it runs beside the Modal client, which has no NumPy.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: The published rulers' first seed and the distance between seeds.
FIRST_SEED = 9_100_000
STEP = 100_003
#: An effect this small or smaller is looked for with ten seeds a look.
SMALL_EFFECT = 0.006


def _density(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _grid(half: float, points: int) -> tuple[list[float], list[float]]:
    """`points` (odd) evenly spaced points over (-half, half) and their
    Simpson weights."""
    step = 2.0 * half / (points - 1)
    xs = [-half + i * step for i in range(points)]
    weights = [step / 3.0 * (1 if i in (0, points - 1) else 4 if i % 2 else 2) for i in range(points)]
    return xs, weights


def stopping(bound: float, looks: int, drift: float = 0.0, points: int = 201) -> list[tuple[float, float]]:
    """For each look, the chance the comparison stops there with z above
    `bound` and with z below `-bound`, when each look adds as many deals and
    the true difference moves z by `drift` a look's worth of deals (zero
    with no difference). The sum over the looks and both sides is the chance
    of any stop; with no drift that is the error the bound allows.

    The partial sums of the looks' standardised differences are a random
    walk, and a look stops it when the sum leaves `bound` times the square
    root of the looks so far. The density of the walks still going is
    carried from look to look on a grid and integrated by Simpson's rule
    (as Armitage, McPherson and Rowe computed such bounds)."""
    if looks < 1 or points < 3 or points % 2 == 0:
        raise ValueError("looks must be at least one and points odd and at least three")
    xs, weights = _grid(bound, points)
    # Look 1: the walk is one step, normal about `drift`; `alive` is the
    # density of the walks it leaves going.
    out = [(1.0 - _normal(bound - drift), _normal(-bound - drift))]
    alive = [_density(x - drift) for x in xs]
    for look in range(2, looks + 1):
        half = bound * math.sqrt(look)
        mass = [w * f for w, f in zip(weights, alive)]
        # What leaves at this look, above and below: the walks still going
        # after the last look, carried one step and found outside.
        upper = sum(m * (1.0 - _normal(half - x - drift)) for m, x in zip(mass, xs))
        lower = sum(m * _normal(-half - x - drift) for m, x in zip(mass, xs))
        out.append((upper, lower))
        new_xs, weights = _grid(half, points)
        alive = [sum(m * _density(s - x - drift) for m, x in zip(mass, xs)) for s in new_xs]
        xs = new_xs
    return out


def _normal(x: float) -> float:
    """The standard normal's distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@functools.cache
def pocock_bound(looks: int, alpha: float = 0.05) -> float:
    """The one bound on |z| that every look shares and that holds the
    chance of any crossing, with no difference, to `alpha`: 2.178 for two
    looks, 2.289 for three, 2.361 for four (Pocock, 1977)."""
    if looks == 1:
        # One look is the ordinary two-sided test.
        low, high = 0.0, 10.0
    else:
        low, high = 1.0, 5.0

    def error(bound: float) -> float:
        return sum(up + down for up, down in stopping(bound, looks))

    for _ in range(60):
        middle = 0.5 * (low + high)
        if error(middle) > alpha:
            low = middle
        else:
            high = middle
        if high - low < 1e-5:
            break
    return round(0.5 * (low + high), 4)


@dataclass
class Design:
    """A ruler's rule, written down before its first seed is spawned:
    which seeds, in which order, how many a look, how many looks, and the
    bound every look is read against. `what` says what is compared and
    `expected` the effect it was designed to find, for whoever reads it
    later; `registered` is when it was written."""

    first_seed: int = FIRST_SEED
    step: int = STEP
    per_look: int = 5
    looks: int = 4
    alpha: float = 0.05
    bound: float = 0.0
    expected: float | None = None
    what: str = ""
    registered: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def __post_init__(self) -> None:
        for name in ("per_look", "looks", "step"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive whole number")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between nought and one")
        if not self.bound:
            self.bound = pocock_bound(self.looks, self.alpha)

    @property
    def seeds(self) -> list[int]:
        """Every seed of the design, in the order they are read."""
        return [self.first_seed + k * self.step for k in range(self.per_look * self.looks)]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Design:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def design(expected: float | None = None, what: str = "", first_seed: int = FIRST_SEED) -> Design:
    """The rule for a comparison expected to differ by `expected` placement:
    five seeds a look, four looks, twenty seeds; ten a look, forty seeds,
    when the effect looked for is 0.006 or less, where twenty seeds (an
    error about 0.0034 on the difference of two rulers of one lineage)
    would find it about half the time. None designs for the usual 20."""
    small = expected is not None and abs(expected) <= SMALL_EFFECT
    return Design(first_seed=first_seed, per_look=10 if small else 5, looks=4, expected=expected, what=what)


def differences(report: dict) -> list[float]:
    """A call's per-deal differences, positive when the first player is the
    stronger: for a paired call the first challenger's edge less the
    second's, deal by deal; for one challenger its edge over the incumbent,
    2.5 less its placement."""
    if "arms" in report:
        first, second = report["arms"][:2]
        return [b - a for a, b in zip(first["by_deal"], second["by_deal"], strict=True)]
    return [2.5 - placement for placement in report["by_deal"]]


def against(first: dict, second: dict) -> list[float]:
    """Two single rulers on the same seed, deal by deal: the first's edge
    less the second's."""
    if first["seed"] != second["seed"] or len(first["by_deal"]) != len(second["by_deal"]):
        raise ValueError("the two rulers were not played on the same deals")
    return [b - a for a, b in zip(first["by_deal"], second["by_deal"], strict=True)]


@dataclass
class Look:
    """One look of a series: the seeds and deals it covers, the mean
    difference, its error, z, and what the rule says."""

    look: int
    seeds: int
    deals: int
    edge: float
    error: float
    z: float
    decision: str


def read(rule: Design, by_seed: dict[int, list[float]]) -> tuple[list[Look], str]:
    """The looks the results so far allow, by the rule, and the state of
    the comparison: "first" or "second" once a look stopped it with that
    side the stronger, "level" once the last look did not, and "waiting"
    while the next look's seeds are not all in. Results for seeds the rule
    does not name are left out, as are those past a look that stopped."""
    order = rule.seeds
    taken: list[Look] = []
    pooled: list[float] = []
    for look in range(1, rule.looks + 1):
        wanted = order[(look - 1) * rule.per_look : look * rule.per_look]
        if not all(seed in by_seed for seed in wanted):
            return taken, "waiting"
        for seed in wanted:
            pooled.extend(by_seed[seed])
        n = len(pooled)
        # In floats rather than `statistics`' exact fractions: tens of
        # thousands of deals a look, and nothing here needs the last bit.
        edge = math.fsum(pooled) / n
        error = math.sqrt(math.fsum((value - edge) ** 2 for value in pooled) / (n - 1) / n) if n > 1 else 0.0
        z = edge / error if error > 0 else (0.0 if edge == 0 else math.copysign(math.inf, edge))
        if z > rule.bound:
            decision = "first"
        elif z < -rule.bound:
            decision = "second"
        elif look == rule.looks:
            decision = "level"
        else:
            decision = "go on"
        taken.append(Look(look, look * rule.per_look, n, edge, error, z, decision))
        if decision != "go on":
            return taken, decision
    return taken, "level"


def load_folder(folder: Path) -> dict[int, dict]:
    """The reports in a folder, one a seed, each `seed-<seed>.json` as the
    cloud keeps them (see `modal_app.ruler`)."""
    reports = {}
    for path in sorted(Path(folder).glob("seed-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["seed"]) in reports:
            raise ValueError(f"two reports for seed {report['seed']} in {folder}")
        reports[int(report["seed"])] = report
    return reports


def describe(rule: Design, looks: list[Look], state: str, names: tuple[str, str]) -> str:
    """The looks as a table, and what they say, in words."""
    first, second = names
    lines = [
        f"rule: {rule.per_look} seeds a look, {rule.looks} looks ({rule.per_look * rule.looks} seeds from "
        f"{rule.first_seed}, every {rule.step}), stop when |z| > {rule.bound} "
        f"(Pocock, {rule.alpha:.0%} two-sided over all looks); registered {rule.registered}"
        + (f"; expected {rule.expected:+.4f}" if rule.expected is not None else ""),
    ]
    if rule.what:
        lines.append(f"comparing: {rule.what}")
    lines.append("look  seeds   deals   difference   error      z     reading")
    for look in looks:
        lines.append(f"{look.look:>4}  {look.seeds:>5}  {look.deals:>6}  {look.edge:+10.4f}  {look.error:7.4f}  "
                     f"{look.z:+6.2f}   {look.decision}")
    if state == "first":
        lines.append(f"stopped at look {looks[-1].look}: {first} is the stronger")
    elif state == "second":
        lines.append(f"stopped at look {looks[-1].look}: {second} is the stronger")
    elif state == "level":
        lines.append(f"level: no look crossed {rule.bound} by seed {rule.per_look * rule.looks}")
    else:
        lines.append(f"waiting for the seeds of look {len(looks) + 1}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="one report a seed, and the series' design.json")
    parser.add_argument("--against", type=Path, default=None,
                        help="a second series of single rulers on the same seeds, read deal by deal")
    parser.add_argument("--design", type=Path, default=None, help="the rule, if not the folder's design.json")
    parser.add_argument("--json", action="store_true", help="print the looks as JSON")
    args = parser.parse_args()

    path = args.design or args.folder / "design.json"
    if path.exists():
        rule = Design.load(path)
    else:
        rule = design()
        print(f"no design at {path}: reading by the usual rule, which was not registered for this series")
    reports = load_folder(args.folder)
    if args.against is not None:
        others = load_folder(args.against)
        by_seed = {seed: against(reports[seed], others[seed]) for seed in reports if seed in others}
        names = (str(args.folder), str(args.against))
    else:
        by_seed = {seed: differences(report) for seed, report in reports.items()}
        sample = next(iter(reports.values()), {})
        if "arms" in sample:
            names = tuple(arm.get("challenger", f"challenger {i + 1}") for i, arm in enumerate(sample["arms"][:2]))
        else:
            names = (sample.get("challenger", "the challenger"), sample.get("incumbent", "the incumbent"))
    looks, state = read(rule, by_seed)
    if args.json:
        print(json.dumps({"design": asdict(rule), "looks": [asdict(look) for look in looks], "state": state},
                         indent=1))
    else:
        print(describe(rule, looks, state, names))


if __name__ == "__main__":
    main()
