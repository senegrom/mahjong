"""Running rulers on Modal from the desktop: a series read look by look, and
the timing of the old duel against the new.

A series is registered under a tag before its first call: its players and how
they are asked (`series.json`) and the rule it is read by (`design.json`, see
`neural.looks`), both kept in the local folder and beside the reports on the
volume. `series` then spawns one look's seeds at a time, waits for them, reads
the look, and spawns the next only if the rule says to go on, so a series that
stops at its first look pays for five seeds rather than twenty. It can be
stopped and run again: the calls it spawned are in `calls.txt`, and the
reports are kept on the volume (`rulers/<tag>/`, see `modal_app.ruler`) and
mirrored locally.

    python -m neural.ruler_cloud series <tag> <challenger> <incumbent> [--also <second>]
        [--expected 0.006] [--questions fixed --graphs --skip-forced --preview-reach]
        [--run fusion-long] [--local rulers]
    python -m neural.ruler_cloud looks <tag> [--against <tag>] [--local rulers]
    python -m neural.ruler_cloud timing <challenger> <incumbent> [--run fusion-long] [--seed 9100000]

Checkpoints are named as `modal_app` names them: within `--run`, or by their
path from the volume's root (`zoo/mortal_298k`). Run with the Modal client's
Python from the repository root; it needs nothing but the client.

`timing` plays one call of 1,000 deals three ways on the same seed and says
how they compare: the old duel (`neural.duel`, its four tables one after
another), the ruler asking as the old duel asked (`--questions seating`), and
the ruler asking in fixed rows with every switch that then changes no answer.
What it should print, with the old duel's figures from the published rulers
(about 21 minutes a 4,000-game call on an L40S) and the rest estimated from
the desktop (see docs/RULER.md):

    old duel             placement 2.4110 +- 0.0170 over 1000 deals; play 1260 s ...
    ruler, seating       placement 2.4110 ...; the same placements on 1000 of 1000 deals
    ruler, fixed 256     placement 2.4... ...; other games, as expected: ...
    the seating ruler plays the old duel's games bit for bit, x.xx as fast

The seating ruler must place exactly as the old duel on every deal; if it
does not, it has changed games and must not be used for a series measured
the old way. The roadmap keeps the port only if it is at least 1.5 times as
fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import looks

APP = "mahjong-train"
VOLUME = "mahjong-train"
#: The folder the reports are kept in on the volume (see `modal_app.RULERS`).
RULERS = "rulers"
#: How often a waiting series asks Modal how its calls are doing.
POLL = 60
#: How often a seed whose call failed is spawned again, with its own seed.
RESPAWNS = 2
#: The players and the asking a series keeps from its first call to its last.
SERIES = ("challenger", "also", "incumbent", "games", "questions", "rows", "graphs", "skip_forced",
          "preview_reach", "run")


def say(*words) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *words, flush=True)


class Series:
    """A registered series: its local folder, what it plays, its rule, and
    the calls it has spawned, a line each (`seed<TAB>call id`)."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    @property
    def tag(self) -> str:
        return self.folder.name

    def register(self, settings: dict, rule: looks.Design) -> looks.Design:
        """Writes what the series plays and its rule, or, registered before,
        checks that nothing has changed and answers the rule it was
        registered with: a series read by a rule chosen after its first look
        would make the rule's error rate fiction."""
        self.folder.mkdir(parents=True, exist_ok=True)
        kept = self.folder / "series.json"
        if kept.exists():
            before = json.loads(kept.read_text(encoding="utf-8"))
            if before != settings:
                changed = sorted(name for name in set(before) | set(settings) if before.get(name) != settings.get(name))
                raise SystemExit(f"{self.tag} was registered with other {', '.join(changed)}: "
                                 "a series keeps what it plays; register a new tag")
            registered = looks.Design.load(self.folder / "design.json")
            if (registered.per_look, registered.looks, registered.first_seed, registered.step) != (
                    rule.per_look, rule.looks, rule.first_seed, rule.step):
                raise SystemExit(f"{self.tag} was registered with another rule: {asdict(registered)}")
            return registered
        kept.write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")
        rule.save(self.folder / "design.json")
        return rule

    def calls(self) -> dict[int, list[str]]:
        """The calls spawned for each seed, oldest first, leaving out a line
        a kill cut short."""
        found: dict[int, list[str]] = {}
        path = self.folder / "calls.txt"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    found.setdefault(int(parts[0]), []).append(parts[1])
        return found

    def note(self, seed: int, call_id: str) -> None:
        with open(self.folder / "calls.txt", "a", encoding="utf-8") as handle:
            handle.write(f"{seed}\t{call_id}\n")
            handle.flush()

    def reports(self) -> dict[int, dict]:
        return looks.load_folder(self.folder)

    def keep(self, report: dict) -> None:
        path = self.folder / f"seed-{report['seed']}.json"
        staged = path.with_suffix(".partial")
        staged.write_text(json.dumps(report, indent=1), encoding="utf-8")
        staged.replace(path)


def upload(modal, series: Series) -> None:
    """Puts the series' registration beside its reports on the volume."""
    volume = modal.Volume.from_name(VOLUME)
    with volume.batch_upload(force=True) as batch:
        for name in ("series.json", "design.json"):
            batch.put_file(str(series.folder / name), f"{RULERS}/{series.tag}/{name}")


def mirror(modal, series: Series) -> int:
    """Copies the series' reports the volume holds and the folder does not.
    Answers how many it copied."""
    volume = modal.Volume.from_name(VOLUME)
    have = {path.name for path in series.folder.glob("seed-*.json")}
    copied = 0
    try:
        entries = volume.listdir(f"{RULERS}/{series.tag}")
    except modal.exception.NotFoundError:
        return 0
    for entry in entries:
        name = entry.path.rstrip("/").rsplit("/", 1)[-1]
        if not name.startswith("seed-") or name in have:
            continue
        text = b"".join(volume.read_file(f"{RULERS}/{series.tag}/{name}")).decode("utf-8")
        series.keep(json.loads(text))
        copied += 1
    return copied


def unasked(modal) -> tuple[type, ...]:
    """The client's errors that say Modal could not be asked, or refused a
    launch for now (the spend limit, a full queue), rather than that a call
    failed: what is met with asking again later, never with a respawn,
    which would pay for a seed twice."""
    names = ("ConnectionError", "ServiceError", "InternalError", "ResourceExhaustedError")
    found = (getattr(modal.exception, name, None) for name in names)
    return tuple(error for error in found if isinstance(error, type))


def outcome(modal, call_id: str):
    """("done", report), ("playing", None) or ("failed", why) for one call;
    a call Modal could not be asked about is still playing."""
    try:
        answer = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError as error:
        # The builtin is get()'s "not yet"; the call's own timeout carries a message.
        return ("failed", str(error)[-200:]) if error.args else ("playing", None)
    except unasked(modal):
        return "playing", None
    except Exception as error:  # noqa: BLE001 - whatever ended the call
        return "failed", f"{type(error).__name__}: {str(error)[-200:]}"
    report = json.loads(answer) if isinstance(answer, str) and answer.strip() else answer
    return ("done", report) if isinstance(report, dict) and report else ("failed", "an empty answer")


def spawn_seed(modal, series: Series, settings: dict, seed: int) -> str:
    ruler = modal.Function.from_name(APP, "ruler")
    arguments = {name: settings[name] for name in SERIES if name != "also"}
    call = ruler.spawn(also=settings["also"] or "", seed=seed, tag=series.tag, **arguments)
    series.note(seed, call.object_id)
    say(f"{series.tag}: seed {seed} spawned as {call.object_id}")
    return call.object_id


def settle(modal, series: Series, settings: dict, seeds: list[int]) -> bool:
    """Spawns whatever of `seeds` has no call yet, keeps the reports of the
    calls that have finished, and spawns a failed seed again, at most
    `RESPAWNS` times. True once every one of `seeds` has a report."""
    reports = series.reports()
    calls = series.calls()
    for seed in seeds:
        if seed in reports:
            continue
        spawned = calls.get(seed, [])
        if not spawned:
            spawn_seed(modal, series, settings, seed)
            continue
        state, what = outcome(modal, spawned[-1])
        if state == "done":
            series.keep(what)
        elif state == "failed":
            if len(spawned) > RESPAWNS:
                raise SystemExit(f"{series.tag}: seed {seed} failed {len(spawned)} times, last: {what}")
            say(f"{series.tag}: seed {seed} failed ({what}); spawning it again")
            spawn_seed(modal, series, settings, seed)
    mirror(modal, series)
    return all(seed in series.reports() for seed in seeds)


def run_series(modal, series: Series, settings: dict, rule: looks.Design, poll: float = POLL) -> str:
    """Plays a registered series look by look until its rule stops it, and
    answers how it ended: "first", "second" or "level"."""
    order = rule.seeds
    # The difference is the first challenger's edge less the second's, or,
    # with one challenger, its edge over the incumbent.
    names = (settings["challenger"], settings["also"] or settings["incumbent"])
    shown = 0
    while True:
        by_seed = {seed: looks.differences(report) for seed, report in series.reports().items()}
        taken, state = looks.read(rule, by_seed)
        if len(taken) > shown:
            print(looks.describe(rule, taken, state, names), flush=True)
            shown = len(taken)
        if state != "waiting":
            return state
        wanted = order[: (len(taken) + 1) * rule.per_look]
        try:
            ready = settle(modal, series, settings, wanted)
        except unasked(modal) as error:
            say(f"{series.tag}: Modal could not be asked, or refused a launch for now "
                f"({type(error).__name__}); asking again")
            ready = False
        if not ready:
            time.sleep(poll)


def names_of(series: Series) -> tuple[str, str]:
    """The two players a series' difference is between, as its reports
    name them: two challengers, or a challenger and its incumbent."""
    sample = next(iter(series.reports().values()), {})
    arms = sample.get("arms")
    if arms:
        return arms[0].get("challenger", "the first"), arms[1].get("challenger", "the second")
    return sample.get("challenger", "the challenger"), sample.get("incumbent", "the incumbent")


def reading(modal, series: Series, against: Series | None = None) -> list[str]:
    """A series' looks so far, by its registered rule; with `against`, a
    second series of single rulers on the same seeds, read deal by deal
    (the first's edge less the second's). A folder with no registered
    rule is read by the usual one, and says so."""
    path = series.folder / "design.json"
    lines = []
    if path.exists():
        rule = looks.Design.load(path)
    else:
        rule = looks.design()
        lines.append(f"no design at {path}: read by the usual rule, which was not registered for it")
    reports = series.reports()
    if against is not None:
        mirror(modal, against)
        others = against.reports()
        by_seed = {seed: looks.against(report, others[seed]) for seed, report in reports.items() if seed in others}
        names = (names_of(series)[0], names_of(against)[0])
    else:
        by_seed = {seed: looks.differences(report) for seed, report in reports.items()}
        names = names_of(series)
    taken, state = looks.read(rule, by_seed)
    lines.append(looks.describe(rule, taken, state, names))
    return lines


def wait_for(modal, call_ids: dict[str, str], poll: float = POLL) -> dict[str, dict]:
    """The reports of the calls named, once all have finished; a failure
    ends the wait with its reason."""
    done: dict[str, dict] = {}
    while len(done) < len(call_ids):
        for name, call_id in call_ids.items():
            if name in done:
                continue
            state, what = outcome(modal, call_id)
            if state == "done":
                done[name] = what
                say(f"{name} finished")
            elif state == "failed":
                raise SystemExit(f"{name} ({call_id}) failed: {what}")
        if len(done) < len(call_ids):
            time.sleep(poll)
    return done


def compare(reports: dict[str, dict]) -> list[str]:
    """What the timing says: each way's placement, play time and parts, how
    many deals each ruler placed as the old duel did, and how much faster."""
    old = reports["old duel"]
    lines = []
    for name, report in reports.items():
        play = report.get("play", {})
        n = len(report["by_deal"])
        line = (f"{name:<20} placement {report['placement']:.4f} +- {report['standard_error']:.4f} over {n} deals; "
                f"play {play.get('seconds', float('nan')):.0f} s, loading {play.get('loading', 0):.0f} s, "
                f"{play.get('decisions_per_second') or 0:.0f} decisions a second")
        lines.append(line)
        parts = play.get("parts", {})
        if parts:
            lines.append(" " * 21 + ", ".join(f"{part} {seconds:.0f}" for part, seconds in parts.items())
                         + f"; encoded aside {play.get('aside', 0):.0f}")
        if report is not old:
            same = sum(1 for a, b in zip(old["by_deal"], report["by_deal"]) if a == b)
            speed = old["play"]["seconds"] / play["seconds"] if play.get("seconds") else float("nan")
            lines.append(" " * 21 + f"the same placements as the old duel on {same} of {n} deals; "
                         f"{speed:.2f} times as fast")
    seating = reports.get("ruler, seating")
    if seating is not None:
        same = seating["by_deal"] == old["by_deal"]
        speed = old["play"]["seconds"] / seating["play"]["seconds"]
        if not same:
            lines.append("the seating ruler changed games: it must not be used for a series measured the old way")
        elif speed >= 1.5:
            lines.append(f"the seating ruler plays the old duel's games bit for bit, {speed:.2f} times as fast: keep it")
        else:
            lines.append(f"the seating ruler plays the old duel's games bit for bit but only {speed:.2f} times as "
                         "fast, short of the 1.5 the roadmap asks")
    return lines


def timing(modal, folder: Path, challenger: str, incumbent: str, run: str, seed: int, games: int,
           rows: int, poll: float = POLL) -> list[str]:
    """Spawns the three timing calls (or finds them, spawned before, in
    `folder/timing-calls.json`), waits for them, keeps their reports in the
    folder and answers the comparison."""
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / "timing-calls.json"
    call_ids = json.loads(kept.read_text(encoding="utf-8")) if kept.exists() else {}
    if not call_ids:
        duel = modal.Function.from_name(APP, "duel")
        ruler = modal.Function.from_name(APP, "ruler")
        common = dict(challenger=challenger, incumbent=incumbent, games=games, seed=seed, run=run)
        call_ids = {
            "old duel": duel.spawn(timing=True, **common).object_id,
            # Graphs at the sizes asked change no answer, so the seating
            # ruler is timed with every switch that keeps the old games.
            "ruler, seating": ruler.spawn(questions="seating", graphs=True, timing=True, **common).object_id,
            f"ruler, fixed {rows}": ruler.spawn(questions="fixed", rows=rows, graphs=True, skip_forced=True,
                                                preview_reach=True, timing=True, **common).object_id,
        }
        kept.write_text(json.dumps(call_ids, indent=1) + "\n", encoding="utf-8")
        say("spawned " + ", ".join(f"{name} {call_id}" for name, call_id in call_ids.items()))
    reports = wait_for(modal, call_ids, poll)
    for name, report in reports.items():
        (folder / f"timing-{name.replace(',', '').replace(' ', '-')}.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
    return compare(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("series", help="register a series and play it look by look")
    play.add_argument("tag")
    play.add_argument("challenger")
    play.add_argument("incumbent")
    play.add_argument("--also", default="", help="a second challenger on the same deals")
    play.add_argument("--expected", type=float, default=None,
                      help="the effect looked for; 0.006 or less is read over forty seeds")
    play.add_argument("--first-seed", type=int, default=looks.FIRST_SEED)
    play.add_argument("--games", type=int, default=1000)
    play.add_argument("--questions", choices=("seating", "fixed"), default="seating")
    play.add_argument("--rows", type=int, default=256)
    play.add_argument("--graphs", action="store_true")
    play.add_argument("--skip-forced", action="store_true")
    play.add_argument("--preview-reach", action="store_true")
    play.add_argument("--run", default="w1012-run")
    play.add_argument("--local", type=Path, default=Path("rulers"))
    read = commands.add_parser("looks", help="mirror a series' reports and read its looks")
    read.add_argument("tag")
    read.add_argument("--against", default=None, help="a second series of single rulers on the same seeds")
    read.add_argument("--local", type=Path, default=Path("rulers"))
    clock = commands.add_parser("timing", help="the old duel against the ruler, one call each")
    clock.add_argument("challenger")
    clock.add_argument("incumbent")
    clock.add_argument("--run", default="w1012-run")
    clock.add_argument("--seed", type=int, default=looks.FIRST_SEED)
    clock.add_argument("--games", type=int, default=1000)
    clock.add_argument("--rows", type=int, default=256)
    clock.add_argument("--local", type=Path, default=Path("rulers") / "timing")
    args = parser.parse_args()

    import modal

    if args.command == "timing":
        for line in timing(modal, args.local, args.challenger, args.incumbent, args.run, args.seed, args.games,
                           args.rows):
            print(line)
        return
    series = Series(args.local / args.tag)
    if args.command == "looks":
        say(f"{mirror(modal, series)} new reports from the volume")
        for line in reading(modal, series, Series(args.local / args.against) if args.against else None):
            print(line)
        return
    settings = {
        "challenger": args.challenger, "also": args.also, "incumbent": args.incumbent, "games": args.games,
        "questions": args.questions, "rows": args.rows, "graphs": args.graphs, "skip_forced": args.skip_forced,
        "preview_reach": args.preview_reach, "run": args.run,
    }
    what = (f"{args.challenger} against {args.also} at {args.incumbent}'s table" if args.also
            else f"{args.challenger} against {args.incumbent}") + f", questions {args.questions}"
    rule = series.register(settings, looks.design(args.expected, what, args.first_seed))
    upload(modal, series)
    state = run_series(modal, series, settings, rule)
    say(f"{series.tag}: {state}")
    sys.exit(0)


if __name__ == "__main__":
    main()
