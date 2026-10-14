"""Running rulers on Modal from the desktop: a series read look by look, and
the timing of the old duel against the new.

A series is registered under a tag before its first call: its players, how
they are asked and what it compares (`series.json`), and the rule it is read
by (`design.json`, see `neural.looks`), both kept in the local folder and
beside the reports on the volume. `series` then spawns one look's seeds at a
time, waits for them, reads the look, and spawns the next only if the rule
says to go on, so a series that stops at its first look pays for five seeds
rather than twenty. It can be stopped and run again: the calls it spawned are
in `calls.txt`, and the reports are kept on the volume (`rulers/<tag>/`, see
`modal_app.ruler`) and mirrored locally.

The rule stops a series on the comparison it was registered with:

- with `--also`, the first challenger against the second, deal by deal, the
  two played in every call against the same incumbent;
- with `--against <tag>`, the challenger against the challenger of another
  series of single rulers, deal by deal on the same seeds: an old ruler kept
  with `keep`, say, whose games a series asked by seating plays again bit for
  bit, so that only the new challenger is paid for;
- with neither, the challenger against the incumbent.

The generations measured lately are all far ahead of published Mortal (910
crossed the bound at the first look, z +5.95), so a series of one of them
against it stops after five seeds, and two such series read against each
other afterwards (`looks --against`) never get past their first look. Two
generations are compared by a series registered with `--also` or
`--against`.

    python -m neural.ruler_cloud series <tag> <challenger> <incumbent> --local <folder>
        [--also <second> | --against <tag>] [--expected 0.006] [--graphs]
        [--questions fixed --graphs --skip-forced --preview-reach] [--run fusion-long]
    python -m neural.ruler_cloud looks <tag> --local <folder> [--against <tag>]
    python -m neural.ruler_cloud keep <tag> <call file> --local <folder>
    python -m neural.ruler_cloud timing <challenger> <incumbent> --local <folder>
        [--run fusion-long] [--seed 9100000]

`keep` saves the finished calls of a ruler measured the old way
(`modal-duel-many.py`, which writes a seed and a call id a line) as a
series' reports, so a new series can be read against it deal by deal after
Modal has let its calls go.

At the spend limit Modal refuses a launch, which is asked again later, or
takes it and leaves the call in its queue with no container. A call in the
queue is waiting, not playing: the launchers say when one starts waiting,
again every hour it waits, and when it leaves the queue, and never spawn it
again, since it has not failed.

Checkpoints are named as `modal_app` names them: within `--run`, or by their
path from the volume's root (`zoo/mortal_298k`). Run with the Modal client's
Python from the repository root; it needs nothing but the client.

`timing` plays one call of 1,000 deals four ways on the same seed and says
how they compare: the old duel (`neural.duel`, its four tables one after
another); the ruler asking as the old duel asked (`--questions seating`),
once without graphs and once with graphs at the sizes asked, so that a ruler
that changed games is known to have done it by its arena or by its graphs;
and the ruler asking in fixed rows with every switch, none of which then
changes an answer. It prints a block for each, always in that order, then
the verdict; for 910 against published Mortal on seed 9,100,000 it should
read about so:

    old duel                 placement 2.4110 +- 0.0183 over 1000 deals; play about 1250 s, ...
                             engine ..., follow ..., encode ..., challengers ..., incumbent ..., other ...
    ruler, seating           placement 2.4110 +- 0.0183 over 1000 deals; play ...
                             ...
                             the same placements as the old duel on 1000 of 1000 deals; x.xx times as fast
    ruler, seating, graphs   placement 2.4110 +- 0.0183 over 1000 deals; play ...
                             ...
                             ... graphs recorded in ... s, ... questions answered from them
                             the same placements as the old duel on 1000 of 1000 deals; x.xx times as fast
    ruler, fixed 256         placement 2.4... over 1000 deals; play ...
                             ...
                             the same placements as the old duel on many of the 1000 deals, not all; ...
    the seating ruler with graphs plays the old duel's games bit for bit, x.xx times as fast: keep it

The old duel's 2.4110 +- 0.0183 is the published ruler's first seed, read
again if nothing in the engine or the image has changed since 14 October;
its play time is most of the 21 minutes a published call took. A seating
ruler must place exactly as the old duel on every deal, or it has changed
games and must not continue a series measured the old way. One that changed
games without graphs did it by its arena or its weights cast once, and then
no seating ruler may; one that changed them only with graphs did it by its
graphs, and a series continued the old way is then asked without them. The
roadmap keeps the port only if a seating ruler that kept the games is at
least 1.5 times as fast. On the desktop, at 8 deals a seating, the seating
ruler was 1.5 times as fast without graphs and 7.3 with them, and the fixed
one 6.9, but there small questions are nearly all of the work, and at 1,000
deals the encoder, the same in every way, weighs far more: expect less, and
read the parts to see where the rest goes. The fixed ruler plays other
games, so its placement differs within its error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
#: How often a launcher says again that a call still waits in Modal's queue.
REMIND = 60 * 60
#: How often a seed whose call failed is spawned again, with its own seed.
RESPAWNS = 2
#: What `modal_app.ruler` is told for every seed of a series.
CALLED = ("challenger", "incumbent", "games", "questions", "rows", "graphs", "skip_forced", "preview_reach", "run")
#: The players, the asking and the comparison a series keeps from its first
#: call to its last: its calls' arguments, the second challenger played in
#: every call (`also`), and the series of single rulers it is read against
#: (`against`).
SERIES = (*CALLED, "also", "against")
#: A tag names a folder here and on the volume, so it is held to what
#: `modal_app.ruler` takes for one (see `cloud_runs.validate_run`), before
#: any folder is made.
TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
#: Named every time rather than taken for granted: a series' folder holds
#: its registration, and one written into the checkout by accident would be
#: lost with it.
LOCAL = "the folder the series' folders are kept in, outside the checkout"
#: The timing's ways of playing, in the order it prints them; the ruler in
#: fixed rows comes after them, named with its rows.
OLD, SEATING, GRAPHED = "old duel", "ruler, seating", "ruler, seating, graphs"
WAYS = (OLD, SEATING, GRAPHED)


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

    def beside(self, tag: str) -> Series:
        """The series of `tag` kept in the same folder as this one."""
        return Series(self.folder.parent / tag)

    def settings(self) -> dict:
        """What the series was registered to play and compare, or nothing
        for a folder never registered, as an old ruler kept with `keep` is."""
        path = self.folder / "series.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

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
            # Every field but when it was written down: a bound or an error
            # rate changed after the first look falsifies the error rate as
            # surely as more looks or other seeds would.
            mine, theirs = asdict(rule), asdict(registered)
            changed = sorted(name for name in mine if name != "registered" and mine[name] != theirs.get(name))
            if changed:
                raise SystemExit(f"{self.tag} was registered with another rule ({', '.join(changed)} differ): "
                                 f"{theirs}")
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
        self.folder.mkdir(parents=True, exist_ok=True)
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
    """("done", report), ("waiting", None) while the call waits in Modal's
    queue for a container, ("playing", None) once one has taken it up, or
    ("failed", why). A call Modal could not be asked about, or whose state
    its call graph does not say, is taken to be playing, as it was before
    the queue could be told apart."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        answer = call.get(timeout=0)
    except TimeoutError as error:
        # The builtin is get()'s "not yet"; the call's own timeout carries a message.
        if error.args:
            return "failed", str(error)[-200:]
        return ("waiting", None) if queued(call, call_id) else ("playing", None)
    except unasked(modal):
        return "playing", None
    except Exception as error:  # noqa: BLE001 - whatever ended the call
        return "failed", f"{type(error).__name__}: {str(error)[-200:]}"
    report = json.loads(answer) if isinstance(answer, str) and answer.strip() else answer
    return ("done", report) if isinstance(report, dict) and report else ("failed", "an empty answer")


def queued(call, call_id: str) -> bool:
    """Whether Modal holds an unfinished call in its queue still: no input of
    it has been given to a container (a task) in its call graph. False when
    the graph cannot be read or names no input of the call, as it may, for
    Modal keeps it only as best it can."""
    try:
        inputs = call.get_call_graph()
    except Exception:  # noqa: BLE001 - the graph only tells waiting from playing, never the outcome
        return False
    own = [info for info in inputs if getattr(info, "function_call_id", None) == call_id]
    return bool(own) and not any(getattr(info, "task_id", "") for info in own)


class Heard:
    """What a launcher last heard of each of its calls, so it says when one
    starts waiting in Modal's queue, again every `REMIND` seconds it goes on
    waiting, and when it leaves the queue."""

    def __init__(self) -> None:
        # The state of each call, since when, and when it was last said.
        self.last: dict[str, tuple[str, float, float]] = {}

    def __call__(self, label: str, state: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        before, since, said = self.last.get(label, ("", now, now))
        if state != before:
            self.last[label] = (state, now, now)
            if state == "waiting":
                say(f"{label} waits in Modal's queue: no container has taken it up "
                    "(at the spend limit Modal takes launches and runs none)")
            elif before == "waiting":
                say(f"{label} left Modal's queue after {(now - since) / 60:.0f} minutes: {state}")
        elif state == "waiting" and now - said >= REMIND:
            self.last[label] = (state, since, now)
            say(f"{label} has waited {(now - since) / 3600:.1f} hours in Modal's queue")


def spawn_seed(modal, series: Series, settings: dict, seed: int) -> str:
    ruler = modal.Function.from_name(APP, "ruler")
    arguments = {name: settings[name] for name in CALLED}
    call = ruler.spawn(also=settings["also"] or "", seed=seed, tag=series.tag, **arguments)
    series.note(seed, call.object_id)
    say(f"{series.tag}: seed {seed} spawned as {call.object_id}")
    return call.object_id


def settle(modal, series: Series, settings: dict, seeds: list[int], heard: Heard | None = None) -> bool:
    """Spawns whatever of `seeds` has no call yet, keeps the reports of the
    calls that have finished, and spawns a failed seed again, at most
    `RESPAWNS` times; a call that waits or plays is waited for. True once
    every one of `seeds` has a report."""
    heard = heard or Heard()
    # First what the volume kept: a launcher started again after Modal let a
    # finished call's answer go would otherwise take it to have failed, and
    # pay for its seed twice.
    mirror(modal, series)
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
        heard(f"{series.tag}: seed {seed}", state)
        if state == "done":
            series.keep(what)
        elif state == "failed":
            if len(spawned) > RESPAWNS:
                raise SystemExit(f"{series.tag}: seed {seed} failed {len(spawned)} times, last: {what}")
            say(f"{series.tag}: seed {seed} failed ({what}); spawning it again")
            spawn_seed(modal, series, settings, seed)
    mirror(modal, series)
    return all(seed in series.reports() for seed in seeds)


def differences_of(series: Series, settings: dict) -> dict[int, list[float]]:
    """Each seed's per-deal differences, by the comparison the series was
    registered with (see the module's notes), positive when its first
    player is the stronger."""
    reports = series.reports()
    if settings.get("against"):
        return looks.against_each(reports, series.beside(settings["against"]).reports())
    return {seed: looks.differences(report) for seed, report in reports.items()}


def compared(series: Series, settings: dict) -> tuple[str, str]:
    """The two players a series' rule reads against each other."""
    if settings.get("also"):
        return settings["challenger"], settings["also"]
    if settings.get("against"):
        return settings["challenger"], names_of(series.beside(settings["against"]))[0]
    return settings["challenger"], settings["incumbent"]


def run_series(modal, series: Series, settings: dict, rule: looks.Design, poll: float = POLL) -> str:
    """Plays a registered series look by look until its rule stops it, and
    answers how it ended: "first", "second" or "level"."""
    order = rule.seeds
    names = compared(series, settings)
    heard = Heard()
    shown = 0
    while True:
        taken, state = looks.read(rule, differences_of(series, settings))
        if len(taken) > shown:
            print(looks.describe(rule, taken, state, names), flush=True)
            shown = len(taken)
        if state != "waiting":
            return state
        wanted = order[: (len(taken) + 1) * rule.per_look]
        if settings.get("against") and set(wanted) <= set(series.reports()):
            # The look's own seeds are in and it waits all the same: the
            # series it is read against lacks some, which no call of this
            # one can bring. Registration refuses such a series; this is
            # one whose reports went missing since.
            others = series.beside(settings["against"]).reports()
            missing = ", ".join(str(seed) for seed in wanted if seed not in others)
            raise SystemExit(f"{series.tag}: the seeds of look {len(taken) + 1} are in, "
                             f"but {settings['against']} holds no report for {missing}")
        try:
            ready = settle(modal, series, settings, wanted, heard)
        except unasked(modal) as error:
            say(f"{series.tag}: Modal could not be asked, or refused a launch for now "
                f"({type(error).__name__}); asking again")
            ready = False
        if not ready:
            time.sleep(poll)


def keep_calls(modal, series: Series, lines: list[str]) -> tuple[int, list[str]]:
    """Keeps the reports of finished calls, a `seed<TAB>call id` line each,
    as `modal-duel-many.py` writes them, as the series' own: the old
    rulers' results otherwise last only as long as Modal keeps their calls,
    about a fortnight. Answers how many it kept, and the seeds it could
    not, with why."""
    series.folder.mkdir(parents=True, exist_ok=True)
    have = series.reports()
    kept, missing = 0, []
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or int(parts[0]) in have:
            continue
        state, what = outcome(modal, parts[1])
        if state == "done":
            series.keep(what)
            kept += 1
        else:
            missing.append(f"seed {parts[0]}: {state}{f' ({what})' if what else ''}")
    return kept, missing


def checkpoint_name(named: str) -> str:
    """A checkpoint as a report names it, as the volume names it. A ruler's
    report names it so (see `modal_app.ruler`); the old duel's names the
    container's copy, the volume's name with each "/" made "--" and ".pt"
    after it (see `modal_app.duel`)."""
    if named.endswith(".pt"):
        return named.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".pt").replace("--", "/")
    return named


def names_of(series: Series) -> tuple[str, str]:
    """The two players a series' reports set against each other: two
    challengers, or a challenger and its incumbent."""
    sample = next(iter(series.reports().values()), {})
    arms = sample.get("arms")
    if arms:
        return (checkpoint_name(arms[0].get("challenger", "the first")),
                checkpoint_name(arms[1].get("challenger", "the second")))
    return (checkpoint_name(sample.get("challenger", "the challenger")),
            checkpoint_name(sample.get("incumbent", "the incumbent")))


def asking_problem(settings: dict) -> str | None:
    """Why a series cannot ask its players as `settings` say, if it cannot:
    found before anything is registered, where `modal_app.ruler` and
    `neural.ruler` would find it only in each call, failing every one."""
    if settings["questions"] != "fixed" and (settings["skip_forced"] or settings["preview_reach"]):
        # Each changes the batches a network is asked in, and with them the
        # last bits of answers, unless every batch has one size.
        return "--skip-forced and --preview-reach change answers unless the questions are fixed (--questions fixed)"
    if settings["questions"] == "fixed" and settings["rows"] <= 0:
        return "--rows must be a positive whole number"
    if settings["games"] <= 0:
        return "--games must be a positive whole number"
    return None


def against_problem(settings: dict, rule: looks.Design, other: Series) -> str | None:
    """Why the series of `settings` cannot be read against `other` by `rule`,
    if it cannot: `other` must be a series of single rulers holding a report
    of as many deals for every seed the rule reads, played against the same
    incumbent and asked the same way. A ruler asked in fixed rows plays other
    games than one asked by seating, whose games are the old duel's, so a
    folder never registered, an old ruler kept with `keep`, counts as asked
    by seating."""
    reports = other.reports()
    if not reports:
        return f"{other.folder} holds no reports"
    missing = [seed for seed in rule.seeds if seed not in reports]
    if missing:
        shown = ", ".join(str(seed) for seed in missing[:3]) + (", ..." if len(missing) > 3 else "")
        return (f"it holds {len(rule.seeds) - len(missing)} of the {len(rule.seeds)} seeds the rule reads "
                f"(not {shown}), and a look waits for every one of them")
    for seed in rule.seeds:
        if "arms" in reports[seed]:
            return "it is a series of paired rulers; a series is read against one of single rulers"
        if reports[seed].get("games_per_seat") != settings["games"]:
            return f"it played {reports[seed].get('games_per_seat')} deals at seed {seed}, not {settings['games']}"
    theirs = other.settings()
    asked = theirs.get("questions", "seating")
    if asked != settings["questions"]:
        return f"it was asked {asked} and this series {settings['questions']}, which play other games"
    incumbent = theirs.get("incumbent") or checkpoint_name(reports[rule.seeds[0]].get("incumbent", ""))
    if incumbent != settings["incumbent"]:
        return f"its incumbent is {incumbent}, not {settings['incumbent']}: name the incumbent as it does"
    return None


def what_of(settings: dict) -> str:
    """What a series compares, in words, as its rule records it."""
    if settings["also"]:
        pair = f"{settings['challenger']} against {settings['also']} at {settings['incumbent']}'s table"
    elif settings["against"]:
        pair = (f"{settings['challenger']} against the challenger of {settings['against']}, deal by deal, "
                f"both at {settings['incumbent']}'s table")
    else:
        pair = f"{settings['challenger']} against {settings['incumbent']}"
    return f"{pair}, questions {settings['questions']}"


def reading(modal, series: Series, against: Series | None = None) -> list[str]:
    """A series' looks so far, by its registered rule and the comparison it
    was registered with; with `against`, read instead against that series
    of single rulers on the same seeds, deal by deal (the first's edge less
    the second's), which is said when the rule was registered for another.
    A folder with no registered rule is read by the usual one, and says so."""
    path = series.folder / "design.json"
    lines = []
    if path.exists():
        rule = looks.Design.load(path)
    else:
        rule = looks.design()
        lines.append(f"no design at {path}: read by the usual rule, which was not registered for it")
    registered = series.settings().get("against", "")
    if against is None and registered:
        against = series.beside(registered)
    elif against is not None and path.exists() and against.tag != registered:
        lines.append(f"read against {against.tag}, a comparison {series.tag}'s rule was not registered for: "
                     "its looks did not decide when the series stopped")
    reports = series.reports()
    if against is not None:
        mirror(modal, against)
        by_seed = looks.against_each(reports, against.reports())
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
    heard = Heard()
    while len(done) < len(call_ids):
        for name, call_id in call_ids.items():
            if name in done:
                continue
            state, what = outcome(modal, call_id)
            heard(name, state)
            if state == "done":
                done[name] = what
                say(f"{name} finished")
            elif state == "failed":
                raise SystemExit(f"{name} ({call_id}) failed: {what}")
        if len(done) < len(call_ids):
            time.sleep(poll)
    return done


def order(name: str) -> tuple[int, str]:
    """Where the timing prints a way of playing: the old duel, the seating
    ruler without graphs and with them, then the rest by name."""
    return (WAYS.index(name), "") if name in WAYS else (len(WAYS), name)


def speed_of(old: dict, report: dict) -> float:
    """How many times as fast as the old duel a way played; NaN if either
    was not timed."""
    seconds = report.get("play", {}).get("seconds")
    return old.get("play", {}).get("seconds", math.nan) / seconds if seconds else math.nan


def compare(reports: dict[str, dict]) -> list[str]:
    """What the timing says: each way's placement, play time and parts, how
    many deals each ruler placed as the old duel did, and how much faster,
    in the same order whichever call finished first; then the verdict."""
    old = reports[OLD]
    lines = []
    for name in sorted(reports, key=order):
        report = reports[name]
        play = report.get("play", {})
        n = len(report["by_deal"])
        lines.append(f"{name:<24} placement {report['placement']:.4f} +- {report['standard_error']:.4f} over {n} "
                     f"deals; play {play.get('seconds', math.nan):.0f} s, loading {play.get('loading', 0):.0f} s, "
                     f"{play.get('decisions_per_second') or 0:.0f} decisions a second")
        parts = play.get("parts", {})
        if parts:
            lines.append(" " * 25 + ", ".join(f"{part} {seconds:.0f}" for part, seconds in parts.items())
                         + f"; encoded aside {play.get('aside', 0):.0f}")
        if "graphs" in play:
            # Recorded while playing, at each size asked often, so inside the
            # play's own time: at 1,000 deals perhaps hundreds a network.
            lines.append(" " * 25 + f"{play['graphs']} graphs recorded in {play.get('graph_seconds', 0):.0f} s, "
                         f"{play.get('from_graphs', 0)} questions answered from them")
        if name != OLD:
            same = sum(1 for a, b in zip(old["by_deal"], report["by_deal"]) if a == b)
            lines.append(" " * 25 + f"the same placements as the old duel on {same} of {n} deals; "
                         f"{speed_of(old, report):.2f} times as fast")
    return lines + verdict(reports)


def verdict(reports: dict[str, dict]) -> list[str]:
    """Whether a seating ruler may continue a series measured the old way,
    and which: one that placed exactly as the old duel on every deal, the
    faster if both did. Without graphs the seating ruler asks the old
    duel's very batches, so one that changed games there did it by its
    arena or its weights cast once, which no graphs can undo; one that
    changed games only with graphs did it by its graphs."""
    old = reports[OLD]
    plain, graphed = reports.get(SEATING), reports.get(GRAPHED)
    if plain is None and graphed is None:
        return []
    if plain is not None and plain["by_deal"] != old["by_deal"]:
        return ["the seating ruler changed games without graphs: its arena or its weights cast once play otherwise "
                "than the old duel here, so no seating ruler may continue a series measured the old way"]
    lines = []
    if graphed is not None and graphed["by_deal"] != old["by_deal"]:
        if plain is None:
            lines.append("the seating ruler changed games with graphs, and with no call without them the arena "
                         "cannot be told from the graphs: it must not continue a series measured the old way")
        else:
            lines.append("the seating ruler changed games with graphs and not without: graphs at the sizes asked "
                         "turned answers here, so a series measured the old way is continued without --graphs")
    kept = [(speed_of(old, report), name) for name, report in ((SEATING, plain), (GRAPHED, graphed))
            if report is not None and report["by_deal"] == old["by_deal"]]
    timed = [(speed, name) for speed, name in kept if not math.isnan(speed)]
    if timed:
        # As fast either way, the ruler without graphs, which has less to go wrong.
        speed, name = max(timed, key=lambda item: (item[0], item[1] == SEATING))
        which = "the seating ruler " + ("with graphs" if name == GRAPHED else "without graphs")
        if speed >= 1.5:
            lines.append(f"{which} plays the old duel's games bit for bit, {speed:.2f} times as fast: keep it")
        else:
            lines.append(f"{which} plays the old duel's games bit for bit but only {speed:.2f} times as fast, "
                         "short of the 1.5 the roadmap asks")
    return lines


def timing(modal, folder: Path, challenger: str, incumbent: str, run: str, seed: int, games: int,
           rows: int, poll: float = POLL) -> list[str]:
    """Spawns the four timing calls, or those not spawned yet for these
    arguments (`folder/timing-calls.json`, written as each is spawned, so a
    launch refused partway loses none of the calls before it), waits for
    them, keeps their reports in the folder and answers the comparison."""
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / "timing-calls.json"
    arguments = {"challenger": challenger, "incumbent": incumbent, "run": run, "seed": seed, "games": games,
                 "rows": rows}
    saved = json.loads(kept.read_text(encoding="utf-8")) if kept.exists() else {"arguments": arguments, "calls": {}}
    if saved["arguments"] != arguments:
        raise SystemExit(f"{kept} holds the calls spawned for {saved['arguments']}: name another folder")
    common = dict(challenger=challenger, incumbent=incumbent, games=games, seed=seed, run=run)
    ways = {
        # 48 GiB, as the published rulers' launcher asks: a fusion's duel died
        # of signal 9 at the function's own 32 once in twenty calls.
        OLD: ("duel", {"memory": 49152}, {}),
        # The old duel's very batches, without graphs and with graphs at the
        # sizes asked: either should keep the old duel's games, and the two
        # tell which of the arena and the graphs changed them if one did not.
        SEATING: ("ruler", {}, {"questions": "seating"}),
        GRAPHED: ("ruler", {}, {"questions": "seating", "graphs": True}),
        f"ruler, fixed {rows}": ("ruler", {}, {"questions": "fixed", "rows": rows, "graphs": True,
                                               "skip_forced": True, "preview_reach": True}),
    }
    for name, (function, options, asked) in ways.items():
        if name in saved["calls"]:
            continue
        spawner = modal.Function.from_name(APP, function)
        if options:
            spawner = spawner.with_options(**options)
        try:
            call = spawner.spawn(timing=True, **common, **asked)
        except unasked(modal) as error:
            raise SystemExit(f"Modal could not be asked, or refused a launch for now ({type(error).__name__}): "
                             f"the {len(saved['calls'])} calls spawned before it are kept in {kept}; "
                             "run the same command again later to spawn the rest") from None
        saved["calls"][name] = call.object_id
        kept.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
        say(f"spawned {name} as {saved['calls'][name]}")
    reports = wait_for(modal, saved["calls"], poll)
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
    second = play.add_mutually_exclusive_group()
    second.add_argument("--also", default="",
                        help="a second challenger on the same deals in every call; the rule reads the two "
                             "challengers against each other")
    second.add_argument("--against", default="",
                        help="the tag of a series of single rulers on the same seeds, kept beside this one "
                             "(an old ruler kept with keep, say); the rule reads the challenger against its "
                             "challenger, deal by deal")
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
    play.add_argument("--local", type=Path, required=True, help=LOCAL)
    read = commands.add_parser("looks", help="mirror a series' reports and read its looks")
    read.add_argument("tag")
    read.add_argument("--against", default=None,
                      help="a second series of single rulers on the same seeds, in place of the comparison "
                           "the series was registered with")
    read.add_argument("--local", type=Path, required=True, help=LOCAL)
    clock = commands.add_parser("timing", help="the old duel against the ruler, one call each way")
    clock.add_argument("challenger")
    clock.add_argument("incumbent")
    clock.add_argument("--run", default="w1012-run")
    clock.add_argument("--seed", type=int, default=looks.FIRST_SEED)
    clock.add_argument("--games", type=int, default=1000)
    clock.add_argument("--rows", type=int, default=256)
    clock.add_argument("--local", type=Path, required=True, help="where the four reports are kept")
    hold = commands.add_parser("keep", help="keep an old ruler's finished calls as a series' reports")
    hold.add_argument("tag")
    hold.add_argument("calls", type=Path, help="its call file, a seed and a call id a line")
    hold.add_argument("--local", type=Path, required=True, help=LOCAL)
    args = parser.parse_args()

    # Refused before the client is asked or any folder is made.
    for tag in (getattr(args, "tag", None), getattr(args, "against", None)):
        if tag and not TAG.fullmatch(tag):
            parser.error(f"{tag!r} is not a tag: one folder name of letters, digits, '_', '.' and '-', "
                         "beginning with a letter or digit, as the volume takes")
    if args.command == "series":
        settings = {name: getattr(args, name) for name in SERIES}
        problem = asking_problem(settings)
        if problem:
            parser.error(problem)

    import modal

    if args.command == "timing":
        for line in timing(modal, args.local, args.challenger, args.incumbent, args.run, args.seed, args.games,
                           args.rows):
            print(line)
        return
    series = Series(args.local / args.tag)
    if args.command == "keep":
        kept, missing = keep_calls(modal, series, args.calls.read_text(encoding="utf-8").splitlines())
        say(f"{series.tag}: kept {kept} reports, {len(series.reports())} in all")
        for line in missing:
            print(f"  not kept: {line}")
        sys.exit(1 if missing else 0)
    if args.command == "looks":
        say(f"{mirror(modal, series)} new reports from the volume")
        for line in reading(modal, series, series.beside(args.against) if args.against else None):
            print(line)
        return
    rule = looks.design(args.expected, what_of(settings), args.first_seed)
    if args.against:
        other = series.beside(args.against)
        mirror(modal, other)
        problem = against_problem(settings, rule, other)
        if problem:
            raise SystemExit(f"{series.tag} cannot be read against {other.tag}: {problem}")
    rule = series.register(settings, rule)
    upload(modal, series)
    state = run_series(modal, series, settings, rule)
    say(f"{series.tag}: {state}")
    sys.exit(0)


if __name__ == "__main__":
    main()
