"""Running rulers on Modal from the desktop (`neural.ruler_cloud`), with the
Modal client stood in for: a series is registered before its first call and
keeps what it plays and every field of its rule; it spawns one look's seeds
at a time and stops where its rule says, on the comparison it was registered
with (two challengers played in each call, or one read against another
series deal by deal), never on the challenger against the incumbent once a
second player is named; it spawns a failed seed again with its own seed,
waits for a call in Modal's queue without taking it to be playing, keeps
every report locally and reads what the volume kept; what it could not play
or read is refused before anything is registered; and the timing says, in a
fixed order, whether the ruler played the old duel's games, by its arena or
by its graphs if not, and how much faster."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from neural import looks, ruler_cloud


class NotFound(Exception):
    pass


class Refused(Exception):
    """As Modal refuses a launch at the spend limit."""


class Modal:
    """The parts of the Modal client the launcher uses. A spawned call first
    passes through `queue(function, arguments)` (or `queue` itself), a
    state a poll: "queued", given no container yet, or "running". Then it
    finishes with what `answer(function, arguments)` gives, a report or an
    exception it raises, and a ruler's report lands on the volume as
    `modal_app.ruler` keeps it. The first `refusals` launches are refused."""

    def __init__(self, answer, refusals: int = 0, queue=()):
        self.answer = answer
        self.refusals = refusals
        self.queue = queue if callable(queue) else (lambda name, arguments: queue)
        self.spawned: list[tuple[str, dict]] = []
        self.results: dict[str, object] = {}
        # The states each call has still to pass through, its state at the
        # last poll, and the files it leaves on the volume as it finishes.
        self.ahead: dict[str, list[str]] = {}
        self.state: dict[str, str] = {}
        self.landing: dict[str, dict[str, bytes]] = {}
        self.files: dict[str, bytes] = {}
        self.uploaded: dict[str, str] = {}
        self.options: dict[str, dict] = {}
        self.exception = SimpleNamespace(NotFoundError=NotFound, ResourceExhaustedError=Refused)
        self.Function = SimpleNamespace(from_name=self.function)
        self.FunctionCall = SimpleNamespace(from_id=self.call)
        self.Volume = SimpleNamespace(from_name=lambda name: self)

    def function(self, app, name):
        options = {}

        def with_options(**chosen):
            options.update(chosen)
            self.options[name] = dict(options)
            return function

        def spawn(**arguments):
            if self.refusals:
                self.refusals -= 1
                raise Refused("billing cycle spend limit")
            call_id = f"fc-{len(self.spawned)}"
            self.spawned.append((name, arguments))
            self.ahead[call_id] = list(self.queue(name, arguments))
            try:
                report = self.answer(name, arguments)
            except Exception as error:  # noqa: BLE001 - the call's own failure
                self.results[call_id] = error
            else:
                text = json.dumps(report)
                self.results[call_id] = text
                if name == "ruler" and arguments.get("tag"):
                    self.landing[call_id] = {f"rulers/{arguments['tag']}/seed-{arguments['seed']}.json": text.encode()}
            return SimpleNamespace(object_id=call_id)

        function = SimpleNamespace(spawn=spawn, with_options=with_options)
        return function

    def call(self, call_id):
        def get(timeout=None):
            ahead = self.ahead.get(call_id)
            if ahead:
                self.state[call_id] = ahead.pop(0)
                raise TimeoutError()
            self.state[call_id] = "finished"
            self.files.update(self.landing.pop(call_id, {}))
            result = self.results[call_id]
            if isinstance(result, Exception):
                raise result
            return result

        def get_call_graph():
            task = "" if self.state.get(call_id) == "queued" else "ta-1"
            return [SimpleNamespace(function_call_id=call_id, task_id=task, status=0)]

        return SimpleNamespace(get=get, get_call_graph=get_call_graph)

    def listdir(self, folder):
        found = [SimpleNamespace(path=path) for path in self.files if path.startswith(folder + "/")]
        if not found:
            raise NotFound(folder)
        return found

    def read_file(self, path):
        yield self.files[path]

    @contextlib.contextmanager
    def batch_upload(self, force=False):
        yield SimpleNamespace(put_file=lambda local, remote: self.uploaded.update(
            {remote: Path(local).read_text(encoding="utf-8")}))


def placements(seed: int, spread: int, base: float, deals: int = 50) -> list[float]:
    """A challenger's placement on each of a seed's deals: `base` on
    average, exactly, with a little spread that `spread` sets apart from
    another challenger's."""
    return [base + 0.1 * ((deal * spread + seed) % 5 - 2) for deal in range(deals)]


def paired_report(seed: int, gap: float, deals: int = 50, base: float = 2.5) -> dict:
    """A paired call's report: both challengers placing `base` on average
    against the incumbent, the first `gap` ahead of the second on every
    deal."""
    first = placements(seed, 7, base - gap, deals)
    second = placements(seed, 3, base, deals)
    return {"seed": seed, "games_per_seat": deals,
            "arms": [{"challenger": "a", "by_deal": first}, {"challenger": "b", "by_deal": second}]}


def single_report(seed: int, by_deal: list[float], challenger: str, incumbent: str = "zoo/mortal_298k") -> dict:
    """One challenger's report, as the old duel and the ruler write one."""
    return {"seed": seed, "games_per_seat": len(by_deal), "placement": sum(by_deal) / len(by_deal),
            "standard_error": 0.02, "by_deal": by_deal, "challenger": challenger, "incumbent": incumbent}


def old_ruler(seed: int) -> dict:
    """The published ruler of 910 at `seed`, as the old duel wrote it: the
    checkpoints named as the container's copies, and half a place ahead of
    the incumbent."""
    return single_report(seed, placements(seed, 7, 2.0), "/scratch/duel-x/fusion-long--history--gen-00910.pt",
                         "/scratch/duel-x/zoo--mortal_298k.pt")


SETTINGS = {"challenger": "history/gen-00910", "also": "history/gen-00810", "against": "",
            "incumbent": "zoo/mortal_298k", "games": 1000, "questions": "fixed", "rows": 256, "graphs": True,
            "skip_forced": True, "preview_reach": True, "run": "fusion-long"}
#: One challenger read against the old ruler of 910 kept as `g910`, asked as
#: the old duel asked, on rulers of fifty deals.
AGAINST = {**SETTINGS, "challenger": "history/gen-01010", "also": "", "against": "g910", "games": 50,
           "questions": "seating", "skip_forced": False, "preview_reach": False}


class SeriesTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def play(self, modal, tag="g910-g810", expected=None):
        series = ruler_cloud.Series(self.root / tag)
        rule = series.register(dict(SETTINGS), looks.design(expected, ruler_cloud.what_of(SETTINGS)))
        ruler_cloud.upload(modal, series)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            state = ruler_cloud.run_series(modal, series, dict(SETTINGS), rule, poll=0)
        return series, state, out.getvalue()

    def test_a_clear_difference_stops_after_one_look_of_seeds(self):
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.3))
        series, state, out = self.play(modal)
        self.assertEqual(state, "first")
        seeds = [arguments["seed"] for _name, arguments in modal.spawned]
        self.assertEqual(seeds, looks.design().seeds[:5])
        for name, arguments in modal.spawned:
            self.assertEqual(name, "ruler")
            self.assertEqual({key: arguments[key] for key in ruler_cloud.CALLED},
                             {key: SETTINGS[key] for key in ruler_cloud.CALLED})
            self.assertEqual((arguments["also"], arguments["tag"]), ("history/gen-00810", "g910-g810"))
            self.assertNotIn("against", arguments)
        self.assertEqual(sorted(series.reports()), seeds)
        self.assertIn("stopped at look 1: history/gen-00910 is the stronger", out)
        self.assertEqual(json.loads(modal.uploaded["rulers/g910-g810/series.json"]), SETTINGS)
        self.assertEqual(json.loads(modal.uploaded["rulers/g910-g810/design.json"])["per_look"], 5)

    def test_two_challengers_are_read_against_each_other_not_against_the_incumbent(self):
        """Both arms half a place ahead of the incumbent, as every generation
        is of published Mortal, and level with each other: the series plays
        every look and reads level, where a rule reading the first challenger
        against the incumbent stops at the first look."""
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.0, base=2.0))
        _series, state, out = self.play(modal)
        self.assertEqual(state, "level")
        self.assertEqual(len(modal.spawned), 20)
        self.assertIn("comparing: history/gen-00910 against history/gen-00810 at zoo/mortal_298k's table", out)
        rule = looks.design()
        alone = {seed: [2.5 - placement for placement in paired_report(seed, 0.0, base=2.0)["arms"][0]["by_deal"]]
                 for seed in rule.seeds}
        taken, against_incumbent = looks.read(rule, alone)
        self.assertEqual((len(taken), against_incumbent), (1, "first"))

    def test_a_level_series_plays_every_look_and_a_small_effect_forty_seeds(self):
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.0))
        _series, state, out = self.play(modal, expected=0.005)
        self.assertEqual(state, "level")
        self.assertEqual(len(modal.spawned), 40)
        self.assertEqual(out.count("look  seeds"), 4)

    def test_a_failed_seed_is_spawned_again_with_its_own_seed(self):
        failures = {9_300_006: 1}

        def answer(name, arguments):
            if failures.get(arguments["seed"]):
                failures[arguments["seed"]] -= 1
                raise RuntimeError("container lost")
            return paired_report(arguments["seed"], 0.3)

        modal = Modal(answer)
        with contextlib.redirect_stdout(io.StringIO()):
            series, state, _out = self.play(modal)
        self.assertEqual(state, "first")
        seeds = [arguments["seed"] for _name, arguments in modal.spawned]
        self.assertEqual(seeds.count(9_300_006), 2)
        self.assertEqual(len(seeds), 6)
        self.assertEqual(series.calls()[9_300_006], ["fc-2", "fc-5"])

    def test_a_refused_launch_is_asked_again_and_never_counted_as_a_failure(self):
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.3), refusals=3)
        series, state, out = self.play(modal)
        self.assertEqual(state, "first")
        self.assertEqual(len(modal.spawned), 5)
        self.assertEqual(sorted(series.calls()), looks.design().seeds[:5])
        self.assertTrue(all(len(calls) == 1 for calls in series.calls().values()))
        self.assertIn("refused a launch for now", out)

    def test_a_call_in_the_queue_is_waited_for_and_never_spawned_again(self):
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.3),
                      queue=("queued", "queued", "running"))
        series, state, out = self.play(modal)
        self.assertEqual(state, "first")
        self.assertEqual(len(modal.spawned), 5)
        self.assertTrue(all(len(calls) == 1 for calls in series.calls().values()))
        self.assertEqual(out.count("waits in Modal's queue"), 5)
        self.assertEqual(out.count("left Modal's queue after 0 minutes: playing"), 5)

    def test_a_report_the_volume_kept_is_never_paid_for_again(self):
        """A launcher started again after Modal let a finished call's answer
        go: the report is on the volume, and the seed is not spawned again."""
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.3))
        series = ruler_cloud.Series(self.root / "again")
        series.register(dict(SETTINGS), looks.design())
        seed = looks.design().seeds[0]
        series.note(seed, "fc-gone")
        modal.results["fc-gone"] = RuntimeError("OutputExpiredError")
        modal.files[f"rulers/again/seed-{seed}.json"] = json.dumps(paired_report(seed, 0.3)).encode()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(ruler_cloud.settle(modal, series, dict(SETTINGS), [seed]))
        self.assertEqual(modal.spawned, [])
        self.assertEqual(sorted(series.reports()), [seed])

    def test_a_seed_that_keeps_failing_ends_the_series(self):
        modal = Modal(lambda name, arguments: (_ for _ in ()).throw(RuntimeError("no card")))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.play(modal)

    def test_a_registered_series_keeps_what_it_plays_and_its_rule(self):
        series = ruler_cloud.Series(self.root / "kept")
        first = series.register(dict(SETTINGS), looks.design())
        again = series.register(dict(SETTINGS), looks.design())
        self.assertEqual(again, first)
        with self.assertRaisesRegex(SystemExit, "also"):
            series.register({**SETTINGS, "also": "history/gen-00710"}, looks.design())
        with self.assertRaisesRegex(SystemExit, "another rule \\(expected, per_look differ"):
            series.register(dict(SETTINGS), looks.design(0.004))
        # Every field of the rule but when it was written: the bound, and the
        # error rate it was computed for, as much as the looks.
        with self.assertRaisesRegex(SystemExit, "another rule \\(bound differ"):
            series.register(dict(SETTINGS), looks.Design(bound=2.5))
        with self.assertRaisesRegex(SystemExit, "another rule \\(alpha, bound differ"):
            series.register(dict(SETTINGS), looks.Design(alpha=0.01, bound=2.9))
        with self.assertRaisesRegex(SystemExit, "another rule \\(what differ"):
            series.register(dict(SETTINGS), looks.design(None, "910 against 610"))

    def test_reading_mirrors_what_the_volume_kept(self):
        modal = Modal(lambda name, arguments: paired_report(arguments["seed"], 0.3))
        for seed in looks.design().seeds[:5]:
            modal.files[f"rulers/kept/seed-{seed}.json"] = json.dumps(paired_report(seed, 0.3)).encode()
        series = ruler_cloud.Series(self.root / "kept")
        series.folder.mkdir()
        self.assertEqual(ruler_cloud.mirror(modal, series), 5)
        self.assertEqual(ruler_cloud.mirror(modal, series), 0)
        lines = ruler_cloud.reading(modal, series)
        self.assertIn("not registered", lines[0])
        self.assertIn("stopped at look 1: a is the stronger", lines[1])
        nothing = ruler_cloud.Series(self.root / "nothing")
        nothing.folder.mkdir()
        self.assertEqual(ruler_cloud.mirror(modal, nothing), 0)


class AgainstTests(unittest.TestCase):
    """One challenger read against an old ruler of 910 kept with `keep`,
    whose games a series asked by seating plays again."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        # Kept as the old duel wrote it, and never registered.
        self.kept = ruler_cloud.Series(self.root / "g910")
        for seed in looks.design().seeds:
            self.kept.keep(old_ruler(seed))

    def play(self, gap: float):
        """A series of 1010, `gap` ahead of 910 on every deal and, like 910,
        half a place ahead of the incumbent."""
        modal = Modal(lambda name, arguments: single_report(
            arguments["seed"], placements(arguments["seed"], 3, 2.0 - gap), "history/gen-01010"))
        series = ruler_cloud.Series(self.root / "g1010")
        rule = looks.design(None, ruler_cloud.what_of(AGAINST))
        self.assertIsNone(ruler_cloud.against_problem(AGAINST, rule, self.kept))
        rule = series.register(dict(AGAINST), rule)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            state = ruler_cloud.run_series(modal, series, dict(AGAINST), rule, poll=0)
        return modal, series, state, out.getvalue()

    def test_level_with_the_kept_ruler_it_plays_every_look(self):
        """Read against the incumbent, it would stop at the first look."""
        modal, _series, state, out = self.play(0.0)
        self.assertEqual(state, "level")
        self.assertEqual(len(modal.spawned), 20)
        for _name, arguments in modal.spawned:
            self.assertEqual((arguments["challenger"], arguments["also"], arguments["questions"]),
                             ("history/gen-01010", "", "seating"))
            self.assertNotIn("against", arguments)
        self.assertIn("comparing: history/gen-01010 against the challenger of g910, deal by deal", out)
        self.assertIn("level: no look crossed", out)

    def test_a_clear_gain_over_the_kept_ruler_stops_at_the_first_look(self):
        modal, series, state, out = self.play(0.3)
        self.assertEqual(state, "first")
        self.assertEqual(len(modal.spawned), 5)
        self.assertIn("stopped at look 1: history/gen-01010 is the stronger", out)
        # Read again later, by the comparison it was registered with; read
        # against another series, it says its rule was not registered so.
        lines = ruler_cloud.reading(modal, series)
        self.assertEqual(len(lines), 1)
        self.assertIn("stopped at look 1: history/gen-01010 is the stronger", lines[0])
        other = ruler_cloud.Series(self.root / "g810")
        for seed in looks.design().seeds[:5]:
            other.keep(single_report(seed, placements(seed, 7, 1.5), "history/gen-00810"))
        lines = ruler_cloud.reading(modal, series, other)
        self.assertIn("not registered for", lines[0])
        self.assertIn("stopped at look 1: history/gen-00810 is the stronger", lines[-1])

    def test_a_series_whose_kept_reports_went_missing_stops_rather_than_wait(self):
        modal = Modal(lambda name, arguments: single_report(
            arguments["seed"], placements(arguments["seed"], 3, 2.0), "history/gen-01010"))
        series = ruler_cloud.Series(self.root / "g1010")
        rule = series.register(dict(AGAINST), looks.design(None, ruler_cloud.what_of(AGAINST)))
        (self.kept.folder / "seed-9200003.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "no report for 9200003"):
            ruler_cloud.run_series(modal, series, dict(AGAINST), rule, poll=0)
        self.assertEqual(len(modal.spawned), 5)

    def test_what_could_not_be_read_against_it_is_said(self):
        rule = looks.design()

        def problem(settings=AGAINST, rule=rule, other=self.kept):
            return ruler_cloud.against_problem(settings, rule, other)

        self.assertIsNone(problem())
        # An effect small enough for forty seeds, where the old ruler has twenty.
        self.assertIn("20 of the 40 seeds", problem(rule=looks.design(0.004)))
        self.assertIn("asked seating", problem({**AGAINST, "questions": "fixed"}))
        self.assertIn("incumbent is zoo/mortal_298k", problem({**AGAINST, "incumbent": "history/gen-00466"}))
        self.assertIn("50 deals", problem({**AGAINST, "games": 1000}))
        self.assertIn("holds no reports", problem(other=ruler_cloud.Series(self.root / "nothing")))
        paired = ruler_cloud.Series(self.root / "paired")
        for seed in rule.seeds:
            paired.keep(paired_report(seed, 0.0))
        self.assertIn("paired rulers", problem(other=paired))
        # A registered series is taken at its word on how it was asked.
        fixed = ruler_cloud.Series(self.root / "fixed")
        fixed.register({**AGAINST, "against": "", "questions": "fixed"}, rule)
        for seed in rule.seeds:
            fixed.keep(single_report(seed, placements(seed, 7, 2.0), "history/gen-00910"))
        self.assertIn("asked fixed", problem(other=fixed))
        self.assertIsNone(problem({**AGAINST, "questions": "fixed"}, other=fixed))


class QueueTests(unittest.TestCase):
    def test_a_call_in_the_queue_waits_and_one_with_a_container_plays(self):
        def modal_with(graph):
            def get(timeout=None):
                raise TimeoutError()

            call = SimpleNamespace(get=get, get_call_graph=graph)
            return SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda call_id: call),
                                   exception=SimpleNamespace())

        def graph(*inputs):
            return lambda: [SimpleNamespace(function_call_id=call_id, task_id=task, status=0)
                            for call_id, task in inputs]

        def broken():
            raise RuntimeError("no graph for this call")

        self.assertEqual(ruler_cloud.outcome(modal_with(graph(("fc-1", ""))), "fc-1"), ("waiting", None))
        self.assertEqual(ruler_cloud.outcome(modal_with(graph(("fc-1", "ta-1"))), "fc-1"), ("playing", None))
        # A graph that cannot be read, or names no input of the call, says
        # nothing, and the call is taken to be playing, as it always was.
        for unknown in (graph(), graph(("fc-2", "")), broken):
            self.assertEqual(ruler_cloud.outcome(modal_with(unknown), "fc-1"), ("playing", None))

    def test_a_long_wait_is_said_again_every_hour(self):
        heard = ruler_cloud.Heard()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for now, state in ((0.0, "waiting"), (1800.0, "waiting"), (3600.0, "waiting"), (5400.0, "waiting"),
                               (7300.0, "playing"), (9000.0, "playing")):
                heard("seed 9100000", state, now=now)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("seed 9100000 waits in Modal's queue", lines[0])
        self.assertIn("seed 9100000 has waited 1.0 hours in Modal's queue", lines[1])
        self.assertIn("seed 9100000 left Modal's queue after 122 minutes: playing", lines[2])


class KeepTests(unittest.TestCase):
    def test_an_old_rulers_finished_calls_are_kept_as_a_series(self):
        """As `modal-duel-many.py` lists them: a seed and a call id a line,
        and a line a kill cut short. Kept once; a call that failed, waits in
        the queue or still plays is said, and kept when run again once it
        has finished."""
        modal = Modal(lambda name, arguments: None)
        report = {"seed": 9_100_000, "placement": 2.411, "by_deal": [2.5, 2.0]}
        modal.results = {"fc-a": json.dumps(report), "fc-b": RuntimeError("lost"),
                         "fc-c": json.dumps({**report, "seed": 9_300_006}),
                         "fc-d": json.dumps({**report, "seed": 9_500_012})}
        modal.ahead["fc-d"] = ["queued"]
        lines = ["9100000\tfc-a", "9200003\tfc-b", "9300006\tfc-c", "9500012\tfc-d", "9400009"]
        with tempfile.TemporaryDirectory() as folder:
            series = ruler_cloud.Series(Path(folder) / "g910")
            kept, missing = ruler_cloud.keep_calls(modal, series, lines)
            self.assertEqual(kept, 2)
            self.assertEqual(len(missing), 2)
            self.assertIn("seed 9200003: failed", missing[0])
            self.assertEqual(missing[1], "seed 9500012: waiting")
            self.assertEqual(series.reports()[9_100_000], report)
            # The call that waited has finished since; the failed one is said
            # again.
            again, missing = ruler_cloud.keep_calls(modal, series, lines)
            self.assertEqual((again, len(missing)), (1, 1))
            self.assertEqual(sorted(series.reports()), [9_100_000, 9_300_006, 9_500_012])
            # Read against another series of the same seeds, deal by deal.
            other = ruler_cloud.Series(Path(folder) / "g810")
            other.folder.mkdir()
            other.keep({"seed": 9_100_000, "placement": 2.25, "by_deal": [2.5, 2.0]})
            lines = ruler_cloud.reading(modal, other, series)
            self.assertIn("waiting for the seeds of look 1", lines[-1])


class Untouchable:
    """Stands in for the Modal client where nothing may be asked of it."""

    def __getattr__(self, name):
        raise AssertionError(f"modal.{name} was asked for before the refusal")


class CommandLineTests(unittest.TestCase):
    def run_main(self, *arguments, modal=None):
        """The launcher's exit code, or its message, and what it wrote to
        standard error."""
        err = io.StringIO()
        with patch.object(sys, "argv", ["ruler_cloud", *map(str, arguments)]), \
                patch.dict(sys.modules, {"modal": modal or Untouchable()}), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(SystemExit) as stop:
            ruler_cloud.main()
        return stop.exception.code, err.getvalue()

    def test_what_a_series_could_not_ask_is_refused_before_anything_is_registered(self):
        """Every call of such a series would fail on Modal, three times a
        seed; it is refused here, before the client is asked or a folder is
        made."""
        with tempfile.TemporaryDirectory() as folder:
            series = ("series", "t", "history/gen-01010", "zoo/mortal_298k", "--local", folder)
            for extra in (["--skip-forced"], ["--preview-reach"], ["--questions", "seating", "--skip-forced"],
                          ["--questions", "fixed", "--rows", "0"], ["--games", "0"],
                          ["--also", "history/gen-00910", "--against", "g910"]):
                with self.subTest(extra=extra):
                    code, err = self.run_main(*series, *extra)
                    self.assertEqual(code, 2)
            _code, err = self.run_main(*series, "--preview-reach")
            self.assertIn("unless the questions are fixed", err)
            for tag in ("../up", "a/b", ".hidden", "a b"):
                with self.subTest(tag=tag):
                    code, err = self.run_main("series", tag, "x", "y", "--local", folder)
                    self.assertEqual(code, 2)
                    self.assertIn("is not a tag", err)
            for command in (("series", "t", "x", "y", "--against", "../g910"), ("looks", "t", "--against", "a/b"),
                            ("keep", "a/b", "calls.txt")):
                with self.subTest(command=command):
                    code, err = self.run_main(*command, "--local", folder)
                    self.assertIn("is not a tag", err)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_a_comparison_that_could_not_be_read_is_refused_before_it_is_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            kept = ruler_cloud.Series(Path(folder) / "g910")
            for seed in looks.design().seeds:
                kept.keep(old_ruler(seed))
            modal = Modal(lambda name, arguments: None)
            code, _err = self.run_main("series", "g1010", "history/gen-01010", "zoo/mortal_298k", "--against", "g910",
                                       "--games", "50", "--expected", "0.004", "--local", folder, modal=modal)
            self.assertIn("g1010 cannot be read against g910: it holds 20 of the 40 seeds", code)
            self.assertFalse((Path(folder) / "g1010").exists())
            self.assertEqual((modal.spawned, modal.uploaded), ([], {}))


class TimingTests(unittest.TestCase):
    def report(self, by_deal, seconds, **play):
        return {"seed": 9_100_000, "placement": sum(by_deal) / len(by_deal), "standard_error": 0.0183,
                "by_deal": by_deal,
                "play": {"seconds": seconds, "parts": {"engine": 1.0, "encode": seconds - 1.0}, "aside": 0.0,
                         "decisions_per_second": 1000.0, **play}}

    def test_the_four_ways_are_compared_with_the_old_duel_in_one_order(self):
        old = [2.0, 3.0, 2.5, 2.5]
        answers = {"duel": self.report(old, 1200.0), "seating": self.report(old, 900.0, loading=21.6),
                   "graphs": self.report(old, 600.0, loading=22.4, graphs=412, graph_seconds=38.4,
                                         from_graphs=9876),
                   "fixed": self.report([2.0, 3.0, 3.0, 2.0], 400.0, loading=25.0)}

        def way(name, arguments):
            if name == "duel":
                return "duel"
            if arguments["questions"] == "fixed":
                return "fixed"
            return "graphs" if arguments.get("graphs") else "seating"

        def answer(name, arguments):
            self.assertTrue(arguments["timing"])
            self.assertEqual(arguments["seed"], 9_100_000)
            return answers[way(name, arguments)]

        # The calls finish in another order than they were spawned in: the
        # fixed ruler first and the old duel last.
        polls = {"fixed": 0, "graphs": 1, "seating": 2, "duel": 3}
        modal = Modal(answer, queue=lambda name, arguments: ("running",) * polls[way(name, arguments)])
        arguments = ("history/gen-00910", "zoo/mortal_298k", "fusion-long", 9_100_000, 1000, 256)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            lines = ruler_cloud.timing(modal, Path(folder), *arguments, poll=0)
            kept = json.loads((Path(folder) / "timing-calls.json").read_text())
            # Run again, it reads the calls it spawned rather than spawning
            # more, and calls spawned for other arguments are not mixed in.
            again = ruler_cloud.timing(modal, Path(folder), *arguments, poll=0)
            with self.assertRaisesRegex(SystemExit, "name another folder"):
                ruler_cloud.timing(modal, Path(folder), "x", "y", "z", 1, 1, 1, poll=0)
            files = sorted(path.name for path in Path(folder).iterdir())
        self.assertEqual(len(modal.spawned), 4)
        self.assertEqual(set(kept["calls"]), {"old duel", "ruler, seating", "ruler, seating, graphs",
                                              "ruler, fixed 256"})
        self.assertEqual(kept["arguments"]["rows"], 256)
        self.assertEqual(files, ["timing-calls.json", "timing-old-duel.json", "timing-ruler-fixed-256.json",
                                 "timing-ruler-seating-graphs.json", "timing-ruler-seating.json"])
        # The old duel with the room the published rulers' launcher gives it.
        self.assertEqual(modal.options, {"duel": {"memory": 49152}})
        spawned = {way(name, arguments): arguments for name, arguments in modal.spawned}
        self.assertTrue(spawned["fixed"]["graphs"] and spawned["fixed"]["skip_forced"]
                        and spawned["fixed"]["preview_reach"])
        # Asked by seating without graphs and with them, and with no switch
        # that changes the old duel's batches.
        self.assertNotIn("graphs", spawned["seating"])
        self.assertTrue(spawned["graphs"]["graphs"])
        for name in ("seating", "graphs"):
            self.assertEqual(spawned[name]["questions"], "seating")
            self.assertNotIn("skip_forced", spawned[name])
        heads = [line.split(" placement ")[0].strip() for line in lines if " placement " in line]
        self.assertEqual(heads, ["old duel", "ruler, seating", "ruler, seating, graphs", "ruler, fixed 256"])
        self.assertIn("placement 2.5000 +- 0.0183 over 4 deals", lines[0])
        # A ruler's loading is said; the old duel does not time its own.
        self.assertIn("deals; play 1200 s, 1000 decisions a second", lines[0])
        self.assertIn("deals; play 900 s, loading 22 s, 1000 decisions a second", lines[2])
        text = "\n".join(lines)
        self.assertIn("the same placements as the old duel on 4 of 4 deals; 1.33 times as fast", text)
        self.assertIn("the same placements as the old duel on 4 of 4 deals; 2.00 times as fast", text)
        self.assertIn("the same placements as the old duel on 2 of 4 deals; 3.00 times as fast", text)
        # How many graphs the seating ruler recorded as it played, and in how long.
        self.assertEqual(text.count("graphs recorded"), 1)
        self.assertIn("412 graphs recorded in 38 s, 9876 questions answered from them", text)
        self.assertEqual(lines[-1], "the seating ruler with graphs plays the old duel's games bit for bit, "
                                    "2.00 times as fast: keep it")
        self.assertEqual(again, lines)

    def test_a_launch_refused_partway_keeps_the_calls_before_it(self):
        """As at the spend limit: the calls already spawned are kept and
        waited for when the command is run again, and only the rest are
        spawned then."""
        modal = None

        def answer(name, arguments):
            plain = name == "ruler" and arguments["questions"] == "seating" and not arguments.get("graphs")
            if plain:
                modal.refusals = 1  # the next launch, the seating ruler with graphs
            return self.report([2.0, 3.0], 1000.0 if name == "duel" else 500.0 if plain else 400.0)

        modal = Modal(answer)
        arguments = ("history/gen-00910", "zoo/mortal_298k", "fusion-long", 9_100_000, 2, 256)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "the 2 calls spawned before it are kept"):
                ruler_cloud.timing(modal, Path(folder), *arguments, poll=0)
            kept = json.loads((Path(folder) / "timing-calls.json").read_text())
            lines = ruler_cloud.timing(modal, Path(folder), *arguments, poll=0)
        self.assertEqual(list(kept["calls"]), ["old duel", "ruler, seating"])
        self.assertEqual(len(modal.spawned), 4)
        self.assertEqual(lines[-1], "the seating ruler with graphs plays the old duel's games bit for bit, "
                                    "2.50 times as fast: keep it")

    def test_changed_games_are_laid_to_the_arena_or_to_the_graphs(self):
        old = self.report([2.0, 3.0], 1000.0)
        same, other = [2.0, 3.0], [3.0, 2.0]
        arena = ruler_cloud.compare({"old duel": old, "ruler, seating": self.report(other, 800.0),
                                     "ruler, seating, graphs": self.report(same, 400.0)})
        self.assertIn("changed games without graphs: its arena or its weights cast once", arena[-1])
        graphs = ruler_cloud.compare({"old duel": old, "ruler, seating": self.report(same, 500.0),
                                      "ruler, seating, graphs": self.report(other, 300.0)})
        self.assertIn("changed games with graphs and not without", graphs[-2])
        self.assertEqual(graphs[-1], "the seating ruler without graphs plays the old duel's games bit for bit, "
                                     "2.00 times as fast: keep it")
        unsure = ruler_cloud.compare({"old duel": old, "ruler, seating, graphs": self.report(other, 300.0)})
        self.assertIn("cannot be told from the graphs", unsure[-1])
        slow = ruler_cloud.compare({"old duel": old, "ruler, seating": self.report(same, 900.0),
                                    "ruler, seating, graphs": self.report(same, 800.0)})
        self.assertIn("with graphs plays the old duel's games bit for bit but only 1.25 times as fast, "
                      "short of the 1.5", slow[-1])


if __name__ == "__main__":
    unittest.main()
