"""Running rulers on Modal from the desktop (`neural.ruler_cloud`), with the
Modal client stood in for: a series is registered before its first call and
keeps what it plays, spawns one look's seeds at a time and stops where its
rule says, spawns a failed seed again with its own seed, keeps every report
locally and reads what the volume kept; and the timing says whether the
ruler played the old duel's games and how much faster."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from neural import looks, ruler_cloud


class NotFound(Exception):
    pass


class Refused(Exception):
    """As Modal refuses a launch at the spend limit."""


class Modal:
    """The parts of the Modal client the launcher uses. Every spawned call
    finishes at once with what `answer(function, arguments)` gives: a report,
    or an exception it raises; the ruler's reports are kept on the volume
    as `modal_app.ruler` keeps them. The first `refusals` launches are
    refused."""

    def __init__(self, answer, refusals: int = 0):
        self.answer = answer
        self.refusals = refusals
        self.spawned: list[tuple[str, dict]] = []
        self.results: dict[str, object] = {}
        self.files: dict[str, bytes] = {}
        self.uploaded: dict[str, str] = {}
        self.exception = SimpleNamespace(NotFoundError=NotFound, ResourceExhaustedError=Refused)
        self.Function = SimpleNamespace(from_name=self.function)
        self.FunctionCall = SimpleNamespace(from_id=self.call)
        self.Volume = SimpleNamespace(from_name=lambda name: self)

    def function(self, app, name):
        def spawn(**arguments):
            if self.refusals:
                self.refusals -= 1
                raise Refused("billing cycle spend limit")
            call_id = f"fc-{len(self.spawned)}"
            self.spawned.append((name, arguments))
            try:
                report = self.answer(name, arguments)
            except Exception as error:  # noqa: BLE001 - the call's own failure
                self.results[call_id] = error
            else:
                text = json.dumps(report)
                self.results[call_id] = text
                if name == "ruler" and arguments.get("tag"):
                    self.files[f"rulers/{arguments['tag']}/seed-{arguments['seed']}.json"] = text.encode()
            return SimpleNamespace(object_id=call_id)

        return SimpleNamespace(spawn=spawn)

    def call(self, call_id):
        def get(timeout=None):
            result = self.results[call_id]
            if isinstance(result, Exception):
                raise result
            return result

        return SimpleNamespace(get=get)

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


def paired_report(seed: int, gap: float, deals: int = 50) -> dict:
    """A paired call's report: the first challenger `gap` ahead on every
    deal, with a little spread."""
    first = [2.5 - gap + 0.1 * ((deal * 7 + seed) % 5 - 2) for deal in range(deals)]
    second = [2.5 + 0.1 * ((deal * 3 + seed) % 5 - 2) for deal in range(deals)]
    return {"seed": seed, "arms": [{"challenger": "a", "by_deal": first}, {"challenger": "b", "by_deal": second}]}


SETTINGS = {"challenger": "history/gen-00910", "also": "history/gen-00810", "incumbent": "zoo/mortal_298k",
            "games": 1000, "questions": "fixed", "rows": 256, "graphs": True, "skip_forced": True,
            "preview_reach": True, "run": "fusion-long"}


class SeriesTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def play(self, modal, tag="g910-g810", expected=None):
        series = ruler_cloud.Series(self.root / tag)
        rule = series.register(dict(SETTINGS), looks.design(expected, "910 against 810"))
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
            self.assertEqual({key: arguments[key] for key in SETTINGS}, SETTINGS)
            self.assertEqual(arguments["tag"], "g910-g810")
        self.assertEqual(sorted(series.reports()), seeds)
        self.assertIn("stopped at look 1: history/gen-00910 is the stronger", out)
        self.assertEqual(json.loads(modal.uploaded["rulers/g910-g810/series.json"]), SETTINGS)
        self.assertEqual(json.loads(modal.uploaded["rulers/g910-g810/design.json"])["per_look"], 5)

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
        with self.assertRaisesRegex(SystemExit, "another rule"):
            series.register(dict(SETTINGS), looks.design(0.004))

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


class KeepTests(unittest.TestCase):
    def test_an_old_rulers_finished_calls_are_kept_as_a_series(self):
        """As `modal-duel-many.py` lists them: a seed and a call id a line,
        and a line a kill cut short. Kept once; a call that failed or still
        plays is said, and kept when run again once it has finished."""
        modal = Modal(lambda name, arguments: None)
        report = {"seed": 9_100_000, "placement": 2.411, "by_deal": [2.5, 2.0]}
        modal.results = {"fc-a": json.dumps(report), "fc-b": RuntimeError("lost"),
                         "fc-c": json.dumps({**report, "seed": 9_300_006})}
        lines = ["9100000\tfc-a", "9200003\tfc-b", "9300006\tfc-c", "9400009"]
        with tempfile.TemporaryDirectory() as folder:
            series = ruler_cloud.Series(Path(folder) / "g910")
            kept, missing = ruler_cloud.keep_calls(modal, series, lines)
            self.assertEqual(kept, 2)
            self.assertEqual(len(missing), 1)
            self.assertIn("seed 9200003: failed", missing[0])
            self.assertEqual(series.reports()[9_100_000], report)
            again, _missing = ruler_cloud.keep_calls(modal, series, lines)
            self.assertEqual(again, 0)
            # Read against another series of the same seeds, deal by deal.
            other = ruler_cloud.Series(Path(folder) / "g810")
            other.folder.mkdir()
            other.keep({"seed": 9_100_000, "placement": 2.25, "by_deal": [2.5, 2.0]})
            lines = ruler_cloud.reading(modal, other, series)
            self.assertIn("waiting for the seeds of look 1", lines[-1])


class TimingTests(unittest.TestCase):
    def report(self, by_deal, seconds, **play):
        return {"seed": 9_100_000, "placement": sum(by_deal) / len(by_deal), "standard_error": 0.017,
                "by_deal": by_deal,
                "play": {"seconds": seconds, "parts": {"engine": 1.0, "encode": seconds - 1.0}, "aside": 0.0,
                         "decisions_per_second": 1000.0, **play}}

    def test_the_three_ways_are_compared_with_the_old_duel(self):
        old = [2.0, 3.0, 2.5, 2.5]
        answers = {"duel": self.report(old, 1200.0),
                   "seating": self.report(old, 600.0),
                   "fixed": self.report([2.0, 3.0, 3.0, 2.0], 400.0)}

        def answer(name, arguments):
            self.assertTrue(arguments["timing"])
            self.assertEqual(arguments["seed"], 9_100_000)
            return answers["duel" if name == "duel" else arguments["questions"]]

        modal = Modal(answer)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            lines = ruler_cloud.timing(modal, Path(folder), "fusion-long/history/gen-00910", "zoo/mortal_298k",
                                       "fusion-long", 9_100_000, 1000, 256, poll=0)
            kept = json.loads((Path(folder) / "timing-calls.json").read_text())
            # Run again, it reads the calls it spawned rather than spawning more.
            again = ruler_cloud.timing(modal, Path(folder), "x", "y", "z", 1, 1, 1, poll=0)
        self.assertEqual(len(modal.spawned), 3)
        self.assertEqual(set(kept), {"old duel", "ruler, seating", "ruler, fixed 256"})
        fixed = next(arguments for name, arguments in modal.spawned if arguments.get("questions") == "fixed")
        self.assertTrue(fixed["graphs"] and fixed["skip_forced"] and fixed["preview_reach"])
        # Asked by seating with every switch that keeps the old duel's games.
        seating = next(arguments for name, arguments in modal.spawned if arguments.get("questions") == "seating")
        self.assertTrue(seating["graphs"])
        self.assertNotIn("skip_forced", seating)
        text = "\n".join(lines)
        self.assertIn("the same placements as the old duel on 4 of 4 deals; 2.00 times as fast", text)
        self.assertIn("the same placements as the old duel on 2 of 4 deals; 3.00 times as fast", text)
        self.assertIn("bit for bit, 2.00 times as fast: keep it", text)
        self.assertEqual(again, lines)

    def test_changed_games_or_too_little_speed_are_said(self):
        old = self.report([2.0, 3.0], 1000.0)
        changed = ruler_cloud.compare({"old duel": old, "ruler, seating": self.report([3.0, 2.0], 500.0)})
        self.assertIn("changed games", changed[-1])
        slow = ruler_cloud.compare({"old duel": old, "ruler, seating": self.report([2.0, 3.0], 800.0)})
        self.assertIn("short of the 1.5", slow[-1])


if __name__ == "__main__":
    unittest.main()
