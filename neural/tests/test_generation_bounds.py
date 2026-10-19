"""Every generation's record says what bounded it, and reading that changes
nothing played or learned: the card and the processors it had, the
container's quota of them and any time it was held back, the processors'
time it used, the seated others' encoding, and, with the round played
ahead, how much of the learning went on beside that round and how much
after it was done, in seconds and in updates."""

import contextlib
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from neural import host, observe, ppo_loop, selfplay
from neural.tests.test_baseline_from_play import varied
from neural.tests.test_bfloat16_players import checkpoints, fusion
from neural.tests.test_encode_aside import Inline
from neural.tests.test_play_ahead import Rounds, Trainer
from neural.tests.test_preview_reach import Steered

SPLIT = ("learning_beside_round_seconds", "learning_alone_seconds", "updates_beside_round",
         "updates_alone")


class LearningClockTests(unittest.TestCase):
    def clock(self, began: float, updates: list[float]) -> ppo_loop.LearningClock:
        clock = ppo_loop.LearningClock()
        clock.began, clock.updates = began, updates
        return clock

    def test_a_round_that_ends_part_way_cuts_the_learning_there(self):
        split = self.clock(100.0, [101.0, 102.0, 105.0, 106.0]).split(103.5, now=107.0)
        self.assertEqual(split, {"learning_beside_round_seconds": 3.5, "learning_alone_seconds": 3.5,
                                 "updates_beside_round": 2, "updates_alone": 2})

    def test_a_round_that_outlasts_the_learning_has_all_of_it_beside(self):
        split = self.clock(100.0, [101.0, 102.0]).split(None, now=104.0)
        self.assertEqual(split, {"learning_beside_round_seconds": 4.0, "learning_alone_seconds": 0.0,
                                 "updates_beside_round": 2, "updates_alone": 0})

    def test_with_no_round_beside_it_the_learning_is_all_alone(self):
        clock = self.clock(100.0, [101.0, 102.0])
        for ended in (clock.began, 50.0):
            with self.subTest(ended=ended):
                self.assertEqual(clock.split(ended, now=104.0),
                                 {"learning_beside_round_seconds": 0.0, "learning_alone_seconds": 4.0,
                                  "updates_beside_round": 0, "updates_alone": 2})

    def test_an_update_as_the_round_ends_counts_beside_it(self):
        split = self.clock(100.0, [101.0, 102.0, 103.0]).split(102.0, now=103.0)
        self.assertEqual((split["updates_beside_round"], split["updates_alone"]), (2, 1))

    def test_updates_are_noted_as_they_are_taken(self):
        clock = ppo_loop.LearningClock()
        clock.update()
        clock.update()
        self.assertEqual(len(clock.updates), 2)
        self.assertLessEqual(clock.began, clock.updates[0])
        self.assertLessEqual(clock.updates[0], clock.updates[1])


class RoundEndTests(unittest.TestCase):
    def test_a_rounds_end_is_noted_until_it_is_taken(self):
        torch.manual_seed(2)
        learner = torch.nn.Linear(3, 2)
        go = threading.Event()

        def play(_player, generation, **_options):
            self.assertTrue(go.wait(10))
            return generation

        ahead = ppo_loop.PlayAhead(learner, torch.nn.Linear(3, 2), "cpu", play)
        self.addCleanup(ahead.close)
        ahead.start(7)
        self.assertNotIn(7, ahead.ended)
        started = time.time()
        go.set()
        # Played, and not yet taken.
        ahead.round[1].result()
        self.assertGreaterEqual(ahead.ended[7], started)
        self.assertEqual(ahead.take(7)[0], 7)
        # Taken, it is no longer the round any learning is beside.
        self.assertNotIn(7, ahead.ended)

    def test_a_round_that_fails_is_noted_and_let_go_too(self):
        def play(_player, generation, **_options):
            raise ValueError(f"round {generation} broke")

        ahead = ppo_loop.PlayAhead(torch.nn.Linear(3, 2), torch.nn.Linear(3, 2), "cpu", play)
        self.addCleanup(ahead.close)
        ahead.start(3)
        with self.assertRaises(ValueError):
            ahead.round[1].result()
        self.assertIn(3, ahead.ended)
        with self.assertRaisesRegex(ValueError, "round 3 broke"):
            ahead.take(3)
        self.assertEqual(ahead.ended, {})


class HostTests(unittest.TestCase):
    def root(self, files: dict[str, str]) -> Path:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding="utf-8")
        return root

    def test_what_a_linux_container_shows_is_read(self):
        root = self.root({
            "proc/cpuinfo": "processor\t: 0\nvendor_id\t: AuthenticAMD\n"
                            "model name\t: AMD EPYC 9454 48-Core Processor\ncpu MHz\t\t: 2750.000\n\n"
                            "processor\t: 1\nmodel name\t: AMD EPYC 9454 48-Core Processor\n",
            "sys/fs/cgroup/cpu.max": "1600000 100000\n",
            "sys/fs/cgroup/cpu.stat": "usage_usec 9000000\nuser_usec 8000000\nsystem_usec 1000000\n"
                                      "nr_periods 120\nnr_throttled 30\nthrottled_usec 2500000\n",
            "proc/stat": "cpu  100 0 50 1000 3 0 2 800 0 0\ncpu0 50 0 25 500 1 0 1 400 0 0\n",
            "proc/pressure/cpu": "some avg10=0.00 avg60=0.00 avg300=0.00 total=1500000\n"
                                 "full avg10=0.00 avg60=0.00 avg300=0.00 total=700000\n",
        })
        with patch.dict(host.os.environ, {"RAYON_NUM_THREADS": "16"}):
            described = host.processors(root)
        self.assertEqual(described["cpu_model"], "AMD EPYC 9454 48-Core Processor")
        self.assertEqual(described["cpu_quota"], 16.0)
        self.assertEqual(described["encoder_threads"], 16)
        self.assertEqual(described["cpus"], host.os.cpu_count())
        with patch.object(host, "_ticks_per_second", return_value=100):
            counted = host.counters(root)
        self.assertEqual(counted, {"throttled_periods": 30, "throttled_seconds": 2.5,
                                   "steal_seconds": 8.0, "cpu_pressure_seconds": 1.5})

    def test_a_container_without_a_quota_and_cgroup_version_one(self):
        self.assertEqual(host.processors(self.root({"sys/fs/cgroup/cpu.max": "max 100000\n"}))["cpu_quota"],
                         "unlimited")
        old = self.root({"sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000\n",
                         "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
                         "sys/fs/cgroup/cpu/cpu.stat": "nr_periods 9\nnr_throttled 4\nthrottled_time 3000000000\n"})
        self.assertEqual(host.processors(old)["cpu_quota"], 2.0)
        self.assertEqual(host.counters(old), {"throttled_periods": 4, "throttled_seconds": 3.0})
        free = self.root({"sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n",
                          "sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n"})
        self.assertEqual(host.processors(free)["cpu_quota"], "unlimited")

    def test_what_the_platform_does_not_show_is_left_out(self):
        """Windows shows none of the files, and a sandbox only some: what
        is missing or unreadable is left out, never an error."""
        empty = self.root({})
        garbled = self.root({"sys/fs/cgroup/cpu.max": "lots\n", "proc/stat": "cpu x y\n",
                             "sys/fs/cgroup/cpu.stat": "nr_throttled many\n",
                             "proc/pressure/cpu": "some total=never\n"})
        for root in (empty, garbled):
            with self.subTest(root=root), patch.dict(host.os.environ, {"RAYON_NUM_THREADS": ""}):
                described = host.processors(root)
                self.assertEqual(set(described) - {"cpus_allowed"}, {"cpus"})
                self.assertEqual(host.counters(root), {})

    def test_a_generation_records_what_each_counter_grew_by(self):
        self.assertEqual(host.change({"steal_seconds": 1.0, "throttled_periods": 3},
                                     {"steal_seconds": 1.25, "throttled_periods": 7, "cpu_pressure_seconds": 2.0}),
                         {"steal_seconds": 0.25, "throttled_periods": 4})

    def test_on_the_processor_there_is_no_card(self):
        self.assertIsNone(host.card("cpu"))
        with patch.object(host.torch.cuda, "is_available", return_value=False):
            self.assertIsNone(host.card("cuda"))
        with patch.object(host.torch.cuda, "is_available", return_value=True), \
                patch.object(host.torch.cuda, "get_device_name", return_value="NVIDIA L40S"):
            self.assertEqual(host.card("cuda"), "NVIDIA L40S")

    def test_the_processors_time_grows_as_the_process_works(self):
        # Windows counts it in ticks of about sixteen milliseconds.
        spent = host.cpu_seconds()
        deadline = time.time() + 10
        while host.cpu_seconds() <= spent and time.time() < deadline:
            sum(range(100_000))
        self.assertGreater(host.cpu_seconds(), spent)


class SeatedEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory()
        cls.paths = checkpoints(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.folder.cleanup()

    def round(self, seated: bool, inline: bool = False):
        torch.manual_seed(31)
        net = varied(fusion(16, 2)).eval()
        learner = Steered(lambda planes, mask: net.decision(planes, mask)[:2])
        options = {}
        if seated:
            args = SimpleNamespace(opponents=[self.paths["mortal"], "club"], compile=False,
                                   seat_share=0.5, opponent_share=0.0)
            with contextlib.redirect_stdout(None):
                roster, others = ppo_loop.seat_others(args, "cpu")
            options = {"opponents": others, "seat_share": 0.5, "population": roster}
        torch.manual_seed(5)
        with patch.object(observe, "encoder_thread", return_value=Inline()) if inline \
                else contextlib.nullcontext():
            return selfplay.play(learner, games=4, seed=808, device="cpu", **options)

    def test_a_round_with_a_seated_mortal_says_what_its_views_cost(self):
        """Encoded by the worker or at once, the seated Mortal's views are
        counted, and they are not a part of the round's play record."""
        for inline in (False, True):
            with self.subTest(inline=inline):
                batch = self.round(seated=True, inline=inline)
                self.assertGreater(batch.seated_encode_seconds, 0.0)
                self.assertNotIn("seated_encode", batch.timing)

    def test_a_round_with_nobody_seated_has_none(self):
        self.assertEqual(self.round(seated=False).seated_encode_seconds, 0.0)


class TrainerRecordTests(unittest.TestCase):
    def test_every_generation_says_what_it_ran_on(self):
        for flags in ([], ["--play-ahead"]):
            with self.subTest(flags=flags):
                records = Trainer(self).run(flags, rounds=2)
                for record in records:
                    self.assertIsNone(record["card"])
                    self.assertEqual(record["host"]["cpus"], host.os.cpu_count())
                    self.assertGreaterEqual(record["cpu_seconds"], 0.0)
                    self.assertIsInstance(record["throttling"], dict)
                    self.assertEqual(record["seated_encode_seconds"], 0.0)
                    self.assertEqual(all(name in record for name in SPLIT), bool(flags))

    def test_the_learning_is_split_where_the_round_beside_it_ended(self):
        """Generation 0 learns beside round 1, which is held until the
        record of generation 0 is made: all of that learning is beside it.
        Generation 1 learns beside round 2, which ends once generation 1
        has taken its first update: the update is beside the round, and the
        two parts add up to the learning. Generation 2 has no round beside
        it, being the last: all alone. However the parts fall, nothing
        played or learned changes, which the play-ahead tests hold."""
        recorded = threading.Event()
        stepped = threading.Event()
        clocks = []
        real_finish = ppo_loop.finish
        real_init = ppo_loop.LearningClock.__init__
        real_update = ppo_loop.LearningClock.update

        def finish(args, generation, entry, *rest, **options):
            if generation == 0:
                recorded.set()
            return real_finish(args, generation, entry, *rest, **options)

        def begin(clock):
            real_init(clock)
            clocks.append(clock)

        def update(clock):
            real_update(clock)
            # Generation 1's learning has taken an update.
            if len(clocks) > 1 and clock is clocks[1]:
                stepped.set()

        def then(generation, _options):
            if generation == 1:
                self.assertTrue(recorded.wait(60))
            elif generation == 2:
                self.assertTrue(stepped.wait(60))

        records = Trainer(self).run(
            ["--play-ahead"], rounds=3, play=Rounds(then=then),
            patches=[patch.object(ppo_loop, "finish", side_effect=finish),
                     patch.object(ppo_loop.LearningClock, "__init__", begin),
                     patch.object(ppo_loop.LearningClock, "update", update)])
        self.assertEqual([record["generation"] for record in records], [0, 1, 2])
        first, second, last = records
        for record in records:
            self.assertEqual(record["updates_beside_round"] + record["updates_alone"],
                             record["optimizer_updates"])
            self.assertGreater(record["optimizer_updates"], 0)
        self.assertEqual(first["updates_alone"], 0)
        self.assertEqual(first["learning_alone_seconds"], 0.0)
        self.assertGreaterEqual(second["updates_beside_round"], 1)
        self.assertEqual(last["updates_beside_round"], 0)
        self.assertEqual(last["learning_beside_round_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
