"""A reach's tile asked ahead (`--preview-reach`): every row that may
declare riichi is asked, in the forward that decides on the reach, which
tile it would throw as it would stand having declared, from the follower's
preview of the reach, and nobody is told anything. The states are the ones
telling would have made, so the planes recorded are the same to the bit;
the draws are made in the same order from batches of the same shapes, so
the moves are the same; and the answers come from one larger batch, so
they agree to the float's last bits. Off, every question is asked as it
always was."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import riichi_py
import torch

from neural import (
    combined, mortal_learner, mortal_model, policy_inference, ppo_loop, selfplay, train_combined,
    train_mortal, zoo,
)
from neural.model import PolicyValueNet
from neural.observe import Views
from neural.tests import test_mixed_tables
from neural.tests.test_baseline_from_play import TO_MORTAL, Taught, fake_round, varied
from neural.tests.test_bfloat16_players import checkpoints, fusion
from neural.tests.test_forced_rows import Telling, random_rows, row_values
from neural.tests.test_reuse_phi import chunks
from neural.tests.test_step_planes import deciding

CUDA = torch.cuda.is_available()


class Counting:
    """The real follower, counting what it is told."""

    def __init__(self, follower):
        self.follower = follower
        self.told = 0

    def tell(self, *args):
        self.told += 1
        return self.follower.tell(*args)

    def __getattr__(self, name):
        return getattr(self.follower, name)


def counted(made: list):
    """Makes views as self-play does, their followers counting tells."""

    def make(*args, **kwargs):
        views = Views(*args, **kwargs)
        views.observer.follower = Counting(views.observer.follower)
        made.append(views)
        return views

    return make


class Steered:
    """A learner pushed hard, in each step's first question, towards the
    heuristic player's move and towards a reach wherever one is open, and
    left to its network for every reach's tile. A random network seldom
    holds a closed ready hand and the heuristic player often does, so a
    round has many reaches, each tile drawn from the network's own
    probabilities, from the batch the switch puts it in. `forward(planes,
    mask)` answers the logits and the value; `asked` counts the rows of
    each question, a list a step. A generator to draw from, which a round
    drawing apart hands its learner, goes on to the deciding."""

    kind = "mortal"

    def __init__(self, forward, device: str = "cpu"):
        self.forward = forward
        self.device = device
        self.timing: dict = {}
        self.asked: list[list[int]] = []

    def eval(self):
        return self

    @torch.no_grad()
    def decide(self, views, rows, players, legal, greedy=False, explore_share=0.0, wanderer=None,
               preview_reach=False, **draws):
        taught = np.frombuffer(views.arena.teacher(), dtype=np.uint8)[rows].astype(np.int64)
        asked: list[int] = []
        self.asked.append(asked)

        def score(planes, mask):
            logits, value = self.forward(planes, mask)
            logits = logits.float()
            if not asked:
                # The step's own rows lead the first question; whatever
                # follows them, and all of a second question, is a reach's
                # tile, answered by the network alone.
                push = torch.zeros_like(logits)
                n = len(taught)
                at = torch.arange(n, device=logits.device)
                push[at, torch.from_numpy(TO_MORTAL[taught]).to(logits.device)] = 30.0
                push[:n, zoo.MORTAL_RIICHI] += 60.0
                logits = logits + push
            asked.append(len(mask))
            return logits, value

        return mortal_learner.decide_in_mortal_space(
            score, views, rows, players, legal, greedy, self.device, self.timing,
            explore_share=explore_share, wanderer=wanderer, preview_reach=preview_reach, **draws)


def reaching(choice: np.ndarray) -> np.ndarray:
    """Which of our moves are reaches."""
    return (choice >= zoo.RIICHI_DISCARD) & (choice < zoo.TSUMO)


def learners():
    """The two learners that decide in Mortal's moves, each by its forward."""
    torch.manual_seed(31)
    joined = varied(fusion(16, 2)).eval()
    yield "joined", lambda planes, mask: joined.decision(planes, mask)[:2]
    torch.manual_seed(37)
    mortal = mortal_learner.MortalLearner(mortal_model.build(16, 2)).eval()
    with torch.no_grad():
        mortal.value_head.weight.normal_()
    yield "mortal", mortal.policy


def same_planes(test, new, old):
    for name in ("indptr", "indices", "values"):
        got, want = getattr(new, name), getattr(old, name)
        test.assertEqual(got.dtype, want.dtype, name)
        np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8), name)


class RoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_a_round_is_played_as_by_telling(self):
        """On the processor, with a little exploration: the same moves,
        the same planes to the bit, the tile of every reach included, the
        same probabilities to the float's last bits, one question a step
        and nothing told."""
        for name, forward in learners():
            rounds, told, learners_ = {}, {}, {}
            for preview in (False, True):
                made: list = []
                learners_[preview] = Steered(forward)
                torch.manual_seed(5)
                with patch.object(selfplay, "Views", side_effect=counted(made)):
                    rounds[preview] = selfplay.play(
                        learners_[preview], games=4, seed=41, device="cpu", explore_share=0.05,
                        preview_reach=preview)
                told[preview] = made[0].observer.follower.told
            old, new = rounds[False], rounds[True]
            with self.subTest(learner=name):
                reaches = int((old.actions == zoo.MORTAL_RIICHI).sum())
                self.assertGreater(reaches, 20)
                self.assertEqual(told, {False: reaches, True: 0})
                for field in ("actions", "legal", "explored", "returns"):
                    self.assertTrue(torch.equal(getattr(new, field), getattr(old, field)), field)
                np.testing.assert_array_equal(new.final_scores, old.final_scores)
                same_planes(self, new.observations, old.observations)
                torch.testing.assert_close(new.log_probs, old.log_probs, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(new.values, old.values, rtol=1e-5, atol=1e-5)
                # The tiles were the network's own draws, not foregone ones.
                self.assertGreater(int((old.log_probs < -0.1).sum()), reaches // 4)
                # Two questions wherever a reach was chosen, against one.
                self.assertGreater(sum(len(asked) == 2 for asked in learners_[False].asked), 10)
                self.assertTrue(all(len(asked) == 1 for asked in learners_[True].asked))

    def test_the_tiles_asked_ahead_keep_their_values_and_vectors(self):
        """Asked to keep Mortal's vectors (`--reuse-phi`), a round that asks
        ahead records for each reach's tile the value and the vector of the
        state the reach is declared in, as one that tells does: the two
        rounds keep the same, to the float's last bits, and each vector is
        Mortal's own of the planes recorded beside it."""
        torch.manual_seed(11)
        net = varied(fusion()).eval()
        rounds = {}
        for preview in (False, True):
            player = Taught(net)
            player.keep_phi = True
            rounds[preview] = selfplay.play(player, games=2, seed=53, device="cpu", preview_reach=preview)
        old, new = rounds[False], rounds[True]
        self.assertGreater(int((new.actions == zoo.MORTAL_RIICHI).sum()), 5)
        self.assertTrue(torch.equal(new.actions, old.actions))
        same_planes(self, new.observations, old.observations)
        torch.testing.assert_close(new.values, old.values, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new.phi, old.phi, rtol=1e-5, atol=1e-5)
        with torch.no_grad():
            for start, planes in chunks(new.observations, "cpu"):
                torch.testing.assert_close(new.phi[start:start + len(planes)], net.mortal.features(planes),
                                           rtol=1e-5, atol=1e-5)

    def lockstep(self, device: str, net, steps: int = 500):
        """Two tables dealt alike, each step decided greedily by telling at
        one and asking ahead at the other, both then played as the telling
        one chose, for `steps` steps. Yields the step's rows, the moves our
        engine allows them, which may reach, and what each way chose,
        recorded and was asked."""
        games = 8
        tables = [riichi_py.Arena(games=games, seed=20261006) for _ in range(2)]
        views = [Views(table, games) for table in tables]
        seen: list = []

        def forward(planes, mask):
            seen.append((planes.clone(), mask.clone()))
            return net.decision(planes, mask)[:2]

        learner = Steered(forward, device)
        for _step in range(steps):
            if tables[0].all_finished():
                break
            for view in views:
                view.advance()
            live, players = deciding(tables[0], games)
            legal = np.frombuffer(tables[0].legal_mask(), dtype=np.uint8).reshape(games, -1).astype(bool)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                want, old = learner.decide(views[0], live, players, legal[live], greedy=True)
                told = seen[:]
                seen.clear()
                got, new = learner.decide(views[1], live, players, legal[live], greedy=True,
                                          preview_reach=True)
                previewed = seen[:]
                seen.clear()
            yield (live, legal[live], legal[live, zoo.RIICHI_DISCARD:zoo.TSUMO].any(axis=1),
                   want, old, told, got, new, previewed)
            choice = np.zeros(games, dtype=np.int64)
            choice[live] = want
            for table in tables:
                table.step(choice.tolist())

    def test_the_first_question_holds_every_row_that_may_reach(self):
        """Asked ahead: after the step's own rows, one row for each that
        may declare, each with only the tiles its reach may throw, from
        the planes telling would have made; and the step's records are
        those of telling, in the same order."""
        torch.manual_seed(41)
        net = fusion(16, 2).eval()
        reached = 0
        for live, legal, ready, want, old, told, got, new, previewed in self.lockstep("cpu", net):
            np.testing.assert_array_equal(got, want)
            for field in ("masks", "actions", "slots", "forced"):
                np.testing.assert_array_equal(getattr(new, field), getattr(old, field), field)
            same_planes(self, new.planes, old.planes)
            np.testing.assert_allclose(new.log_probs, old.log_probs, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(new.values, old.values, rtol=1e-5, atol=1e-5)
            (planes, mask), = previewed
            self.assertEqual(len(mask), len(live) + int(ready.sum()))
            np.testing.assert_array_equal(mask[: len(live)].numpy(), told[0][1].numpy())
            tiles = mask[len(live):].numpy()
            np.testing.assert_array_equal(tiles[:, :34], legal[ready, zoo.RIICHI_DISCARD:zoo.TSUMO])
            self.assertFalse(tiles[:, 34:].any())
            self.assertEqual(len(told), 2 if reaching(want).any() else 1)
            if len(told) == 2:
                # Every reach chosen was asked ahead, from the same planes.
                at = np.searchsorted(np.flatnonzero(ready), np.flatnonzero(reaching(want)))
                self.assertTrue(torch.equal(planes[len(live):][at], told[1][0]))
                np.testing.assert_array_equal(mask[len(live):][at].numpy(), told[1][1].numpy())
                reached += len(at)
        self.assertGreater(reached, 10)

    @unittest.skipUnless(CUDA, "bfloat16 is the card's")
    def test_on_the_card_the_tiles_agree_to_bfloat16(self):
        """As the trainer plays, by a copy with its weights cast once,
        under autocast: the same planes to the bit, and each reach's
        tiles valued within bfloat16's rounding of the telling batch's."""
        torch.manual_seed(43)
        net = policy_inference.precast(varied(fusion(32, 3)).cuda().eval())
        reached = 0
        largest = 0.0
        for live, _legal, ready, want, old, told, _got, new, previewed in self.lockstep("cuda", net):
            same_planes(self, new.planes, old.planes)
            np.testing.assert_array_equal(new.slots, old.slots)
            if len(told) == 2:
                (planes, mask), = previewed
                at = torch.from_numpy(
                    np.searchsorted(np.flatnonzero(ready), np.flatnonzero(reaching(want)))).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    alone = net.decision(*told[1])[0].float()
                    together = net.decision(planes, mask)[0].float()[len(live):][at]
                tiles = told[1][1]
                gap = (torch.log_softmax(together, dim=1) - torch.log_softmax(alone, dim=1))[tiles]
                largest = max(largest, float(gap.abs().max()))
                reached += len(at)
        self.assertGreater(reached, 10)
        # A few bfloat16 steps at most, at the size of these logits.
        self.assertLess(largest, 0.1)


class ChoiceTests(unittest.TestCase):
    def test_the_same_choices_from_one_question(self):
        """Values a row at a time, the same whatever else is asked with
        them: the reaches' tiles asked ahead choose what the second
        question chose, with nothing told and nothing asked twice."""
        rng = np.random.default_rng(29)
        ahead_rows = 0
        for trial in range(300):
            rows, players, legal = random_rows(rng)
            for skip_forced in (False, True):
                questions = []

                def ask(who, fresh, allowed, ahead=()):
                    questions.append((list(who), fresh, allowed.copy(), list(ahead)))
                    values, own = row_values(who, fresh)
                    if len(ahead):
                        # Their tiles from the state the reach is declared in.
                        later, _own = row_values(ahead, True)
                        values = np.concatenate([values, later])
                    return values, own

                old_views, new_views = Telling(), Telling()
                old_stats, new_stats = {}, {}
                want = zoo.choose_in_mortal_space(
                    lambda who, fresh, allowed: row_values(who, fresh), old_views, rows, players, legal,
                    old_stats, skip_forced=skip_forced)
                got = zoo.choose_in_mortal_space(ask, new_views, rows, players, legal, new_stats,
                                                 skip_forced=skip_forced, preview_reach=True)
                with self.subTest(trial=trial, skip_forced=skip_forced):
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(new_views.told, [])
                    self.assertEqual(new_stats, old_stats)
                    self.assertLessEqual(len(questions), 1)
                    for who, fresh, allowed, ahead in questions:
                        self.assertFalse(fresh)
                        self.assertEqual(len(allowed), len(who) + len(ahead))
                        tiles = allowed[len(who):]
                        self.assertFalse(tiles[:, 34:].any())
                        # Only a reach with a choice of tile is asked ahead
                        # when rows with one move go unasked.
                        if skip_forced:
                            self.assertTrue((tiles.sum(axis=1) > 1).all())
                        ahead_rows += len(ahead)
        self.assertGreater(ahead_rows, 100)

    def test_off_every_question_is_as_it_was(self):
        """Without the switch a question names no rows ahead, and a reach's
        tile is asked again after telling (see test_forced_rows)."""
        rng = np.random.default_rng(31)
        asked = []

        def ask(who, fresh, allowed, *more):
            self.assertEqual(more, ())
            asked.append(fresh)
            return row_values(who, fresh)

        for _trial in range(100):
            rows, players, legal = random_rows(rng)
            zoo.choose_in_mortal_space(ask, Telling(), rows, players, legal)
        self.assertIn(True, asked)


class SeatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def players(self):
        """A seated player of each kind that answers in Mortal's moves."""
        torch.manual_seed(3)
        net = mortal_model.build(16, 2).eval()
        with patch.object(zoo.mortal_model, "load", return_value=net):
            mortal = zoo.MortalPlayer("unused", "cpu")
        yield "mortal", mortal
        torch.manual_seed(5)
        yield "reheaded", zoo.MortalSpacePlayer(PolicyValueNet(16, 2, actions=46).eval(), "cpu")
        torch.manual_seed(7)
        yield "fused", fusion().eval()

    def test_seated_players_choose_as_by_telling(self):
        """Two tables dealt alike, one asked as before and one asking ahead,
        stay in step move for move. Each player is pushed towards the
        heuristic player's move and a reach wherever one is open, in the
        step's own rows only, so its reaches are many and their tiles its
        network's own."""
        mortal_of = np.isfinite(zoo.PRIORITY).argmax(axis=0)
        for name, player in self.players():
            for skip_forced in (False, True):
                games = 8
                tables = [riichi_py.Arena(games=games, seed=20261005) for _ in range(2)]
                views = [Views(table, games) for table in tables]
                for view in views:
                    view.observer.follower = Counting(view.observer.follower)
                reaches = 0

                def steered(table, teacher):
                    def ask(who, fresh, allowed, ahead=()):
                        values, own = player._ask(views[table], who, fresh, allowed, ahead)
                        if not fresh:
                            for row, (game, _player) in enumerate(who):
                                move = int(teacher[game])
                                if move != 0xFF:
                                    values[row, mortal_of[move]] += 100.0
                                    values[row, zoo.MORTAL_RIICHI] += 200.0 * allowed[row, zoo.MORTAL_RIICHI]
                        return values, own
                    return ask

                for _step in range(250):
                    if tables[0].all_finished():
                        break
                    for view in views:
                        view.advance()
                    live, players = deciding(tables[0], games)
                    mask = np.frombuffer(tables[0].legal_mask(), dtype=np.uint8).reshape(games, -1)
                    legal = mask.astype(bool)[live]
                    teacher = np.frombuffer(tables[0].teacher(), dtype=np.uint8)
                    for view in views:
                        view.prepare(live, players)
                    with torch.no_grad():
                        want = zoo.choose_in_mortal_space(steered(0, teacher), views[0], live, players,
                                                          legal, skip_forced=skip_forced)
                        got = zoo.choose_in_mortal_space(steered(1, teacher), views[1], live, players,
                                                         legal, skip_forced=skip_forced, preview_reach=True)
                    with self.subTest(player=name, skip_forced=skip_forced, step=_step):
                        np.testing.assert_array_equal(got, want)
                    reaches += int(reaching(want).sum())
                    choice = np.zeros(games, dtype=np.int64)
                    choice[live] = want
                    for table in tables:
                        table.step(choice.tolist())
                with self.subTest(player=name, skip_forced=skip_forced):
                    self.assertGreater(reaches, 10)
                    self.assertEqual(views[0].observer.follower.told, reaches)
                    self.assertEqual(views[1].observer.follower.told, 0)


class SwitchTests(unittest.TestCase):
    def test_seated_players_ask_ahead_only_when_the_trainer_says(self):
        legal = np.zeros((1, riichi_py.ACTIONS), dtype=bool)
        legal[0, 0] = True
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            names = ("mortal", "reheaded", "fused")
            for options in ({"preview_reach": False}, {"preview_reach": True}, {}):
                args = SimpleNamespace(opponents=[*(paths[name] for name in names), "club"],
                                       compile=False, **options)
                seated = ppo_loop.load_others(args, "cpu")
                self.assertFalse(hasattr(seated[-1], "preview_reach"))
                for name, player in zip(names, seated):
                    with self.subTest(player=name, **options):
                        passed = {}

                        def choose(*_args, **kwargs):
                            passed.update(kwargs)
                            return np.zeros(1, dtype=np.int64)

                        with patch.object(zoo, "choose_in_mortal_space", side_effect=choose):
                            player.choose(None, np.zeros(1), np.zeros(1), legal)
                        self.assertIs(passed["preview_reach"], options.get("preview_reach", False))
                        self.assertIs(passed["skip_forced"], False)
        # Players nobody seated, as a duel or a measurement loads them, and
        # the learner when it is measured.
        self.assertFalse(zoo.MortalPlayer.preview_reach)
        self.assertFalse(zoo.MortalSpacePlayer.preview_reach)
        self.assertFalse(combined.Combined.preview_reach)

    def test_play_tells_the_learner_what_it_was_told(self):
        for preview in (False, True):
            told = []

            class Learner(test_mixed_tables.Seated):
                timing: dict = {}

                def decide(self, views, rows, players, legal, greedy=False, **options):
                    told.append(options["preview_reach"])
                    from neural.tests.test_selfplay_contract import FirstLegalDecider
                    return FirstLegalDecider().decide(views, rows, players, legal, greedy)

            selfplay.play(Learner(), games=1, seed=3, device="cpu", preview_reach=preview)
            with self.subTest(preview=preview):
                self.assertTrue(told)
                self.assertEqual(set(told), {preview})

    def test_the_trainers_and_the_cloud_leave_it_off_unless_asked(self):
        for module in (train_combined, train_mortal):
            with self.subTest(trainer=module.__name__):
                with patch.object(sys, "argv", ["trainer"]):
                    self.assertFalse(module.parse_args().preview_reach)
                with patch.object(sys, "argv", ["trainer", "--preview-reach"]):
                    self.assertTrue(module.parse_args().preview_reach)
        launch = test_mixed_tables.CloudLaunchTests.launch
        for trainer in ("train_mortal", "train_combined"):
            with self.subTest(cloud=trainer):
                self.assertNotIn("--preview-reach", launch(self, trainer))
                self.assertIn("--preview-reach", launch(self, trainer, preview_reach=True))

    def test_the_trainers_play_with_it_as_told(self):
        """Each trainer hands its switch to the round it plays."""
        for module in (train_combined, train_mortal):
            for flags in ([], ["--preview-reach"]):
                played = []

                def play(*_args, **kwargs):
                    played.append(kwargs.get("preview_reach"))
                    return fake_round()

                with self.subTest(trainer=module.__name__, flags=flags), \
                        tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    origin, out = root / "source.pt", root / "run"
                    if module is train_mortal:
                        net = mortal_model.build(8, 1)
                        torch.save({**net.state(), "config": {"resnet": {"conv_channels": 8, "num_blocks": 1},
                                                              "control": {"version": 4}}}, origin)
                    else:
                        torch.save(fusion(8, 1).state(), origin)
                    argv = ["trainer", "--mortal" if module is train_mortal else "--resume", str(origin),
                            "--out", str(out), "--rounds", "1", "--batch", "4", "--epochs", "1",
                            "--games", "1", "--measure-games", "1", *flags]
                    if module is train_combined:
                        argv += ["--fixed", "none"]
                    torch.set_num_threads(1)
                    with patch.object(sys, "argv", argv), \
                            patch.object(torch.cuda, "is_available", return_value=False), \
                            patch.object(module.selfplay, "play", side_effect=play), \
                            patch.object(module.selfplay, "measure",
                                         return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                    self.assertEqual(played, [bool(flags)])
                    json.loads((out / "log.jsonl").read_text().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
