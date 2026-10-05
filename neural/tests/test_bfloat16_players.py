"""Players that play on the card in bfloat16 hold their convolutions' and
linear layers' weights that way, cast once, instead of having autocast cast
them on every call: the seated others from their loading, and the joined
player through a copy refreshed from the learner before every round. The
cast is autocast's own, so every answer is bit for bit what it was."""

import contextlib
import copy
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
from torch import nn

from neural import combined, mortal_model, policy_inference, ppo_loop, selfplay, train_combined, zoo
from neural.model import MORTAL_PLANES, PolicyValueNet
from neural.observe import Planes, Views
from neural.tests.test_step_planes import deciding, play_on

CUDA = torch.cuda.is_available()
MORTAL_CONFIG = {"resnet": {"conv_channels": 16, "num_blocks": 2}, "control": {"version": 4}}


def fusion(channels=16, blocks=2):
    net = combined.Combined(PolicyValueNet(channels, blocks, actions=46), mortal_model.build(16, 2))
    net.mortal_config = MORTAL_CONFIG
    with torch.no_grad():
        # A head that has learned something, so it has a say in the answer.
        for parameter in net.fuse.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return net


def checkpoints(root: Path) -> dict[str, Path]:
    """One checkpoint of every kind a table can seat, by name."""
    torch.manual_seed(11)
    mortal = mortal_model.build(16, 2)
    ours = PolicyValueNet(16, 2, actions=46)
    engine = PolicyValueNet(16, 2, MORTAL_PLANES)
    payloads = {
        "mortal": {**mortal.state(), "config": MORTAL_CONFIG},
        "reheaded": {"model": ours.state_dict(), **ours.payload_fields()},
        "fused": fusion().state(),
        "ours, our moves": {"model": engine.state_dict(), **engine.payload_fields()},
    }
    paths = {}
    for name, payload in payloads.items():
        paths[name] = root / f"{name}.pt"
        torch.save(payload, paths[name])
    return paths


def network(player):
    return getattr(player, "net", player)


def kinds_of_parameters(net: nn.Module) -> dict[torch.dtype, set[str]]:
    """Which modules' parameters and buffers are held in which precision."""
    found: dict[torch.dtype, set[str]] = {}
    for module in net.modules():
        for tensor in [*module.parameters(recurse=False), *module.buffers(recurse=False)]:
            if tensor.is_floating_point():
                found.setdefault(tensor.dtype, set()).add(type(module).__name__)
    return found


def bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


class PrecastTests(unittest.TestCase):
    def test_only_convolutions_and_linear_layers_are_cast(self):
        net = policy_inference.precast(fusion())
        found = kinds_of_parameters(net)
        self.assertEqual(found[torch.bfloat16], {"Conv1d", "Linear"})
        # Normalisation, Mortal's statistics and the head's own weighing of
        # the three answers stay as they were.
        self.assertEqual(found[torch.float32], {"GroupNorm", "BatchNorm1d", "Fuse"})
        for module in net.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                self.assertTrue(all(p.dtype == torch.bfloat16 for p in module.parameters()))

    def test_only_players_on_the_card_in_bfloat16_are_cast(self):
        """And only those that play eagerly: compiled, the weights cast
        once are arranged into another graph than the one that casts them
        itself, and the answers change, so compiled players are left as
        they were. The compiler does not run here; it would only on the
        first move."""
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            for device in ("cpu", "cuda") if CUDA else ("cpu",):
                for compiled in (False, True):
                    args = SimpleNamespace(opponents=[*paths.values(), "club"], compile=compiled)
                    seated = dict(zip([*paths, "club"], ppo_loop.load_others(args, device)))
                    for name, player in seated.items():
                        with self.subTest(device=device, compiled=compiled, player=name):
                            if name == "club":
                                self.assertEqual(player.kind, "club")
                                continue
                            found = kinds_of_parameters(network(player))
                            cast = device == "cuda" and name != "ours, our moves" and not compiled
                            self.assertEqual(torch.bfloat16 in found, cast)
                            if compiled:
                                self.assertEqual(set(found), {torch.float32})


@unittest.skipUnless(CUDA, "the bfloat16 path is the card's")
class SameAnswersOnTheCardTests(unittest.TestCase):
    def test_seated_players_answer_bit_for_bit_as_before(self):
        """Over real positions, the first question and a fresh one, every
        kind of seated player answers the same bits cast or not."""
        with tempfile.TemporaryDirectory() as folder:
            paths = checkpoints(Path(folder))
            games = 16
            arena = riichi_py.Arena(games=games, seed=707)
            views = Views(arena, games)
            players = {}
            for name in ("mortal", "reheaded", "fused"):
                players[name] = (zoo.load_player(paths[name], "cuda"),
                                 zoo.play_in_bfloat16(zoo.load_player(paths[name], "cuda"), "cuda"))
            rows = 0
            for step in range(60):
                views.advance()
                live, deciders = deciding(arena, games)
                legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, -1)
                allowed = zoo.translatable(legal.astype(bool)[live])
                allowed[~allowed.any(axis=1), zoo.MORTAL_PASS] = True
                views.prepare(live, deciders)
                who = list(zip(live.tolist(), deciders.tolist()))
                if step % 10 == 0:
                    for name, (before, cast) in players.items():
                        for fresh in (False, True):
                            with self.subTest(player=name, step=step, fresh=fresh):
                                want, want_own = before._ask(views, who, fresh, allowed)
                                got, got_own = cast._ask(views, who, fresh, allowed)
                                np.testing.assert_array_equal(bits(got), bits(want))
                                np.testing.assert_array_equal(got_own, want_own)
                    rows += len(who)
                play_on(arena, games)
            self.assertGreater(rows, 60)

    def test_the_learners_copy_plays_bit_for_bit_as_the_learner(self):
        torch.manual_seed(13)
        net = fusion(32, 3).to("cuda")
        actor = policy_inference.precast(copy.deepcopy(net)).requires_grad_(False)
        # A round's learning moves the learner; the copy is refreshed from it.
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(torch.randn_like(parameter) * 1e-3)
        actor.load_state_dict(net.state_dict())
        for mode in combined.Combined.MODES:
            net.set_mode(mode)
            net.eval()
            torch.manual_seed(29)
            learner_round = selfplay.play(net, games=2, seed=99, device="cuda", amp=True)
            torch.manual_seed(29)
            actor_round = selfplay.play(actor, games=2, seed=99, device="cuda", amp=True)
            with self.subTest(mode=mode):
                self.assertGreater(learner_round.decisions, 100)
                self.assertTrue(torch.equal(actor_round.actions, learner_round.actions))
                self.assertTrue(torch.equal(actor_round.log_probs.view(torch.int32),
                                            learner_round.log_probs.view(torch.int32)))
                self.assertTrue(torch.equal(actor_round.legal, learner_round.legal))
                np.testing.assert_array_equal(actor_round.final_scores, learner_round.final_scores)
            if mode != "none":
                break  # one frozen part is enough to show the caching does not matter

    def test_the_trainer_plays_each_round_with_a_fresh_copy(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin, out = root / "source.pt", root / "run"
            torch.save(fusion().state(), origin)
            seen, played_by = [], []

            def collect(learner, **_kwargs):
                # The checkpoint written last is the learner as it is now.
                expected, saved = combined.load(out / "latest.pt" if seen else origin, "cuda")
                seen.append(int(saved.get("generation", 0)))
                self.assertIsInstance(learner, combined.Combined)
                self.assertEqual(learner.fuse.tiles[0].weight.dtype, torch.bfloat16)
                wanted = expected.state_dict()
                for name, got in learner.state_dict().items():
                    self.assertTrue(torch.equal(got, wanted[name].to(got.dtype)), name)
                played_by.append({name: tensor.clone() for name, tensor in learner.state_dict().items()})
                n = 8
                planes = Planes(np.arange(n + 1, dtype=np.int64), np.zeros(n, dtype=np.uint16),
                                np.ones(n, dtype=np.float16))
                return SimpleNamespace(
                    decisions=n, observations=planes, legal=torch.ones(n, 46, dtype=torch.bool),
                    actions=torch.zeros(n, dtype=torch.int64), returns=torch.linspace(-1, 1, n),
                    log_probs=torch.zeros(n), held=torch.full((n, 3, 34), 1 / 34),
                    games=1, hands=1, timing={})

            argv = ["trainer", "--resume", str(origin), "--out", str(out), "--rounds", "2",
                    "--batch", "4", "--epochs", "1", "--games", "1", "--measure-games", "1",
                    "--measure-every", "5", "--fixed", "none", "--amp", "--lr", "1e-2"]
            with patch.object(sys, "argv", argv), \
                    patch.object(train_combined.selfplay, "play", side_effect=collect), \
                    patch.object(train_combined.selfplay, "measure",
                                 return_value={"placement": 2.5, "score": 0.0, "wins": 0.25}), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_combined.main()
            self.assertEqual(seen, [0, 1])
            records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            # The first round's learning moved the weights the second played with.
            first, second = played_by
            self.assertTrue(any(not torch.equal(first[name], second[name]) for name in first))


if __name__ == "__main__":
    unittest.main()
