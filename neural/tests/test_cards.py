"""Every function of the Modal app runs on an L40S. The owner moved training
to the L40S on 19 October 2026, after it played a generation in 9.5 minutes
against an H100's 9.0 for half the card's price, and asked that nothing
switch back without the owner's word. A function that asks for another card,
or for a list of cards to fall back along, fails here. A card asked for at
the call (`with_options`) is outside this test, as a benchmark of another
card, which the owner asked for of the L4 and the A10G, must be.

The settings are read as Modal receives them: the app is loaded with a
stand-in for the Modal client whose `App.function` keeps every function's
keywords, however they were written, a `gpu=` in the decorator or a dict of
shared settings spread into it.

Read the same way, no function declares retries: a failed call on a card
ends, and is started again only by a caller that asked for that at the call
(decided in November 2026, see `TRAINER` in the app)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "modal_app.py"


class Image:
    """Stands in for `modal.Image`: every step builds the same image."""

    def __getattr__(self, method: str):
        return lambda *arguments, **keywords: self


def settings() -> dict[str, dict]:
    """Every function the app registers with Modal, by name, with the
    keywords it registers it with, from the app loaded with a stand-in for
    the Modal client."""
    registered: dict[str, dict] = {}

    class App:
        def __init__(self, *arguments, **keywords): pass

        def function(self, **keywords):
            def register(function):
                registered[function.__name__] = keywords
                return function
            return register

        def local_entrypoint(self, **keywords): return lambda function: function

        def __getattr__(self, name: str):
            # Another way to put a function on a card (a class, a web
            # endpoint, ...) is not read here, so it fails the test instead.
            raise AssertionError(f"modal_app uses App.{name}, which this test does not read")

    modal = SimpleNamespace(
        Image=SimpleNamespace(debian_slim=lambda **keywords: Image()), App=App,
        Volume=SimpleNamespace(from_name=lambda *arguments, **keywords: SimpleNamespace()))
    spec = importlib.util.spec_from_file_location("cards_modal_app", APP)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"modal": modal}):
        spec.loader.exec_module(module)
    return registered


def cards() -> dict[str, object]:
    """The card each function of the app asks for, by name, and the
    trainers' shared settings as TRAINER, read from the source without
    importing the Modal client."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TRAINER"
                                                for target in node.targets):
            for keyword in node.value.keywords:
                if keyword.arg == "gpu":
                    found["TRAINER"] = ast.literal_eval(keyword.value)
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "function":
                    for keyword in decorator.keywords:
                        if keyword.arg == "gpu":
                            found[node.name] = ast.literal_eval(keyword.value)
    return found


class CardTests(unittest.TestCase):
    def test_every_function_modal_receives_asks_for_an_l40s_and_nothing_else(self):
        # Read from the source, a function whose card came from shared
        # settings other than the trainers' was never compared at all.
        registered = settings()
        self.assertGreaterEqual(set(registered), {
            "train", "train_mortal", "train_combined", "placement_pretest", "noise_pretest", "arena",
            "duel", "ruler", "distil", "rehead", "smoke"})
        for name, keywords in registered.items():
            with self.subTest(function=name):
                self.assertEqual(keywords.get("gpu"), "L40S")

    def test_every_function_asks_for_an_l40s_and_nothing_else(self):
        found = cards()
        # The trainers' shared settings and every function that names a card:
        # the placement pre-test, the arena, the duel, the ruler, distillation,
        # the head refit and the smoke test.
        self.assertIn("TRAINER", found)
        self.assertGreaterEqual(len(found), 8, found)
        for name, card in found.items():
            with self.subTest(function=name):
                self.assertEqual(card, "L40S")

    def test_the_trainers_take_their_card_from_their_shared_settings(self):
        tree = ast.parse(APP.read_text(encoding="utf-8"))
        trainers = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and any(isinstance(decorator, ast.Call) and any(keyword.arg is None and isinstance(keyword.value, ast.Name)
                                                                   and keyword.value.id == "TRAINER"
                                                                   for keyword in decorator.keywords)
                            for decorator in node.decorator_list)]
        self.assertEqual(sorted(trainers), ["train", "train_combined", "train_mortal"])

    def test_no_function_has_modal_start_a_failed_call_again(self):
        # Retries declared on a function start every failed call of it again
        # on the card, watched or not (see `TRAINER` in the app). A caller
        # that means its call to go on asks for them at the call
        # (`with_options`), which is outside this test, as a card asked for
        # there is.
        registered = settings()
        self.assertGreaterEqual(set(registered), {"train", "train_mortal", "train_combined"})
        for name, keywords in registered.items():
            with self.subTest(function=name):
                self.assertNotIn("retries", keywords)


if __name__ == "__main__":
    unittest.main()
