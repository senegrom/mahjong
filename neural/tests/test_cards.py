"""Every function of the Modal app runs on an L40S. The owner moved training
to the L40S on 19 October 2026, after it played a generation in 9.5 minutes
against an H100's 9.0 for half the card's price, and asked that nothing
switch back without the owner's word. A function that asks for another card,
or for a list of cards to fall back along, fails here. A card asked for at
the call (`with_options`) is outside this test, as a benchmark of another
card, which the owner asked for of the L4 and the A10G, must be."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

APP = Path(__file__).resolve().parents[1] / "modal_app.py"


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


if __name__ == "__main__":
    unittest.main()
