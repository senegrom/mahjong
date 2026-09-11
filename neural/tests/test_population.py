"""Who sits at the tables, and whether the round can say so afterwards.

    python -m unittest neural.tests.test_population -v
"""

from __future__ import annotations

import unittest

import numpy as np

from neural import population


class TheRosterIsWhatItSaysItIs(unittest.TestCase):
    def test_the_fine_tuned_mortal_is_a_fixed_reference(self):
        """It is the strongest player here and the fusion is built around
        it, so a round that never meets it cannot say whether the fusion
        has learned anything the Mortal did not already know."""
        names = [member.name for member in population.REFERENCES]
        self.assertIn("mortal-run/latest", names)
        mortal = next(m for m in population.REFERENCES if m.name == "mortal-run/latest")
        self.assertEqual(mortal.role, "reference")
        self.assertGreaterEqual(
            mortal.weight,
            max(m.weight for m in population.REFERENCES),
            "the hardest opponent should be met at least as often as the others",
        )

    def test_the_published_mortal_is_kept_as_an_outside_yardstick(self):
        names = [member.name for member in population.REFERENCES]
        self.assertIn("zoo/mortal_298k", names)

    def test_the_champion_outweighs_the_rest(self):
        roster = population.Population.around(
            champion="leashed-run/champion", recent=["a", "b"], older=["c"]
        )
        champion = next(m for m in roster.members if m.role == "champion")
        for member in roster.members:
            if member.role in {"recent", "older"}:
                self.assertGreater(champion.weight, member.weight)

    def test_the_roster_describes_itself_for_the_log(self):
        roster = population.Population.around(champion="x", recent=["y"])
        rows = roster.describe()
        self.assertEqual({row["name"] for row in rows} & {"x", "y"}, {"x", "y"})
        self.assertAlmostEqual(sum(row["share"] for row in rows), 1.0, places=2)
        for row in rows:
            self.assertTrue(row["note"], f"{row['name']} says nothing about itself")


class SeatingIsProportional(unittest.TestCase):
    def test_a_share_of_zero_seats_nobody(self):
        roster = population.Population.around(champion="x")
        seated = roster.seat(200, 0.0, np.random.default_rng(0))
        self.assertTrue(np.all(seated == -1))

    def test_members_are_drawn_by_their_weight(self):
        roster = population.Population(
            members=[
                population.Member("heavy", "reference", "seated often", 3.0),
                population.Member("light", "reference", "seated seldom", 1.0),
            ]
        )
        seated = roster.seat(20_000, 1.0, np.random.default_rng(7))
        heavy = float((seated == 0).mean())
        self.assertAlmostEqual(heavy, 0.75, delta=0.02)

    def test_the_share_decides_how_many_tables_have_anyone_else(self):
        roster = population.Population.around(champion="x")
        seated = roster.seat(20_000, 0.25, np.random.default_rng(1))
        self.assertAlmostEqual(float((seated >= 0).mean()), 0.25, delta=0.02)


class MatchupsAreKeptApart(unittest.TestCase):
    def test_each_opponent_is_reported_on_its_own(self):
        members = [
            population.Member("strong", "reference", "hard", 1.0),
            population.Member("weak", "older", "easy", 1.0),
        ]
        seated = np.array([0, 0, 1, 1, -1, -1])
        # Beaten by the strong one, beating the weak one, level alone.
        placements = np.array([3.0, 3.0, 2.0, 2.0, 2.5, 2.5])
        rows = population.matchups(seated, placements, members)
        got = {row["name"]: row["placement"] for row in rows}
        self.assertEqual(got["strong"], 3.0)
        self.assertEqual(got["weak"], 2.0)
        self.assertEqual(got["itself"], 2.5)

    def test_an_opponent_never_met_is_left_out_rather_than_reported_as_level(self):
        members = [population.Member("absent", "reference", "never seated", 1.0)]
        rows = population.matchups(np.array([-1, -1]), np.array([2.5, 2.5]), members)
        self.assertEqual([row["name"] for row in rows], ["itself"])

    def test_the_rows_are_not_averaged_together(self):
        """The whole point: a mean of these would hide specialisation."""
        members = [
            population.Member("a", "reference", "", 1.0),
            population.Member("b", "recent", "", 1.0),
        ]
        rows = population.matchups(
            np.array([0, 1]), np.array([3.0, 2.0]), members
        )
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["placement"], rows[1]["placement"])



class ARoundSaysWhoItPlayed(unittest.TestCase):
    """The end-to-end promise: seat somebody, and the batch can name them."""

    def test_the_batch_reports_a_row_for_each_opponent_met(self):
        from neural import selfplay
        from neural.tests.test_selfplay_contract import FirstLegal

        roster = population.Population(
            members=[
                population.Member("stand-in", "reference", "a stand-in", 1.0),
            ]
        )
        batch = selfplay.play(
            FirstLegal(),
            games=8,
            seed=99,
            device="cpu",
            opponents=[FirstLegal()],
            opponent_share=0.5,
            population=roster,
        )
        self.assertIsNotNone(batch.seated)
        self.assertEqual(len(batch.seated), 8)
        names = {row["name"] for row in batch.matchups}
        self.assertIn("stand-in", names, f"nobody was reported: {batch.matchups}")
        self.assertIn("itself", names, "the tables it held alone are a row too")
        for row in batch.matchups:
            self.assertGreater(row["games"], 0)
            self.assertTrue(1.0 <= row["placement"] <= 4.0, row)

    def test_without_a_roster_nothing_is_claimed(self):
        from neural import selfplay
        from neural.tests.test_selfplay_contract import FirstLegal

        batch = selfplay.play(FirstLegal(), games=4, seed=99, device="cpu")
        self.assertEqual(batch.matchups, [])

if __name__ == "__main__":
    unittest.main()
