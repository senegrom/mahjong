"""A step's planes are served faster without a bit of them changing: rows
gathered by slicing, values halved by the encoder's own workers, and planes
encoded this step checked for their columns but not value by value."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import riichi_py
from libriichi.follow import Follower

from neural import zoo
from neural.observe import VERSION, WIDTH, Planes, Views


def rows_by_index(planes, rows):
    """`Planes.rows` as it was before it sliced: the place of every entry
    of the selection in the source, then one gather by those places. Kept
    as the reference the slicing has to reproduce."""
    rows = np.asarray(rows, dtype=np.int64)
    starts = planes.indptr[rows]
    counts = planes.indptr[rows + 1] - starts
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    within = np.arange(total, dtype=np.int64) - np.repeat(indptr[:-1], counts)
    flat = np.repeat(starts, counts) + within
    return Planes(indptr, np.asarray(planes.indices[flat]), np.asarray(planes.values[flat]))


def step_sized(rows=1200, seed=5, trusted=False):
    """Planes of a step's shape: about two thousand entries a row, and a
    few rows with none."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(1400, 2500, size=rows)
    counts[rng.choice(rows, size=rows // 50, replace=False)] = 0
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    indices = rng.integers(0, WIDTH, size=total).astype(np.uint16)
    values = rng.random(total, dtype=np.float32).astype(np.float16)
    return Planes(indptr, indices, values, trusted=trusted)


def deciding(arena, games):
    """The games owing a decision and the player owing each, as self-play
    works them out."""
    seats = np.frombuffer(arena.seats(), dtype=np.uint8)
    live = np.flatnonzero(seats != 0xFF)
    players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
    return live, players[live, seats[live]].astype(np.int64)


def play_on(arena, games):
    """Every deciding seat plays the heuristic player's move."""
    choice = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
    choice[choice == 0xFF] = 0
    arena.step(choice.tolist())


def assert_same(test, new, old):
    """`new` holds the same arrays as `old`, byte for byte, each an array
    of its own rather than a memory map."""
    for name in Planes.ARRAYS:
        got, want = getattr(new, name), getattr(old, name)
        test.assertIs(type(got), np.ndarray, name)
        test.assertEqual(got.dtype, want.dtype, name)
        np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8), name)


class RowsBySlicingTests(unittest.TestCase):
    def test_rows_are_the_index_gathers_bit_for_bit(self):
        planes = step_sized()
        n = len(planes)
        rng = np.random.default_rng(7)
        selections = {
            "sorted": np.sort(rng.choice(n, size=n // 2, replace=False)),
            "unsorted, with repeats": rng.integers(0, n, size=n),
            "every row in order": np.arange(n),
            "one row": np.array([n - 1]),
            "rows with no entries": np.flatnonzero(np.diff(planes.indptr) == 0),
            "nothing": np.array([], dtype=np.int64),
            "a list of ints": [3, 1, 2],
        }
        for name, rows in selections.items():
            with self.subTest(name):
                new = planes.rows(rows)
                assert_same(self, new, rows_by_index(planes, rows))
                # Copies, not views: a record holding a view would keep the
                # whole step's buffer alive until the round is gathered.
                self.assertFalse(np.shares_memory(new.indices, planes.indices))
                self.assertFalse(np.shares_memory(new.values, planes.values))

    def test_every_bit_pattern_of_a_value_is_copied_as_it_is(self):
        # All 65,536 float16 patterns, NaN payloads, infinities, signed
        # zeros and subnormals among them, in 256 rows of 256.
        bits = np.arange(1 << 16).astype(np.uint16)
        planes = Planes(np.arange(0, (1 << 16) + 1, 256, dtype=np.int64),
                        (bits % WIDTH).astype(np.uint16), bits.view(np.float16))
        rows = np.random.default_rng(3).permutation(len(planes))
        assert_same(self, planes.rows(rows), rows_by_index(planes, rows))

    def test_memory_maps_give_the_same_rows_as_arrays_of_their_own(self):
        planes = step_sized(rows=600)
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            planes.save(Path(folder), "observations")
            mapped = Planes.load(Path(folder), "observations", mmap=True)
            self.assertIsInstance(mapped.indices, np.memmap)
            for rows in (np.sort(rng.choice(600, size=300, replace=False)),
                         rng.permutation(600), np.array([], dtype=np.int64)):
                with self.subTest(rows=len(rows)):
                    assert_same(self, mapped.rows(rows), rows_by_index(mapped, rows))
                    assert_same(self, mapped.rows(rows), planes.rows(rows))
            # Windows will not remove a file that is still mapped.
            del mapped

    def test_rows_outside_the_planes_are_refused_as_before(self):
        planes = step_sized(rows=10)
        for rows, error in (([10], IndexError), ([-1], ValueError)):
            with self.subTest(rows=rows):
                with self.assertRaises(error):
                    rows_by_index(planes, rows)
                with self.assertRaises(error):
                    planes.rows(rows)


class TrustedPlanesTests(unittest.TestCase):
    def one_entry(self, value, column=0, trusted=False):
        return Planes(np.array([0, 1], dtype=np.int64), np.array([column], dtype=np.uint16),
                      np.array([value], dtype=np.float16), trusted=trusted)

    def test_planes_encoded_this_step_skip_the_value_check_and_only_that(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.one_entry(value).dense("cpu")
                dense = self.one_entry(value, trusted=True).dense("cpu").reshape(-1)
                self.assertFalse(np.isfinite(dense[0].item()))
                # Asked for outright, validation is the whole check.
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.one_entry(value, trusted=True).validate()
        # A column past the row's end would be written into the next row's
        # planes, so the bound is checked whoever wrote the planes.
        for trusted in (False, True):
            with self.subTest(trusted=trusted):
                with self.assertRaisesRegex(ValueError, "bounds"):
                    self.one_entry(1.0, column=WIDTH, trusted=trusted).dense("cpu")

    def test_trust_follows_rows_and_slices_not_stacking_or_saving(self):
        planes = step_sized(rows=20, trusted=True)
        self.assertTrue(planes.rows([3, 1]).trusted)
        self.assertTrue(planes.slice(2, 9).trusted)
        self.assertFalse(step_sized(rows=20).rows([3, 1]).trusted)
        self.assertFalse(Planes.cat([planes.rows([0]), planes.rows([1])]).trusted)
        self.assertFalse(Planes.empty().trusted)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            planes.save(Path(folder), "observations")
            self.assertFalse(Planes.load(Path(folder), "observations", mmap=False).trusted)

    def test_views_serve_this_steps_encoding_trusted_and_unchanged(self):
        """Prepared for the step or encoded fresh, the planes a table's
        views serve are trusted, and are the follower's float32 encoding
        halved, bit for bit."""
        games = 6
        arena = riichi_py.Arena(games=games, seed=81)
        views = Views(arena, games)
        for _ in range(40):
            views.advance()
            live, players = deciding(arena, games)
            views.prepare(live, players)
            follower = views.observer.follower
            indptr, indices, values, masks = follower.encode(list(zip(live.tolist(),
                                                                      players.tolist())))
            reference = Planes(np.asarray(indptr, dtype=np.int64), indices,
                               np.asarray(values, dtype=np.float16))
            served = (views.sparse_and_masks(live, players),
                      views.sparse_and_masks(live, players, fresh=True))
            for planes, own in served:
                self.assertTrue(planes.trusted)
                assert_same(self, planes, reference)
                np.testing.assert_array_equal(own, masks)
            play_on(arena, games)


class HalvedByTheEncoderTests(unittest.TestCase):
    def test_the_encoders_float16_is_numpys_rounding_of_its_float32(self):
        """Over real positions from whole hands: the values the encoder
        halves itself are the bits numpy makes of its float32 values,
        everything else it returns is unchanged, and the planes take the
        halved values as they are."""
        games = 12
        arena = riichi_py.Arena(games=games, seed=20261004)
        follower = Follower(games, VERSION)
        rows = entries = rounded = previews = 0
        for _ in range(300):
            if arena.all_finished():
                break
            follower.feed(arena.mjai_all())
            live, players = deciding(arena, games)
            who = list(zip(live.tolist(), players.tolist()))
            single = follower.encode(who)
            halved = follower.encode(who, half=True)
            self.assert_halved(single, halved)
            # Neither converted again nor copied on the way in.
            self.assertIs(Planes.from_follower(*halved[:3]).values, halved[2])
            rows += len(who)
            entries += len(single[2])
            exact = np.asarray(single[2], dtype=np.float16).astype(np.float32)
            rounded += int(np.count_nonzero(exact != single[2]))
            # And for the view a reach is previewed in.
            ready = [pair for pair, mask in zip(who, single[3]) if mask[zoo.MORTAL_RIICHI]]
            if ready:
                self.assert_halved(follower.encode(ready, after_reach=True),
                                   follower.encode(ready, after_reach=True, half=True))
                previews += len(ready)
            play_on(arena, games)
        self.assertGreater(rows, 2000)
        self.assertGreater(entries, 2_000_000)
        self.assertGreater(previews, 0)
        # Values that float16 cannot hold were among them, so the rounding
        # itself was compared, not only values it keeps as they are.
        self.assertGreater(rounded, 0)

    def assert_halved(self, single, halved):
        indptr, indices, values, masks = halved
        self.assertEqual(values.dtype, np.float16)
        self.assertEqual(single[2].dtype, np.float32)
        for got, want in ((indptr, single[0]), (indices, single[1]), (masks, single[3])):
            self.assertEqual(got.dtype, want.dtype)
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(values.view(np.uint16),
                                      np.asarray(single[2], dtype=np.float16).view(np.uint16))


if __name__ == "__main__":
    unittest.main()
