"""The follower encodes recorded games exactly as it always has.

Mortal's encoder (`engine/libriichi`, reached through
`libriichi.follow.Follower`) is the network's whole view of the table, and
its efficiency lookahead is nearly all of what encoding costs, so it is the
code most worth making faster and the code where a faster version must not
change a bit. The fixture in `engine/libriichi/testdata/encoder-golden` holds
whole games the engine's heuristic player played at a fixed seed, kept as
self-play feeds them to the follower: step by step, every game's new mjai
events and the players then asked for, with whether each may declare riichi
(self-play previews those players' reach). Every hand's encodings -- float32
and float16, the reach previews included -- are digested here and compared
with the digests the encoder gave when the fixture was blessed
(`follower.digests`), so a change meant to be exact has to reproduce the old
bytes, not merely pass the other tests. The crate's own `state::golden` test
replays the same games through `PlayerState` and also holds the lookahead's
own tables to their old values.

The digests were made on Windows and are checked on Linux. Apart from +, -,
*, / and comparisons, which IEEE 754 fixes, the only arithmetic in the
planes of Mortal's version 4 is the river's decay, exp(-0.2 k) for whole k,
and every libm the crate is built with rounds those to the same float32:
`state::golden` checks this platform's against the correctly rounded values,
so a mismatch there names the platform rather than the encoder.

To record the games again, or to bless new digests after a change that is
meant to change the planes, see `encoder_golden_fixture.py`.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from libriichi.follow import Follower

FIXTURE = Path(__file__).resolve().parents[2] / "engine" / "libriichi" / "testdata" / "encoder-golden"
STEPS = FIXTURE / "steps.jsonl.gz"
DIGESTS = FIXTURE / "follower.digests"
# Mortal's observation version 4, the one the trainer encodes
# (`neural.observe.VERSION`); not imported, so that this needs no torch.
VERSION = 4
# A prefix of the games, encoded again with one rayon thread in a process of
# its own: the global pool is sized once per process.
ONE_THREAD_GAMES = 2


def load_steps(path: Path = STEPS) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def hand_digests(steps: list[dict], games: int | None = None) -> dict[tuple[int, int], str]:
    """One digest per hand of each game, of every row the follower encodes
    in it, in order: the plain view as float32 and as float16, and the
    view with the reach previewed for a player who may declare, likewise.
    Each row is digested as its own slice of the batch, with its length,
    so the digests do not depend on which games shared a step. `games`
    keeps only the first that many games' rows."""
    total = 1 + max(game for step in steps for game, _lines in step["feed"])
    follower = Follower(total, VERSION)
    hands = [-1] * total
    digests: dict[tuple[int, int], hashlib.blake2b] = {}
    for step in steps:
        lines = [[] for _ in range(total)]
        for game, game_lines in step["feed"]:
            lines[game] = game_lines
            hands[game] += sum(json.loads(line)["type"] == "start_kyoku" for line in game_lines)
        follower.feed(lines)
        who = [(game, player) for game, player, _preview in step["who"]
               if games is None or game < games]
        ready = [(game, player) for game, player, preview in step["who"]
                 if preview and (games is None or game < games)]
        for pairs, after_reach in ((who, False), (ready, True)):
            if not pairs:
                continue
            for half in (False, True):
                indptr, indices, values, masks = follower.encode(pairs, after_reach=after_reach, half=half)
                for row, (game, player) in enumerate(pairs):
                    start, end = int(indptr[row]), int(indptr[row + 1])
                    digest = digests.setdefault((game, hands[game]), hashlib.blake2b(digest_size=8))
                    digest.update(f"{player} {after_reach} {values.dtype.str} {end - start} ".encode())
                    digest.update(np.ascontiguousarray(indices[start:end]).tobytes())
                    digest.update(np.ascontiguousarray(values[start:end]).tobytes())
                    digest.update(np.ascontiguousarray(masks[row]).tobytes())
    return {hand: digest.hexdigest() for hand, digest in sorted(digests.items())}


def read_digests(path: Path = DIGESTS) -> dict[tuple[int, int], str]:
    digests = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            game, hand, digest = line.split()
            digests[int(game), int(hand)] = digest
    return digests


def mismatches(got: dict, want: dict) -> list[str]:
    return [f"game {game} hand {hand}: {got.get((game, hand))} where it was {want.get((game, hand))}"
            for game, hand in sorted(set(got) | set(want)) if got.get((game, hand)) != want.get((game, hand))]


class EncoderGoldenTests(unittest.TestCase):
    def test_the_recorded_games_encode_as_they_always_have(self):
        steps = load_steps()
        rows = sum(len(step["who"]) for step in steps)
        previews = sum(preview for step in steps for _game, _player, preview in step["who"])
        # What the fixture has to exercise to guard anything: the crate's
        # golden test counts the costly kinds of rows among these.
        self.assertGreater(rows, 10_000)
        self.assertGreater(previews, 100)
        want = read_digests()
        got = hand_digests(steps)
        self.assertGreater(len(want), 150)
        wrong = mismatches(got, want)
        self.assertFalse(wrong, f"{len(wrong)} of {len(want)} hands encode differently, first "
                                f"{wrong[:5]}; a change meant to be exact is not")

    def test_one_thread_encodes_as_many_do(self):
        """The follower encodes each row on its own and collects them in
        order, so the planes cannot depend on how many threads share the
        work; checked in a process of its own, since rayon sizes its pool
        once a process."""
        script = (
            "import json, sys\n"
            "from neural.tests import test_encoder_golden as golden\n"
            f"got = golden.hand_digests(golden.load_steps(), games={ONE_THREAD_GAMES})\n"
            "print(json.dumps([[game, hand, digest] for (game, hand), digest in got.items()]))\n"
        )
        environment = dict(os.environ, RAYON_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env=environment, cwd=FIXTURE.parents[3], check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        got = {(game, hand): digest for game, hand, digest in json.loads(result.stdout)}
        want = {hand: digest for hand, digest in read_digests().items() if hand[0] < ONE_THREAD_GAMES}
        self.assertGreater(len(want), 10)
        self.assertEqual(mismatches(got, want), [])


if __name__ == "__main__":
    unittest.main()
