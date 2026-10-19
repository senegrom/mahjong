"""Records the encoder's golden fixture and blesses its digests.

    python -m neural.tests.encoder_golden_fixture record [games] [seed]
    python -m neural.tests.encoder_golden_fixture bless

`record` plays whole games with the engine's heuristic player in every seat
and writes `engine/libriichi/testdata/encoder-golden/steps.jsonl.gz`: one
line a step, holding every game's new mjai events ("feed", as [game,
[lines]]) and the players self-play would then have the follower encode
("who", as [game, player, preview], where preview is 1 for a player our
rules let declare riichi, whose reach self-play previews). A last line
feeds the events after the final step and asks for nobody.

`bless` writes `follower.digests` beside it, the digests the libriichi on
the path gives for those steps (see `test_encoder_golden`). The crate's own
digests are blessed by its test: `ENCODER_GOLDEN_BLESS=1 cargo test
--no-default-features --lib state::golden` in engine/libriichi.

Bless only from a build of a commit whose planes are to be the reference,
and never to make a change that is meant to be exact pass: that is the very
thing the digests are there to catch.
"""
from __future__ import annotations

import gzip
import json
import sys

import numpy as np

from neural.tests import test_encoder_golden as golden

GAMES, SEED = 16, 20261019
# Our engine's actions: 34 discards, then the 34 riichi discards.
RIICHI_DISCARD, TSUMO = 34, 68


def record(games: int, seed: int) -> None:
    import riichi_py

    arena = riichi_py.Arena(games=games, seed=seed)
    steps = []
    while True:
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        lines = arena.mjai_all()
        feed = [[game, game_lines] for game, game_lines in enumerate(lines) if game_lines]
        if not live.any():
            steps.append({"feed": feed, "who": []})
            break
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, riichi_py.ACTIONS)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        index = np.flatnonzero(live)
        deciding = players[index, seats[index]]
        preview = mask[index, RIICHI_DISCARD:TSUMO].any(axis=1)
        steps.append({"feed": feed, "who": [[int(g), int(p), int(r)]
                                            for g, p, r in zip(index, deciding, preview)]})
        advice = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
        advice[~live] = 0
        arena.step(advice.tolist())
    # Written without a timestamp, so that recording the same games again
    # gives the same bytes.
    with gzip.GzipFile(golden.STEPS, "wb", mtime=0) as raw:
        for step in steps:
            raw.write((json.dumps(step, separators=(",", ":")) + "\n").encode("utf-8"))
    rows = sum(len(step["who"]) for step in steps)
    previews = sum(r for step in steps for _g, _p, r in step["who"])
    print(f"{games} games at seed {seed}: {len(steps)} steps, {rows} rows, {previews} previews, "
          f"{golden.STEPS.stat().st_size} bytes")


def bless() -> None:
    import libriichi

    digests = golden.hand_digests(golden.load_steps())
    with open(golden.DIGESTS, "w", encoding="utf-8", newline="\n") as out:
        out.write("# game hand digest: Follower.encode of every row of the hand, float32 and\n"
                  "# float16, reach previews included (see neural/tests/test_encoder_golden.py)\n")
        for (game, hand), digest in digests.items():
            out.write(f"{game} {hand} {digest}\n")
    print(f"{len(digests)} hands blessed from {libriichi.__file__}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "record":
        record(int(sys.argv[2]) if len(sys.argv) > 2 else GAMES,
               int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
    elif command == "bless":
        bless()
    else:
        sys.exit(__doc__)
