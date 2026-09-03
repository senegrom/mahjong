"""Reads an mjai log back and checks it describes a possible game.

The Rust tests check the events before they are written; this checks the
written text, so a mistake in the JSON itself cannot slip through. It reads
from a file or standard input:

    riichi-cli log --seed 42 --games 3 | python engine/riichi-cli/check-log.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SUITS = "mps"
HONOURS = ["E", "S", "W", "N", "P", "F", "C"]
TILES = [f"{rank}{suit}" for suit in SUITS for rank in range(1, 10)] + HONOURS


def check(lines: list[str]) -> dict[str, int]:
    """Walks the log, raising on anything that could not have happened."""
    hands: list[Counter] = [Counter() for _ in range(4)]
    melded: list[Counter] = [Counter() for _ in range(4)]
    pond: list[list[str]] = [[] for _ in range(4)]
    indicators: list[str] = []
    scores = [0, 0, 0, 0]
    counts = Counter()
    open_hand = False
    games = 0

    for number, line in enumerate(lines, start=1):
        where = f"line {number}"
        event = json.loads(line)
        kind = event["type"]
        counts[kind] += 1

        for tile in tiles_in(event):
            assert tile in TILES, f"{where}: {tile} is not a tile"

        if kind == "start_game":
            assert len(event["names"]) == 4, where
            games += 1
        elif kind == "start_kyoku":
            assert not open_hand, f"{where}: a hand was already open"
            open_hand = True
            hands = [Counter(hand) for hand in event["tehais"]]
            for who, hand in enumerate(hands):
                assert sum(hand.values()) == 13, f"{where}: player {who} was not dealt thirteen"
            melded = [Counter() for _ in range(4)]
            pond = [[] for _ in range(4)]
            indicators = [event["dora_marker"]]
            scores = list(event["scores"])
            assert 1 <= event["kyoku"] <= 4, where
            assert event["bakaze"] in ("E", "S", "W", "N"), where
        elif kind == "tsumo":
            hands[event["actor"]][event["pai"]] += 1
        elif kind == "dahai":
            actor, tile = event["actor"], event["pai"]
            assert hands[actor][tile] > 0, f"{where}: player {actor} discarded {tile} unheld"
            hands[actor][tile] -= 1
            pond[actor].append(tile)
        elif kind in ("chi", "pon", "daiminkan"):
            actor, target, tile = event["actor"], event["target"], event["pai"]
            assert actor != target, f"{where}: a player cannot claim their own discard"
            assert pond[target] and pond[target][-1] == tile, f"{where}: {tile} was not on the table"
            pond[target].pop()
            for member in event["consumed"]:
                assert hands[actor][member] > 0, f"{where}: {member} was not held"
                hands[actor][member] -= 1
                melded[actor][member] += 1
            melded[actor][tile] += 1
            wanted = {"chi": 2, "pon": 2, "daiminkan": 3}[kind]
            assert len(event["consumed"]) == wanted, where
        elif kind == "kakan":
            actor, tile = event["actor"], event["pai"]
            assert hands[actor][tile] > 0, f"{where}: {tile} was not held"
            assert melded[actor][tile] >= 3, f"{where}: no triplet of {tile} to add to"
            hands[actor][tile] -= 1
            melded[actor][tile] += 1
        elif kind == "ankan":
            actor = event["actor"]
            assert len(event["consumed"]) == 4, where
            for member in event["consumed"]:
                assert hands[actor][member] > 0, f"{where}: {member} was not held"
                hands[actor][member] -= 1
                melded[actor][member] += 1
        elif kind == "dora":
            indicators.append(event["dora_marker"])
            assert len(indicators) <= 5, f"{where}: a sixth indicator"
        elif kind in ("reach", "reach_accepted"):
            assert 0 <= event["actor"] <= 3, where
        elif kind in ("hora", "ryukyoku"):
            assert sum(event["deltas"]) + sum(scores) == sum(event["scores"]), (
                f"{where}: the points do not add up"
            )
            scores = list(event["scores"])
        elif kind == "end_kyoku":
            assert open_hand, f"{where}: no hand was open"
            open_hand = False
            seen = Counter()
            for who in range(4):
                seen.update(hands[who])
                seen.update(melded[who])
                seen.update(pond[who])
            seen.update(indicators)
            for tile, count in seen.items():
                assert count <= 4, f"{where}: {count} copies of {tile}"
        elif kind == "end_game":
            assert not open_hand, f"{where}: the game ended mid-hand"
        else:
            raise AssertionError(f"{where}: unknown event {kind}")

    assert not open_hand, "the log ends inside a hand"
    assert games >= 1, "no game was logged"
    return counts


def tiles_in(event: dict) -> list[str]:
    """Every tile named by an event."""
    found = []
    for key in ("pai", "dora_marker"):
        if key in event:
            found.append(event[key])
    for key in ("consumed", "uradora_markers"):
        found.extend(event.get(key, []))
    for hand in event.get("tehais", []):
        found.extend(hand)
    return found


def main() -> int:
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    lines = [line for line in text.splitlines() if line.strip()]
    counts = check(lines)
    print(f"{len(lines)} events over {counts['start_game']} game(s), all consistent")
    for kind, count in counts.most_common():
        print(f"  {kind:16} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
