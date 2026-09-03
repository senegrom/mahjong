"""Scores the same hands twice and reports where the two disagree.

The scorer is the part of these rules with the most places to be quietly
wrong, and a test suite only checks what its author thought of. So this
takes the hands `riichi-cli dump` writes and scores them again with the
MIT-licensed `mahjong` library, which was itself validated against millions
of hands from Tenhou, and prints every disagreement.

    riichi-cli dump --games 100000 --seed 1 > hands.jsonl
    python engine/riichi-cli/differential.py hands.jsonl

Disagreements are not automatically bugs. The two implement different rule
sets, so the library is configured for the EMA rules as far as its options
reach, and the differences that remain are listed by kind so each can be
traced to a sentence in the rulebook or to a fault here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

WINDS = [EAST, SOUTH, WEST, NORTH]
SUIT_BASE = {"m": 0, "p": 36, "s": 72}


class Physical:
    """Hands the library distinct numbers for the four copies of a kind.

    The library works in 136 tiles rather than 34 kinds, so every copy needs
    its own number and no number may be used twice in one hand.
    """

    def __init__(self) -> None:
        self.used: Counter[str] = Counter()

    def take(self, name: str) -> int:
        copy = self.used[name]
        self.used[name] += 1
        if copy > 3:
            raise ValueError(f"a fifth {name}")
        rank = int(name[0])
        suit = name[1]
        if suit == "z":
            return 108 + (rank - 1) * 4 + copy
        return SUIT_BASE[suit] + (rank - 1) * 4 + copy


# The EMA rules, as far as the library's options reach: no red fives, four
# han at thirty minipoints rounds up to a limit hand, a hand counted to
# thirteen han is not a yakuman, and no yakuman is worth double.
EMA = OptionalRules(
    has_open_tanyao=True,
    has_aka_dora=False,
    has_double_yakuman=False,
    kazoe_limit=HandConfig.KAZOE_SANBAIMAN,
    kiriage=True,
)

MELD_KIND = {
    "chi": Meld.CHI,
    "pon": Meld.PON,
    "kan": Meld.KAN,
    "ankan": Meld.KAN,
}


def rescore(record: dict) -> dict | None:
    """What the library makes of one hand, or None if it refuses it."""
    physical = Physical()
    tiles: list[int] = []
    melds: list[Meld] = []

    for meld in record["melds"]:
        members = [physical.take(name) for name in meld["tiles"]]
        tiles.extend(members)
        melds.append(
            Meld(
                meld_type=MELD_KIND[meld["kind"]],
                tiles=members,
                opened=meld["opened"],
            )
        )

    concealed = [physical.take(name) for name in record["concealed"]]
    tiles.extend(concealed)

    # The winning tile has to be one of the tiles in hand, so it is found
    # among them rather than made up again.
    wanted = record["winning"]
    winning = None
    held = list(record["concealed"])
    for index, name in zip(concealed, held):
        if name == wanted:
            winning = index
            break
    if winning is None:
        return None

    indicators = [Physical().take(name) for name in record["dora_indicators"]]
    ura = [Physical().take(name) for name in record["ura_indicators"]]

    config = HandConfig(
        is_tsumo=record["tsumo"],
        is_riichi=record["riichi"] and not record["double_riichi"],
        is_daburu_riichi=record["double_riichi"],
        is_ippatsu=record["ippatsu"],
        is_rinshan=record["after_quad"],
        is_chankan=record["robbing_quad"],
        is_haitei=record["under_the_sea"],
        is_houtei=record["under_the_river"],
        player_wind=WINDS[record["seat"]],
        round_wind=WINDS[record["round"]],
        options=EMA,
    )

    result = HandCalculator().estimate_hand_value(
        tiles,
        winning,
        melds=melds or None,
        dora_indicators=(indicators + ura) or None,
        config=config,
    )
    if result.error:
        return {"error": result.error}
    return {
        "han": result.han,
        "fu": result.fu,
        "yaku": sorted(str(yaku) for yaku in result.yaku),
    }


# Two differences are the rules, not faults, and both are quoted here so a
# reader can check them rather than take this file's word for it.
#
# EMA 2025 section 4.1.1: "A pair of both seat and round winds is worth only
# 2 minipoints." This is new in the 2025 edition, which lists it among its
# changes as "treated as 2 minipoints instead of 4". The library awards 4,
# which is the older convention, and after rounding that shows up as ten
# more minipoints.
#
# EMA 2025 section 4.2: "Yakuman are not cumulative." The library adds them,
# so a hand that is two yakuman under Japanese rules comes back at 26 han
# where this engine stops at 13.
DOUBLE_WIND = "EMA 4.1.1: a pair of both seat and round wind is 2 minipoints, not 4"
YAKUMAN_SUM = "EMA 4.2: yakuman are not cumulative"


def has_double_wind_pair(record: dict) -> bool:
    """Whether the hand's pair is a wind that is both the seat and the round."""
    if record["seat"] != record["round"]:
        return False
    wind = f"{record['seat'] + 1}z"
    return record["concealed"].count(wind) == 2


def compare(path: Path, limit: int | None) -> int:
    agreed = 0
    refused = 0
    expected: Counter[str] = Counter()
    disagreements: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    total = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if limit is not None and total >= limit:
                break
            total += 1
            record = json.loads(line)
            try:
                theirs = rescore(record)
            except (ValueError, KeyError, IndexError) as error:
                disagreements[f"could not be rebuilt: {type(error).__name__}"] += 1
                continue
            if theirs is None:
                refused += 1
                continue
            if "error" in theirs:
                kind = f"the library refused it: {theirs['error']}"
                disagreements[kind] += 1
                examples.setdefault(kind, record)
                continue

            ours_han = record["han"]
            ours_fu = record["fu"]

            # A yakuman is worth a fixed amount, so this engine does not
            # work out minipoints for one and reports none. The library
            # computes them anyway. Neither is wrong and no payment depends
            # on it, so only the han is compared for those hands.
            reached = record.get("limit")
            fu_matters = not (reached == "yakuman" and ours_fu == 0)

            if ours_han == theirs["han"] and (not fu_matters or ours_fu == theirs["fu"]):
                agreed += 1
                continue

            # The two rules above, recognised so what remains is unexplained.
            if (
                ours_han == theirs["han"]
                and theirs["fu"] - ours_fu == 10
                and has_double_wind_pair(record)
            ):
                expected[DOUBLE_WIND] += 1
                continue
            if (
                reached == "yakuman"
                and ours_han == 13
                and theirs["han"] > 13
                and theirs["han"] % 13 == 0
            ):
                expected[YAKUMAN_SUM] += 1
                continue

            if ours_han != theirs["han"]:
                kind = f"han: ours {ours_han}, theirs {theirs['han']}"
            else:
                kind = f"minipoints: ours {ours_fu}, theirs {theirs['fu']}"
            disagreements[kind] += 1
            examples.setdefault(kind, {**record, "theirs": theirs})

    print(f"hands compared: {total}")
    print(f"agreed on han and minipoints: {agreed}")
    print(f"skipped, no winning tile in hand: {refused}")
    if expected:
        print("differed for a rule, not a fault:")
        for rule, count in expected.most_common():
            print(f"  {count:>7}  {rule}")
    disagreed = sum(disagreements.values())
    print(f"unexplained disagreements: {disagreed}")
    if total - refused:
        rate = disagreed / (total - refused)
        print(f"unexplained rate: {rate:.4%}")

    if disagreements:
        print()
        print("by kind, most common first:")
        for kind, count in disagreements.most_common(25):
            print(f"  {count:>7}  {kind}")
        print()
        print("one example of each of the three most common:")
        for kind, _ in disagreements.most_common(3):
            example = examples.get(kind)
            if example is None:
                continue
            print(f"  {kind}")
            print(f"    {json.dumps(example)[:400]}")
    return 0 if disagreed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hands", type=Path, help="the JSON lines riichi-cli dump wrote")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return compare(args.hands, args.limit)


if __name__ == "__main__":
    sys.exit(main())
