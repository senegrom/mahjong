"""Put the page's own decisions back to the network and see if they agree.

`web/scripts/network-truth-check.mjs` records every trained decision as the
page made it: the planes it sent, the legality mask, the move it took and
its weights. This replays those planes through the same checkpoint and names
the move the network would take. Anything that goes wrong between the two --
the wrong graph, the wrong half of a fusion, a mask that never reached the
head, an observation the browser builds differently -- shows up as a
disagreement on real positions rather than in somebody's game.

The page asks a question Trained AI hesitates over again of the renamed game
(`neural.suits`, web/src/lib/suits.js), so a decision whose weights hesitate
is followed in the recording by the renamed game's. That one must be the
renamed game: its moves the plain ones renamed, its planes the plain planes
renamed on every row but the efficiency lookahead's (`test_suits.EXACT`), and
not the plain planes themselves; and the mean of the two answers must play
the move the mean of the network's two answers plays. A recording made
before the page averaged, without weights, is replayed plain.

    python -m neural.tests.replay_browser_decisions network-truth.json checkpoint.pt
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

POSITIONS = 34


def planes_of(decision: dict) -> np.ndarray:
    """The planes the page sent, (planes, 34)."""
    return np.frombuffer(base64.b64decode(decision["planes"]), dtype=np.float32).reshape(-1, POSITIONS)


def main() -> None:
    recorded = Path(sys.argv[1])
    checkpoint = Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    payload = json.loads(recorded.read_text(encoding="utf-8"))
    decisions = payload["decisions"]
    if not decisions:
        raise SystemExit("the recording holds no decisions")

    from neural import suits, zoo
    from neural.tests.test_suits import EXACT, renamed_planes

    net = zoo.unwrap(zoo.load_player(checkpoint, device))
    agreed = forced = 0
    disagreements = []
    answers = []
    for index, decision in enumerate(decisions):
        planes = torch.from_numpy(planes_of(decision)[None].copy()).to(device)
        mask = np.asarray(decision["mask"], dtype=bool)
        legal = torch.from_numpy(mask.reshape(1, -1).copy()).to(device)
        with torch.no_grad():
            logits, _value, _hands = net.everything(planes, legal)
        said = np.where(mask, logits[0].float().cpu().numpy(), -np.inf)
        answers.append(said)
        best = int(said.argmax())
        took = int(decision["action"])
        if mask.sum() <= 1:
            forced += 1
        if best == took:
            agreed += 1
        else:
            order = np.argsort(said)[::-1]
            place = int(np.nonzero(order == took)[0][0]) + 1
            disagreements.append({"decision": index, "page": took, "network": best,
                                  "page_move_ranked": place, "legal": int(mask.sum())})

    # The questions asked again of the renamed game, each right after its own.
    plain = again = changed = 0
    faults, unlike = [], []
    index = 0
    while index < len(decisions):
        decision = decisions[index]
        weights = decision.get("weights")
        plain += 1
        if weights is None or max(weights) >= suits.GATE:
            index += 1
            continue
        if index + 1 == len(decisions):
            # The recording stopped while its renamed game was asked.
            break
        twin = decisions[index + 1]
        mask = np.asarray(decision["mask"], dtype=bool)
        mine, theirs = planes_of(decision), planes_of(twin)
        if not np.array_equal(np.asarray(twin["mask"], dtype=bool), suits.swap_moves(mask)):
            faults.append({"decision": index + 1, "fault": "its moves are not the question's renamed"})
        elif not np.array_equal(renamed_planes(mine)[EXACT], theirs[EXACT]):
            faults.append({"decision": index + 1, "fault": "its planes are not the question's renamed"})
        elif np.array_equal(mine, theirs):
            faults.append({"decision": index + 1, "fault": "its planes are the question's own"})
        again += 1
        page = np.where(mask, np.sqrt(np.asarray(weights, dtype=np.float64)
                                      * suits.swap_moves(np.asarray(twin["weights"], dtype=np.float64))), -np.inf)
        mean = (answers[index] + suits.swap_moves(answers[index + 1])) / 2
        page_move, network_move = int(page.argmax()), int(mean.argmax())
        changed += network_move != int(answers[index].argmax())
        if page_move != network_move:
            unlike.append({"decision": index, "page": page_move, "network": network_move})
        index += 2

    print(json.dumps({
        "decisions": len(decisions), "same_move": agreed,
        "agreement": round(agreed / len(decisions), 4),
        "forced": forced,
        "disagreements": disagreements[:10],
        "questions": plain, "asked_again": again,
        "share_asked_again": round(again / plain, 4) if plain else None,
        "moves_changed": changed,
        "renaming_faults": faults[:10],
        "means_unlike": unlike[:10],
    }, indent=1))
    if agreed < len(decisions):
        raise SystemExit(f"the page and the network differ on {len(decisions) - agreed} decisions")
    if faults:
        raise SystemExit(f"{len(faults)} questions asked again are not the renamed game")
    if unlike:
        raise SystemExit(f"the page's mean and the network's differ on {len(unlike)} questions")


if __name__ == "__main__":
    main()
