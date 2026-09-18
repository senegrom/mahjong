"""Put the page's own decisions back to the network and see if they agree.

`web/scripts/network-truth-check.mjs` records every trained decision as the
page made it: the planes it sent, the legality mask, and the move it took.
This replays those planes through the same checkpoint and names the move the
network would take. Anything that goes wrong between the two -- the wrong
graph, the wrong half of a fusion, a mask that never reached the head, an
observation the browser builds differently -- shows up as a disagreement on
real positions rather than in somebody's game.

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


def main() -> None:
    recorded = Path(sys.argv[1])
    checkpoint = Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    payload = json.loads(recorded.read_text(encoding="utf-8"))
    decisions = payload["decisions"]
    if not decisions:
        raise SystemExit("the recording holds no decisions")

    from neural import contract, zoo

    net = contract.unwrap(zoo.load_player(checkpoint, device))
    agreed = forced = 0
    disagreements = []
    for index, decision in enumerate(decisions):
        flat = np.frombuffer(base64.b64decode(decision["planes"]), dtype=np.float32)
        planes = torch.from_numpy(flat.reshape(1, -1, POSITIONS).copy()).to(device)
        mask = np.asarray(decision["mask"], dtype=bool)
        legal = torch.from_numpy(mask.reshape(1, -1).copy()).to(device)
        with torch.no_grad():
            logits, _value, _hands = net.everything(planes, legal)
        said = logits[0].float().cpu().numpy()
        best = int(np.where(mask, said, -np.inf).argmax())
        took = int(decision["action"])
        if mask.sum() <= 1:
            forced += 1
        if best == took:
            agreed += 1
        else:
            order = np.argsort(np.where(mask, said, -np.inf))[::-1]
            place = int(np.nonzero(order == took)[0][0]) + 1
            disagreements.append({"decision": index, "page": took, "network": best,
                                  "page_move_ranked": place, "legal": int(mask.sum())})
    print(json.dumps({
        "decisions": len(decisions), "same_move": agreed,
        "agreement": round(agreed / len(decisions), 4),
        "forced": forced,
        "disagreements": disagreements[:10],
    }, indent=1))
    if agreed < len(decisions):
        raise SystemExit(f"the page and the network differ on {len(decisions) - agreed} decisions")


if __name__ == "__main__":
    main()
