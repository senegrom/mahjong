"""Create tiny real exports and ordinary/two-stage-riichi browser inputs.

Used by CI with the shipped reduced WASM runtime. No user checkpoint, cloud
service or production model is downloaded, modified or promoted.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import torch

from neural import combined, export, model, mortal_model
from neural.checkpoints import atomic_save
from neural.observe import Planes


def reach_positions():
    from libriichi.follow import Follower
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p"]
    kinds = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
    pool = [tile for tile in kinds for _ in range(4)]
    for tile in hand + ["9m", "9s"]:
        pool.remove(tile)
    hands = [hand] + [pool[i * 13:(i + 1) * 13] for i in range(3)]
    follower = Follower(1, 4)
    follower.feed([[json.dumps(event) for event in (
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
         "kyotaku": 0, "oya": 0, "scores": [25000] * 4, "dora_marker": "9m", "tehais": hands},
        {"type": "tsumo", "actor": 0, "pai": "9s"},
    )]])
    ptr, col, values, masks = follower.encode([(0, 0)])
    before = Planes.from_follower(ptr, col, values).dense("cpu")
    first = torch.as_tensor(np.asarray(masks).copy(), dtype=torch.bool)
    assert first[0, 37], "fixture must actually permit riichi"
    follower.tell(0, 0, json.dumps({"type": "reach", "actor": 0}))
    ptr, col, values, masks = follower.encode([(0, 0)])
    after = Planes.from_follower(ptr, col, values).dense("cpu")
    second = torch.as_tensor(np.asarray(masks).copy(), dtype=torch.bool)
    assert second[0, :34].sum() >= 2 and not second[0, 34:].any()
    return torch.cat([before, after]), torch.cat([first, second])


def main(folder: Path):
    import onnx
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(81)
    ordinary, legal = export.validation_positions()
    reach, reach_legal = reach_positions()
    planes = torch.cat([ordinary[:4], reach])
    masks = torch.cat([legal[:4], reach_legal])
    records = []
    for fused in (False, True):
        name = "fusion" if fused else "standalone"
        ours = model.PolicyValueNet(8, 1, actions=46)
        net = combined.Combined(ours, mortal_model.build(16, 1)) if fused else ours
        if fused:
            net.mortal_config = {"resnet": {"conv_channels": 16, "num_blocks": 1}, "control": {"version": 4}}
            payload = net.state()
        else:
            payload = {"model": net.state_dict(), **net.payload_fields()}
        checkpoint = folder / (name + ".pt")
        atomic_save({**payload, "generation": 0}, checkpoint)
        loaded, payload = export.load_network(checkpoint)
        export.validate_browser_model(loaded, payload)
        wrapped = export.playable(loaded).eval()
        path = folder / (name + ".onnx")
        names = ["planes", "legal"] if fused else ["planes"]
        example = (planes[:1], masks[:1].float()) if fused else (planes[:1],)
        torch.onnx.export(wrapped, example, str(path), input_names=names,
                          output_names=list(export.OUTPUTS),
                          dynamic_axes={key: {0: "batch"} for key in (*names, *export.OUTPUTS)},
                          opset_version=17, dynamo=False)
        # Small freshly initialized networks share many equal constants. Remove
        # internal aliases with the exporter's own cleanup, never graph outputs.
        graph = onnx.load(str(path))
        export.without_identities(graph)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        export.check_operators(path)
        export.check_runs(path, wrapped, planes, masks)
        with torch.no_grad():
            for row in range(len(planes)):
                allowed = masks[row:row + 1]
                args = (planes[row:row + 1], allowed.float()) if fused else (planes[row:row + 1],)
                policy, value, hands = wrapped(*args)
                records.append({"model": path.name, "takes_legal": fused,
                                "stage": "riichi-discard" if row == len(planes) - 1 else "riichi-declaration" if row == len(planes) - 2 else "ordinary",
                                "planes": planes[row].reshape(-1).tolist(), "mask": allowed[0].tolist(),
                                "action": int(policy.masked_fill(~allowed, -torch.inf).argmax()),
                                "logits": [float(x) if torch.isfinite(x) else None for x in policy[0]],
                                "value": float(value.reshape(-1)[0]), "hands": hands.reshape(-1).tolist()})
    (folder / "inputs.json").write_text(json.dumps(records, allow_nan=False))
    print(f"Created {len(records)} real worker cases, including both riichi stages")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
