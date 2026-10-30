"""Create tiny real exports and ordinary/two-stage-riichi browser inputs, and
the questions the page averages over the suits.

Used by CI with the shipped reduced WASM runtime. No user checkpoint, cloud
service or production model is downloaded, modified or promoted.

`suits.json` holds questions of a real game, each with the game's planes and
the renamed game's, as `neural.suits`'s averaging player asks them of a fused
export's PyTorch twin: the plain and the renamed answers' logits, whether
the plain one hesitates, and the answer the player plays, which the page's
own averaging (web/src/lib/suits.js) must give from the worker's weights.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import torch

from neural import combined, export, model, mortal_model, suits
from neural.checkpoints import atomic_save
from neural.observe import Planes

#: The arena seed whose game the averaged questions are taken from.
SUITS_SEED = 23
#: How much the averaging network's fusion weights are scaled: a fresh
#: fusion plays its Mortal's values, which are too close to one another for
#: any question with a choice to clear the gate, and at a hundred times some
#: do, as a trained network's mostly do, while most still hesitate.
SUITS_SCALE = 100.0
#: Questions within this of the gate, or whose two best moves are within
#: this of each other, plain or averaged, are passed over: a page and a card
#: may round them apart.
MARGIN = 1e-3


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


def exported(folder: Path, name: str, net, fused: bool, planes, masks):
    """The network saved, loaded as the site's export loads it, exported to
    ONNX and checked; its file and its PyTorch twin."""
    import onnx
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
    return path, wrapped


def weights_of(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """A row's logits as the page's weights: the softmax over the moves
    allowed, in double precision, nought elsewhere."""
    values = np.where(allowed, np.asarray(values, dtype=np.float64), -np.inf)
    raised = np.exp(values - values.max())
    return raised / raised.sum()


def apart(weights: np.ndarray) -> bool:
    """Whether a row's two best weights are further apart than MARGIN."""
    best = np.sort(weights)[::-1]
    return len(best) < 2 or best[0] - best[1] > MARGIN


def suit_cases(wrapped, model_name: str) -> list[dict]:
    """Questions of one game the heuristic player plays at every seat, put to
    `neural.suits`'s averaging player over the export's twin as a ruler puts
    them (`suits.averaged`), a reach's tile previewed: one sure of a question
    with a choice, one hesitating and keeping its move, one hesitating and
    changing it, and one hesitating over a reach's tile."""
    import riichi_py
    from neural import observe, zoo

    def answered(planes, allowed):
        with torch.no_grad():
            policy, value, hands = wrapped(planes, torch.as_tensor(np.asarray(allowed)).float())
        return policy.numpy(), value.reshape(-1).numpy(), hands.reshape(len(planes), -1).numpy()

    def ask(views, who, fresh, allowed, ahead=()):
        blocks = []
        if who:
            blocks.append(views.sparse_and_masks(np.array([g for g, _ in who]), np.array([p for _, p in who]),
                                                 fresh=fresh)[0].dense("cpu"))
        if ahead:
            blocks.append(views.sparse_and_masks(np.array([g for g, _ in ahead]), np.array([p for _, p in ahead]),
                                                 after_reach=True)[0].dense("cpu"))
        policy, _value, _hands = answered(torch.cat(blocks), allowed)
        return np.where(allowed, policy, -np.inf).astype(np.float32), np.zeros(np.shape(allowed), dtype=bool)

    account = suits.Averaging()
    averaged = suits.averaged(ask, account)
    arena = riichi_py.Arena(games=1, seed=SUITS_SEED)
    views = observe.Views(arena, 1, renamed=[0])
    found: dict[str, dict] = {}
    wanted = ("sure", "kept", "changed", "reach tile")
    while not arena.all_finished() and len(found) < len(wanted):
        views.advance()
        seat = int(np.frombuffer(arena.seats(), dtype=np.uint8)[0])
        moves = np.frombuffer(arena.teacher(), dtype=np.uint8).astype(np.int64)
        moves[moves == 0xFF] = 0
        if seat != 0xFF:
            player = int(np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(1, 4)[0, seat])
            legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(1, zoo.ACTIONS).astype(bool)
            allowed = zoo.translatable(legal)
            tiles = legal[0, zoo.RIICHI_DISCARD:zoo.TSUMO]
            if allowed.sum() > 1:
                ahead = [(0, player)] if allowed[0, zoo.MORTAL_RIICHI] and tiles.sum() > 1 else []
                questions = allowed
                if ahead:
                    after = np.zeros((1, zoo.MORTAL_ACTIONS), dtype=bool)
                    after[0, :zoo.RIICHI_DISCARD] = tiles
                    questions = np.concatenate([allowed, after])
                played, _own = averaged(views, [(0, player)], False, questions, ahead)
                for row, mask in enumerate(questions):
                    after_reach = row == 1
                    plain = views.sparse_and_masks([0], [player], after_reach=after_reach)[0].dense("cpu")
                    renamed = views.renamed.sparse_and_masks([0], [player], after_reach=after_reach)[0].dense("cpu")
                    logits, value, hands = answered(plain, mask[None])
                    renamed_mask = suits.swap_moves(mask)
                    renamed_logits, renamed_value, renamed_hands = answered(renamed, renamed_mask[None])
                    top = float(suits.top_weight(np.where(mask, logits, -np.inf))[0])
                    weights, plain_weights = weights_of(played[row], mask), weights_of(logits[0], mask)
                    if abs(top - suits.GATE) <= MARGIN or not apart(plain_weights) or not apart(weights):
                        continue
                    hesitant = top < suits.GATE
                    action = int(np.where(mask, played[row], -np.inf).argmax())
                    kind = ("reach tile" if after_reach else "sure") if not hesitant else (
                        "reach tile" if after_reach else "kept" if action == int(plain_weights.argmax()) else "changed")
                    if kind in found or (kind == "reach tile" and not hesitant):
                        continue
                    found[kind] = {
                        "model": model_name, "takes_legal": True, "stage": kind,
                        "planes": plain[0].reshape(-1).tolist(), "renamed": renamed[0].reshape(-1).tolist(),
                        "mask": mask.tolist(), "hesitant": hesitant, "top": top,
                        "logits": [float(x) if np.isfinite(x) else None for x in logits[0]],
                        "value": float(value[0]), "hands": hands[0].tolist(),
                        "renamed_logits": [float(x) if np.isfinite(x) else None for x in renamed_logits[0]],
                        "renamed_value": float(renamed_value[0]), "renamed_hands": renamed_hands[0].tolist(),
                        "action": action, "weights": weights.tolist(),
                    }
        arena.step(moves.tolist())
    missing = [kind for kind in wanted if kind not in found]
    assert not missing, f"the game of seed {SUITS_SEED} asks no question of these kinds: {missing}"
    assert account.again and account.changed, "the averaging player asked the renamed game, and the mean changed a move"
    return [found[kind] for kind in wanted]


def main(folder: Path):
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
        path, wrapped = exported(folder, name, net, fused, planes, masks)
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
    # The averaging player's questions, of a fused network whose weights are
    # scaled up so that some questions with a choice clear the gate.
    torch.manual_seed(82)
    net = combined.Combined(model.PolicyValueNet(8, 1, actions=46), mortal_model.build(16, 1))
    with torch.no_grad():
        net.fuse.weights.mul_(SUITS_SCALE)
    path, wrapped = exported(folder, "suits", net, True, planes, masks)
    cases = suit_cases(wrapped, path.name)
    (folder / "suits.json").write_text(json.dumps({"gate": suits.GATE, "moves": suits.MOVES.tolist(), "cases": cases},
                                                  allow_nan=False))
    print(f"Created {len(records)} real worker cases, including both riichi stages, "
          f"and {len(cases)} questions averaged over the suits: {', '.join(case['stage'] for case in cases)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
