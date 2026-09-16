"""Generate disposable small-network exports for the real browser-worker CI test.

No user checkpoint, model manifest or shipped network is read or changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import riichi_py

from neural import combined, export, model, mortal_model, zoo
from neural.observe import Planes, Views
from neural.tests.test_distillation import ready_follower


def cases():
    trial, legal = export.validation_positions()
    found = [('ordinary', trial[:1], legal[:1])]
    follower = ready_follower()
    for after, label in ((False, 'riichi_declaration'), (True, 'riichi_discard')):
        indptr, indices, values, masks = follower.encode([(0, 0)], after_reach=after)
        masks = torch.as_tensor(np.asarray(masks).copy(), dtype=torch.bool)
        assert masks[0, 37] if not after else not masks[0, 34:].any()
        found.append((label, Planes.from_follower(indptr, indices, values).dense('cpu'), masks))
    arena = riichi_py.Arena(games=1, seed=2, bot_places=[])
    views = Views(arena, 1, {'mortal'})
    for _ in range(200):
        views.advance()
        mask = np.frombuffer(arena.legal_mask(), np.uint8).reshape(1, 78).astype(bool)
        if mask[0, 70] and mask[0, 71:75].any():
            seat = np.frombuffer(arena.seats(), np.uint8)[0]
            player = np.frombuffer(arena.seat_players(), np.uint8).reshape(1, 4)[0, seat]
            rows = np.array([0]); people = np.array([player])
            views.prepare(rows, people)
            planes, _ = views.sparse_and_masks(rows, people)
            found.append(('call', planes.dense('cpu'), torch.from_numpy(zoo.translatable(mask))))
            break
        arena.step([int(np.flatnonzero(mask[0])[0])])
    else: raise AssertionError('no native call fixture')
    return found


def generate(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(31)
    samples = cases()
    positions = torch.cat([x for _, x, _ in samples])
    masks = torch.cat([m for _, _, m in samples])
    reports = []
    for kind in ('single', 'fusion'):
        ours = model.PolicyValueNet(8, 1, actions=46).eval()
        net = ours if kind == 'single' else combined.Combined(ours, mortal_model.build(16, 1)).eval()
        if kind == 'fusion':
            net.mortal_config = {'resnet': {'conv_channels': 16, 'num_blocks': 1}, 'control': {'version': 4}}
            payload = net.state()
        else:
            payload = {'model': net.state_dict(), **net.payload_fields()}
        checkpoint = out / (kind + '.pt'); graph = out / (kind + '.onnx')
        torch.save({**payload, 'generation': 0}, checkpoint)
        with patch('sys.argv', ['export', str(checkpoint), str(graph), '--float32']), \
             patch.object(export, 'validation_positions', return_value=(positions, masks)):
            export.main()
        wrapped = export.playable(net)
        records = []
        for index, (label, x, allowed) in enumerate(samples):
            with torch.no_grad():
                policy, value, hands = wrapped(x, allowed.float()) if kind == 'fusion' else wrapped(x)
            logits = policy[0].masked_fill(~allowed[0], -torch.inf)
            weights = torch.softmax(logits, dim=0)
            name = f'{kind}-{index}.f32'
            (out/name).write_bytes(x.numpy().astype('<f4').tobytes())
            records.append({'label': label, 'planes': name, 'mask': allowed[0].int().tolist(),
                            'action': int(logits.argmax()), 'weights': weights.tolist(),
                            'value': float(value.reshape(-1)[0]), 'hands': hands.reshape(-1).tolist()})
        data = graph.read_bytes()
        reports.append({'kind': kind, 'graph': graph.name, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(), 'cases': records})
    (out/'fixtures.json').write_text(json.dumps(reports, allow_nan=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
