"""Build ephemeral real ONNX fixtures for the actual browser policy worker.

Run as a module with an output directory. No production checkpoint, manifest or
model is read or overwritten. Native follower inputs include a riichi second stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from neural import combined, export, mortal_model
from neural.model import PolicyValueNet
from neural.observe import Planes
from neural.tests.test_distillation import ready_follower


def make(folder: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(461)
    ordinary, masks = export.validation_positions()
    follower = ready_follower()
    ptr, columns, values, allowed = follower.encode([(0, 0)])
    root = Planes.from_follower(ptr, columns, values).dense('cpu')
    allowed = torch.as_tensor(np.asarray(allowed), dtype=torch.bool)
    assert allowed[0, 37]
    follower.tell(0, 0, json.dumps({'type': 'reach', 'actor': 0}))
    ptr, columns, values, tiles = follower.encode([(0, 0)])
    after = Planes.from_follower(ptr, columns, values).dense('cpu')
    tiles = torch.as_tensor(np.asarray(tiles), dtype=torch.bool)
    assert tiles[0, :34].any() and not tiles[0, 37:].any()
    positions = [('ordinary', ordinary[:1], masks[:1]), ('riichi-root', root, allowed),
                 ('riichi-discard', after, tiles)]
    folder.mkdir(parents=True, exist_ok=True)
    fixture = []
    for joined in (False, True):
        name = 'fusion' if joined else 'standalone'
        where = folder / name; where.mkdir()
        net = PolicyValueNet(8, 1, actions=46).eval()
        if joined:
            net = combined.Combined(net, mortal_model.build(16, 1)).eval()
            net.mortal_config = {'resnet': {'conv_channels': 16, 'num_blocks': 1}, 'control': {'version': 4}}
            state = net.state()
        else:
            state = {'model': net.state_dict(), **net.payload_fields()}
        checkpoint = where / 'actor.pt'; torch.save(state, checkpoint)
        graph = where / 'model.onnx'
        # Some ONNX exporters emit removable internal Identity nodes. The real
        # reduced runtime, not a widened generic runtime, is exercised below.
        with patch('sys.argv', ['export', str(checkpoint), str(graph), '--float32', '--allow-any-operator']):
            export.main()
        checkpoint.unlink()
        wrapped = export.playable(net)
        cases = []
        for label, x, legal in positions:
            (where / f'{label}.bin').write_bytes(x.numpy().astype('<f4').tobytes())
            with torch.no_grad():
                logits, value, hands = wrapped(x, legal.float()) if joined else wrapped(x)
                weights = logits.float().masked_fill(~legal, -torch.inf).softmax(1)[0].tolist()
            cases.append({'name': label, 'mask': legal[0].tolist(),
                          'action': int(np.argmax(weights)), 'weights': weights,
                          'value': float(value.reshape(-1)[0]), 'hands': hands.flatten().tolist()})
        body = graph.read_bytes()
        fixture.append({'name': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                        'inputs': ['planes', 'legal'] if joined else ['planes'], 'cases': cases})
    (folder / 'fixtures.json').write_text(json.dumps(fixture, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    make(parser.parse_args().folder)
