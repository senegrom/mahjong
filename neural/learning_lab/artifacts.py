"""Immutable, checked experimental data and safe actor construction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

from ..checkpoints import sync_directory


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest_file(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_state(state) -> str:
    """Tensor identity without pickle timestamps, paths, or random draws."""
    h = hashlib.sha256()
    def walk(value):
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().contiguous()
            h.update(b'tensor\0' + canonical([str(v.dtype), list(v.shape)]))
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            h.update(b'dict\0' + canonical(len(value)))
            for key in sorted(value):
                h.update(canonical(key)); walk(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(b'sequence\0' + canonical(len(value)))
            for x in value: walk(x)
        else:
            h.update(b'scalar\0' + canonical(value) + b'\0')
    walk(state)
    return h.hexdigest()


def engine_identity() -> dict:
    """Reject trace replays under changed game/follower/observation semantics."""
    import riichi_py
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    files = sorted((root / 'engine/riichi-core/src').rglob('*.rs'))
    files += sorted((root / 'engine/riichi-py/src').rglob('*.rs'))
    files += sorted((root / 'engine/libriichi/src').rglob('*.rs'))
    files += [root / 'neural/observe.py']
    if not files or any(not p.is_file() for p in files):
        raise ValueError('Learning experiments require a source checkout')
    for p in files:
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return dict(training_api=int(riichi_py.TRAINING_API_VERSION),
                search_api=int(riichi_py.SEARCH_API_VERSION), sources=h.hexdigest())


def save(path: Path, kind: str, meta: dict, arrays: dict[str, np.ndarray]) -> str:
    """Exclusive directory; the completion manifest is always published last."""
    path = Path(path)
    if not isinstance(meta, dict) or not isinstance(kind, str) or not arrays:
        raise ValueError('Expected nonempty experimental artifact')
    canonical(meta)
    for name, a in arrays.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,60}', name) or not isinstance(a, np.ndarray) or a.dtype.hasobject:
            raise ValueError('Invalid array name or unsafe dtype')
    path.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, a in sorted(arrays.items()):
        file = path / (name + '.npy')
        with file.open('xb') as stream:
            np.save(stream, a, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        files[name] = dict(sha256=digest_file(file), shape=list(a.shape), dtype=a.dtype.str)
    manifest = dict(version=1, kind=kind, complete=True, meta=meta, files=files)
    identity = hashlib.sha256(canonical(manifest)).hexdigest()
    manifest['identity'] = identity
    staged = path / 'manifest.partial'
    with staged.open('xb') as stream:
        stream.write(canonical(manifest)); stream.flush(); os.fsync(stream.fileno())
    os.replace(staged, path / 'manifest.json'); sync_directory(path)
    return identity


def load(path: Path, kind: str) -> tuple[dict, dict, str]:
    path = Path(path)
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file() or manifest_path.stat().st_size > 4_000_000:
        raise ValueError('Missing or oversized completion manifest')
    m = json.loads(manifest_path.read_text())
    if (type(m.get('version')) is not int or m['version'] != 1 or m.get('kind') != kind
            or m.get('complete') is not True or not isinstance(m.get('files'), dict)):
        raise ValueError('Wrong experimental data contract')
    identity = m.pop('identity', None)
    if hashlib.sha256(canonical(m)).hexdigest() != identity:
        raise ValueError('Experimental manifest identity changed')
    arrays = {}
    for name, spec in m['files'].items():
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,60}', name):
            raise ValueError('Unsafe array path')
        file = path / (name + '.npy')
        if file.is_symlink() or digest_file(file) != spec['sha256']:
            raise ValueError('Experimental array checksum mismatch')
        a = np.load(file, mmap_mode='r', allow_pickle=False)
        if a.dtype.str != spec['dtype'] or list(a.shape) != spec['shape']:
            raise ValueError('Experimental array schema changed')
        arrays[name] = a
    return m['meta'], arrays, identity


def actor_payload(net) -> dict:
    return net.state() if hasattr(net, 'state') else dict(model=net.state_dict(), **net.payload_fields())


def actor_from_payload(payload: dict, device='cpu'):
    from .. import combined, model, mortal_model
    net = model.from_payload(payload, device)
    if 'combined' in payload:
        mortal = mortal_model.build(**mortal_model.shape_of(payload)).to(device)
        mortal.brain.load_state_dict(payload['mortal'])
        mortal.dqn.load_state_dict(payload['current_dqn'])
        net = combined.Combined(net, mortal).to(device)
        net.fuse.load_state_dict(payload['combined'], strict=True)
        net.mortal_config = payload['config']
    if net.actions != 46 or net.planes != 1012:
        raise ValueError('Learning lab requires current Mortal-v4 / 46-action policies')
    return net


def tensor(a, device='cpu'):
    return torch.from_numpy(np.asarray(a).copy()).to(device)


def finite(tree):
    from ..train_search import _finite_tree
    _finite_tree(tree)


def generation(payload):
    value = payload.get('generation', 0)
    if type(value) is not int or value < 0:
        raise ValueError('checkpoint generation must be a nonnegative integer')
    return value


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None
