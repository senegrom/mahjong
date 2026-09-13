"""Immutable search snapshots with a single atomic publication pointer.

A recording directory contains snapshots/<id>/ and current.json. Readers resolve
that pointer ONCE and use that immutable snapshot for every array. No reader sees
an in-place copy, a directory-swap gap, or files from different generations.
Interrupted writes leave the previous pointer intact. Historical snapshots are
never deleted here. Cloud progress and accepted results use separate directories.

This module stays importable on thin cloud launchers without NumPy or Torch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

from .checkpoints import staging_file, sync_directory

FORMAT = 2
SEARCH_BACKUP_VERSION = 2
REWARD_VERSION = 1
ARRAYS = (
    'root-indptr', 'root-indices', 'root-values', 'candidates', 'values',
    'per_world', 'policy', 'search', 'chair', 'game', 'step', 'sure', 'legal',
)
FILES = tuple(f'{name}.npy' for name in ARRAYS) + ('meta.json',)


def digest_file(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    with staging_file(path) as staged:
        with staged.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        sync_directory(path.parent)


def experiment(checkpoint: Path, generation: int, settings: dict) -> dict:
    """Bind a unique attempt to the actual checkpoint bytes and every option."""
    identity = {
        'checkpoint_sha256': digest_file(checkpoint), 'generation': generation,
        'settings': dict(settings), 'reward_version': REWARD_VERSION,
        'search_backup_version': SEARCH_BACKUP_VERSION,
    }
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode('utf-8')
    identity['configuration_sha256'] = hashlib.sha256(encoded).hexdigest()
    identity['experiment_id'] = identity['configuration_sha256'][:24] + '-' + uuid4().hex
    return identity


def resolve_recording(folder: Path) -> Path:
    """Pin a snapshot; legacy flat directories remain readable by diagnostics."""
    folder = Path(folder)
    pointer = folder / 'current.json'
    if pointer.exists():
        state = json.loads(pointer.read_text(encoding='utf-8'))
        name = state.get('snapshot') if isinstance(state, dict) else None
        if (not isinstance(state, dict) or state.get('format') != FORMAT or not isinstance(name, str)
                or re.fullmatch(r'[0-9a-f]{32}', name) is None):
            raise ValueError('Invalid recording publication pointer')
        snapshot = folder / 'snapshots' / name
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError('Recording pointer does not name an immutable snapshot')
        return snapshot
    if (folder / 'meta.json').is_file():
        return folder
    raise FileNotFoundError(f'No published recording at {folder}')


def validate_snapshot(folder: Path, *, require_complete: bool = True,
                      require_manifest: bool = True) -> dict:
    """Check provenance, checksums and array schemas before accepting a snapshot."""
    import numpy as np
    from .observe import Planes

    folder = Path(folder)
    for name in FILES:
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Recording is missing a regular file: {name}')
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    if (not isinstance(meta, dict) or meta.get('recording_format') != FORMAT
            or meta.get('search_backup_version') != SEARCH_BACKUP_VERSION
            or meta.get('reward_version') != REWARD_VERSION):
        raise ValueError('Legacy or incompatible search targets; collect a new recording')
    rows = meta.get('rows')
    if type(rows) is not int or rows < 0 or type(meta.get('complete')) is not bool:
        raise ValueError('Invalid recording row count or completeness flag')
    if require_complete and not meta['complete']:
        raise ValueError('Incomplete search progress is not an accepted recording')
    if re.fullmatch(r'[0-9a-f]{32}', str(meta.get('snapshot_id', ''))) is None:
        raise ValueError('Recording needs an immutable snapshot id')
    if require_manifest:
        manifest_path = folder / 'manifest.json'
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError('Recording has no completion manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected = manifest.get('files') if isinstance(manifest, dict) else None
        if not isinstance(manifest, dict) or manifest.get('format') != FORMAT or not isinstance(expected, dict) or set(expected) != set(FILES):
            raise ValueError('Invalid recording manifest')
        for name in FILES:
            path = folder / name
            if expected[name] != {'bytes': path.stat().st_size, 'sha256': digest_file(path)}:
                raise ValueError(f'Recording checksum mismatch: {name}')
    Planes.load(folder, 'root').validate(expected_rows=rows)
    data = {name: np.load(folder / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            for name in ARRAYS if not name.startswith('root-')}
    candidates, values, worlds = (data[name] for name in ('candidates', 'values', 'per_world'))
    if (candidates.dtype != np.int64 or candidates.ndim != 2 or candidates.shape[0] != rows
            or values.dtype != np.float32 or values.shape != candidates.shape
            or worlds.dtype != np.float32 or worlds.ndim != 3 or worlds.shape[:2] != candidates.shape):
        raise ValueError('Invalid candidate/value/world array schema')
    for name in ('policy', 'search', 'chair', 'game', 'step'):
        if data[name].dtype != np.int64 or data[name].shape != (rows,):
            raise ValueError(f'Invalid recording array: {name}')
    if (data['sure'].dtype != np.float32 or data['sure'].shape != (rows,)
            or data['legal'].dtype != np.bool_ or data['legal'].shape != (rows, 78)):
        raise ValueError('Invalid confidence or legal-mask array')
    for start in range(0, rows, 256):
        c, v, w = candidates[start:start + 256], values[start:start + 256], worlds[start:start + 256]
        valid = c >= 0
        if (np.any(c < -1) or np.any(c >= 78) or np.any(valid.sum(1) < 2)
                or np.any(valid[:, 1:] & ~valid[:, :-1]) or np.isinf(w).any()
                or np.any(~np.isnan(v[~valid])) or np.any(~np.isnan(w[~valid]))
                or np.any(~np.isfinite(v[valid]))
                or np.any(~np.isfinite(w).any(axis=2)[valid])):
            raise ValueError('Invalid candidates, padding or nonfinite search values')
        policy, search = data['policy'][start:start + 256], data['search'][start:start + 256]
        if np.any(c[:, 0] != policy) or np.any(~(c == search[:, None]).any(1)):
            raise ValueError('Policy/search choices are not the recorded candidates')
        legal = data['legal'][start:start + 256]
        if not np.all(np.take_along_axis(legal, np.maximum(c, 0), axis=1) | ~valid):
            raise ValueError('A recorded candidate is illegal')
        if any(len(np.unique(row[row >= 0])) != np.sum(row >= 0) for row in c):
            raise ValueError('Duplicate recorded candidates')
        sure = data['sure'][start:start + 256]
        if not np.isfinite(sure).all() or np.any((sure < 0) | (sure > 1)):
            raise ValueError('Invalid policy confidence')
        for name, upper in (('chair', 4), ('game', None), ('step', None)):
            vector = data[name][start:start + 256]
            if np.any(vector < 0) or upper is not None and np.any(vector >= upper):
                raise ValueError(f'Invalid recording {name}')
    return meta


def _manifest(folder: Path) -> None:
    files = {}
    for name in FILES:
        path = folder / name
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest_file(path)}
    atomic_json(folder / 'manifest.json', {'format': FORMAT, 'files': files})


def write_snapshot(folder: Path, writer, meta: dict) -> Path:
    """Publish complete bytes or leave the prior pointer and every snapshot intact."""
    folder = Path(folder)
    snapshots = folder / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    name = uuid4().hex
    writing = Path(tempfile.mkdtemp(prefix='.writing-', dir=snapshots))
    destination = snapshots / name
    try:
        writer(writing, {**meta, 'recording_format': FORMAT, 'snapshot_id': name,
                        'reward_version': REWARD_VERSION,
                        'search_backup_version': SEARCH_BACKUP_VERSION})
        validate_snapshot(writing, require_complete=False, require_manifest=False)
        _manifest(writing)
        os.rename(writing, destination)
        sync_directory(snapshots)
        atomic_json(folder / 'current.json', {'format': FORMAT, 'snapshot': name})
        return destination
    finally:
        # Only this invocation's unpublished staging area is disposable.
        # Never remove destination after a rename: a signal may follow commit.
        if writing.exists():
            shutil.rmtree(writing)


def copy_recording(source: Path, destination: Path, *, require_complete: bool) -> dict:
    """Copy a pinned snapshot, validate the copied bytes, then move one pointer.

    No in-place fallback is allowed on filesystems which cannot rename. An
    unsuccessful copy/publication must leave the existing accepted result alone.
    """
    source = resolve_recording(source)
    meta = validate_snapshot(source, require_complete=require_complete)
    destination = Path(destination)
    snapshots = destination / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    name = meta['snapshot_id']
    published = snapshots / name
    if published.exists():
        validate_snapshot(published, require_complete=require_complete)
        if (published / 'manifest.json').read_bytes() != (source / 'manifest.json').read_bytes():
            raise ValueError('A snapshot id cannot be reused for different bytes')
    else:
        writing = Path(tempfile.mkdtemp(prefix='.copy-', dir=snapshots))
        try:
            shutil.copytree(source, writing, dirs_exist_ok=True)
            validate_snapshot(writing, require_complete=require_complete)
            _manifest(writing)
            os.rename(writing, published)
            sync_directory(snapshots)
        finally:
            if writing.exists():
                shutil.rmtree(writing)
    atomic_json(destination / 'current.json', {'format': FORMAT, 'snapshot': name})
    return meta
