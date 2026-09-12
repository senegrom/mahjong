"""Apply pinned, reconciled review repairs only inside the preparation checkout."""
from pathlib import Path
import hashlib
import lzma
import subprocess

base = '393d9ba2393f5b765883b8bc03f1b197483498d8'
# These are disposable validation checkouts, not user working directories.
subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', base], check=True)
subprocess.run(['git', 'checkout', base, '--', 'neural', 'engine', 'Cargo.toml', 'Cargo.lock'], check=True)
parts = [Path(f'.review/repairs.{i}.xzpart').read_bytes() for i in range(4)]
exclude = ['--exclude=' + p for p in (
    'neural/contract.py', 'neural/selfplay.py', 'neural/train_combined.py', 'neural/searched.py')]
patches = [
    (lzma.decompress(b''.join(parts)),
     '2df4c505411c8603bb4821946c8dcfc7aaef2ec204e64357032b05f5482b8e2b', exclude),
    (Path('.review/correction.patch').read_bytes(),
     '097d17da1fb96d8d00a2b700b6e2a088b731827032933666f7b7b1698700cdd3', []),
    (lzma.decompress(Path('.review/reconcile.patch.xz').read_bytes()),
     '34b94527d4425001408d547c1c6c9c72e3888782c1c3fc1bac42d3f79aa6535e', []),
    (Path('.review/native-mask.patch').read_bytes(),
     'cc0873a3313b0d999e281209e3b9d2bfa6009d1b5c6b4cbb4688c9c2099a0082', []),
]
for patch, expected, options in patches:
    if hashlib.sha256(patch).hexdigest() != expected:
        raise RuntimeError('Reviewed patch hash mismatch')
    subprocess.run(['git', 'apply', '--check', *options, '-'], input=patch, check=True)
    subprocess.run(['git', 'apply', *options, '-'], input=patch, check=True)
    print('Applied exact reviewed patch:', expected, flush=True)
subprocess.run(['git', 'diff', '--exit-code', base, '--', 'engine', 'Cargo.toml', 'Cargo.lock'], check=True)
print('Preserved concurrent engine fixes from', base, flush=True)
