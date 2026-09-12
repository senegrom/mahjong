"""Apply the exact reviewed patch, on the preparation branch only."""
from pathlib import Path
import hashlib
import lzma
import subprocess

parts = [Path(f'.review/repairs.{i}.xzpart').read_bytes() for i in range(4)]
patch = lzma.decompress(b''.join(parts))
expected = '2df4c505411c8603bb4821946c8dcfc7aaef2ec204e64357032b05f5482b8e2b'
if hashlib.sha256(patch).hexdigest() != expected:
    raise RuntimeError('Reviewed patch hash mismatch')
subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, check=True)
print('Applied exact reviewed patch:', expected, flush=True)
