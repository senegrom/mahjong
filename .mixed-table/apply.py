"""Install only reviewed files on exact validated preimages."""
import hashlib
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path('.mixed-table/manifest.json').read_text())

def blob(path):
    if not path.exists():
        return None
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()

for name, hashes in manifest.items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts and path.parts[0] in ('docs', 'engine', 'web')
    assert blob(path) == hashes['before'], f'Concurrent source change: {name}'
for part in ('engine', 'ui', 'session'):
    patch = f'.mixed-table/{part}.patch'
    subprocess.run(['git', 'apply', '--check', '--recount', '--whitespace=error', patch], check=True)
    subprocess.run(['git', 'apply', '--recount', '--whitespace=error', patch], check=True)
for name, hashes in manifest.items():
    path = Path(name)
    if hashes['before'] is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path('.mixed-table/new') / path).read_bytes())
    assert blob(path) == hashes['after'], f'Patch checksum mismatch: {name}'
# Start the two browser tests from positions that actually have progress and
# a pending network response. Keep their cancellation and recovery assertions.
patch = '.mixed-table/fixtures.patch'
subprocess.run(['git', 'apply', '--check', '--whitespace=error', patch], check=True)
subprocess.run(['git', 'apply', '--whitespace=error', patch], check=True)
assert blob(Path('web/scripts/mixed-opponents-check.mjs')) == '2f6cc2a90e5d645f5baea4e8be3afbfd27babbe5', 'Fixture patch checksum mismatch'
Path('/tmp/mixed-files.json').write_text(json.dumps(list(manifest)))
print(f'Applied {len(manifest)} source files with verified pre/post hashes')
