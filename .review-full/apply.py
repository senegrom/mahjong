"""Apply the reviewed source patch only to its exact validated base files."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

payload = ''.join(Path(f'.review-full/payload-{i}.txt').read_text() for i in range(3))
raw = gzip.decompress(base64.b64decode(payload, validate=True))
assert hashlib.sha256(raw).hexdigest() == 'de7f5f99d17f100f05f609abc3e04ef4d852ada8119944ee8621cf1813e61b0a', 'Patch transfer checksum mismatch'
packet = json.loads(raw)

def blob(path):
    if not path.exists():
        return None
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()

for name, hashes in packet['files'].items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert path.parts[0] in ('engine', 'web')
    assert blob(path) == hashes['before'], f'Base changed: {name}'
subprocess.run(['git', 'apply', '--check', '--whitespace=error', '-'], input=packet['patch'], text=True, check=True)
subprocess.run(['git', 'apply', '--whitespace=error', '-'], input=packet['patch'], text=True, check=True)
for name, hashes in packet['files'].items():
    assert blob(Path(name)) == hashes['after'], f'Applied file checksum mismatch: {name}'
print(f"Applied {len(packet['files'])} checked source files")
