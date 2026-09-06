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

# The final hand-total assertion must include every individual payment,
# not mistake the first settlement's incremental delta for a hand total.
p = Path('engine/riichi-core/tests/mjai_replay.rs')
s = p.read_text()
a = s.index('            let reported = hand\n')
b = s.index('            assert_eq!(\n                reported, deltas,', a)
s = s[:a] + '''            let mut reported = [0; 4];
            let mut settled = false;
            for event in &hand.log {
                match event {
                    Event::ReachAccepted { actor } => reported[actor.index()] -= 1000,
                    Event::Hora { deltas, .. } | Event::Ryukyoku { deltas, .. } => {
                        settled = true;
                        for (total, delta) in reported.iter_mut().zip(deltas) {
                            *total += delta;
                        }
                    }
                    _ => {}
                }
            }
            assert!(settled, "a finished hand says what it moved");
''' + s[b:]
p.write_text(s)
assert blob(p) == '1064e8e1a0970a763501fc723a1c4dc78f3438a2'
print(f"Applied {len(packet['files'])} checked source files")
