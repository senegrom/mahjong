"""Reconcile Vineyard A with the concurrently approved Potter's Table without dropping either."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path.cwd()
TARGET = 'f730b66b3cc238d76d8e7feb795b25526151d02e'
SOURCE = 'docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Man5.svg'
MANIFEST = ROOT / 'web/public/tiles/van-gogh/manifest.json'
shared = ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md',
          'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html',
          'web/scripts/export-van-gogh-tiles.mjs', 'web/src/lib/van-gogh-faces.js',
          'web/tests/tile-faces.test.js']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old!r}')
    return text.replace(old, new, 1)


original_doc = (ROOT / shared[0]).read_text()
section = original_doc[original_doc.index('\n## The Red Vineyard A — 5 characters\n'):]
svg = (ROOT / SOURCE).read_bytes()
assert digest(svg) == '167d067218dcb1a9ae71ba43e0f4d4552ef39057c84c6f0d7f52ec566a3a9280'
subprocess.run(['git', 'fetch', 'origin', TARGET], check=True)
result = subprocess.run(['git', 'merge', '--no-commit', '--no-ff', TARGET])
conflicts = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=U'], text=True).splitlines()
assert set(conflicts).issubset(set(shared)), f'Unexpected conflicts: {conflicts}'
assert result.returncode == 0 or conflicts
# Reapply only our small integration edits onto the incoming canonical files.
for p in shared:
    (ROOT / p).write_bytes(subprocess.check_output(['git', 'show', f'{TARGET}:{p}']))
subprocess.run(['git', 'add', '--', *shared], check=True)
old_manifest = json.loads(MANIFEST.read_text())
assert len(old_manifest['tiles']) == 21 and '5m' in old_manifest['remaining']
assert next(t for t in old_manifest['tiles'] if t['tile'] == '9p')['candidate'] == 'Disks C'
before = {}
for item in subprocess.check_output(['git', 'ls-tree', '-r', '-z', TARGET]).split(b'\0'):
    if not item:
        continue
    metadata, raw_path = item.split(b'\t', 1)
    p = raw_path.decode('utf-8')
    if not p.startswith(('docs/design/', 'web/public/tiles/')) or Path(p).suffix.lower() not in ('.svg', '.png', '.webp', '.jpg', '.jpeg'):
        continue
    data = (ROOT / p).read_bytes()
    expected_blob = metadata.split()[2].decode('ascii')
    assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == expected_blob, p
    before[p] = digest(data)

exporter = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
s = replace_once(s,
    "  pottersTable: 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg',",
    "  pottersTable: 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg',\n"
    f"  fiveVineyard: '{SOURCE}',")
s = replace_once(s,
    "  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],",
    "  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],\n"
    "  ['Vineyard A', '5m', 'Five characters', 'fiveVineyard', [0, 0, 300, 400]],")
s, count = re.subn(r"(  approval: '[^\n]*)(',\n)",
    lambda m: m[1] + ' Carl selected the first option, The Red Vineyard, for 5 characters (5m / 五萬) and explicitly approved GitHub deployment. The complete approved composition and colours are preserved; the wheat and iris alternatives are not used.' + m[2], s)
assert count == 1
s = replace_once(s, 'Includes The Potter’s Table C (9p)',
                 'Includes The Red Vineyard A (5m), The Potter’s Table C (9p)')
s = replace_once(s,
    "const featured = ['9p', '3p', '9s', '6s', '5s', '8s', '7s', '4s', '3s', '2s', '2m', '3m'];",
    "const featured = ['5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '5s', '8s', '7s', '4s'];")
s = replace_once(s, 'approved faces, including The Potter’s Table C',
                 'approved faces, including The Red Vineyard for 5 characters, The Potter’s Table C')
s = replace_once(s, 'Three Café Lanterns (3 disks), Moonlit Wind Chime (9 bamboo)',
                 'The Red Vineyard (5 characters), Three Café Lanterns (3 disks), Moonlit Wind Chime (9 bamboo)')
exporter.write_text(s)
subprocess.run(['node', str(exporter), '--only=5m'], check=True)
new_manifest = json.loads(MANIFEST.read_text())
assert len(new_manifest['tiles']) == 22 and len(new_manifest['remaining']) == 12
assert new_manifest['tiles'][:-1] == old_manifest['tiles']
assert new_manifest['sources'][:-1] == old_manifest['sources']
assert new_manifest['superseded'] == old_manifest['superseded']
assert new_manifest['rejected'] == old_manifest['rejected']
assert new_manifest['remaining'] == [t for t in old_manifest['remaining'] if t != '5m']
assert (ROOT / RUNTIME).read_bytes() == svg
for path, expected in before.items():
    assert digest((ROOT / path).read_bytes()) == expected, f'Existing artwork changed: {path}'
print(f'PRESERVED {len(before)} existing artwork/source files byte-for-byte.')

shared_tests = ROOT / 'web/tests/tile-faces.test.js'
s = shared_tests.read_text()
old_approved = ', '.join(repr(t['tile']) for t in old_manifest['tiles'])
s = replace_once(s, f'const approved = [{old_approved}];',
                 f"const approved = [{old_approved}, '5m'];")
old_candidates = ', '.join(repr(t['candidate']) for t in old_manifest['tiles'])
s = replace_once(s, f'set.tiles.map(tile => tile.candidate), [{old_candidates}]',
                 f"set.tiles.map(tile => tile.candidate), [{old_candidates}, 'Vineyard A']")
s = replace_once(s,
    "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s', '4s', '6s', '7s', '8s', '5s', '9s', '3p'])",
    "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s', '4s', '6s', '7s', '8s', '5s', '9s', '3p', '5m'])")
# Recent tests derive the inventory count dynamically; retain that convention.
s = s.replace("url.startsWith('tiles/van-gogh/')).length, 21)",
              "url.startsWith('tiles/van-gogh/')).length, VAN_GOGH_APPROVED.length)")
shared_tests.write_text(s)

readme = ROOT / 'web/public/tiles/van-gogh/README.md'
s = readme.read_text()
s = replace_once(s, 'The twenty-one approved faces', 'The twenty-two approved faces')
s = replace_once(s, 'plus **Three Café Lanterns B**',
                 'plus **The Red Vineyard A** for 5 characters, **Three Café Lanterns B**')
s = replace_once(s, 'The other 13 identities', 'The other 12 identities')
s = replace_once(s, 'The raft and Green Still Life B use documented quality-80',
                 'The vineyard, raft and Green Still Life B use documented quality-80')
readme.write_text(s.rstrip() + '\n' + section)
design = ROOT / 'docs/design/van-gogh/README.md'
design.write_text(design.read_text().rstrip() + '\n' + section)
# A second export must be deterministic and must not rewrite older art.
outputs = [SOURCE, RUNTIME, 'web/public/tiles/van-gogh/manifest.json',
           'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js']
first_export = {p: (ROOT / p).read_bytes() for p in outputs}
subprocess.run(['node', str(exporter), '--only=5m'], check=True)
assert all((ROOT / p).read_bytes() == data for p, data in first_export.items())
assert all(digest((ROOT / p).read_bytes()) == sha for p, sha in before.items())
print('Verified exact image hashes, unchanged manifest entries, and repeatable selective export.')
