"""Branch-only import of Carl's selected B, The Provencal Kitchen, for 6p."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

STAGE = Path('.kitchen-transfer')
SOURCE = 'docs/design/van-gogh/studies/22-six-disks-provencal-kitchen-b-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Pin6.svg'
MANIFEST = 'web/public/tiles/van-gogh/manifest.json'
EXPORTER = 'web/scripts/export-van-gogh-tiles.mjs'
PROVENANCE = 'docs/design/van-gogh/six-disks-kitchen.json'
TEST = 'web/tests/van-gogh-six-disks.test.js'
CHUNKS = [
    'ee28ae4abf13b940d6bf19ccf4bb9ae300698d14',
    '721914d4d476f5460f60872172f381f0c5a7c1c7',
    'c3b9d740dab6c24a7d9b48e0851de6761298a454',
    'dee2b4c56e27dea41d11b453e50b081d8647b666',
    'fbe69e292e1a1db145f859bb124d73cfce42aaff',
    '641b4686413d1f66d4c276de0a3cdc8cf6cbf13a',
    '0c29ce7e586f9d165436fe266e426361cfa29697',
]
RASTER_HASH = '95d81654ea7db8776840592f0f28e79a33c936fc201b58c2a154bb4fbe9d9e46'
SVG_HASH = '9c0e3ae5048973c0e1e825346d5cabed550411aaef46c6ddf600cd7e488e1303'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old!r}')
    return text.replace(old, new, 1)


before = json.loads(Path(MANIFEST).read_text())
assert '6p' in before['remaining']
assert not any(t['tile'] == '6p' for t in before['tiles'])
assert not Path(SOURCE).exists() and not Path(RUNTIME).exists()
protected = {}
for folder in ('docs/design', 'web/public/tiles'):
    for p in Path(folder).rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.svg', '.png', '.webp', '.avif', '.jpg', '.jpeg'):
            protected[p] = digest(p.read_bytes())
for s in before['sources']:
    p = Path(s['source'])
    protected[p] = digest(p.read_bytes())
    assert protected[p] == s['sha256'], f'Existing source checksum mismatch: {p}'

parts = []
for index, expected in enumerate(CHUNKS):
    part = (STAGE / f'part-{index}.bin').read_bytes()
    assert hashlib.sha1(f'blob {len(part)}\0'.encode() + part).hexdigest() == expected
    parts.append(part)
raster = b''.join(parts)
assert len(raster) == 57930 and digest(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">Six disks — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,'
       + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == SVG_HASH
Path(SOURCE).write_bytes(svg)
shutil.copyfile(STAGE / 'provenance.json', PROVENANCE)
shutil.copyfile(STAGE / 'kitchen.test.js', TEST)

s = Path(EXPORTER).read_text()
s = replace_once(s, '};\nconst definitions = [', f"  sixKitchen: '{SOURCE}',\n}};\nconst definitions = [")
s = replace_once(s, '];\nconst onlyArgument =', "  ['Kitchen B', '6p', 'Six disks', 'sixKitchen', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
approval = ' Carl selected B, The Provençal Kitchen, the middle painting with six ceramic plates in two columns of three, as 6 disks (6p / Pin6) and explicitly requested GitHub deployment. The full selected crop is fitted to the shared tile format without regeneration, repainting or recolouring; the sunflower and fishing-float alternatives are not used.'
s = replace_once(s, "',\n  fallback: 'classic',", approval + "',\n  fallback: 'classic',")
s = replace_once(s, 'Includes Starry Olive Grove (8m)', 'Includes The Provencal Kitchen B (6p), Starry Olive Grove (8m)')
s = replace_once(s, 'approved faces, including Starry Olive Grove', 'approved faces, including The Provençal Kitchen B for 6 disks, Starry Olive Grove')
# Preserve all thirteen featured earlier tiles and the existing Classic fallback.
Path(EXPORTER).write_text(s)
subprocess.run(['node', EXPORTER, '--only=6p'], check=True)
after = json.loads(Path(MANIFEST).read_text())
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['rejected'] == before['rejected'] and after['superseded'] == before['superseded']
assert after['remaining'] == [t for t in before['remaining'] if t != '6p']
assert len(after['tiles']) == len(before['tiles']) + 1
assert Path(RUNTIME).read_bytes() == svg
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Existing artwork changed: {p}'
outputs = [SOURCE, RUNTIME, MANIFEST, 'web/src/lib/van-gogh-faces.js', 'web/public/tiles/van-gogh/preview.html']
first_export = {p: Path(p).read_bytes() for p in outputs}
subprocess.run(['node', EXPORTER, '--only=6p'], check=True)
for p, expected in first_export.items():
    assert Path(p).read_bytes() == expected, f'Non-repeatable export: {p}'
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Repeat export changed existing artwork: {p}'

p = Path('web/tests/tile-faces.test.js')
s = p.read_text()
match = re.search(r"  const approved = \[([^\n]+)\];", s)
assert match and "'8m'" in match[1] and "'6p'" not in match[1]
s = replace_once(s, match[0], match[0][:-2] + ", '6p'];")
s = replace_once(s, "'Oranges A', 'Starry Olive Grove']);", "'Oranges A', 'Starry Olive Grove', 'Kitchen B']);")
# Add the newly painted disk to the established component dora checks when present.
for candidate in re.finditer(r"for \(const tile of \[([^\n]+)\]\) \{", s):
    if "'6m'" in candidate[1] and "'8m'" in candidate[1]:
        assert "'6p'" not in candidate[1]
        s = replace_once(s, candidate[0], candidate[0].replace(']) {', ", '6p']) {"))
        break
p.write_text(s)

count, remaining = len(after['tiles']), len(after['remaining'])
section = '''\n## The Provençal Kitchen B — 6 disks\n\nCarl selected **B, the middle kitchen painting**, as `6p` / `Pin6`: exactly six decorated plates in two columns of three on an ochre wall, with a blue shutter and the dresser still life below. The sunflower and fishing-float alternatives are not selected.\n\nThe source `docs/design/van-gogh/studies/22-six-disks-provencal-kitchen-b-approved.svg` and runtime `web/public/tiles/van-gogh/approved/Pin6.svg` are identical. They embed a 300 × 400 quality-90 WebP export of the complete 464 × 873 selected crop. Fitting the tall crop to the shared face changes its aspect ratio; no objects, colours or painted details were regenerated. The shared rounded clipping and 1% bleed are retained.\n\nThe untouched 1491 × 1055 board and lossless selected crop are supplied in `Van_Gogh_6_Disks_Kitchen_B_Approved.zip` in chat. `docs/design/van-gogh/six-disks-kitchen.json` records original, crop, production and SVG checksums and exact processing.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=6p` to reproduce this face and its registration without rewriting any other tile artwork. The gallery includes the kitchen; the existing fourteen-tile example hand is unchanged.\n'''
p = Path('docs/design/van-gogh/README.md')
s = p.read_text()
s, n = re.subn(r'\*\*Current set:\*\* \d+ painted faces and \d+ Classic fallbacks;', f'**Current set:** {count} painted faces and {remaining} Classic fallbacks;', s, count=1)
assert n == 1
s = replace_once(s, 'Starry Olive Grove is active for 8 characters,', 'The Provençal Kitchen B is active for 6 disks, Starry Olive Grove for 8 characters,')
p.write_text(s.rstrip() + '\n' + section)
p = Path('web/public/tiles/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, 'The twenty-six approved faces', 'The twenty-seven approved faces')
s = replace_once(s, 'plus **Starry Olive Grove**', 'plus **The Provençal Kitchen B** for 6 disks, **Starry Olive Grove**')
s = replace_once(s, 'The other 8 identities awaiting Van Gogh artwork', 'The other 7 identities awaiting Van Gogh artwork')
p.write_text(s.rstrip() + '\n' + section)
print(f'PRESERVED {len(protected)} existing artwork/source files byte-for-byte.')
print(f'REPRODUCIBLE: {count} approved; {remaining} remaining. All prior manifest entries and the fourteen-tile hand are preserved.')
