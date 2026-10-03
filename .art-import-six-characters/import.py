"""Branch-only import of Carl's approved six-characters lemon landscape."""
from pathlib import Path
import base64
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path.cwd()
TEMP = ROOT / '.art-import-six-characters'
PUBLIC = Path('web/public/tiles/van-gogh')
SOURCE = Path('docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg')
RUNTIME = PUBLIC / 'approved/Man6.svg'
RECORD = Path('docs/design/van-gogh/six-characters-lemon-terrace.json')
TEST = Path('web/tests/van-gogh-six-characters.test.js')
RASTER_HASH = '1fb6ccae9c621b881d676d89d79e452ce6b79e44f365dfc89df404de2ea80cf0'
SVG_HASH = 'dcd95ef26e4a2dec009d4d5c12aa8b87cd00a57be03f395349707b5fb3808418'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one patch anchor: {old!r}')
    return text.replace(old, new, 1)

def js_list(values):
    return '[' + ', '.join(repr(value) for value in values) + ']'

before = json.loads((PUBLIC / 'manifest.json').read_text())
assert '6m' in before['remaining']
assert not any(tile['tile'] == '6m' for tile in before['tiles'])
for target in (SOURCE, RUNTIME, RECORD, TEST):
    assert not target.exists(), f'Refusing to overwrite {target}'
tracked = subprocess.check_output(['git', 'ls-files', '-z', 'web/public/tiles', 'docs/design']).decode().split('\0')
existing = {name: digest(Path(name).read_bytes()) for name in tracked if name and Path(name).suffix.lower() in {'.png', '.svg', '.webp', '.jpg', '.jpeg'}}
raster = b''.join((TEMP / f'chunk{i}.bin').read_bytes() for i in range(4))
assert len(raster) == 53298 and digest(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Six characters — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == SVG_HASH
SOURCE.parent.mkdir(parents=True, exist_ok=True)
SOURCE.write_bytes(svg)
shutil.copyfile(TEMP / 'provenance.json', RECORD)
shutil.copyfile(TEMP / 'six-characters.test.js', TEST)

exporter = Path('web/scripts/export-van-gogh-tiles.mjs')
s = exporter.read_text()
s = replace_once(s, '};\nconst definitions = [', f"  sixLemonTerrace: '{SOURCE.as_posix()}',\n}};\nconst definitions = [")
s = replace_once(s, '];\nconst onlyArgument =', "  ['Lemon Terrace', '6m', 'Six characters', 'sixLemonTerrace', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
approval = re.findall(r"^  approval: '.+',$", s, re.M)
assert len(approval) == 1
sentence = ' Carl approved the latest red 六萬 calligraphy with lemons, cypresses, lake and village as 6 characters (6m). The complete approved canvas and colours are retained; the sunflower study is not selected.'
s = replace_once(s, approval[0], approval[0][:-2] + sentence + "',")
s = replace_once(s, 'Includes The Red Vineyard A (5m)', 'Includes Lemon Terrace (6m), The Red Vineyard A (5m)')
featured = re.findall(r'^const featured = \[.*\];$', s, re.M)
assert len(featured) == 1
s = replace_once(s, featured[0], "const featured = ['6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '5s', '8s', '7s'];")
s = replace_once(s, 'approved faces, including The Red Vineyard for 5 characters,', 'approved faces, including Lemon Terrace for 6 characters, The Red Vineyard for 5 characters,')
s = replace_once(s, 'Earlier selected art is preserved in lossless PNG crops.', 'Earlier selected art is preserved in lossless PNG crops. Lemon Terrace fits its entire approved 1295 × 1214 canvas to the shared 300 × 400 face; the original is separately preserved and the aspect ratio changes.')
exporter.write_text(s)

shared = Path('web/tests/tile-faces.test.js')
s = shared.read_text()
ids = [tile['tile'] for tile in before['tiles']]
candidates = [tile['candidate'] for tile in before['tiles']]
s = replace_once(s, '  const approved = ' + js_list(ids) + ';', '  const approved = ' + js_list(ids + ['6m']) + ';')
s = replace_once(s, '  assert.deepEqual(set.tiles.map(tile => tile.candidate), ' + js_list(candidates) + ');', '  assert.deepEqual(set.tiles.map(tile => tile.candidate), ' + js_list(candidates + ['Lemon Terrace']) + ');')
dora = [line for line in s.splitlines() if line.startswith("    for (const tile of ['1z', '4z',")]
assert len(dora) == 1 and dora[0].endswith(']) {')
s = replace_once(s, dora[0], dora[0][:-4] + ", '6m']) {")
shared.write_text(s)

subprocess.run(['node', str(exporter), '--only=6m'], check=True)
after = json.loads((PUBLIC / 'manifest.json').read_text())
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['superseded'] == before['superseded']
assert after['rejected'] == before['rejected']
assert after['remaining'] == [tile for tile in before['remaining'] if tile != '6m']
assert len(after['tiles']) == len(before['tiles']) + 1
assert RUNTIME.read_bytes() == svg
assert after['tiles'][-1]['svgSha256'] == SVG_HASH
outputs = [PUBLIC / 'manifest.json', PUBLIC / 'preview.html', Path('web/src/lib/van-gogh-faces.js'), RUNTIME]
first = {str(p): p.read_bytes() for p in outputs}
subprocess.run(['node', str(exporter), '--only=6m'], check=True)
assert all(Path(name).read_bytes() == data for name, data in first.items())
assert all(Path(name).exists() and digest(Path(name).read_bytes()) == value for name, value in existing.items())

note = '\n## Lemon Terrace — 6 characters (六萬)\n\nCarl approved the latest red 六萬 calligraphy with lemons, cypresses, a lake and village for `6m` / `Man6`. This is the selected character painting, not the earlier sunflower study. No existing tile artwork is changed.\n\nThe complete 1295 × 1214 source is fitted to the shared 300 × 400 face using Lanczos and quality-80 WebP, embedded in a self-contained SVG with the usual rounded clip and 1% bleed. Fitting changes the aspect ratio; it is not a proportional resize. There is no selective cropping, repainting or colour edit. The unchanged full-resolution original is preserved in `Van_Gogh_6_Characters_Lemon_Terrace_Approved.zip` supplied in chat. See `docs/design/van-gogh/six-characters-lemon-terrace.json` for original, raster and SVG hashes.\n\nThe set now has **23 painted faces and 11 Classic fallbacks**. `node web/scripts/export-van-gogh-tiles.mjs --only=6m` copies the approved source to the game exactly without rewriting other faces. Regression tests cover hashes, registration, preloading, hidden tiles, the fourteen-tile preview hand and repeated selective export.\n'
runtime_doc = PUBLIC / 'README.md'
s = runtime_doc.read_text()
s = replace_once(s, 'The twenty-two approved faces', 'The twenty-three approved faces')
s = replace_once(s, 'The other 12 identities awaiting Van Gogh artwork', 'The other 11 identities awaiting Van Gogh artwork')
s = replace_once(s, 'plus **The Red Vineyard A** for 5 characters,', 'plus **Lemon Terrace** for 6 characters, **The Red Vineyard A** for 5 characters,')
runtime_doc.write_text(s.rstrip() + '\n' + note)
design_doc = Path('docs/design/van-gogh/README.md')
s = design_doc.read_text()
s, n = re.subn(r'^\*\*Current set:\*\*.*$', '**Current set:** 23 painted faces and 11 Classic fallbacks; all nine bamboo identities are covered. Lemon Terrace is active for 6 characters, The Red Vineyard for 5 characters, and The Potter’s Table for 9 disks. The history below retains earlier milestone counts.', s, count=1, flags=re.M)
assert n == 1
design_doc.write_text(s.rstrip() + '\n' + note)
print(f'PRESERVED {len(existing)} pre-existing artwork/source files byte-for-byte.')
print('Verified exact approved raster and SVG, all prior manifest entries, and repeatable selective export.')
