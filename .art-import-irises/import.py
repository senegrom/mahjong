"""Branch-only verified import of Carl's selected seven-characters blue irises."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path.cwd()
TEMP = ROOT / '.art-import-irises'
def digest(data):
    return hashlib.sha256(data).hexdigest()
def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one exact patch anchor: {old!r}; got {text.count(old)}')
    return text.replace(old, new, 1)
def write(relative, content):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))

manifest_path = ROOT / 'web/public/tiles/van-gogh/manifest.json'
before = json.loads(manifest_path.read_text())
assert '7m' in before['remaining']
assert not any(tile['tile'] == '7m' for tile in before['tiles'])
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
artwork = {p: digest((ROOT / p).read_bytes()) for p in tracked if p and p.startswith(('docs/design/', 'web/public/tiles/')) and Path(p).suffix.lower() in {'.svg', '.png', '.webp', '.jpg', '.jpeg'}}
record = json.loads((TEMP / 'provenance.json').read_text())
raster = b''.join((TEMP / f'chunk{i}.bin').read_bytes() for i in range(4))
assert len(raster) == 57100
assert digest(raster) == record['rasterSha256']
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Seven characters — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode('utf-8')
assert digest(svg) == record['svgSha256']
assert not (ROOT / record['source']).exists()
assert not (ROOT / record['runtime']).exists()
write(record['source'], svg)
write('docs/design/van-gogh/seven-characters-irises.json', json.dumps(record, ensure_ascii=False, indent=2) + '\n')
shutil.copyfile(TEMP / 'seven-characters.test.js', ROOT / 'web/tests/van-gogh-seven-characters.test.js')

exporter = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
anchor = "  sixLemonTerrace: 'docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg',"
s = replace_once(s, anchor, anchor + "\n  sevenCharactersIrises: '" + record['source'] + "',")
anchor = "  ['Lemon Terrace', '6m', 'Six characters', 'sixLemonTerrace', [0, 0, 300, 400]],"
s = replace_once(s, anchor, anchor + "\n  ['Irises C', '7m', 'Seven characters', 'sevenCharactersIrises', [0, 0, 300, 400]],")
line = next(line for line in s.splitlines() if line.startswith('  approval: '))
assert line.endswith("',")
addition = ' Carl selected the last seven-characters option with blue iris flowers and golden 萬, Irises at Dusk C, and explicitly requested GitHub deployment as 7m / 七萬. The complete composition and proportions are preserved without repainting; the olive and wheat alternatives are not used.'
s = replace_once(s, line, line[:-2] + addition + "',")
s = replace_once(s, 'Includes Lemon Terrace (6m),', 'Includes Irises at Dusk C (7m), Lemon Terrace (6m),')
line = next(line for line in s.splitlines() if line.startswith('const featured = '))
s = replace_once(s, line, "const featured = ['7m', '6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '8s', '7s'];")
s = replace_once(s, 'including Lemon Terrace for 6 characters,', 'including Irises at Dusk C for 7 characters, Lemon Terrace for 6 characters,')
s = replace_once(s, 'The Red Vineyard (5 characters), Three Café Lanterns', 'Irises at Dusk (7 characters), The Red Vineyard (5 characters), Three Café Lanterns')
exporter.write_text(s)

shared = ROOT / 'web/tests/tile-faces.test.js'
s = shared.read_text()
line = next(line for line in s.splitlines() if line.startswith('  const approved = ['))
s = replace_once(s, line, line[:-2] + ", '7m'];")
line = next(line for line in s.splitlines() if 'assert.deepEqual(set.tiles.map(tile => tile.candidate),' in line)
assert line.endswith(']);')
s = replace_once(s, line, line[:-3] + ", 'Irises C']);")
# The existing dora coverage list is distinct from the general all-tile loops.
lines = [line for line in s.splitlines() if line.strip().startswith("for (const tile of ['1z', '4z'")]
if len(lines) != 1:
    raise RuntimeError('Expected one explicit dora coverage loop')
line = lines[0]
s = replace_once(s, line, line.replace(']) {', ", '7m']) {"))
shared.write_text(s)

section = '''\n## Irises at Dusk C — 7 characters (七萬)\n\nCarl selected the last of the three seven-characters designs: blue-violet irises form 七 above a golden 萬, against the painted sunset lake and village. This is the exact approved C painting, not the olive-grove or wheat alternatives. No repainting, recolouring, additional lettering or selective cropping is applied.\n\nThe full 1086 × 1448 portrait is resized proportionally to a 300 × 400 quality-80 WebP and embedded in the standard rounded SVG with 1% bleed. The source `docs/design/van-gogh/studies/20-seven-characters-irises-c-approved.svg` and runtime `web/public/tiles/van-gogh/approved/Man7.svg` are identical. Original, raster and SVG hashes are recorded in `docs/design/van-gogh/seven-characters-irises.json`. This committed SVG is a game export, not the full-resolution original; the untouched PNG is preserved in `Van_Gogh_7_Characters_Irises_Approved.zip` supplied in chat.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=7m` to reproduce this addition without rewriting another tile. The set now has **24 painted faces and 10 Classic fallbacks**. All 23 previous faces, including restored Nine Stars I, the complete bamboo suit, Vineyard and Lemon Terrace, remain unchanged. The new regressions verify exact artwork hashes, image geometry, registration, preloading, hidden faces, the fourteen-tile preview hand and repeated selective export.\n'''
public_readme = ROOT / 'web/public/tiles/van-gogh/README.md'
s = public_readme.read_text()
s = replace_once(s, 'The twenty-three approved faces', 'The twenty-four approved faces')
s = replace_once(s, 'The other 11 identities awaiting', 'The other 10 identities awaiting')
s = replace_once(s, 'plus **Lemon Terrace** for 6 characters,', 'plus **Irises at Dusk C** for 7 characters, **Lemon Terrace** for 6 characters,')
public_readme.write_text(s.rstrip() + '\n' + section)
design_readme = ROOT / 'docs/design/van-gogh/README.md'
s = design_readme.read_text()
s, count = re.subn(r'(\*\*Current set:\*\* )\d+ painted faces and \d+ Classic fallbacks', r'\g<1>24 painted faces and 10 Classic fallbacks', s, count=1)
if count != 1:
    raise RuntimeError('Expected current-set summary in design README')
design_readme.write_text(s.rstrip() + '\n' + section)

subprocess.run(['node', str(exporter), '--only=7m'], check=True)
after = json.loads(manifest_path.read_text())
assert [tile for tile in after['tiles'] if tile['tile'] != '7m'] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['remaining'] == [tile for tile in before['remaining'] if tile != '7m']
assert after['superseded'] == before['superseded']
assert after['rejected'] == before['rejected']
assert len(after['tiles']) == len(before['tiles']) + 1 == 24
assert (ROOT / record['runtime']).read_bytes() == svg
check_files = ['web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js', record['source'], record['runtime']]
first_export = {p: (ROOT / p).read_bytes() for p in check_files}
subprocess.run(['node', str(exporter), '--only=7m'], check=True)
assert all((ROOT / p).read_bytes() == data for p, data in first_export.items())
for p, expected in artwork.items():
    assert digest((ROOT / p).read_bytes()) == expected, f'Existing artwork changed: {p}'
print(f'PRESERVED {len(artwork)} pre-existing artwork/source files byte-for-byte.')
print('Verified exact selected iris raster, original aspect ratio, unchanged existing manifest entries, and repeated selective export.')
