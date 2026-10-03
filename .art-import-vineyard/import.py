"""Branch-only import of Carl's approved first, vineyard, five-characters painting."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path.cwd()
TEMP = ROOT / '.art-import-vineyard'
SOURCE = 'docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Man5.svg'
MANIFEST = ROOT / 'web/public/tiles/van-gogh/manifest.json'
PROVENANCE = 'docs/design/van-gogh/five-characters-vineyard.json'
TEST = 'web/tests/van-gogh-five-characters.test.js'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old!r}')
    return text.replace(old, new, 1)


old_manifest = json.loads(MANIFEST.read_text())
assert len(old_manifest['tiles']) == 21
assert '5m' in old_manifest['remaining']
assert not (ROOT / SOURCE).exists() and not (ROOT / RUNTIME).exists()
# Audit all existing source and runtime artwork across every face set, not only this set.
tracked = subprocess.check_output(['git', 'ls-files', '-z'], text=True).split('\0')
images = [p for p in tracked if p.startswith(('docs/design/', 'web/public/tiles/'))
          and Path(p).suffix.lower() in ('.svg', '.png', '.webp', '.jpg', '.jpeg')]
before = {p: digest((ROOT / p).read_bytes()) for p in images}
raster = b''.join((TEMP / f'chunk{i}.bin').read_bytes() for i in range(4))
assert len(raster) == 55626
assert digest(raster) == 'c7b5707a3a4ac45425326ba9f38d29b47d5cb52b0a3f2a9d80d1ee5e561e7922'
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" '
       'role="img" aria-labelledby="title"><title id="title">Five characters — Van Gogh</title>'
       '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" '
       'preserveAspectRatio="none" href="data:image/webp;base64,'
       + base64.b64encode(raster).decode('ascii') + '"/></svg>\n').encode('utf-8')
assert digest(svg) == '167d067218dcb1a9ae71ba43e0f4d4552ef39057c84c6f0d7f52ec566a3a9280'
(ROOT / SOURCE).write_bytes(svg)
shutil.copyfile(TEMP / 'provenance.json', ROOT / PROVENANCE)
shutil.copyfile(TEMP / 'five-characters.test.js', ROOT / TEST)

exporter = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
s = replace_once(s,
    "  threeLanterns: 'docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg',",
    "  threeLanterns: 'docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg',\n"
    f"  fiveVineyard: '{SOURCE}',")
s = replace_once(s,
    "  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],",
    "  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],\n"
    "  ['Vineyard A', '5m', 'Five characters', 'fiveVineyard', [0, 0, 300, 400]],")
s, count = re.subn(r"(  approval: '[^\n]*)(',\n)",
    lambda m: m[1] + ' Carl selected the first option, The Red Vineyard, for 5 characters (5m / 五萬) and explicitly approved GitHub deployment. The complete approved composition and colours are preserved; the wheat and iris alternatives are not used.' + m[2], s)
assert count == 1
s = replace_once(s, 'Includes Three Cafe Lanterns B (3p)',
                 'Includes The Red Vineyard A (5m), Three Cafe Lanterns B (3p)')
s = replace_once(s,
    "const featured = ['3p', '9s', '6s', '5s', '8s', '7s', '4s', '3s', '2s', '2m', '3m', '4m'];",
    "const featured = ['5m', '2m', '3m', '4m', '3p', '9s', '6s', '5s', '8s', '7s', '4s', '3s'];")
s = replace_once(s, 'approved faces, including Three Café Lanterns',
                 'approved faces, including The Red Vineyard for 5 characters, Three Café Lanterns')
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

section = '''\n## The Red Vineyard A — 5 characters\n\nCarl selected the **first option, The Red Vineyard**, for `5m` / `Man5` (`五萬`) and explicitly requested deployment. Copper-red vines and grape clusters form the characters over the golden vineyard landscape. The wheat and iris alternatives are not used. The entire approved composition and colours are retained without repainting or recolouring.\n\nThe complete 1086 × 1448 portrait is resized proportionally with Lanczos to a 300 × 400 quality-80 WebP, following the raft and green still-life export convention. The source `docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg` and game `web/public/tiles/van-gogh/approved/Man5.svg` are identical and use the standard rounded clip and 1% bleed. This is an optimized game export, not the full-resolution original PNG. Original, raster and SVG checksums are recorded in `docs/design/van-gogh/five-characters-vineyard.json`; the untouched original is preserved in `Van_Gogh_5_Characters_Vineyard_Approved.zip` supplied in chat.\n\nRegenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=5m`. All 21 previously approved Van Gogh faces and all other sets remain unchanged. This addition brings the set to **22 painted faces and 12 Classic fallbacks**, with all nine bamboo faces preserved.\n'''
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
