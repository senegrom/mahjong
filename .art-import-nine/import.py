"""One-off, branch-only import of the user-approved moonlit nine-bamboo chime."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path.cwd()
STAGE = ROOT / '.art-import-nine'
SVG_SOURCE = 'docs/design/van-gogh/studies/16-nine-bamboo-wind-chime-a-approved.svg'
SVG_RUNTIME = 'web/public/tiles/van-gogh/approved/Sou9.svg'
PROVENANCE = 'docs/design/van-gogh/nine-bamboo-wind-chime.json'
MANIFEST = 'web/public/tiles/van-gogh/manifest.json'
EXPORTER = 'web/scripts/export-van-gogh-tiles.mjs'
TEST = 'web/tests/van-gogh-nine-bamboo.test.js'
CHUNKS = ['1ff7cc4daedb2cc441720642a801ba1c916f4921', '2442805033e2e67f6a2b3335987de00143899dda', '91b24f5aaf4b9b08de75a6371b9a6e5ed4f47650', '64107ad8dbce21e02ac2e602e49cec7f3a727b8a']
RASTER_HASH = 'ab954a7208567a80d87d3be0f0ba15373dd5cc644feb8c766880f404435dc2f3'
SVG_HASH = '8085bc2a65d372c8b28fc673063a4fd3de090849e4de6749a96de6c719da74ac'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one exact patch anchor: {old!r}')
    return text.replace(old, new, 1)


before = json.loads(Path(MANIFEST).read_text())
assert '9s' in before['remaining']
assert not any(tile['tile'] == '9s' for tile in before['tiles'])
# Preserve every existing playable image and every recorded source, not just this suit.
protected = {p: digest(p.read_bytes()) for p in Path('web/public/tiles').rglob('*') if p.is_file() and p.suffix in ('.svg', '.png', '.webp')}
for source in before['sources']:
    p = Path(source['source'])
    protected[p] = digest(p.read_bytes())
parts = []
for index, expected in enumerate(CHUNKS):
    part = (STAGE / f'part-{index}.bin').read_bytes()
    assert hashlib.sha1(f'blob {len(part)}\0'.encode() + part).hexdigest() == expected
    parts.append(part)
raster = b''.join(parts)
assert len(raster) == 48396 and digest(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">Nine bamboo — Van Gogh</title>'
       '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" '
       'href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == SVG_HASH
Path(SVG_SOURCE).write_bytes(svg)
shutil.copyfile(STAGE / 'provenance.json', PROVENANCE)
shutil.copyfile(STAGE / 'nine.test.js', TEST)

s = Path(EXPORTER).read_text()
anchor = "  sixStillLife: 'docs/design/van-gogh/studies/15-six-bamboo-green-still-life-b-approved.svg',"
s = replace_once(s, anchor, anchor + f"\n  nineWindChime: '{SVG_SOURCE}',")
anchor = "  ['Copper Sunset', '5s', 'Five bamboo', 'copperFive', [0, 0, 300, 400]],"
s = replace_once(s, anchor, anchor + "\n  ['Wind Chime A', '9s', 'Nine bamboo', 'nineWindChime', [0, 0, 300, 400]],")
s = replace_once(s, "its complete composition and colours are preserved.',", "its complete composition and colours are preserved. Carl selected the first nine-bamboo option, Moonlit Bamboo Wind Chime, and explicitly approved deployment; the nine hanging tubes and complete painted composition are preserved.',")
s = replace_once(s, 'Includes Green Still Life B (6s)', 'Includes Moonlit Wind Chime A (9s), completing the bamboo suit, Green Still Life B (6s)')
s = replace_once(s, "const featured = ['6s',", "const featured = ['9s', '6s',")
s = replace_once(s, 'approved faces, including Green Still Life B', 'approved faces, including Moonlit Wind Chime for 9 bamboo, Green Still Life B')
s = replace_once(s, 'Green Still Life B (6 bamboo) and Bamboo Raft preserve', 'Moonlit Wind Chime (9 bamboo), Green Still Life B (6 bamboo) and Bamboo Raft preserve')
Path(EXPORTER).write_text(s)
subprocess.run(['node', EXPORTER, '--only=9s'], check=True)
after = json.loads(Path(MANIFEST).read_text())
assert len(after['tiles']) == len(before['tiles']) + 1
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['remaining'] == [t for t in before['remaining'] if t != '9s']
assert Path(SVG_RUNTIME).read_bytes() == svg
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Existing artwork changed: {p}'
print(f'PRESERVED {len(protected)} existing artwork/source files byte-for-byte.')

test_path = Path('web/tests/tile-faces.test.js')
s = test_path.read_text()
old = "  const approved = ['1p', '5p', '3s', '3m', '7z', '1s', '2p', '9p', '6s', '5z', '1z', '4z', '2m', '4m', '2s', '4s', '7s', '8s', '5s'];"
s = replace_once(s, old, old[:-2] + ", '9s'];")
s = replace_once(s, "'Seven C', 'Bamboo Raft', 'Copper Sunset']);", "'Seven C', 'Bamboo Raft', 'Copper Sunset', 'Wind Chime A']);")
s = replace_once(s, "TILE_IMAGE_URLS.filter(url => url.startsWith('tiles/van-gogh/')).length, 19", "TILE_IMAGE_URLS.filter(url => url.startsWith('tiles/van-gogh/')).length, VAN_GOGH_APPROVED.length")
s = s.replace('assert.equal(set.tiles.length, 19);', 'assert.equal(set.tiles.length, VAN_GOGH_APPROVED.length);')
s = s.replace('assert.equal(set.remaining.length, 15);', 'assert.equal(set.remaining.length, 34 - VAN_GOGH_APPROVED.length);')
s = replace_once(s, "'4s', '6s', '7s', '8s', '5s'])", "'4s', '6s', '7s', '8s', '5s', '9s'])")
test_path.write_text(s)

section = '''\n## Moonlit Bamboo Wind Chime — 9 bamboo\n\nCarl selected the **first option, Wind Chime A**, for `9s` / `Sou9` and explicitly requested deployment. Nine hanging bamboo tubes are arranged in three groups of three against a swirling cobalt sky and golden moon. The support rail, copper-orange cords, foliage and village remain part of the exact approved painting. No repainting, recolouring or additional symbols were applied.\n\nThe source `docs/design/van-gogh/studies/16-nine-bamboo-wind-chime-a-approved.svg` and playable `web/public/tiles/van-gogh/approved/Sou9.svg` are byte-for-byte identical. They embed a proportional 300 × 400 quality-80 WebP export of the complete 1086 × 1448 portrait, with the shared rounded clipping and 1% bleed. This is a game-sized export, not the full-resolution original. Original, raster and SVG hashes are recorded in `docs/design/van-gogh/nine-bamboo-wind-chime.json`; the untouched original is preserved in `Van_Gogh_9_Bamboo_Wind_Chime_Approved.zip` supplied in chat.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=9s` to reproduce the approved SVG, registration, manifest and preview without rewriting any other tile artwork.\n\nThis addition completes **all nine bamboo identities**. The set now has **20 painted faces and 14 Classic fallbacks**. All nineteen previously approved faces are unchanged.\n'''
p = Path('web/public/tiles/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, 'The nineteen approved faces appear', 'The twenty approved faces appear')
s = replace_once(s, 'The other 15 identities awaiting Van Gogh artwork', 'The other 14 identities awaiting Van Gogh artwork')
s = replace_once(s, '**Bamboo Raft** for 8 bamboo', '**Moonlit Wind Chime A** for 9 bamboo, **Bamboo Raft** for 8 bamboo')
p.write_text(s.rstrip() + '\n' + section)
p = Path('docs/design/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, '# Van Gogh\n', '# Van Gogh\n\n**Current set:** 20 painted faces and 14 Classic fallbacks; all nine bamboo identities are now covered. The approved Moonlit Wind Chime A is active for 9 bamboo. The development history below retains earlier milestone counts.\n')
p.write_text(s.rstrip() + '\n' + section)
# The branch-only importer must never become part of the final main-branch diff.
shutil.rmtree(STAGE)
Path('.github/workflows/import-van-gogh-nine.yml').unlink()
print('Approved Wind Chime A imported. Temporary transport files and workflow removed.')
