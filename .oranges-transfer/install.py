"""Branch-only, checksum-verified import of Carl's approved Four Oranges A."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

STAGE = Path('.oranges-transfer')
SOURCE = 'docs/design/van-gogh/studies/20-four-disks-oranges-a-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Pin4.svg'
PROVENANCE = 'docs/design/van-gogh/four-disks-oranges.json'
EXPORTER = 'web/scripts/export-van-gogh-tiles.mjs'
MANIFEST = 'web/public/tiles/van-gogh/manifest.json'
NEW_TEST = 'web/tests/van-gogh-four-oranges.test.js'
EXPECTED = ['88e7826f57c84c34c48d67e36d9a2619be61c1c1', '96192867163aba02ace8fb3f2be65180b0f5c4a2', '572b7afcee99f5c5a8542757fcf14b950bb8de73', '6cbc0209888c7703c2e66da58ca8ab913be280e0', '9793d092c780251d9a8afa80bac48825df9fcd5f', '0ad0508e2e5064e82469c79204e92b5a0666e169']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_hash(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one patch anchor: {old!r}')
    return text.replace(old, new, 1)


before = json.loads(Path(MANIFEST).read_text())
assert '4p' in before['remaining']
assert not any(t['tile'] == '4p' for t in before['tiles'])
protected = {p: digest(p.read_bytes()) for p in Path('web/public/tiles').rglob('*') if p.is_file() and p.suffix in ('.svg', '.png', '.webp')}
for s in before['sources']:
    p = Path(s['source'])
    protected[p] = digest(p.read_bytes())
parts = []
for i, expected in enumerate(EXPECTED):
    part = (STAGE / f'part-{i}.bin').read_bytes()
    if i == 1:
        # Repair the two verified transport bytes, before any source validation.
        assert git_hash(part) == '61eaeba750edc7c5d4979d1484d10df25d7419f6'
        assert len(part) == 12000 and part[8982:8984] == bytes([102, 119])
        part = part[:8982] + bytes([100, 215]) + part[8984:]
    assert git_hash(part) == expected, f'Chunk {i} checksum mismatch'
    parts.append(part)
raster = b''.join(parts)
assert len(raster) == 66718
assert digest(raster) == 'e2dcad697285d2baa13e0b4f227b1e2f7016279f15252cd4aa5f53fe6922a9ba'
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">Four disks — Van Gogh</title>'
       '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" '
       'href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == '52d7428647df0da97da13594c0913a4849e723ed9ff5d921b5fe2b3b6944a4d1'
Path(SOURCE).write_bytes(svg)
shutil.copyfile(STAGE / 'provenance.json', PROVENANCE)
shutil.copyfile(STAGE / 'oranges.test.js', NEW_TEST)

s = Path(EXPORTER).read_text()
anchor = "  sixLemonTerrace: 'docs/design/van-gogh/studies/19-six-characters-lemon-terrace-approved.svg',"
s = replace_once(s, anchor, anchor + f"\n  fourOranges: '{SOURCE}',")
anchor = "  ['Lemon Terrace', '6m', 'Six characters', 'sixLemonTerrace', [0, 0, 300, 400]],"
s = replace_once(s, anchor, anchor + "\n  ['Oranges A', '4p', 'Four disks', 'fourOranges', [0, 0, 300, 400]],")
s = replace_once(s, "the sunflower study is not selected.',", "the sunflower study is not selected. Carl selected A, Four Oranges, the left panel with four whole oranges on blue cloth, for 4 disks (4p / Pin4) and explicitly requested GitHub deployment. The complete approved crop is retained without repainting or recolouring; the bowl and orange-slice alternatives are not used.',")
s = replace_once(s, 'Includes Lemon Terrace (6m)', 'Includes Four Oranges A (4p), Lemon Terrace (6m)')
s = replace_once(s, "const featured = ['6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '5s', '8s', '7s'];", "const featured = ['4p', '6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s', '5s', '8s'];")
s = replace_once(s, 'approved faces, including Lemon Terrace', 'approved faces, including Four Oranges A for 4 disks, Lemon Terrace')
Path(EXPORTER).write_text(s)
subprocess.run(['node', EXPORTER, '--only=4p'], check=True)
after = json.loads(Path(MANIFEST).read_text())
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['remaining'] == [t for t in before['remaining'] if t != '4p']
assert Path(RUNTIME).read_bytes() == svg

p = Path('web/tests/tile-faces.test.js')
s = p.read_text()
s = replace_once(s, "'9s', '3p', '5m', '6m'];", "'9s', '3p', '5m', '6m', '4p'];")
s = replace_once(s, "'Lanterns B', 'Vineyard A', 'Lemon Terrace']);", "'Lanterns B', 'Vineyard A', 'Lemon Terrace', 'Oranges A']);")
s = replace_once(s, "'9s', '3p', '5m', '6m'])", "'9s', '3p', '5m', '6m', '4p'])")
p.write_text(s)

section = '''\n## Four Oranges A — 4 disks\n\nCarl selected **A**, the left panel with **four whole oranges on blue cloth**, for `4p` / `Pin4` and explicitly requested GitHub deployment. The blue-and-white jug, leafy branch, yellow wall, window and distant village are preserved. The bowl (B) and orange slices (C) are not used.\n\nThe source `docs/design/van-gogh/studies/20-four-disks-oranges-a-approved.svg` and playable `web/public/tiles/van-gogh/approved/Pin4.svg` are identical. They embed a quality-90 300 × 400 WebP of the complete approved crop `[25, 138, 442, 796]` from the 1448 × 1086 concept board, with the set's 26-unit corners and 1% bleed. The label and presentation gutters were excluded; nothing was repainted or recoloured. This is a game-sized export, not the full-resolution original.\n\nBoard, full-resolution crop, decoded crop pixels, raster and SVG checksums are recorded in `docs/design/van-gogh/four-disks-oranges.json`. The original board and lossless selected crop are preserved in `Van_Gogh_4_Disks_Oranges_A_Approved.zip`, supplied in chat.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=4p` to reproduce this tile, its registration, manifest and preview without rewriting any other tile artwork. The set now has **24 painted faces and 10 Classic fallbacks**. All 23 previously approved faces remain unchanged.\n'''
p = Path('docs/design/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, '**Current set:** 23 painted faces and 11 Classic fallbacks;', '**Current set:** 24 painted faces and 10 Classic fallbacks;')
s = replace_once(s, 'Lemon Terrace is active for 6 characters,', 'Four Oranges A is active for 4 disks, Lemon Terrace for 6 characters,')
p.write_text(s.rstrip() + '\n' + section)
p = Path('web/public/tiles/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, 'The twenty-three approved faces', 'The twenty-four approved faces')
s = replace_once(s, 'plus **Lemon Terrace** for 6 characters,', 'plus **Four Oranges A** for 4 disks, **Lemon Terrace** for 6 characters,')
s = replace_once(s, 'The other 11 identities awaiting Van Gogh artwork', 'The other 10 identities awaiting Van Gogh artwork')
p.write_text(s.rstrip() + '\n' + section)

outputs = [Path(p) for p in (SOURCE, RUNTIME, MANIFEST, 'web/src/lib/van-gogh-faces.js', 'web/public/tiles/van-gogh/preview.html')]
first = {p: digest(p.read_bytes()) for p in outputs}
subprocess.run(['node', EXPORTER, '--only=4p'], check=True)
assert {p: digest(p.read_bytes()) for p in outputs} == first
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Existing artwork changed: {p}'
print(f'PRESERVED {len(protected)} existing artwork/source files byte-for-byte.')
print(f'REPRODUCIBLE: {len(after["tiles"])} approved; {len(after["remaining"])} remaining.')
