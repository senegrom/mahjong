"""One-off GitHub-only import of Carl's approved Three Cafe Lanterns B."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

STAGE = Path('.lanterns-transfer')
SOURCE = 'docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Pin3.svg'
MANIFEST = Path('web/public/tiles/van-gogh/manifest.json')
EXPORTER = Path('web/scripts/export-van-gogh-tiles.mjs')
CHUNKS = ['95606aa2cc7e414a98ef2537e6d7dd5e77120cc1', '1084dc9404986c91f21e0cb54da7ebeabb01dec7', '323f45145950c6fbe4b9aaa55de37f9bf5351afc', '1009e3441f1aa9627ba4e5c9cf9fc0695d27bcaf', '3ea3de428dd3df3a72aeab4c152a0d9eeba1967d', '54d62c7bae77d7e9c91edc5a61470b80cec4ea56']
RASTER_HASH = '74d9a2bf1807c3f58a4f5728794d83dfacfdeac1b7fd3400051c35077d57e105'
SVG_HASH = 'a5fe322fffbe4717b57897209fca1cbaa737989bb318884a510330c5fb6d8713'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one exact patch anchor: {old!r}')
    return text.replace(old, new, 1)


before = json.loads(MANIFEST.read_text())
assert '3p' in before['remaining']
assert not any(tile['tile'] == '3p' for tile in before['tiles'])
assert len(before['tiles']) == 20
assert not Path(SOURCE).exists() and not Path(RUNTIME).exists()
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
assert len(raster) == 68688 and digest(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">Three disks — Van Gogh</title>'
       '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" '
       'href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == SVG_HASH
Path(SOURCE).write_bytes(svg)
shutil.copyfile(STAGE / 'provenance.json', 'docs/design/van-gogh/three-disks-lanterns.json')
shutil.copyfile(STAGE / 'lanterns.test.js', 'web/tests/van-gogh-three-lanterns.test.js')

s = EXPORTER.read_text()
anchor = "  nineWindChime: 'docs/design/van-gogh/studies/16-nine-bamboo-wind-chime-a-approved.svg',"
s = replace_once(s, anchor, anchor + f"\n  threeLanterns: '{SOURCE}',")
anchor = "  ['Wind Chime A', '9s', 'Nine bamboo', 'nineWindChime', [0, 0, 300, 400]],"
s = replace_once(s, anchor, anchor + "\n  ['Lanterns B', '3p', 'Three disks', 'threeLanterns', [0, 0, 300, 400]],")
s = replace_once(s, "the nine hanging tubes and complete painted composition are preserved.',", "the nine hanging tubes and complete painted composition are preserved. Carl approved B, Three Cafe Lanterns, for 3 disks and explicitly requested deployment. The complete selected middle-panel crop is preserved without repainting or recolouring in a documented quality-90 WebP game export.',")
s = replace_once(s, 'Includes Moonlit Wind Chime A (9s)', 'Includes Three Cafe Lanterns B (3p), Moonlit Wind Chime A (9s)')
s = replace_once(s, "const featured = ['9s',", "const featured = ['3p', '9s',")
s = replace_once(s, 'approved faces, including Moonlit Wind Chime', 'approved faces, including Three Café Lanterns for 3 disks, Moonlit Wind Chime')
s = replace_once(s, 'Moonlit Wind Chime (9 bamboo), Green Still Life B (6 bamboo) and Bamboo Raft preserve', 'Three Café Lanterns (3 disks), Moonlit Wind Chime (9 bamboo), Green Still Life B (6 bamboo) and Bamboo Raft preserve')
EXPORTER.write_text(s)
subprocess.run(['node', str(EXPORTER), '--only=3p'], check=True)
after = json.loads(MANIFEST.read_text())
assert len(after['tiles']) == 21 and len(after['remaining']) == 13
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['remaining'] == [t for t in before['remaining'] if t != '3p']
assert after['rejected'] == before['rejected'] and after['superseded'] == before['superseded']
assert Path(RUNTIME).read_bytes() == svg
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Existing artwork changed: {p}'
print(f'PRESERVED {len(protected)} existing artwork/source files byte-for-byte.')

# Keep the explicit approval contract; change only Van Gogh's first expectation.
test_path = Path('web/tests/tile-faces.test.js')
s = test_path.read_text()
old = '  const approved = ' + repr([t['tile'] for t in before['tiles']]) + ';'
new = '  const approved = ' + repr([t['tile'] for t in after['tiles']]) + ';'
s = replace_once(s, old, new)
old = '  assert.deepEqual(set.tiles.map(tile => tile.candidate), ' + repr([t['candidate'] for t in before['tiles']]) + ');'
new = '  assert.deepEqual(set.tiles.map(tile => tile.candidate), ' + repr([t['candidate'] for t in after['tiles']]) + ');'
s = replace_once(s, old, new)
s = replace_once(s, "'4s', '6s', '7s', '8s', '5s', '9s'])", "'4s', '6s', '7s', '8s', '5s', '9s', '3p'])")
test_path.write_text(s)

section = '''\n## Three Café Lanterns B — 3 disks\n\nCarl selected **B — Three Café Lanterns**, the middle panel of the three-disk concept board, for `3p` / `Pin3` and explicitly approved deployment. Three large golden lanterns hang above the night café. No repainting, recolouring, additional lanterns or symbol substitutions were applied.\n\nThe caption and presentation gutters are excluded with the exact `[503, 127, 442, 860]` crop from the original 1448 × 1086 board. Its entire composition is resized to the shared 300 × 400 game canvas using Lanczos and encoded as quality-90 WebP. This is an optimized game export, not a full-resolution or lossless copy. The untouched board and full-resolution selected PNG remain in the approval archive `van-gogh-3-disks-lanterns-prepared.zip` supplied in chat.\n\nSource `docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg` and runtime `web/public/tiles/van-gogh/approved/Pin3.svg` are byte-for-byte identical. The embedded image uses the existing 26-unit rounded clipping and 1% bleed. Original-board, full-resolution-crop, WebP and SVG checksums are recorded in `docs/design/van-gogh/three-disks-lanterns.json`.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=3p` to reproduce the game SVG, registration, manifest and preview without rewriting other artwork. This addition brings the set to **21 painted faces and 13 Classic fallbacks**. All twenty previously approved faces, including the complete bamboo suit, are unchanged.\n'''
p = Path('docs/design/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, '**Current set:** 20 painted faces and 14 Classic fallbacks; all nine bamboo identities are now covered. The approved Moonlit Wind Chime A is active for 9 bamboo.', '**Current set:** 21 painted faces and 13 Classic fallbacks; all nine bamboo identities are covered. Three Café Lanterns B is active for 3 disks, and Moonlit Wind Chime A remains active for 9 bamboo.')
p.write_text(s.rstrip() + '\n' + section)
p = Path('web/public/tiles/van-gogh/README.md')
s = p.read_text()
s = replace_once(s, 'The twenty approved faces', 'The twenty-one approved faces')
s = replace_once(s, 'The selected studies are A–B, E, G–I and L, plus ', 'The selected studies are A–B, E, G–I and L, plus **Three Café Lanterns B** for 3 disks, ')
s = replace_once(s, 'The other 14 identities awaiting Van Gogh artwork', 'The other 13 identities awaiting Van Gogh artwork')
p.write_text(s.rstrip() + '\n' + section)

# A second selective export must reproduce every generated byte without changing old art.
generated = [MANIFEST, Path(RUNTIME), Path('web/src/lib/van-gogh-faces.js'), Path('web/public/tiles/van-gogh/preview.html')]
first = {p: digest(p.read_bytes()) for p in generated}
subprocess.run(['node', str(EXPORTER), '--only=3p'], check=True)
assert first == {p: digest(p.read_bytes()) for p in generated}
for p, expected in protected.items():
    assert digest(p.read_bytes()) == expected, f'Existing artwork changed on repeat export: {p}'
print('REPRODUCIBLE: selective export is byte-identical; 21 approved, 13 remaining.')
