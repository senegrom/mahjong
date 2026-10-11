"""One-off branch-only integration of Carl's final four-plus-three apple painting."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

STAGE = Path('.apples-transfer')
SOURCE = Path('docs/design/van-gogh/studies/23-seven-disks-apples-corrected-approved.svg')
RUNTIME = Path('web/public/tiles/van-gogh/approved/Pin7.svg')
RECORD = Path('docs/design/van-gogh/seven-disks-apples.json')
EXPORTER = Path('web/scripts/export-van-gogh-tiles.mjs')
MANIFEST = Path('web/public/tiles/van-gogh/manifest.json')
TEST = Path('web/tests/van-gogh-seven-disks.test.js')
PARTS = ['70c0a5f2d60c3d74f1d8a9ae54c84f621028b0a2', '33200164186f983ed40d948f1333f1e046c92e9b', 'eecea12818443fd1820f62e28cf2d6ec19b91687', 'bed86723bd270ce6d73c0d7568cfde73557f36cb', 'f89e37fe2b9bb787fc0d103a788b7e4fe25bf484', '9b7847fdc370f70ab818947215f9536f1412900b', 'b9f524d48ad779c864de7df083fd06e6d85bacc9', 'de8905efd5dbd89ff4c64c617087cec93e053b3c', 'aaf33c719b3058d090776f095bd6f1e2385ce6ab']
RASTER_HASH = '08d3c718a05d824e63f1d9debbd3cef8edb05f00554c44e064224b59debe5668'
SVG_HASH = '20ee711b62a73dd50b54f6fd635a4fb6eb26a7c58b47501d3b38d00e4a3ce4e7'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old!r}; found {text.count(old)}')
    return text.replace(old, new, 1)


before = json.loads(MANIFEST.read_text())
assert len(before['tiles']) == 27 and '7p' in before['remaining']
assert before['fallback'] == 'text'
assert not SOURCE.exists() and not RUNTIME.exists()
protected = {}
for directory in ['web/public/tiles', 'docs/design']:
    for item in Path(directory).rglob('*'):
        if item.is_file() and item.suffix.lower() in ('.svg', '.png', '.webp', '.avif', '.jpg', '.jpeg'):
            protected[item] = digest(item.read_bytes())
chunks = []
for index, expected in enumerate(PARTS):
    part = (STAGE / f'part-{index}.bin').read_bytes()
    assert hashlib.sha1(f'blob {len(part)}\0'.encode() + part).hexdigest() == expected
    chunks.append(part)
raster = b''.join(chunks)
assert len(raster) == 56596 and digest(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">Seven disks — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,'
       + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert digest(svg) == SVG_HASH
SOURCE.write_bytes(svg)
shutil.copyfile(STAGE / 'provenance.json', RECORD)
shutil.copyfile(STAGE / 'apples.test.js', TEST)

s = EXPORTER.read_text()
anchor = "  sixKitchen: 'docs/design/van-gogh/studies/22-six-disks-provencal-kitchen-b-approved.svg',"
s = replace_once(s, anchor, anchor + f"\n  sevenApples: '{SOURCE.as_posix()}',")
anchor = "  ['Kitchen B', '6p', 'Six disks', 'sixKitchen', [0, 0, 300, 400]],"
s = replace_once(s, anchor, anchor + "\n  ['Seven Apples (corrected)', '7p', 'Seven disks', 'sevenApples', [0, 0, 300, 400]],")
approval_line = next(line for line in s.splitlines() if line.startswith('  approval: '))
assert approval_line.endswith("',")
addition = ' Carl approved the final corrected standalone Seven Apples painting for 7 disks (7p / Pin7), with four apples in the bowl and three on the cloth. Its 900 x 1200 crop is exactly 3:4 and uniformly downsampled; all seven apples remain visible without stretching, repainting or recolouring. Earlier wrong-count attempts are not used.'
s = replace_once(s, approval_line, approval_line[:-2] + addition + "',")
s = replace_once(s, 'Includes The Provencal Kitchen B (6p)', 'Includes Seven Apples (7p), The Provencal Kitchen B (6p)')
s = replace_once(s, 'approved faces, including The Provençal Kitchen B for 6 disks', 'approved faces, including Seven Apples at the Window for 7 disks, The Provençal Kitchen B for 6 disks')
EXPORTER.write_text(s)
subprocess.run(['node', str(EXPORTER), '--only=7p'], check=True)
after = json.loads(MANIFEST.read_text())
assert after['tiles'][:-1] == before['tiles']
assert after['sources'][:-1] == before['sources']
assert after['remaining'] == [tile for tile in before['remaining'] if tile != '7p']
assert after['fallback'] == before['fallback']
assert after['superseded'] == before['superseded'] and after['rejected'] == before['rejected']
assert RUNTIME.read_bytes() == svg
for item, expected in protected.items():
    assert digest(item.read_bytes()) == expected, f'Existing artwork changed: {item}'
generated = [MANIFEST, Path('web/src/lib/van-gogh-faces.js'), Path('web/public/tiles/van-gogh/preview.html'), RUNTIME]
first_export = {item: item.read_bytes() for item in generated}
subprocess.run(['node', str(EXPORTER), '--only=7p'], check=True)
for item, expected in first_export.items():
    assert item.read_bytes() == expected, f'Export not repeatable: {item}'

p = Path('web/tests/tile-faces.test.js')
s = p.read_text()
s = replace_once(s, "'4p', '8m', '6p'];", "'4p', '8m', '6p', '7p'];")
s = replace_once(s, "'Starry Olive Grove', 'Kitchen B']);", "'Starry Olive Grove', 'Kitchen B', 'Seven Apples (corrected)']);")
p.write_text(s)

section = '''\n## Seven Apples at the Window — 7 disks\n\nCarl approved the final corrected standalone painting: **four apples in the bowl and three on the cloth**, seven in total. The earlier six-apple and eight-apple attempts are not used.\n\nThe original 1024 × 1536 PNG is cropped at (62, 180) to **900 × 1200, exactly 3:4**. This removes the white border and excess setting above and below, while retaining all seven apples. Uniform Lanczos downsampling gives a 300 × 400 quality-90 WebP; the SVG adds the shared 26-unit corner clipping and 1% bleed. No stretching, regeneration, repainting or recolouring is applied. The new source and runtime SVGs are byte-identical, without needing an additional fit override.\n\nThe original and lossless crop are preserved in `Van_Gogh_7_Disks_Seven_Apples_Approved.zip` supplied in chat. The repository source SVG embeds the game-sized export, not the full-resolution original. `docs/design/van-gogh/seven-disks-apples.json` records the selection, framing, dimensions and checksums. Run `node web/scripts/export-van-gogh-tiles.mjs --only=7p` to reproduce the new face, registry, manifest and gallery.\n\nThis addition brings the set to **28 painted faces and 6 named fallbacks**. Every earlier painting, fit and manifest entry is preserved; the fourteen-tile example hand and gameplay are unchanged.\n'''
p = Path('docs/design/van-gogh/README.md')
s = replace_once(p.read_text(), '**Current set:** 27 painted faces, and 7 tiles', '**Current set:** 28 painted faces, and 6 tiles')
s = replace_once(s, 'The Provençal Kitchen B is active for 6 disks', 'Seven Apples at the Window is active for 7 disks, The Provençal Kitchen B for 6 disks')
p.write_text(s.rstrip() + '\n' + section)
p = Path('web/public/tiles/van-gogh/README.md')
s = replace_once(p.read_text(), 'The twenty-seven approved faces', 'The twenty-eight approved faces')
s = replace_once(s, 'plus **The Provençal Kitchen B**', 'plus **Seven Apples at the Window** for 7 disks, **The Provençal Kitchen B**')
s = replace_once(s, 'The other 7 identities awaiting Van Gogh artwork', 'The other 6 identities awaiting Van Gogh artwork')
p.write_text(s.rstrip() + '\n' + section)
print(f'PRESERVED {len(protected)} existing artwork/source files and all 27 prior manifest entries.')
print('VERIFIED: 28 approved, 6 remaining; exact 3:4 apple crop; reproducible selective export.')
