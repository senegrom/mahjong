from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

root = Path.cwd()
out = root / 'web/public/tiles/van-gogh'
archive = root / 'docs/design/van-gogh/superseded'
original_png = archive / 'nine-stars-Pin9.png'
original_svg = archive / 'nine-stars-Pin9.svg'
potters_archive = archive / 'potters-table-Pin9.svg'
png_hash = '61eb9c2563a8039eb755315b1f5ef670a472fcff2e1facaef4b0e05e1af0267d'
svg_hash = '34f0fd31056df47c064383ca79b67d17aa9bd5a62a7aae1d3f9188701e98a89a'
potters_hash = '7bfed3b794e8a09dc61f2020662d9c70572d883f05194267df07f535778591b1'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch anchor changed: ' + old)
    return text.replace(old, new, 1)

before = json.loads((out / 'manifest.json').read_text())
old_entry = next(t for t in before['tiles'] if t['tile'] == '9p')
assert old_entry['candidate'] == 'Disks C'
assert digest(original_png.read_bytes()) == png_hash
assert digest(original_svg.read_bytes()) == svg_hash
assert struct.unpack('>II', original_png.read_bytes()[16:24]) == (380, 471)
potters = (out / 'approved/Pin9.svg').read_bytes()
assert digest(potters) == potters_hash
assert potters == (root / old_entry['source']).read_bytes()
assert not potters_archive.exists() or potters_archive.read_bytes() == potters
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
artwork = {p: digest((root / p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in {'.svg', '.png', '.webp', '.jpg', '.jpeg'} and (root / p).is_file()}

script_path = root / 'web/scripts/export-van-gogh-tiles.mjs'
script = script_path.read_text()
script = once(script,
    "  pottersTable: 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg',",
    "  pottersTable: 'docs/design/van-gogh/studies/17-nine-disks-potters-table-c-approved.svg',\n  nineStarsOriginal: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png',")
script = once(script,
    "['Disks C', '9p', 'Nine disks', 'pottersTable', [0, 0, 300, 400]]",
    "['I', '9p', 'Nine disks', 'nineStarsOriginal', [0, 0, 380, 471]]")
script = once(script, "new Set(['nightCafe', 'cypressFields'])", "new Set(['nightCafe', 'cypressFields', 'nineStarsOriginal'])")
script = once(script,
    "{ candidate: 'I', tile: '9p', activeCandidate: 'Disks C', source: sources.second, archivedPng: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png', archivedSvg: 'docs/design/van-gogh/superseded/nine-stars-Pin9.svg' }",
    "{ candidate: 'Disks C', tile: '9p', activeCandidate: 'I', source: sources.pottersTable, archivedSvg: 'docs/design/van-gogh/superseded/potters-table-Pin9.svg' }")
script = once(script,
    'The nine patterned plates retain the selected painting; the old face is archived.',
    'The nine patterned plates were deployed and the old face archived. Carl subsequently rejected the eight-star adaptation and requested the original nine-star face as 9 disks. Nine Stars I is now restored byte-for-byte and The Potter’s Table C is archived; 8 disks is unchanged.')
script = once(script, 'The Potter’s Table C (9p), replacing Nine Stars I', 'Nine Stars I (9p), restored from the original archive')
script = once(script, 'The Potter’s Table C for 9 disks', 'Nine Stars I for 9 disks')
script = once(script,
    'The Potter’s Table C uses a documented quality-95 WebP crop; its former Nine Stars face is archived.',
    'Nine Stars I uses its exact original 380 × 471 PNG and original SVG wrapper, with no resizing or repainting. The Potter’s Table C remains preserved in the design archive.')
tile_tests_path = root / 'web/tests/tile-faces.test.js'
tile_tests = once(tile_tests_path.read_text(), "'H', 'Disks C', 'Six B (green)'", "'H', 'I', 'Six B (green)'")
new_tests = (root / '.nine-stars-restore/nine-disks.test.js').read_text()

# All patch anchors and source bytes are validated before changing anything.
archive.mkdir(parents=True, exist_ok=True)
if not potters_archive.exists():
    potters_archive.write_bytes(potters)
script_path.write_text(script)
tile_tests_path.write_text(tile_tests)
(root / 'web/tests/van-gogh-nine-disks.test.js').write_text(new_tests)
record = {
    'tile': '9p', 'name': 'Pin9', 'candidate': 'I', 'title': 'Nine Stars',
    'source': 'docs/design/van-gogh/superseded/nine-stars-Pin9.png',
    'runtimePng': 'web/public/tiles/van-gogh/approved/Pin9.png',
    'runtimeSvg': 'web/public/tiles/van-gogh/approved/Pin9.svg',
    'pngSha256': png_hash, 'svgSha256': svg_hash, 'sourceDimensions': [380, 471],
    'originalStudy': 'docs/design/van-gogh/studies/02-van-gogh-concepts.png',
    'originalStudyCrop': {'x': 838, 'y': 100, 'width': 380, 'height': 471},
    'approval': 'After viewing the original Nine Stars and rejecting its eight-star adaptation, Carl requested deployment of the nine. Restore the original 9 disks only; do not deploy the eight-star adaptation.',
    'processing': 'Exact archived PNG bytes; the existing SVG exporter reproduces the original SVG byte-for-byte. No new painting, resizing, recolouring or recompression.',
    'supersedes': {'candidate': 'Disks C', 'title': 'The Potter’s Table', 'archivedSvg': 'docs/design/van-gogh/superseded/potters-table-Pin9.svg', 'svgSha256': potters_hash, 'source': old_entry['source']},
}
(root / 'docs/design/van-gogh/nine-stars-restored.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
for relative in ('docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md'):
    path = root / relative
    text = path.read_text()
    note = '\n**Current 9 disks:** Nine Stars I has been restored from the exact original PNG/SVG. The Potter’s Table C is preserved in the design archive. The eight-star adaptation is not deployed; 8 disks is unchanged. Earlier sections below describe the design history.\n'
    text = once(text, '# Van Gogh\n', '# Van Gogh\n' + note)
    text += '\n## Nine Stars — original 9 disks restored\n\nThe original nine luminous disks in a 3×3 arrangement are active again for `9p` / `Pin9`. Both the 380×471 PNG and its existing game SVG are byte-for-byte identical to the archived originals. The Potter’s Table C source, original provenance and archived runtime SVG are all retained. No other tile or tile coverage changes.\n\n`docs/design/van-gogh/nine-stars-restored.json` records this decision and the hashes. Run `node web/scripts/export-van-gogh-tiles.mjs --only=9p` to reproduce the restored face without rewriting other artwork.\n'
    path.write_text(text)

command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=9p']
subprocess.run(command, check=True)
assert (out / 'approved/Pin9.png').read_bytes() == original_png.read_bytes()
assert (out / 'approved/Pin9.svg').read_bytes() == original_svg.read_bytes()
assert potters_archive.read_bytes() == potters
for relative, expected in artwork.items():
    if relative != 'web/public/tiles/van-gogh/approved/Pin9.svg':
        assert digest((root / relative).read_bytes()) == expected, relative
updated = json.loads((out / 'manifest.json').read_text())
assert updated['remaining'] == before['remaining']
assert [t['tile'] for t in updated['tiles']] == [t['tile'] for t in before['tiles']]
assert [t for t in updated['tiles'] if t['tile'] != '9p'] == [t for t in before['tiles'] if t['tile'] != '9p']
html = (out / 'preview.html').read_text()
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', html)
assert rack and rack.group(1).count('<img ') == 14
exports = {p: digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
subprocess.run(command, check=True)
assert all(digest(p.read_bytes()) == h for p, h in exports.items())
print(f'PASS: exact original PNG and SVG restored; Potter’s Table archived; {len(artwork)-1} other artwork files unchanged; 8 disks unchanged; all other entries and coverage unchanged; deterministic export.')
