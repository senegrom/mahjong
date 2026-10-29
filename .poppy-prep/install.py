import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = os.environ['BASE_SHA']
FOLDER = 'docs/design/van-gogh'
OUT = 'web/public/tiles/van-gogh'
SOURCE = f'{FOLDER}/studies/36-nine-characters-poppy-hillside-b-approved.svg'
RUNTIME = f'{OUT}/approved/Man9.svg'
ARCHIVE = f'{FOLDER}/superseded/nine-characters-poppy-field-smaller-wan-Man9.svg'
RECORD = f'{FOLDER}/nine-characters-poppy-hillside-b.json'
RASTER_HASH = '046444eb5ba454bbc0a740cd965a6b8863d86898603057de1a919f2f26603bd1'
SOURCE_HASH = '7b62e183e307623673c7f004bcb04ccf5ae4991981472745fdd139dbabde4caa'
RUNTIME_HASH = '2f1303facc7868b37827b9feace26618adca6c69fd2947758b7dc92bb5416c79'
ORIGINAL_HASH = 'fb028118434d2817b2dbaaef5b3ddf9ba5f474cd03e6760c77876d1b75e6e4f9'
sha = lambda data: hashlib.sha256(data).hexdigest()
def read(p):
    return Path(p).read_text()
def write(p, text):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(text)
def replace_once(text, old, new):
    assert text.count(old) == 1, f'Expected exactly one match: {old[:150]}'
    return text.replace(old, new, 1)

before = json.loads(read(f'{OUT}/manifest.json'))
old = next(t for t in before['tiles'] if t['tile'] == '9m')
assert old['candidate'] == 'Poppies B (smaller Wan)'
assert len(before['tiles']) == 34 and not before['remaining']
assert not Path(SOURCE).exists() and not Path(ARCHIVE).exists()
paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--', 'web/public/tiles', 'docs/design']).decode().splitlines()
artwork = {p: sha(Path(p).read_bytes()) for p in paths if Path(p).suffix in {'.svg', '.png', '.webp', '.avif', '.jpg', '.jpeg'}}
write('.poppy-prep/before.json', json.dumps({'manifest': before, 'artwork': artwork}))
assert artwork[RUNTIME] == 'ed88ef4bd1f0531ba649490137076547633e5fc99b8afef9bde9af2beda4b140'
Path(ARCHIVE).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(RUNTIME, ARCHIVE)
raster = b''.join(Path(f'.poppy-prep/part{n}').read_bytes() for n in range(5))
assert len(raster) == 49281 and sha(raster) == RASTER_HASH
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Nine characters — Van Gogh — Poppy Hillside B</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n'
assert sha(svg.encode()) == SOURCE_HASH
write(SOURCE, svg)
record = {
    'tile': '9m', 'name': 'Man9', 'candidate': 'Poppy Hillside B', 'title': 'Poppy Hillside — Floral Nine B',
    'source': SOURCE, 'runtime': RUNTIME,
    'approval': 'Carl selected the middle of the three new Nine Characters candidates and explicitly requested GitHub deployment. Use the exact second painting, with the floral nine over a golden hillside, smaller floral Wan, cypresses, village and swirling sky. Neither the first meadow nor the third winding-path alternative is selected.',
    'originalFilename': 'van_gogh_style_poppy_characters_in_provence.png',
    'originalSha256': ORIGINAL_HASH, 'originalDimensions': [1086, 1448],
    'originalCrop': {'x': 0, 'y': 0, 'width': 1086, 'height': 1448},
    'originalRetention': 'The full-resolution original PNG is retained with the approval conversation and the supplied artwork archive. The repository stores the optimized approved artwork and checksums, not a full-resolution PNG archive.',
    'processing': 'Uniformly downsample the entire exact 3:4 RGB painting using Pillow Lanczos to 300 x 400, then encode AVIF quality 70, speed 0, 4:2:0. No cropping, stretching, repainting or recolouring. The standard exporter places the image at 0,0,300,400 with zero extra bleed. Only the shared rounded-corner clipping applies.',
    'rasterDimensions': [300, 400], 'rasterMimeType': 'image/avif', 'rasterBytes': len(raster),
    'rasterSha256': RASTER_HASH, 'sourceSha256': SOURCE_HASH, 'svgSha256': RUNTIME_HASH, 'runtimeSvgSha256': RUNTIME_HASH,
    'fit': {'painting': [1086, 1448], 'stretch': 0, 'anchor': 0.5, 'bleedPixels': 0},
    'supersededCandidate': old['candidate'], 'supersededSource': old['source'],
    'archivedSvg': ARCHIVE, 'archivedSvgSha256': artwork[RUNTIME],
}
write(RECORD, json.dumps(record, ensure_ascii=False, indent=2) + '\n')

p = 'web/scripts/export-van-gogh-tiles.mjs'
s = read(p)
s = replace_once(s, "  // The revised poppy portrait is wider than 3:4; keep the glyphs' exact proportions.\n  '9m': { painting: [1122, 1402], stretch: 0, anchor: 0.5, bleedPixels: 0 },", "  // Poppy Hillside B is exactly 3:4: retain the full painting and both floral glyphs.\n  '9m': { painting: [1086, 1448], stretch: 0, anchor: 0.5, bleedPixels: 0 },")
s = replace_once(s, 'const sources = {', "const sources = {\n  ninePoppyHillside: '" + SOURCE + "',")
s = replace_once(s, "['Poppies B (smaller Wan)', '9m', 'Nine characters', 'ninePoppyField', [0, 0, 300, 400]]", "['Poppy Hillside B', '9m', 'Nine characters', 'ninePoppyHillside', [0, 0, 300, 400]]")
s = replace_once(s, '  superseded: [', "  superseded: [\n    { candidate: 'Poppies B (smaller Wan)', tile: '9m', activeCandidate: 'Poppy Hillside B', source: sources.ninePoppyField, archivedSvg: '" + ARCHIVE + "' },")
s = replace_once(s, "',\n  fallback: 'text',", " Carl then selected the middle of three new Nine Characters paintings, Poppy Hillside B, and explicitly requested GitHub deployment. The complete 1086 x 1448 portrait and both floral characters are uniformly downsampled with zero stretch and zero extra bleed. The previous Poppy Field with smaller Wan remains archived as its exact playable SVG; its source and provenance are unchanged.',\n  fallback: 'text',")
s = replace_once(s, 'Poppy Field with smaller Wan (9m)', 'Poppy Hillside B (9m), replacing the earlier Poppy Field with smaller Wan')
s = replace_once(s, 'Poppy Field with smaller 萬 for 9 characters', 'Poppy Hillside B for 9 characters')
s = replace_once(s, 'Earlier selected art is preserved in lossless PNG crops.', 'Poppy Hillside B retains the complete selected 1086 × 1448 middle painting in a 300 × 400 quality-70 AVIF, without cropping or stretching and with zero extra bleed. The former Nine Characters playable SVG is archived unchanged. Earlier selected art is preserved in lossless PNG crops.')
write(p, s)

p = 'web/tests/tile-faces.test.js'
s = replace_once(read(p), "'Poppies B (smaller Wan)'", "'Poppy Hillside B'")
write(p, s)
p = 'web/tests/tile-proportions.test.js'
s = replace_once(read(p), "['2z', '3p', '4p', '6p', '7s', '8s', '9m', '9p']", "['2z', '3p', '4p', '6p', '7s', '8s', '9p']")
s = replace_once(s, '  assert.deepEqual(squeezed.sort(),', '  // Poppy Hillside B also replaces a squeezed export: 9m is now native 3:4.\n  assert.deepEqual(squeezed.sort(),')
write(p, s)

p = 'web/tests/van-gogh-nine-characters.test.js'
s = read(p)
for a, b in [
    ('nine-characters-poppy-field.json', 'nine-characters-poppy-hillside-b.json'),
    ('nine characters uses the approved revised poppy field with smaller Wan', 'nine characters uses the exact selected middle Poppy Hillside B'),
    ("'Poppies B (smaller Wan)'", "'Poppy Hillside B'"),
    ('van_gogh_style_poppy_field_characters.png', 'van_gogh_style_poppy_characters_in_provence.png'),
    ('ff3137d35a61f8569d3f6c7ff4c98dc2cc2809fe2b8d0b49be109d20a3ff4ac6', ORIGINAL_HASH),
    ('99f6155e1a7972f082fac869aca5e24559d65508be1ad175c2457b578372825a', SOURCE_HASH),
    ('ed88ef4bd1f0531ba649490137076547633e5fc99b8afef9bde9af2beda4b140', RUNTIME_HASH),
    ('5ddc2af7ede1d69f7a51e71ab308afcca157fbaf9b5a2cdfd8b09e56cc539f03', RASTER_HASH),
    ('Poppies B \\(smaller Wan\\) · Nine characters', 'Poppy Hillside B · Nine characters'),
]:
    s = replace_once(s, a, b)
assert s.count('[1122, 1402]') == 2
s = s.replace('[1122, 1402]', '[1086, 1448]')
s = replace_once(s, '  assert.deepEqual(entry.fit, fitRecord(record.fit, record.rasterDimensions));', '  assert.deepEqual(entry.fit, fitRecord(record.fit, record.rasterDimensions));\n  assert.deepEqual(entry.fit.image, { x: 0, y: 0, width: 300, height: 400 });\n  assert.equal(raster.length, 49281);')
s += """

test('the previous poppy face remains archived exactly and recent replacements stay selected', () => {
  const previous = set.superseded.find(tile => tile.tile === '9m' && tile.candidate === 'Poppies B (smaller Wan)');
  assert.equal(previous.activeCandidate, 'Poppy Hillside B');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(hash(read(record.archivedSvg)), 'ed88ef4bd1f0531ba649490137076547633e5fc99b8afef9bde9af2beda4b140');
  assert.equal(record.archivedSvgSha256, hash(read(record.archivedSvg)));
  assert.equal(hash(read(previous.source)), '99f6155e1a7972f082fac869aca5e24559d65508be1ad175c2457b578372825a');
  const historical = JSON.parse(read('docs/design/van-gogh/nine-characters-poppy-field.json'));
  assert.equal(historical.source, previous.source);
  assert.equal(historical.svgSha256, record.archivedSvgSha256);
  assert.equal(set.tiles.length, 34);
  assert.deepEqual(set.remaining, []);
  for (const [tile, candidate] of Object.entries({
    '2s': 'Flutes B', '3s': 'Vase A', '4s': 'Sunlit Garden B',
    '6m': 'Scarlet Loom C', '5z': 'Almond Blossom C', '7z': 'Painter C',
  })) assert.equal(set.tiles.find(entry => entry.tile === tile).candidate, candidate);
});
"""
write(p, s)

section = '\n\n## Nine Characters — Poppy Hillside B (9m / Man9)\n\nCarl selected the middle of the three new Nine Characters proposals and explicitly requested GitHub deployment. Use the exact second painting, with the floral nine over a golden hillside, smaller floral Wan, cypresses, village and swirling sky. The first meadow and third winding-path alternatives are not selected. The entire 1086 × 1448 portrait is uniformly downsampled to 300 × 400 and encoded as quality-70 AVIF, without cropping, stretching, repainting or recolouring. Runtime placement has zero extra bleed; only standard rounded-corner clipping applies.\n\nThe former Poppy Field with smaller Wan playable SVG is archived byte-for-byte at `docs/design/van-gogh/superseded/nine-characters-poppy-field-smaller-wan-Man9.svg`. Its study and historical provenance remain unchanged. The full-resolution selected PNG is retained with the approval conversation and supplied download archive; the repository contains the optimized source and checksums, not a full-resolution PNG archive. See `docs/design/van-gogh/nine-characters-poppy-hillside-b.json`. Reproduce only this tile with `node web/scripts/export-van-gogh-tiles.mjs --only=9m`. All other 33 Van Gogh faces and all other sets remain unchanged.\n'
for p in (f'{FOLDER}/README.md', f'{OUT}/README.md'):
    write(p, read(p) + section)
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=9m'], check=True)
assert sha(Path(RUNTIME).read_bytes()) == RUNTIME_HASH
print('ASSEMBLED POPPY HILLSIDE B', RUNTIME_HASH)
print(json.dumps(record, ensure_ascii=False, indent=2))
