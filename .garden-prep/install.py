import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE = os.environ['BASE_SHA']
FOLDER = 'docs/design/van-gogh'
OUT = 'web/public/tiles/van-gogh'
SOURCE = f'{FOLDER}/studies/35-four-bamboo-sunlit-garden-b-approved.svg'
RUNTIME = f'{OUT}/approved/Sou4.svg'
OLD_PNG = f'{OUT}/approved/Sou4.png'
ARCHIVE = f'{FOLDER}/superseded/four-bamboo-moonlit-four-Sou4'
RECORD = f'{FOLDER}/four-bamboo-sunlit-garden-b.json'
EXPECTED_RASTER = 'bee554c0d08868e23f20ad7b82dd57df88dbd25b19fc87ea297c2ddb6dfc2b75'
EXPECTED_SOURCE = 'dad4f4cdf389ef5dbadbe82261e036054b5325e98b4e5eb1d562c13cc4749a59'
EXPECTED_RUNTIME = '574e208e0e6dbb4a247b1d58fc09a2e9a610bf43d64d9d7a2948d1f5c2efe841'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return Path(path).read_text()

def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)

def replace_once(text, old, new):
    assert text.count(old) == 1, f'Expected one match: {old[:100]}'
    return text.replace(old, new, 1)

before = json.loads(read(f'{OUT}/manifest.json'))
old = next(t for t in before['tiles'] if t['tile'] == '4s')
assert old['candidate'] == 'Bamboo C (green)' and len(before['tiles']) == 34
paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--', 'web/public/tiles', 'docs/design']).decode().splitlines()
artwork = {p: sha(Path(p).read_bytes()) for p in paths if Path(p).suffix in {'.svg', '.png', '.webp', '.avif'}}
write('.garden-prep/before.json', json.dumps({'manifest': before, 'artwork': artwork}))
for suffix in ('.svg', '.png'):
    Path(ARCHIVE + suffix).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f'{OUT}/approved/Sou4{suffix}', ARCHIVE + suffix)
Path(OLD_PNG).unlink()
raster = b''.join(Path(f'.garden-prep/part{n}').read_bytes() for n in range(4))
assert len(raster) == 48483 and sha(raster) == EXPECTED_RASTER
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Four bamboo — Van Gogh — Sunlit Garden B</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n'
assert sha(svg.encode()) == EXPECTED_SOURCE
write(SOURCE, svg)
record = {
    'tile': '4s', 'name': 'Sou4', 'candidate': 'Sunlit Garden B', 'title': 'Four Bamboo in the Sunlit Garden',
    'source': SOURCE, 'runtime': RUNTIME,
    'approval': 'Carl selected the middle sunlit garden painting from the four-bamboo triptych and explicitly requested GitHub deployment. Neither moonlit alternative is selected.',
    'originalBoardFilename': 'a_triptych_three_panel_painting_illustration_in_a.png',
    'originalBoardDimensions': [1882, 836],
    'originalBoardSha256': '030abbb0fa849e6b87fa9ba77f29576ce88e3890453c260c5e23cbb529b9b4b2',
    'originalCrop': {'x': 640, 'y': 16, 'width': 600, 'height': 800},
    'fullResolutionCropSha256': 'c416e891d268623b2db453e4e11c8eeb5a8080541c90e278d1e0f7b564c311b5',
    'originalRetention': 'The original triptych and lossless selected crop are retained with the approval conversation. This repository stores the optimized selected artwork and provenance, not a full-resolution PNG archive.',
    'processing': 'Mechanically extract the middle panel at x=640,y=16,width=600,height=800, removing the white gutters and panel fringe. Keep all four bamboo stalks, the stone wall and terracotta pot. Uniformly downsample this exact 3:4 crop with Pillow Lanczos to 300 x 400 and encode AVIF quality 70, speed 0, 4:2:0. No stretching, repainting or recolouring. The normal exporter uses zero extra bleed; standard rounded tile clipping is unchanged.',
    'rasterDimensions': [300, 400], 'rasterMimeType': 'image/avif', 'rasterBytes': len(raster),
    'rasterSha256': EXPECTED_RASTER, 'svgSha256': EXPECTED_SOURCE, 'runtimeSvgSha256': EXPECTED_RUNTIME,
    'supersededCandidate': old['candidate'], 'supersededSource': old['source'],
    'archivedSvg': ARCHIVE + '.svg', 'archivedSvgSha256': artwork[RUNTIME],
    'archivedPng': ARCHIVE + '.png', 'archivedPngSha256': artwork[OLD_PNG],
}
write(RECORD, json.dumps(record, ensure_ascii=False, indent=2) + '\n')

p = 'web/scripts/export-van-gogh-tiles.mjs'
s = read(p)
s = replace_once(s, '// studies for 4s, 7s, 8s, 3p, 4p, 6p and 9p were resized from other', '// studies for 7s, 8s, 3p, 4p, 6p and 9p were resized from other')
s = replace_once(s, "  '4s': { painting: [438, 671], stretch: 0.02, anchor: 0.5 },", "  // Sunlit Garden B is an exact 3:4 crop, retaining the four stalks and pot.\n  '4s': { painting: [600, 800], stretch: 0, anchor: 0.5, bleedPixels: 0 },")
s = replace_once(s, 'const sources = {', "const sources = {\n  fourSunlitGarden: '" + SOURCE + "',")
s = replace_once(s, "['Bamboo C (green)', '4s', 'Four bamboo', 'moonlitFourGreen', [0, 0, 300, 400]]", "['Sunlit Garden B', '4s', 'Four bamboo', 'fourSunlitGarden', [0, 0, 300, 400]]")
s = replace_once(s, '  superseded: [', "  superseded: [\n    { candidate: 'Bamboo C (green)', tile: '4s', activeCandidate: 'Sunlit Garden B', source: sources.moonlitFourGreen, archivedPng: '" + ARCHIVE + ".png', archivedSvg: '" + ARCHIVE + ".svg' },")
s = replace_once(s, "',\n  fallback: 'text',", " Carl selected the middle Sunlit Garden B painting for Four Bamboo (4s / Sou4) and explicitly requested GitHub deployment. Its documented 600 x 800 portrait crop retains the four stalks, stone wall and terracotta pot, without stretching or repainting. The previous Moonlit Four PNG and SVG remain archived.',\n  fallback: 'text',")
s = replace_once(s, '. Includes ', '. Includes Sunlit Garden B (4s), replacing Moonlit Four, ')
s = replace_once(s, 'green Moonlit Four for 4 bamboo', 'Sunlit Garden B for 4 bamboo')
s = replace_once(s, 'Earlier selected art is preserved in lossless PNG crops.', 'Sunlit Garden B uses the selected middle-panel 600 × 800 crop in a 300 × 400 quality-70 AVIF export, with zero stretch and zero extra bleed. All four stalks, the stone wall and terracotta pot are retained. The former Moonlit Four PNG and SVG remain archived. Earlier selected art is preserved in lossless PNG crops.')
write(p, s)

p = 'web/tests/tile-faces.test.js'
s = read(p)
s = replace_once(s, "'Flutes B', 'Bamboo C (green)', 'Seven C'", "'Flutes B', 'Sunlit Garden B', 'Seven C'")
s = replace_once(s, "['2s', '3s', '9m', '6z',", "['2s', '3s', '4s', '9m', '6z',")
start = s.index("test('Van Gogh greener Moonlit Four C is the approved four bamboo'")
end = s.index("test('Van Gogh Seven with Irises C", start)
s = s[:start] + """test('Van Gogh Sunlit Garden B replaces four bamboo while Moonlit Four remains recoverable', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '4s');
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.equal(set.tiles.filter(tile => tile.tile === '4s').length, 1);
  assert.equal(entry.candidate, 'Sunlit Garden B');
  assert.equal(entry.name, 'Sou4');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/35-four-bamboo-sunlit-garden-b-approved.svg');
  assert.equal(tileImage('4s', 'van-gogh'), 'tiles/van-gogh/approved/Sou4.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('4s', 'van-gogh')));
  const previous = set.superseded.find(tile => tile.tile === '4s' && tile.candidate === 'Bamboo C (green)');
  assert.equal(previous.activeCandidate, 'Sunlit Garden B');
  const bytes = readFileSync(new URL(`../../${previous.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    '4d1e76c1716b6086524dfcf94ec8141fa9196a78fce5ecffa517e904204304fb');
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/four-bamboo-green.json', import.meta.url), 'utf8'));
  assert.equal(provenance.source, previous.source);
  assert.equal(provenance.sourceSha256, createHash('sha256').update(bytes).digest('hex'));
  assert.deepEqual(provenance.originalCrop, { x: 974, y: 144, width: 438, height: 671 });
  assert.equal(provenance.originalBoardSha256,
    '8aa09c5f197667a94effffd037e506a02064a409444667923e7a46de3d09a450');
});

""" + s[end:]
write(p, s)
p = 'web/tests/tile-proportions.test.js'
s = read(p)
s = replace_once(s, "['2z', '3p', '4p', '4s', '6p', '7s', '8s', '9m', '9p']", "['2z', '3p', '4p', '6p', '7s', '8s', '9m', '9p']")
s = replace_once(s, '// Scarlet Loom, Vase A and Flutes B replace squeezed exports: 6m, 3s and 2s are native 3:4.', '// Scarlet Loom, Vase A, Flutes B and Sunlit Garden B use native 3:4 crops.')
write(p, s)
shutil.copyfile('.garden-prep/four-garden.test.js', 'web/tests/van-gogh-four-garden.test.js')
section = '\n\n## Sunlit Garden B — 4 bamboo (4s / Sou4)\n\nCarl selected the middle sunlit garden painting, replacing Moonlit Four. The 600 × 800 crop at (640, 16) removes the triptych gutters and fringe while retaining all four stalks, the stone wall and terracotta pot. Uniform Lanczos downsampling to 300 × 400 and quality-70 AVIF encoding preserve the selected composition with zero stretch and zero extra bleed; the normal rounded clipping remains unchanged. Neither moonlit alternative is used.\n\nThe prior playable PNG and SVG are archived byte-for-byte under `docs/design/van-gogh/superseded/four-bamboo-moonlit-four-Sou4.*`, with the prior study and provenance unchanged. Original triptych and lossless selected crop remain with the approval conversation; this repository stores the optimized artwork, not a full-resolution PNG archive. See `docs/design/van-gogh/four-bamboo-sunlit-garden-b.json` for checksums and crop coordinates. Reproduce with `node web/scripts/export-van-gogh-tiles.mjs --only=4s`. All 33 other Van Gogh faces and unrelated artwork remain unchanged.\n'
for p in (f'{FOLDER}/README.md', f'{OUT}/README.md'):
    write(p, read(p) + section)
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=4s'], check=True)
assert sha(Path(RUNTIME).read_bytes()) == EXPECTED_RUNTIME
print('ASSEMBLED SUNLIT GARDEN B', EXPECTED_RUNTIME)
print(json.dumps(record, ensure_ascii=False, indent=2))
