from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'web/public/tiles/van-gogh'
SOURCE = 'docs/design/van-gogh/studies/27-south-wind-billowing-courtyard-a-approved.svg'
RUNTIME = 'web/public/tiles/van-gogh/approved/Nan.svg'
RECORD = 'docs/design/van-gogh/south-wind-billowing-courtyard-a.json'
RASTER_HASH = '44801ce1a0171dbd5acb9bab04392d331a61ebcf474792348538a669ae5fcf56'
SOURCE_HASH = '490f86658f1afc854175d81d9fe0e89c90d86e5b18b212cdb79b8fb752bc0764'
RUNTIME_HASH = '0c6f8facecbe662a034af16fe39b0aaa14bd7916bd74882cbb75292d65e74163'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Patch anchor changed: {old!r}')
    return text.replace(old, new, 1)

def put(relative, data):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        target.write_bytes(data)
    else:
        target.write_text(data, encoding='utf-8')

before = json.loads((OUT / 'manifest.json').read_text())
assert all(t['tile'] != '2z' for t in before['tiles']), 'South wind was added concurrently; do not overwrite'
assert not (ROOT / RUNTIME).exists() and not (ROOT / SOURCE).exists()
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
images = {p: sha((ROOT / p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in {'.png', '.svg', '.webp', '.avif', '.jpg', '.jpeg'}}
old_rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', (OUT / 'preview.html').read_text()).group(1)
assert old_rack.count('<img ') == 14
raster = b''.join((ROOT / f'.south-wind-A-transfer/part{i:02d}').read_bytes() for i in range(6))
assert len(raster) == 49477 and sha(raster) == RASTER_HASH, 'Transferred raster differs from the selected A export'
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">South wind — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert sha(svg) == SOURCE_HASH
script_path = 'web/scripts/export-van-gogh-tiles.mjs'
script = (ROOT / script_path).read_text()
script = once(script, 'const fits = {\n', "const fits = {\n  // Keep A's linen structure visible; trim sky and paving instead of squeezing the tall painting.\n  '2z': { painting: [393, 640], stretch: 0.02, anchor: 0.51, bleedPixels: 0 },\n")
script = once(script, 'const sources = {\n', f"const sources = {{\n  southCourtyard: '{SOURCE}',\n")
script = once(script, '];\nconst onlyArgument =', "  ['South A', '2z', 'South wind', 'southCourtyard', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
match = re.search(r"(?m)^(  approval: ')(.*)(',)$", script)
assert match
approval = ' Carl selected A — The Billowing Courtyard for South wind (2z / Nan) and explicitly requested GitHub deployment. The linen courtyard painting is used, not B the boat or C the gate. Its original board and lossless A crop remain in the approval archive; the documented game export uses the existing proportions helper.'
script = script[:match.start()] + match[1] + match[2] + approval + match[3] + script[match.end():]
script = once(script, '. Includes ', '. Includes The Billowing Courtyard A (South wind, 2z), ')
script = once(script, '${approved.length} approved faces, including ', '${approved.length} approved faces, including The Billowing Courtyard A for South wind, ')
script = once(script, 'Earlier selected art is preserved in lossless PNG crops.', 'The Billowing Courtyard A uses a quality-70 AVIF game export and a documented 2% fit of its 393 × 640 painting, trimming surrounding sky and paving while retaining the linen structure. Its original board and lossless selected crop are preserved separately. Earlier selected art is preserved in lossless PNG crops.')

path = 'web/tests/tile-faces.test.js'
tests = (ROOT / path).read_text()
old_ids = [t['tile'] for t in before['tiles']]
old_candidates = [t['candidate'] for t in before['tiles']]
tests = once(tests, f'  const approved = {old_ids!r};', f"  const approved = {old_ids + ['2z']!r};")
tests = once(tests, f'assert.deepEqual(set.tiles.map(tile => tile.candidate), {old_candidates!r});', f"assert.deepEqual(set.tiles.map(tile => tile.candidate), {old_candidates + ['South A']!r});")
tests = once(tests, "['9m', '6z', '1m'].includes(tile)", "['9m', '6z', '1m', '2z'].includes(tile)")
proportions_path = 'web/tests/tile-proportions.test.js'
proportions = (ROOT / proportions_path).read_text()
proportions = once(proportions, "assert.deepEqual(squeezed.sort(), ['2s', '3p'", "assert.deepEqual(squeezed.sort(), ['2s', '2z', '3p'")

record = {
    'tile': '2z', 'name': 'Nan', 'candidate': 'South A', 'title': 'The Billowing Courtyard',
    'source': SOURCE, 'runtime': RUNTIME,
    'originalFilename': 'van_gogh_mahjong_tile_set_design_review.png',
    'originalDimensions': [1448, 1086], 'originalBytes': 3262676,
    'originalSha256': 'b70f75eb381cf9c93f6b236c9e55d88eca27761bd63ccfbc0b0cf0ecb5117abc',
    'originalCrop': {'x': 95, 'y': 208, 'width': 393, 'height': 640},
    'selectedCropFilename': 'south-wind-billowing-courtyard-A-full-crop.png',
    'selectedCropSha256': '2757c24eddbe96f30e023309168cd7d98401aaf64060b0b2e48022206701139c',
    'rasterMimeType': 'image/avif', 'rasterDimensions': [300, 400], 'rasterBytes': len(raster),
    'rasterSha256': RASTER_HASH, 'svgSha256': SOURCE_HASH, 'runtimeSvgSha256': RUNTIME_HASH,
    'fit': {'painting': {'width': 393, 'height': 640}, 'stretch': 0.02, 'anchor': 0.51, 'bleedPixels': 0,
            'image': {'x': 0, 'y': -40.27, 'width': 300, 'height': 478.97}},
    'approval': 'Carl selected A, The Billowing Courtyard, for South wind and requested deployment again. B the boat and C the gate are not selected.',
    'processing': 'Select the exact lossless A crop. Create a 300 x 400 Lanczos/AVIF quality-70 game raster (speed 4, 4:4:4). The face-fit helper restores the tall painting proportions to within 2%, with anchor 0.51 and zero side bleed. The 3:4 face trims sky and foreground paving; the linen structure remains visible. No regeneration, repainting, recolouring or added symbols. This is an optimized game export, not the lossless original.',
    'originalPreservation': 'Untouched A/B/C board and lossless selected A crop are preserved in van-gogh-south-wind-A-prepared.zip supplied in chat, with original checksums. The repository contains the optimized approved source SVG and fitted runtime, not the full-resolution PNG originals.'
}
new_count = len(before['tiles']) + 1
remaining = [t for t in before['remaining'] if t != '2z']
assert remaining == ['3z'], 'Inventory changed; review current documentation wording'
doc_paths = ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']
docs = {}
section = f'''\n## South wind — The Billowing Courtyard A\n\nCarl selected **A — The Billowing Courtyard** as `2z` / `Nan`, South wind. The billowing white linen and its wooden frame are the approved scene; neither the boat B nor the garden gate C is used.\n\nThe original board and exact 393 × 640 A crop are preserved in `van-gogh-south-wind-A-prepared.zip` supplied in chat. This repository stores a 300 × 400 quality-70 AVIF game export inside its approved study SVG. The existing face-fit helper corrects the tall painting's proportions to within 2%, trimming surrounding sky and paving rather than squeezing the whole image into the tile. The runtime differs from the study only in image placement. No repainting or recolouring was applied.\n\nChecksums, crop, encoding and fit are recorded in `south-wind-billowing-courtyard-a.json` in the design directory. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=2z`. All prior artwork, including 7 and 8 disks and The Potter’s Table 9 disks, is unchanged; the existing fourteen-tile preview hand is preserved. The set now has **{new_count} painted faces**, with **West wind** the only remaining unpainted identity.\n'''
for rel in doc_paths:
    text = (ROOT / rel).read_text()
    if rel.startswith('docs/'):
        text, n = re.subn(r'(?m)^\*\*Current set:\*\* .*$', f'**Current set:** {new_count} painted faces and 1 unpainted identity, West wind. The Billowing Courtyard A is active for South wind. All three numbered suits and all dragons are complete. Earlier sections retain historical counts.', text, count=1)
        assert n == 1
    else:
        text = once(text, f'The {len(before["tiles"])} approved faces', f'The {new_count} approved faces')
        text = once(text, 'The other 2 identities awaiting Van Gogh artwork, South and West winds, display their names.', 'West wind is the only identity awaiting Van Gogh artwork and displays its name.')
        text = once(text, '**The Red Sail C** is active', '**The Billowing Courtyard A** is active for South wind (2z / Nan). **The Red Sail C** is active')
    docs[rel] = text + section

# No repository files are touched before every patch anchor and transfer hash is verified.
put(SOURCE, svg)
put(RECORD, json.dumps(record, ensure_ascii=False, indent=2) + '\n')
put(script_path, script)
put(path, tests)
put(proportions_path, proportions)
for rel, text in docs.items():
    put(rel, text)
command = ['node', script_path, '--only=2z']
subprocess.run(command, check=True)
after = json.loads((OUT / 'manifest.json').read_text())
assert len(after['tiles']) == new_count and after['remaining'] == remaining
assert [t for t in after['tiles'] if t['tile'] != '2z'] == before['tiles']
assert [s for s in after['sources'] if s['source'] != SOURCE] == before['sources']
for key in ('superseded', 'rejected', 'fallback', 'facePresentation', 'canvas'):
    assert after[key] == before[key], key
for p, expected in images.items():
    assert (ROOT / p).is_file() and sha((ROOT / p).read_bytes()) == expected, f'Existing artwork changed: {p}'
assert sha((ROOT / RUNTIME).read_bytes()) == RUNTIME_HASH
entry = next(t for t in after['tiles'] if t['tile'] == '2z')
assert entry['fit'] == record['fit'] and entry['source'] == SOURCE
assert entry['rasterSha256'] == RASTER_HASH and entry['svgSha256'] == RUNTIME_HASH
preview = (OUT / 'preview.html').read_text()
assert re.search(r'<div class="rack" id="rack">(.*?)</div>', preview).group(1) == old_rack
assert 'South A · South wind' in preview
first_export = {p: sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
subprocess.run(command, check=True)
assert first_export == {p: sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
print(f'PASS: South A transfer and source/runtime hashes; {len(images)} prior images and {len(before["tiles"])} prior entries unchanged; {new_count} painted / {len(remaining)} unpainted; fourteen-tile hand identical; deterministic selective export.')
