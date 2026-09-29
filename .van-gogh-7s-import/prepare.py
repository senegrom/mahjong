from pathlib import Path
import base64
import hashlib
import json
import subprocess

ROOT = Path.cwd()
STAGING = ROOT / '.van-gogh-7s-import'
OUT = ROOT / 'web/public/tiles/van-gogh'
SOURCE = 'docs/design/van-gogh/studies/14-seven-bamboo-irises-c.webp'
SOURCE_HASH = '68713fe1b285477dd38fd38bb329f59753f02fb263dd76bbe06d72eb4ae5cf4f'
EXPECTED = [
    'd905dcafc6144e418457859a98d82794c2de1c81',
    '6fe677eb993a868524fbfbaa4427991f623e7ee5',
    '643d2159655a95e31595d3a7cf0715388f0835cc',
    'b2818c95f9977cfb3c216424ce9bc6ed0c2e1e17',
    'b6f00ecf397e4ec071eccdc02c83057e47715050',
    '0a01095e4ddbef8b537d6e77e3d8447cdca867b5',
]

def gitsha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def replace_once(text, before, after):
    assert text.count(before) == 1, f'Expected one occurrence: {before!r}'
    return text.replace(before, after, 1)

parts = []
for i, expected in enumerate(EXPECTED):
    data = (STAGING / f'part-{i}.bin').read_bytes()
    if i == 1 and gitsha(data) != expected:
        assert gitsha(data) == '4aa47db21517405e58bc013510aabc927fc29825'
        encoded = base64.b64encode(data).decode('ascii')
        encoded = replace_once(encoded, 'YcWLAD89LTxlz/uyyeKbiu4bDjUVYYfebY', 'YcWLAD89LTxlz/uyyeKbiu4bDjUVYfebY') + 'm'
        data = base64.b64decode(encoded, validate=True)
    assert gitsha(data) == expected, f'Incorrect source chunk {i}'
    parts.append(data)
source = b''.join(parts)
assert len(source) == 87020
assert sha256(source) == SOURCE_HASH
assert gitsha(source) == '50ae06b7e0fe01ea174089f262c1a701cac1e412'
assert not (ROOT / SOURCE).exists()
old_art = {p: sha256(p.read_bytes()) for p in (OUT / 'approved').iterdir() if p.is_file()}
old_manifest = json.loads((OUT / 'manifest.json').read_text())
assert len(old_manifest['tiles']) == 16
assert all(entry['tile'] != '7s' for entry in old_manifest['tiles'])
(ROOT / SOURCE).write_bytes(source)
provenance = {
    'tile': '7s', 'name': 'Sou7', 'candidate': 'Seven C', 'title': 'Seven with Irises',
    'source': SOURCE, 'sourceSha256': SOURCE_HASH,
    'originalBoardDimensions': [1536, 1024],
    'originalBoardSha256': '4908b6b85b891c064037d783005a0a5f11c78f7d23001c134b4c9fccb1e8bd0c',
    'originalCrop': {'x': 1043, 'y': 115, 'width': 479, 'height': 793},
    'fullResolutionCropSha256': '553840fd892b8e4ffe15528ace354965f7b8910b8c26743aaabb9bf66f8a3825',
    'productionDimensions': [300, 400],
    'encoding': {'format': 'WebP', 'quality': 95, 'method': 6, 'resize': 'Lanczos'},
    'generationId': '574e8b87-564a-4801-8ba3-778307fe264e',
    'approval': 'Carl selected C and requested deployment as 7 bamboo on 29 September 2026.',
    'notes': 'Mechanically cropped and resized from selected C; no repainting. The repository source is an optimized export, not the full-resolution lossless original. The study-sheet heading, gutters and caption are excluded.',
}
(ROOT / 'docs/design/van-gogh/seven-bamboo-irises.json').write_text(json.dumps(provenance, indent=2) + '\n')

exporter = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
text = exporter.read_text()
text = replace_once(text, 'the green 2s, 3s and 4s studies', 'the 2s, 3s, 4s and 7s studies')
anchor = "  moonlitFourGreen: 'docs/design/van-gogh/studies/13-four-bamboo-moonlit-four-green.webp',"
text = replace_once(text, anchor, anchor + f"\n  sevenIrises: '{SOURCE}',")
anchor = "  ['Bamboo C (green)', '4s', 'Four bamboo', 'moonlitFourGreen', [0, 0, 300, 400]],"
text = replace_once(text, anchor, anchor + "\n  ['Seven C', '7s', 'Seven bamboo', 'sevenIrises', [0, 0, 300, 400]],")
text = replace_once(text, 'The user also approved greener Moonlit Four C for 4s.', 'The user also approved greener Moonlit Four C for 4s. On 29 September 2026 Carl selected Seven with Irises C for 7s and requested deployment.')
text = replace_once(text, 'and green Moonlit Four C (4s) included;', 'green Moonlit Four C (4s) and Seven with Irises C (7s) included;')
old_hand = "const handTiles = ['4s', '3s', '2s', '2m', '3m', '4m', ...approved.filter(tile => !['4s', '3s', '2s', '2m', '3m', '4m'].includes(tile)).slice(0, 6), '2z', '6z'];"
new_hand = "const handTiles = ['7s', '4s', '3s', '2s', '2m', '3m', '4m', ...approved.filter(tile => !['7s', '4s', '3s', '2s', '2m', '3m', '4m'].includes(tile)).slice(0, 5), '2z', '6z'];"
text = replace_once(text, old_hand, new_hand)
text = replace_once(text, 'approved faces, including green Moonlit Four', 'approved faces, including Seven with Irises for 7 bamboo, green Moonlit Four')
text = replace_once(text, 'Garden Rhythm 2 bamboo, Triple Shoots 3 bamboo and Moonlit Four 4 bamboo use high-quality, web-optimized crops of their approved green studies;', 'Garden Rhythm 2 bamboo, Triple Shoots 3 bamboo, Moonlit Four 4 bamboo and Seven with Irises 7 bamboo use high-quality, web-optimized crops of their approved studies;')
exporter.write_text(text)

tests = ROOT / 'web/tests/tile-faces.test.js'
text = tests.read_text()
text = replace_once(text, "'2m', '4m', '2s', '4s'];", "'2m', '4m', '2s', '4s', '7s'];")
text = replace_once(text, "'Bamboo B (green)', 'Bamboo C (green)']);", "'Bamboo B (green)', 'Bamboo C (green)', 'Seven C']);")
text = replace_once(text, "url.startsWith('tiles/van-gogh/')).length, 16", "url.startsWith('tiles/van-gogh/')).length, 17")
text = replace_once(text, "['1z', '4z', '2m', '4m', '2s', '3s', '4s']", "['1z', '4z', '2m', '4m', '2s', '3s', '4s', '7s']")
text = replace_once(text, 'assert.equal(set.remaining.length, 18);', 'assert.equal(set.remaining.length, 17);')
text += """

test('Van Gogh Seven with Irises C is the selected seven bamboo with verified source provenance', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '7s');
  assert.ok(entry);
  assert.equal(entry.candidate, 'Seven C');
  assert.equal(entry.name, 'Sou7');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/14-seven-bamboo-irises-c.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  const sourceHash = createHash('sha256').update(source).digest('hex');
  assert.equal(sourceHash, '68713fe1b285477dd38fd38bb329f59753f02fb263dd76bbe06d72eb4ae5cf4f');
  const png = readFileSync(new URL(`tiles/van-gogh/${entry.png}`, publicRoot));
  assert.equal(png.readUInt32BE(16), 300);
  assert.equal(png.readUInt32BE(20), 400);
  assert.equal(tileImage('7s', 'van-gogh'), 'tiles/van-gogh/approved/Sou7.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('7s', 'van-gogh')));
  assert.ok(!set.remaining.includes('7s'));
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/seven-bamboo-irises.json', import.meta.url), 'utf8'));
  assert.equal(provenance.source, entry.source);
  assert.equal(provenance.sourceSha256, sourceHash);
  assert.equal(provenance.candidate, entry.candidate);
  assert.equal(provenance.title, 'Seven with Irises');
  assert.deepEqual(provenance.originalCrop, { x: 1043, y: 115, width: 479, height: 793 });
  assert.equal(provenance.originalBoardSha256,
    '4908b6b85b891c064037d783005a0a5f11c78f7d23001c134b4c9fccb1e8bd0c');
  assert.equal(provenance.fullResolutionCropSha256,
    '553840fd892b8e4ffe15528ace354965f7b8910b8c26743aaabb9bf66f8a3825');
});
"""
tests.write_text(text)

section = """
## Seven with Irises: 7 bamboo

On 29 September 2026 Carl selected **C — Seven with Irises** from the 7 bamboo study board and requested deployment. It is active for `7s` (`Sou7`). The six green stalks and central reddish-brown stalk form seven distinct bamboo motifs, with blue-purple irises, golden fields and a small cottage. This is the selected C artwork, not a repaint.

The study-sheet heading, gutters and caption were removed with the exact crop `[1043, 115, 479, 793]` from the 1536 × 1024 board. The entire cropped composition was resized to the shared 300 × 400 game canvas and saved as quality-95 WebP, following the newer bamboo source convention. This committed source is a web-optimized export, not the full-resolution lossless original. The original board, full-resolution crop and production-source hashes are recorded in `docs/design/van-gogh/seven-bamboo-irises.json`.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=7s` to reproduce the playable PNG, self-contained SVG, registration, manifest and preview. All sixteen previously approved tile images remain byte-for-byte unchanged. The set now has **17 painted faces and 17 Classic fallbacks**.
"""
for name in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = ROOT / name
    text = p.read_text()
    if name.startswith('docs/'):
        text = replace_once(text, 'sixteen distinct faces', 'seventeen distinct faces')
        text = replace_once(text, 'The remaining 18 tile identities', 'The remaining 17 tile identities')
        text = replace_once(text, 'all sixteen faces and the preview', 'all seventeen faces and the preview')
    else:
        text = replace_once(text, 'The sixteen approved faces', 'The seventeen approved faces')
        text = replace_once(text, 'The other 18 identities', 'The other 17 identities')
        text = replace_once(text, 'The newer green bamboo sources', 'The newer 2, 3, 4 and 7 bamboo sources')
    text = replace_once(text, 'plus green Moonlit Four C for 4 bamboo', 'plus Seven with Irises C for 7 bamboo, green Moonlit Four C for 4 bamboo')
    text = replace_once(text, 'The set now has 16 painted faces and 18 Classic fallbacks.', 'That addition brought the set to 16 painted faces and 18 Classic fallbacks.')
    p.write_text(text.rstrip() + '\n' + section)

assert subprocess.check_output(['identify', '-format', '%wx%h', SOURCE], text=True) == '300x400'
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=7s'], check=True)
for p, old_hash in old_art.items():
    assert sha256(p.read_bytes()) == old_hash, f'Existing artwork changed: {p}'
new_manifest = json.loads((OUT / 'manifest.json').read_text())
assert len(new_manifest['tiles']) == 17 and len(new_manifest['remaining']) == 17
assert new_manifest['tiles'][:-1] == old_manifest['tiles']
assert new_manifest['sources'][:-1] == old_manifest['sources']
assert set(old_manifest['remaining']) - set(new_manifest['remaining']) == {'7s'}
assert new_manifest['rejected'] == old_manifest['rejected']
assert new_manifest['superseded'] == old_manifest['superseded']
for p in [ROOT / SOURCE, OUT / 'approved/Sou7.png']:
    pixels = subprocess.check_output(['convert', str(p), '-alpha', 'off', '-depth', '8', 'rgb:-'])
    if p == ROOT / SOURCE:
        source_pixels = pixels
    else:
        assert pixels == source_pixels, 'PNG does not preserve decoded selected-source pixels'
print(f'VERIFIED: exact selected C source {SOURCE_HASH}; 7s registered; all {len(old_art)} existing art files unchanged; PNG decoded pixels identical.')
