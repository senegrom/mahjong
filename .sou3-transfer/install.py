"""One-time checked replacement of the approved Van Gogh 3 bamboo."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'web/public/tiles/van-gogh'
SOURCE = 'docs/design/van-gogh/studies/12-three-bamboo-triple-shoots-green.webp'
EXPECTED = 'f2e1617a2e894622a28922f309747a8c2d0bb681fedd4ca37c1ba4b810135c7c'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch anchor changed: ' + old)
    return text.replace(old, new, 1)

def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)

image = b''.join((ROOT / f'.sou3-transfer/part-{n:02d}').read_bytes() for n in range(9))
assert len(image) == 80618 and sha(image) == EXPECTED, 'Artwork transfer checksum mismatch'
assert image[:4] == b'RIFF' and image[8:12] == b'WEBP'
assert not (ROOT / SOURCE).exists(), 'Do not replace an unrelated source'
old_manifest = json.loads((OUT / 'manifest.json').read_text())
assert len(old_manifest['tiles']) == 15 and len(old_manifest['remaining']) == 19
old_three = next(t for t in old_manifest['tiles'] if t['tile'] == '3s')
assert old_three['candidate'] == 'C'
all_art = {p: sha(p.read_bytes()) for p in (ROOT / 'web/public/tiles').rglob('*')
           if p.is_file() and p.suffix in ('.png', '.svg', '.webp')}
old_sources = {ROOT / s['source']: s['sha256'] for s in old_manifest['sources']}

script_path = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
script = script_path.read_text()
script = once(script,
    "  gardenRhythmGreen: 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp',",
    "  gardenRhythmGreen: 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp',\n  tripleShootsGreen: '" + SOURCE + "',")
script = once(script,
    "  ['C', '3s', 'Three bamboo', 'first', [844, 122, 356, 463]],",
    "  ['Bamboo A (green)', '3s', 'Three bamboo', 'tripleShootsGreen', [0, 0, 300, 400]],")
script = once(script, 'the green 2s study uses a documented optimized source.',
    'the green 2s and 3s studies use documented optimized sources.')
script = once(script, 'The user also approved the green Garden Rhythm B for 2s.',
    'The user also approved the green Garden Rhythm B for 2s and selected green Triple Shoots A to overwrite C for 3s.')
script = once(script,
    "    { candidate: 'D', tile: '3m', activeCandidate: 'Characters A', source: sources.first },",
    "    { candidate: 'D', tile: '3m', activeCandidate: 'Characters A', source: sources.first },\n    { candidate: 'C', tile: '3s', activeCandidate: 'Bamboo A (green)', source: sources.first },")
script = once(script, 'North B and green Garden Rhythm B (2s) included;',
    'North B, green Garden Rhythm B (2s) and green Triple Shoots A (3s) included;')
script = once(script,
    "const handTiles = ['2s', '2m', '3m', '4m', ...approved.filter(tile => !['2s', '2m', '3m', '4m'].includes(tile)).slice(0, 8), '2z', '6z'];",
    "const handTiles = ['3s', '2s', '2m', '3m', '4m', ...approved.filter(tile => !['3s', '2s', '2m', '3m', '4m'].includes(tile)).slice(0, 7), '2z', '6z'];")
script = once(script, 'including green Garden Rhythm for 2 bamboo,',
    'including green Triple Shoots for 3 bamboo, green Garden Rhythm for 2 bamboo,')
script = once(script,
    'Garden Rhythm 2 bamboo uses a high-quality, web-optimized crop of the approved green study; its source and crop provenance are documented.',
    'Garden Rhythm 2 bamboo and Triple Shoots 3 bamboo use high-quality, web-optimized crops of their approved green studies; their source and crop provenance are documented.')

test_path = ROOT / 'web/tests/tile-faces.test.js'
tests = test_path.read_text()
tests = once(tests, "['A', 'B', 'C', 'Characters A'", "['A', 'B', 'Bamboo A (green)', 'Characters A'")
tests = once(tests,
    "for (const tile of ['1z', '4z', '2m', '4m', '2s'])",
    "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s'])")
tests += '''

test('Van Gogh green Triple Shoots A replaces only the existing three bamboo', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '3s');
  assert.equal(set.tiles.length, 15);
  assert.equal(set.remaining.length, 19);
  assert.equal(set.tiles.filter(tile => tile.tile === '3s').length, 1);
  assert.equal(entry.candidate, 'Bamboo A (green)');
  assert.equal(entry.name, 'Sou3');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/12-three-bamboo-triple-shoots-green.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const bytes = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),
    'f2e1617a2e894622a28922f309747a8c2d0bb681fedd4ca37c1ba4b810135c7c');
  assert.equal(tileImage('3s', 'van-gogh'), 'tiles/van-gogh/approved/Sou3.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('3s', 'van-gogh')));
  assert.ok(!set.remaining.includes('3s'));
  const superseded = set.superseded.find(tile => tile.tile === '3s');
  assert.equal(superseded.candidate, 'C');
  assert.equal(superseded.activeCandidate, 'Bamboo A (green)');
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/three-bamboo-green.json', import.meta.url), 'utf8'));
  assert.equal(provenance.originalBoardSha256, '054b1a1a246a33a03d6594027b312a70e35c0c7c8ebbf2b58247dcb5e82ad48f');
  assert.deepEqual(provenance.originalCrop, { x: 38, y: 143, width: 439, height: 673 });
  assert.equal(provenance.productionSha256, createHash('sha256').update(bytes).digest('hex'));
});
'''

# Every patch anchor is validated before changing existing files.
(ROOT / SOURCE).write_bytes(image)
script_path.write_text(script)
test_path.write_text(tests)
for rel in ('docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md'):
    p = ROOT / rel
    text = p.read_text()
    text = text.replace('A–C, E, G–J and L, plus', 'A–B, E, G–J and L, plus green Triple Shoots A for 3 bamboo, green Garden Rhythm B for 2 bamboo,')
    text = text.replace('Nine of those faces remain active after the Almond Branches replacement for 3 of characters.',
        'Eight of those faces remain active after the Almond Branches replacement for 3 of characters and the Triple Shoots replacement for 3 bamboo.')
    text = text.replace('The deployed faces preserve the exact approved source pixels in lossless rectangular crops or full-canvas PNG copies.',
        'The earlier deployed faces preserve the approved source pixels in lossless rectangular crops or full-canvas PNG copies; the green 2s and 3s replacements use documented high-quality WebP game exports.')
    text += '\n## Triple Shoots: green 3 bamboo replacement\n\nThe approved greener **A — Triple Shoots** replaces the original C artwork for `3s` (`Sou3`), using the existing `approved/Sou3.png` and `approved/Sou3.svg` paths. Exactly three bamboo stalks form the tile identity. No other playable tile is changed. Original C remains in the first study sheet and Git history.\n\nThe source is a 300 × 400, quality-95 WebP export of the selected panel, not a lossless full-resolution original. The original board checksum, 439 × 673 crop coordinates and optimized-source checksum are recorded in `docs/design/van-gogh/three-bamboo-green.json`. Run `node web/scripts/export-van-gogh-tiles.mjs --only=3s` to reproduce this replacement. The inventory remains 15 painted faces and 19 Classic fallbacks.\n'
    p.write_text(text)

run('node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=3s')
new = json.loads((OUT / 'manifest.json').read_text())
assert [t['tile'] for t in new['tiles']] == [t['tile'] for t in old_manifest['tiles']]
assert new['remaining'] == old_manifest['remaining']
assert [t for t in new['tiles'] if t['tile'] != '3s'] == [t for t in old_manifest['tiles'] if t['tile'] != '3s']
changed = {p for p, h in all_art.items() if sha(p.read_bytes()) != h}
assert changed == {OUT / 'approved/Sou3.png', OUT / 'approved/Sou3.svg'}, changed
assert all(sha(p.read_bytes()) == h for p, h in old_sources.items())
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', (OUT / 'preview.html').read_text())
assert rack and rack.group(1).count('<img ') == 14
snapshot = {p: sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
run('node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=3s')
assert all(sha(p.read_bytes()) == h for p, h in snapshot.items()), 'Export was not deterministic'
print('Verified: only existing Van Gogh Sou3.png/Sou3.svg artwork changed; all other tile bytes and entries preserved.')
print('Source SHA-256:', sha(image))
print('Production PNG SHA-256:', sha((OUT / 'approved/Sou3.png').read_bytes()))
