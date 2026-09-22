from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path.cwd()
transfer = root / '.bamboo-transfer'
sha = lambda data: hashlib.sha256(data).hexdigest()
source_rel = 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp'
source = root / source_rel
expected = '528acc47e037fdce2fc9f48f827bbe554da35a1cc0186a4b74f1ac9fea0ad7ec'
parts = ['00', '01', '02-0', '02-1', '03', '04', '05', '06']
data = b''.join((transfer / f'part-{part}.bin').read_bytes() for part in parts)
assert len(data) == 76866 and sha(data) == expected, 'Artwork transfer checksum mismatch'
assert data[:4] == b'RIFF' and data[8:12] == b'WEBP'
assert not source.exists(), 'Do not overwrite other artwork'

out = root / 'web/public/tiles/van-gogh'
previous = json.loads((out / 'manifest.json').read_text())
assert len(previous['tiles']) == 14 and all(t['tile'] != '2s' for t in previous['tiles'])
before = {p: sha(p.read_bytes()) for p in (out / 'approved').iterdir() if p.is_file()}

def once(text, old, new):
    assert text.count(old) == 1, f'Patch drift: {old!r}'
    return text.replace(old, new, 1)

script_path = root / 'web/scripts/export-van-gogh-tiles.mjs'
script = script_path.read_text()
script = once(script, '\n};\nconst definitions = [',
              f"\n  gardenRhythmGreen: '{source_rel}',\n}};\nconst definitions = [")
script = once(script, '\n];\nconst onlyArgument =',
              "\n  ['Bamboo B (green)', '2s', 'Two bamboo', 'gardenRhythmGreen', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
script = once(script, 'The later L is the active white dragon;',
              'The user also approved the green Garden Rhythm B for 2s. The later L is the active white dragon;')
script = once(script, 'East A and North B included;',
              'East A, North B and green Garden Rhythm B (2s) included;')
script = once(script,
              "const handTiles = ['2m', '3m', '4m', ...approved.filter(tile => !['2m', '3m', '4m'].includes(tile)).slice(0, 9), '2z', '6z'];",
              "const handTiles = ['2s', '2m', '3m', '4m', ...approved.filter(tile => !['2s', '2m', '3m', '4m'].includes(tile)).slice(0, 8), '2z', '6z'];")
script = once(script, 'including Night Cafe for 2 of characters,',
              'including green Garden Rhythm for 2 bamboo, Night Cafe for 2 of characters,')
script = once(script, 'The exact selected pixels are preserved in lossless PNG crops.',
              'Earlier selected art is preserved in lossless PNG crops. Garden Rhythm 2 bamboo uses a high-quality, web-optimized crop of the approved green study; its source and crop provenance are documented.')
script = once(script, '// Lossless extraction of the selected artwork, including Almond Branches for 3m.',
              '// Export selected artwork; the green 2s study uses a documented optimized source.')

test_path = root / 'web/tests/tile-faces.test.js'
tests = test_path.read_text()
tests = once(tests, "'4z', '2m', '4m'];", "'4z', '2m', '4m', '2s'];")
tests = once(tests, "'Characters B', 'Characters C']);", "'Characters B', 'Characters C', 'Bamboo B (green)']);")
tests = once(tests, "url.startsWith('tiles/van-gogh/')).length, 14", "url.startsWith('tiles/van-gogh/')).length, 15")
tests = once(tests, 'assert.equal(set.remaining.length, 20);', 'assert.equal(set.remaining.length, 19);')
tests = once(tests, "for (const tile of ['1z', '4z', '2m', '4m'])", "for (const tile of ['1z', '4z', '2m', '4m', '2s'])")
tests += """

test('Van Gogh green Garden Rhythm B is the approved two bamboo', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '2s');
  assert.ok(entry);
  assert.equal(entry.candidate, 'Bamboo B (green)');
  assert.equal(entry.name, 'Sou2');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/11-two-bamboo-garden-rhythm-green.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'),
    '528acc47e037fdce2fc9f48f827bbe554da35a1cc0186a4b74f1ac9fea0ad7ec');
  assert.equal(tileImage('2s', 'van-gogh'), 'tiles/van-gogh/approved/Sou2.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('2s', 'van-gogh')));
  assert.ok(!set.remaining.includes('2s'));
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/two-bamboo-green.json', import.meta.url), 'utf8'));
  assert.equal(provenance.sourceSha256, createHash('sha256').update(source).digest('hex'));
  assert.equal(provenance.originalSha256, 'ba6cb69a60cc24173125750b412e3899a4406f28557256995ce81d4b34893748');
  assert.deepEqual(provenance.originalCrop, { x: 508, y: 143, width: 433, height: 667 });
});
"""
source.parent.mkdir(parents=True, exist_ok=True)
source.write_bytes(data)
script_path.write_text(script)
test_path.write_text(tests)
provenance = {
    'tile': '2s', 'name': 'Sou2', 'candidate': 'Bamboo B (green)', 'title': 'Garden Rhythm',
    'source': source_rel, 'sourceSha256': expected, 'sourceBytes': len(data),
    'original': 'van_gogh_bamboo_design_review.png',
    'originalSha256': 'ba6cb69a60cc24173125750b412e3899a4406f28557256995ce81d4b34893748',
    'originalDimensions': {'width': 1448, 'height': 1086},
    'originalCrop': {'x': 508, 'y': 143, 'width': 433, 'height': 667},
    'optimization': {'width': 300, 'height': 400, 'encoding': 'WebP', 'quality': 95, 'method': 6, 'resampling': 'Lanczos'},
    'note': 'Approved revised green centre panel, not repainted. This repository source is a web-optimized crop, not the full-resolution original board. The original board and full-resolution crop were preserved in the download archive provided in the conversation.'
}
(root / 'docs/design/van-gogh/two-bamboo-green.json').write_text(json.dumps(provenance, indent=2) + '\n')
for rel in ('docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md'):
    p = root / rel
    text = p.read_text()
    for old, new in [('fourteen distinct faces', 'fifteen distinct faces'), ('fourteen approved faces', 'fifteen approved faces'), ('remaining 20 tile identities', 'remaining 19 tile identities'), ('other 20 identities', 'other 19 identities'), ('all fourteen faces', 'all fifteen faces')]:
        text = text.replace(old, new)
    text += '\n## Garden Rhythm: green 2 bamboo\n\nThe approved revised green B is active for `2s` (`Sou2`). The two bamboo stalks, garden canal and bridge are unchanged in composition. The deployed 300 by 400 source is a high-quality WebP crop, with source checksum, original-board checksum and crop coordinates recorded in `docs/design/van-gogh/two-bamboo-green.json`. Earlier tile artwork is unchanged. Run `node web/scripts/export-van-gogh-tiles.mjs --only=2s` to regenerate this face. The set now has 15 painted faces and 19 Classic fallbacks.\n'
    p.write_text(text)
command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=2s']
subprocess.run(command, check=True)
assert all(p.exists() and sha(p.read_bytes()) == value for p, value in before.items()), 'Existing artwork changed'
manifest = json.loads((out / 'manifest.json').read_text())
assert len(manifest['tiles']) == 15 and len(manifest['remaining']) == 19
assert manifest['tiles'][:-1] == previous['tiles'], 'Existing manifest entries changed'
assert manifest['tiles'][-1]['tile'] == '2s' and manifest['tiles'][-1]['name'] == 'Sou2'
html = (out / 'preview.html').read_text()
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', html)
assert rack is not None and rack.group(1).count('<img ') == 14
assert 'approved/Sou2.svg' in rack.group(1)
exports = {p: sha(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
subprocess.run(command, check=True)
assert all(sha(p.read_bytes()) == value for p, value in exports.items()), 'Non-deterministic export'
print(json.dumps({'tile': '2s', 'newPaintedCount': 15, 'existingArtworkFilesUnchanged': len(before), 'sourceSha256': expected, 'deterministicExport': True}))
