from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path.cwd()
transport = root / '.sou4-transfer'
source_rel = 'docs/design/van-gogh/studies/13-four-bamboo-moonlit-four-green.webp'
source_hash = '4d1e76c1716b6086524dfcf94ec8141fa9196a78fce5ecffa517e904204304fb'
source = b''.join((transport / f'part{i:02d}').read_bytes() for i in range(7))
sha = lambda data: hashlib.sha256(data).hexdigest()
assert len(source) == 72616 and sha(source) == source_hash, 'Artwork transfer mismatch'
assert not (root / source_rel).exists(), 'Do not overwrite an existing source'
out = root / 'web/public/tiles/van-gogh'
old_manifest = json.loads((out / 'manifest.json').read_text())
assert len(old_manifest['tiles']) == 15 and '4s' in old_manifest['remaining']
old_images = {p: sha(p.read_bytes()) for p in (root / 'web/public/tiles').rglob('*')
              if p.is_file() and p.suffix in {'.svg', '.png', '.webp'}}
old_sources = {e['source']: sha((root / e['source']).read_bytes()) for e in old_manifest['sources']}

def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Review changed file before applying: {before!r}')
    return text.replace(before, after, 1)

script_path = root / 'web/scripts/export-van-gogh-tiles.mjs'
script = script_path.read_text()
script = replace(script, 'the green 2s and 3s studies', 'the green 2s, 3s and 4s studies')
script = replace(script, '\n};\nconst definitions = [',
                 f"\n  moonlitFourGreen: '{source_rel}',\n}};\nconst definitions = [")
script = replace(script, '\n];\nconst onlyArgument =',
                 "\n  ['Bamboo C (green)', '4s', 'Four bamboo', 'moonlitFourGreen', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
script = replace(script, 'The later L is the active white dragon;',
                 'The user also approved greener Moonlit Four C for 4s. The later L is the active white dragon;')
script = replace(script, 'and green Triple Shoots A (3s) included;',
                 'green Triple Shoots A (3s) and green Moonlit Four C (4s) included;')
script = replace(script,
                 "const handTiles = ['3s', '2s', '2m', '3m', '4m', ...approved.filter(tile => !['3s', '2s', '2m', '3m', '4m'].includes(tile)).slice(0, 7), '2z', '6z'];",
                 "const handTiles = ['4s', '3s', '2s', '2m', '3m', '4m', ...approved.filter(tile => !['4s', '3s', '2s', '2m', '3m', '4m'].includes(tile)).slice(0, 6), '2z', '6z'];")
script = replace(script, 'including green Triple Shoots for 3 bamboo,',
                 'including green Moonlit Four for 4 bamboo, green Triple Shoots for 3 bamboo,')
script = replace(script, 'Garden Rhythm 2 bamboo and Triple Shoots 3 bamboo use',
                 'Garden Rhythm 2 bamboo, Triple Shoots 3 bamboo and Moonlit Four 4 bamboo use')

tests_path = root / 'web/tests/tile-faces.test.js'
tests = tests_path.read_text()
tests = replace(tests, "'4z', '2m', '4m', '2s'];", "'4z', '2m', '4m', '2s', '4s'];")
tests = replace(tests, "'Characters C', 'Bamboo B (green)']);", "'Characters C', 'Bamboo B (green)', 'Bamboo C (green)']);")
tests = replace(tests, "url.startsWith('tiles/van-gogh/')).length, 15", "url.startsWith('tiles/van-gogh/')).length, 16")
tests = replace(tests, 'assert.equal(set.remaining.length, 19);', 'assert.equal(set.remaining.length, 18);')
tests = replace(tests, "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s'])", "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s', '4s'])")
tests += r'''

test('Van Gogh greener Moonlit Four C is the approved four bamboo', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '4s');
  assert.ok(entry);
  assert.equal(entry.candidate, 'Bamboo C (green)');
  assert.equal(entry.name, 'Sou4');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/13-four-bamboo-moonlit-four-green.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'),
    '4d1e76c1716b6086524dfcf94ec8141fa9196a78fce5ecffa517e904204304fb');
  const png = readFileSync(new URL(`tiles/van-gogh/${entry.png}`, publicRoot));
  assert.equal(png.readUInt32BE(16), 300);
  assert.equal(png.readUInt32BE(20), 400);
  assert.equal(tileImage('4s', 'van-gogh'), 'tiles/van-gogh/approved/Sou4.svg');
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('4s', 'van-gogh')));
  assert.ok(!set.remaining.includes('4s'));
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/four-bamboo-green.json', import.meta.url), 'utf8'));
  assert.equal(provenance.source, entry.source);
  assert.equal(provenance.sourceSha256, createHash('sha256').update(source).digest('hex'));
  assert.deepEqual(provenance.originalCrop, { x: 974, y: 144, width: 438, height: 671 });
  assert.equal(provenance.originalBoardSha256,
    '8aa09c5f197667a94effffd037e506a02064a409444667923e7a46de3d09a450');
});
'''

changes = {script_path: script, tests_path: tests}
for rel in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = root / rel
    text = p.read_text()
    text = text.replace('fifteen distinct faces', 'sixteen distinct faces').replace('fifteen approved faces', 'sixteen approved faces').replace('all fifteen faces', 'all sixteen faces')
    text = text.replace('remaining 19 tile identities', 'remaining 18 tile identities').replace('other 19 identities', 'other 18 identities')
    text = text.replace('plus green Triple Shoots A for 3 bamboo,', 'plus green Moonlit Four C for 4 bamboo, green Triple Shoots A for 3 bamboo,')
    text = text.replace('The set now has 15 painted faces and 19 Classic fallbacks.', 'That addition brought the set to 15 painted faces and 19 Classic fallbacks.')
    text = text.replace('The inventory remains 15 painted faces and 19 Classic fallbacks.', 'That replacement retained 15 painted faces and 19 Classic fallbacks.')
    text = text.replace('The source pixels are cropped losslessly, with no regeneration, repainting or upscaling.', 'Earlier PNG sources are cropped losslessly. The newer green bamboo sources use documented quality-95 WebP game exports, without regeneration or repainting.')
    text += '\n## Moonlit Four: green 4 bamboo\n\nThe approved greener **C — Moonlit Four** is active for `4s` (`Sou4`). Four distinct bamboo stalks stand against the emerald and teal night sky. The committed source is a 300 × 400, quality-95 WebP game export of the selected right panel, not a lossless full-resolution original. The original board checksum and 438 × 671 crop coordinates are recorded in `docs/design/van-gogh/four-bamboo-green.json`. The original review board and full-resolution selected crop were also preserved in the downloadable approval archive.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=4s` to reproduce the playable PNG and SVG. All previously approved artwork is unchanged, including 2 and 3 bamboo. The set now has 16 painted faces and 18 Classic fallbacks.\n'
    changes[p] = text

(root / source_rel).write_bytes(source)
for p, text in changes.items():
    p.write_text(text, encoding='utf-8')
command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=4s']
subprocess.run(command, check=True)
new_manifest = json.loads((out / 'manifest.json').read_text())
assert len(new_manifest['tiles']) == 16 and len(new_manifest['remaining']) == 18
for old in old_manifest['tiles']:
    assert next(e for e in new_manifest['tiles'] if e['tile'] == old['tile']) == old
for key in ['rejected', 'superseded']:
    assert old_manifest[key] == new_manifest[key]
for p, expected in old_images.items():
    assert sha(p.read_bytes()) == expected, f'Other artwork changed: {p}'
for rel, expected in old_sources.items():
    assert sha((root / rel).read_bytes()) == expected, f'Existing study changed: {rel}'
entry = next(e for e in new_manifest['tiles'] if e['tile'] == '4s')
assert entry['source'] == source_rel and entry['candidate'] == 'Bamboo C (green)'
png = (out / entry['png']).read_bytes()
assert sha(png) == entry['pngSha256']
assert (int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')) == (300, 400)
html = (out / 'preview.html').read_text()
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', html)
assert rack and rack.group(1).count('<img ') == 14
assert 'approved/Sou4.svg' in rack.group(1)
first_export = {p: sha(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
registry = (root / 'web/src/lib/van-gogh-faces.js').read_bytes()
subprocess.run(command, check=True)
assert all(sha(p.read_bytes()) == h for p, h in first_export.items()), 'Non-deterministic re-export'
assert (root / 'web/src/lib/van-gogh-faces.js').read_bytes() == registry
print(f'PASS: approved C checksum; 16 faces; unchanged {len(old_images)} existing tile files and all existing manifest entries; 14-tile preview; deterministic export')
