"""One-time, checksum-guarded import of the approved copper 5-bamboo artwork."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

root = Path.cwd()
transport = root / '.van-gogh-five-transfer'
out = root / 'web/public/tiles/van-gogh'
source = 'docs/design/van-gogh/studies/15-five-bamboo-copper-sunset.webp'
source_sha = '16552f26fc7e7090adbc13cf659afcdb9b2549cd1a543441b93eed7d5f742d81'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one patch anchor, found {text.count(old)}: {old!r}')
    return text.replace(old, new, 1)

art = b''.join((transport / f'part{i:02d}.bin').read_bytes() for i in range(8))
assert len(art) == 89232 and sha(art) == source_sha, 'Source transfer checksum mismatch'
assert not (root / source).exists(), 'Source path already exists'
previous = json.loads((out / 'manifest.json').read_text())
assert len(previous['tiles']) == 18 and len(previous['remaining']) == 16
assert '5s' in previous['remaining'] and not any(e['tile'] == '5s' for e in previous['tiles'])
existing_art = {
    p: sha(p.read_bytes())
    for folder in [root / 'web/public/tiles', root / 'docs/design/van-gogh/studies']
    for p in folder.rglob('*')
    if p.is_file() and p.suffix.lower() in {'.png', '.svg', '.webp', '.jpg', '.jpeg'}
}

export_path = root / 'web/scripts/export-van-gogh-tiles.mjs'
text = export_path.read_text()
text = once(text, '\n};\nconst definitions = [',
            f"\n  copperFive: '{source}',\n}};\nconst definitions = [")
text = once(text, '\n];\nconst onlyArgument =',
            "\n  ['Copper Sunset', '5s', 'Five bamboo', 'copperFive', [0, 0, 300, 400]],\n];\nconst onlyArgument =")
text = once(text,
    'Carl selected the raft concept for 8s and approved deployment of the resulting eight-bamboo painting without recolouring.',
    'Carl selected the raft concept for 8s and approved deployment of the resulting eight-bamboo painting without recolouring. Carl approved Copper Sunset for 5s with four green stalks and a reddish-brown central stalk, replacing the earlier all-green concept.')
text = once(text, 'Includes the approved Bamboo Raft (8s)',
            'Includes approved Copper Sunset (5s) and Bamboo Raft (8s)')
text = once(text, "const featured = ['8s', '7s', '4s', '3s', '2s', '2m', '3m', '4m'];",
            "const featured = ['5s', '8s', '7s', '4s', '3s', '2s', '2m', '3m', '4m'];")
text = once(text, "const handTiles = [...featured, ...approved.filter(tile => !featured.includes(tile)).slice(0, 4), '2z', '6z'];",
            "const handTiles = [...featured, ...approved.filter(tile => !featured.includes(tile)).slice(0, 3), '2z', '6z'];")
text = once(text, 'approved faces, including the Bamboo Raft for 8 bamboo,',
            'approved faces, including Copper Sunset for 5 bamboo, the Bamboo Raft for 8 bamboo,')

test_path = root / 'web/tests/tile-faces.test.js'
tests = test_path.read_text()
tests = once(tests,
    "const approved = ['1p', '5p', '3s', '3m', '7z', '1s', '2p', '9p', '6s', '5z', '1z', '4z', '2m', '4m', '2s', '4s', '7s', '8s'];",
    "const approved = ['1p', '5p', '3s', '3m', '7z', '1s', '2p', '9p', '6s', '5z', '1z', '4z', '2m', '4m', '2s', '4s', '7s', '8s', '5s'];")
tests = once(tests, "'Seven C', 'Bamboo Raft']);", "'Seven C', 'Bamboo Raft', 'Copper Sunset']);")
tests = once(tests, "url.startsWith('tiles/van-gogh/')).length, 18)",
                   "url.startsWith('tiles/van-gogh/')).length, 19)")
tests = once(tests, "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s', '4s', '7s', '8s'])",
                   "for (const tile of ['1z', '4z', '2m', '4m', '2s', '3s', '4s', '7s', '8s', '5s'])")
# Only edit inventory assertions inside Van Gogh tests, never the other sets.
counts = {'remaining': 0, 'tiles': 0}
def update_inventory(match):
    block = match.group(0)
    for field, before, after in [('remaining', 16, 15), ('tiles', 18, 19)]:
        old = f'assert.equal(set.{field}.length, {before});'
        counts[field] += block.count(old)
        block = block.replace(old, f'assert.equal(set.{field}.length, {after});')
    return block
tests = re.sub(r"test\('Van Gogh[^\n]*\n[\s\S]*?\n\}\);", update_inventory, tests)
assert counts == {'remaining': 2, 'tiles': 1}, counts
tests += """

test('Van Gogh 5 bamboo uses the approved copper-centred sunset, not the green draft', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const entry = set.tiles.find(tile => tile.tile === '5s');
  assert.ok(entry);
  assert.equal(entry.candidate, 'Copper Sunset');
  assert.equal(entry.name, 'Sou5');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/15-five-bamboo-copper-sunset.webp');
  assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 });
  const source = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'),
    '16552f26fc7e7090adbc13cf659afcdb9b2549cd1a543441b93eed7d5f742d81');
  const provenance = JSON.parse(readFileSync(new URL('../../docs/design/van-gogh/five-bamboo-copper.json', import.meta.url), 'utf8'));
  assert.equal(provenance.sourceSha256, createHash('sha256').update(source).digest('hex'));
  assert.equal(provenance.originalSha256, '20f7812e84d67802c9768baf28c7a2b7d89fa0e68a2a42783fa655380dca51c0');
  assert.deepEqual(provenance.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.ok(TILE_IMAGE_URLS.includes(tileImage('5s', 'van-gogh')));
  assert.ok(!set.remaining.includes('5s'));
});
"""

note = '''\n## Copper Sunset — 5 bamboo\n\nThe approved **Copper Sunset** is active for `5s` / `Sou5`: four green bamboo stalks frame one tall reddish-brown central stalk. The earlier all-green concept is not used. The complete approved composition is retained, with no cropping, recolouring or repainting.\n\n`studies/15-five-bamboo-copper-sunset.webp` is a 300×400, quality-95 WebP game export, following the recent bamboo convention. It is not a lossless full-resolution original. Source and original checksums, full-canvas coordinates and processing details are recorded in `docs/design/van-gogh/five-bamboo-copper.json`; the untouched original is preserved in `van-gogh-five-bamboo-copper-original.zip` supplied in chat.\n\nThe set now contains **19 painted faces and 15 Classic fallbacks**. All 18 previous painted faces and their source artwork are unchanged. Regenerate this addition alone with `node web/scripts/export-van-gogh-tiles.mjs --only=5s`.\n'''
changes = {export_path: text, test_path: tests}
for relative in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    path = root / relative
    doc = path.read_text()
    # Update present-tense overview; retain dated per-tile history below.
    if relative.startswith('docs/'):
        doc = doc.replace('set with seventeen distinct faces', 'set with nineteen distinct faces', 1)
        doc = doc.replace('plus Seven with Irises C for 7 bamboo,',
                          'plus Copper Sunset for 5 bamboo, Bamboo Raft for 8 bamboo, Seven with Irises C for 7 bamboo,', 1)
        doc = doc.replace('The remaining 17 tile identities', 'The remaining 15 tile identities', 1)
        doc = doc.replace('reproduces all seventeen faces', 'reproduces all nineteen faces', 1)
    else:
        doc = doc.replace('The eighteen approved faces', 'The nineteen approved faces', 1)
        doc = doc.replace('The other 16 identities', 'The other 15 identities', 1)
        doc = doc.replace('newer green bamboo sources and Seven with Irises',
                          'newer green bamboo sources, Copper Sunset and Seven with Irises', 1)
    changes[path] = doc.rstrip() + '\n' + note

# No repository files are written until every patch anchor and input has passed.
provenance = json.loads((transport / 'five-bamboo-copper.json').read_text())
assert provenance['sourceSha256'] == source_sha and provenance['source'] == source
assert provenance['originalSha256'] == '20f7812e84d67802c9768baf28c7a2b7d89fa0e68a2a42783fa655380dca51c0'
(root / source).write_bytes(art)
(root / 'docs/design/van-gogh/five-bamboo-copper.json').write_text(json.dumps(provenance, indent=2) + '\n')
for path, content in changes.items():
    path.write_text(content)
command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=5s']
subprocess.run(command, check=True)
manifest = json.loads((out / 'manifest.json').read_text())
assert manifest['tiles'][:-1] == previous['tiles'], 'An existing manifest entry changed'
assert manifest['sources'][:-1] == previous['sources'], 'An existing source entry changed'
assert len(manifest['tiles']) == 19 and len(manifest['remaining']) == 15
assert '5s' not in manifest['remaining']
entry = manifest['tiles'][-1]
assert entry['tile'] == '5s' and entry['candidate'] == 'Copper Sunset'
assert entry['source'] == source and entry['crop'] == {'x': 0, 'y': 0, 'width': 300, 'height': 400}
png = (out / entry['png']).read_bytes()
assert png[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', png[16:24]) == (300, 400)
assert sha(png) == entry['pngSha256']
for path, checksum in existing_art.items():
    assert path.exists() and sha(path.read_bytes()) == checksum, f'Existing art changed: {path}'
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', (out / 'preview.html').read_text(), re.S)
assert rack and rack.group(1).count('<img ') == 14, 'Preview must have a 14-tile hand'
generated = {p: sha(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
registry = root / 'web/src/lib/van-gogh-faces.js'
generated[registry] = sha(registry.read_bytes())
subprocess.run(command, check=True)
assert all(sha(p.read_bytes()) == checksum for p, checksum in generated.items()), 'Non-deterministic re-export'
print(f'PASS: approved copper source verified; {len(existing_art)} existing artwork files unchanged; 18 prior entries preserved; 19 approved / 15 fallback; 14-tile preview; deterministic export.')
