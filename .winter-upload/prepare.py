"""Import Carl's selected winter North only; preserve and verify all other art."""
from pathlib import Path
import base64, hashlib, json, os, re, shutil, subprocess
root = Path.cwd()
folder = root / 'web/public/tiles/van-gogh'
meta = json.loads(Path('.winter-upload/provenance.json').read_text())
source_path = Path(meta['source'])
runtime_path = Path(meta['runtimeSvg'])
hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
art = [Path(s) for s in tracked if s and s.startswith(('web/public/tiles/', 'docs/design/')) and Path(s).suffix in ('.png', '.svg', '.webp', '.avif', '.jpg', '.jpeg')]
allowed = {runtime_path, folder.relative_to(root) / 'approved/Pei.png'}
protected = {str(p): hash_bytes(p.read_bytes()) for p in art if p not in allowed}
before = json.loads((folder / 'manifest.json').read_text())
old_north = next(e for e in before['tiles'] if e['tile'] == '4z')
assert old_north['candidate'] == 'North B', 'North changed since selection; review rather than overwrite'
assert old_north['source'] == meta['supersededSource']
assert not source_path.exists()
raw = b''.join(Path(f'.winter-upload/part{i}.bin').read_bytes() for i in range(4))
assert len(raw) == meta['rasterBytes']
assert hash_bytes(raw) == meta['rasterSha256'], 'Uploaded raster is not the approved export'
encoded = base64.b64encode(raw).decode('ascii')
source = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">North wind — Winter B — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + encoded + '"/></svg>\n'
assert hash_bytes(source.encode()) == meta['sourceSha256']
source_path.write_text(source)
archive_svg, archive_png = Path(meta['archivedSvg']), Path(meta['archivedPng'])
archive_svg.parent.mkdir(parents=True, exist_ok=True)
assert not archive_svg.exists() and not archive_png.exists()
shutil.copyfile(runtime_path, archive_svg)
old_png = folder / 'approved/Pei.png'
assert old_png.exists()
shutil.copyfile(old_png, archive_png)
old_png.unlink()
meta['archivedSvgSha256'] = hash_bytes(archive_svg.read_bytes())
meta['archivedPngSha256'] = hash_bytes(archive_png.read_bytes())
meta['previousNorthEntry'] = old_north
Path('docs/design/van-gogh/north-wind-winter.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')

def one(text, old, new):
    assert text.count(old) == 1, f'Expected one occurrence of {old!r}, found {text.count(old)}'
    return text.replace(old, new, 1)

exporter = Path('web/scripts/export-van-gogh-tiles.mjs')
text = exporter.read_text()
text = one(text, 'const fits = {\n', "const fits = {\n  // Winter B is a documented 3:4 crop, with no stretch or extra bleed.\n  '4z': { painting: [456, 608], stretch: 0, anchor: 0.5, bleedPixels: 0 },\n")
text = one(text, 'const sources = {\n', f"const sources = {{\n  northWinter: '{meta['source']}',\n")
text = one(text, "['North B', '4z', 'North wind', 'north', [0, 0, 1086, 1448]]", "['Winter B', '4z', 'North wind', 'northWinter', [0, 0, 300, 400]]")
lines = text.splitlines(keepends=True)
for i, line in enumerate(lines):
    if line.startswith('  approval: '):
        assert line.endswith("',\n")
        lines[i] = line[:-3] + ' Carl selected the winter middle panel B as the replacement North wind and explicitly requested deployment. Snowbound North replaces Wind Ribbons; the former source and exact playable images remain archived. The winter scene is cropped to 3:4 without stretching or repainting.' + "',\n"
text = ''.join(lines)
text = one(text, '  superseded: [\n', "  superseded: [\n    { candidate: 'North B', tile: '4z', activeCandidate: 'Winter B', source: sources.north, archivedPng: 'docs/design/van-gogh/superseded/north-wind-ribbons-Pei.png', archivedSvg: 'docs/design/van-gogh/superseded/north-wind-ribbons-Pei.svg' },\n")
text = one(text, '. Includes The Billowing Courtyard A', '. Includes Snowbound North B (4z), replacing Wind Ribbons, and The Billowing Courtyard A')
text = one(text, 'Blazing Dawn for East and Wind Ribbons for North.', 'Blazing Dawn for East and Snowbound North B for North.')
text = one(text, '<p class="notes">The Red Sail', '<p class="notes">Snowbound North B uses the selected winter scene, cropped to 456 × 608 and uniformly downsampled to a 300 × 400 quality-60 AVIF, with zero stretch and zero extra bleed. The full selected panel and original triptych are backed up separately, and the old Wind Ribbons source and playable images remain in the design archive. The Red Sail')
exporter.write_text(text)
subprocess.run(['node', str(exporter), '--only=4z'], check=True)
after = json.loads((folder / 'manifest.json').read_text())
assert [e['tile'] for e in after['tiles']] == [e['tile'] for e in before['tiles']]
assert after['remaining'] == before['remaining']
assert [e for e in after['tiles'] if e['tile'] != '4z'] == [e for e in before['tiles'] if e['tile'] != '4z']
assert hash_bytes(runtime_path.read_bytes()) == meta['runtimeSvgSha256']
assert not (folder / 'approved/Pei.png').exists()
for name, digest in protected.items():
    assert hash_bytes(Path(name).read_bytes()) == digest, f'Other artwork changed: {name}'

tests = Path('web/tests/tile-faces.test.js')
text = tests.read_text()
text = one(text, 'Almond Branches, East A and North B,', 'Almond Branches, East A and Winter B,')
text = one(text, "'East A', 'North B', 'Characters B'", "'East A', 'Winter B', 'Characters B'")
text = one(text, "assert.equal(north.source, 'docs/design/van-gogh/studies/04-north-wind-ribbons.png');", f"assert.equal(north.source, '{meta['source']}');")
text = one(text, 'assert.deepEqual(north.crop, { x: 0, y: 0, width: 1086, height: 1448 });', 'assert.deepEqual(north.crop, { x: 0, y: 0, width: 300, height: 400 });')
text = one(text, "'3a51fc2f26cbd476cce085d93b52a5a81c5a17d14d4bcf00a0597a402435588e');", f"'{meta['sourceSha256']}');")
text = one(text, "['9m', '6z', '1m', '2z'].includes(tile)", "['9m', '6z', '1m', '2z', '4z'].includes(tile)")
tests.write_text(text)

# Reuse the repository's isolated export fixture (including Windows newline handling).
text = Path('web/tests/van-gogh-one-red-sail.test.js').read_text()
for old, new in [
    ('26-one-characters-red-sail-c-approved.svg', '28-north-wind-winter-b-approved.svg'),
    ('one-characters-red-sail.json', 'north-wind-winter.json'),
    ('crimson_sail_in_a_mediterranean_harbour.png', 'windy_northern_landscapes_triptych.png'),
    ('7f82fc7cc6ca21f37cba9ccfe5f0292e836eb1f584d5aa7097feaf14ca7ddcf9', meta['sourceSha256']),
    ('85a214007ce059ed61aa47f2257db9ce2c6499ed58c95954d926d88fd2dbcfea', meta['runtimeSvgSha256']),
    ('0a6b2b96aaebe241576e94d6cb1d80acf5fc862a6b7a33f8a5323b2c4bbbcc70', meta['rasterSha256']),
    ('81570fc132f12f48fceb262ecc0f7e6a4f1e9109198d38281a149dea0a8e4d2a', meta['originalSha256']),
    ('Red Sail C', 'Winter B'), ('Man1', 'Pei'), ("'1m'", "'4z'"), ('--only=1m', '--only=4z'),
    ('one character uses the selected red-sail boat, not the sower or window alternatives', 'North wind uses the selected winter B, not the cypress or harbour alternatives'),
    ('the boat keeps the complete 3:4 painting', 'winter North keeps the documented 3:4 crop'),
    ('the boat is registered and preloaded once, completing the characters suit without revealing hidden tiles', 'winter North is registered and preloaded once without revealing hidden tiles'),
    ('the red sail appears', 'winter North appears'),
    ('approved boat exactly', 'approved winter North exactly'),
    ('van-gogh-one-', 'van-gogh-north-winter-'),
    ('[1086, 1448]', '[1448, 1086]'),
    ('{ width: 1086, height: 1448 }', '{ width: 456, height: 608 }'),
    ('36049', '37117'), ('One character', 'North wind'),
]:
    assert old in text, old
    text = text.replace(old, new)
text += """

test('the prior North ribbons and their original source remain recoverable unchanged', () => {
  const archived = manifest.superseded.find(item => item.tile === '4z' && item.candidate === 'North B');
  assert.ok(archived);
  assert.equal(archived.activeCandidate, 'Winter B');
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  assert.equal(hash(read(record.supersededSource)), '3a51fc2f26cbd476cce085d93b52a5a81c5a17d14d4bcf00a0597a402435588e');
  assert.equal(record.previousNorthEntry.candidate, 'North B');
  assert.deepEqual(record.originalCrop, { x: 497, y: 126, width: 456, height: 608 });
  assert.deepEqual(record.fullPanelCrop, { x: 497, y: 108, width: 456, height: 856 });
});
"""
Path('web/tests/van-gogh-north-winter.test.js').write_text(text)

notice = '**Current North wind:** Snowbound North B (`4z` / `Pei`) replaces Wind Ribbons, following Carl\'s winter selection. The snowy scene is cropped to exactly 3:4, not squeezed; the complete 北, moon, village and snowy path remain visible. The former North source and playable PNG/SVG are retained in the design archive. Earlier North sections below describe that superseded design. This replacement does not add a tile or change any other artwork.\n\n'
section = '\n## Snowbound North B — winter replacement\n\nCarl selected the middle winter panel and requested GitHub deployment. The stone marked 北 stands beside a snowy path, warm-lit village and swirling blue-and-gold sky. The cypress and harbour alternatives are not used.\n\nThe crop `[497, 126, 456, 608]` on the original 1448 × 1086 triptych is exactly 3:4. A small strip of upper sky and excess foreground are trimmed, not stretched. The full selected panel and untouched board remain in the separately supplied approval ZIP. Uniform LANCZOS downsampling produces a 300 × 400 quality-60 AVIF; there is no repainting, recolouring or glyph editing. The playable face uses zero stretch and zero extra bleed, apart from the usual rounded-corner mask.\n\nSource, original, crop, raster, runtime and archive checksums are recorded in `docs/design/van-gogh/north-wind-winter.json`. Reproduce only this replacement with `node web/scripts/export-van-gogh-tiles.mjs --only=4z`. All other selected faces and the existing fourteen-tile preview hand remain unchanged.\n'
for name in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = Path(name)
    p.write_text(one(p.read_text(), '# Van Gogh\n\n', '# Van Gogh\n\n' + notice) + section)
report = {'status': 'prepared', 'approved': len(after['tiles']), 'remaining': after['remaining'], 'protectedArtFiles': len(protected), 'sourceSha256': meta['sourceSha256'], 'rasterSha256': meta['rasterSha256'], 'runtimeSvgSha256': meta['runtimeSvgSha256'], 'archivedSvgSha256': meta['archivedSvgSha256'], 'archivedPngSha256': meta['archivedPngSha256']}
Path(os.environ['RUNNER_TEMP'], 'winter-import-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
