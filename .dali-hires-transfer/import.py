"""Import only Carl's two approved high-resolution Dalí bamboo portraits.
Runs on the isolated GitHub deployment branch; never resizes or redraws art.
"""
from pathlib import Path
import base64
import hashlib
import json
import re
import struct

root = Path.cwd()
def read(p):
    return (root / p).read_text()
def write(p, text):
    (root / p).write_text(text)
def hash_bytes(b):
    return hashlib.sha256(b).hexdigest()
def replace(text, old, new):
    if old not in text:
        raise ValueError('Expected patch anchor not found: ' + old[:100])
    return text.replace(old, new)
def dump(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

manifest_path = 'web/public/tiles/dali/manifest.json'
manifest = json.loads(read(manifest_path))
unchanged = {e['tile']: json.dumps(e, sort_keys=True) for e in manifest['tiles'] if e['tile'] not in ('3s', '4s')}
old_art = {str(p): hash_bytes(p.read_bytes()) for p in (root / 'web/public/tiles/dali/approved').glob('*') if p.name not in ('Sou3.svg', 'Sou4.svg')}
fixtures = [
    ('3', 'three', 'A', 'surreal_bamboo_knot_shoreline.png',
     '45ddc490ac33a4cd00da6f61da61204f459c52d7788ebc62c8ec65fb541e177e',
     '98d4ff4e204e5c60068666757ca65c79980fd8db9db4b257100d27d8917eec9f', 0, range(0, 6)),
    ('4', 'four', 'B', 'moonlit_tear_of_the_bamboo_eye.png',
     'dd94c79cc3eb82895ae55d92729d4c72639deab6bdb6b44f1bec03f2d87a45d9',
     'd9e06e6d80d9d5af1931817d8832ccb2093ed9349108a6e55c0a6ee462676611', 4, range(6, 16)),
]
tests_path = 'web/tests/dali-bamboo.test.js'
tests = read(tests_path)
exporter_path = 'web/scripts/export-dali-tiles.mjs'
exporter = read(exporter_path)
summary = []
for rank, word, candidate, original, original_hash, raster_hash, speed, indices in fixtures:
    tile, name = rank + 's', 'Sou' + rank
    entry = next(e for e in manifest['tiles'] if e['tile'] == tile)
    provenance_path = f'docs/design/dali/{word}-bamboo.json'
    prior = json.loads(read(provenance_path))
    raster = b''.join((root / '.dali-hires-transfer' / f'{i:03}.bin').read_bytes() for i in indices)
    assert hash_bytes(raster) == raster_hash, f'{name} transfer hash mismatch'
    assert raster[4:8] == b'ftyp' and raster[8:12] == b'avif'
    ispe = raster.index(b'ispe')
    assert struct.unpack('>II', raster[ispe+8:ispe+16]) == (1086, 1448)
    raster_source = f'docs/design/dali/studies/{word}-bamboo-{candidate.lower()}-hires.avif'
    (root / raster_source).write_bytes(raster)
    metadata = {
        'originalFilename': original, 'originalSha256': original_hash,
        'originalDimensions': [1086, 1448], 'selectedPanel': prior['selectedPanel'],
        'revision': 'approved standalone high-resolution portrait',
        'originalCrop': {'x': 0, 'y': 0, 'width': 1086, 'height': 1448},
        'rasterDimensions': [1086, 1448], 'rasterMimeType': 'image/avif',
        'rasterSha256': raster_hash, 'rasterSource': raster_source,
        'encoding': {'encoder': 'Pillow 12.3.0 AVIF', 'quality': 50, 'speed': speed, 'lossless': False},
        'processing': 'Approved standalone high-resolution portrait encoded at its full 1086 x 1448 pixel dimensions; AVIF quality 50; no resizing, crop, redraw, colour edit, added border or tile texture. The 300 x 400 SVG viewBox controls layout only, not embedded raster resolution.',
    }
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
    svg += f'<title id="title">{rank} bamboo — Dali — {entry["direction"]}</title>'
    svg += '<metadata>' + json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) + '</metadata>'
    svg += '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
    svg += '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n'
    source = entry['source']
    runtime = 'web/public/tiles/dali/' + entry['svg']
    write(source, svg)
    write(runtime, svg)
    svg_hash = hash_bytes(svg.encode())
    entry.update(sourceSha256=svg_hash, svgSha256=svg_hash, crop=metadata['originalCrop'])
    provenance = {'tile': tile, 'name': name, 'direction': entry['direction'], 'source': source, 'runtime': runtime, **metadata,
                  'svgSha256': svg_hash, 'originalStoredInRepository': False,
                  'originalLocation': 'Approved PNG conversation attachment; repository retains the full-dimension AVIF and self-contained SVG, not the original PNG bytes.',
                  'supersedes': prior}
    dump(provenance_path, provenance)
    old_definition = f"'{Path(source).name}', [0, 0, 300, 400]"
    exporter = replace(exporter, old_definition, f"'{Path(source).name}', [0, 0, 1086, 1448]")
    start = tests.index("  {\n    tile: '" + tile + "'")
    end = tests.index('\n  },', start) + len('\n  },')
    fixture = f"""  {{
    tile: '{tile}', name: '{name}', words: '{word} bamboo', direction: '{entry['direction']}',
    source: '{source}',
    artwork: '{svg_hash}',
    original: '{original_hash}',
    raster: '{raster_hash}',
    originalFilename: '{original}',
    originalDimensions: [1086, 1448], rasterDimensions: [1086, 1448],
    mime: 'image/avif',
    processing: /Approved standalone high-resolution portrait encoded at its full 1086 x 1448 pixel dimensions/,
  }},"""
    tests = tests[:start] + fixture + tests[end:]
    summary.append({'tile': tile, 'svgSha256': svg_hash, 'rasterSha256': raster_hash, 'dimensions': [1086, 1448], 'bytes': len(raster)})

new_intro = 'Three and four bamboo use the approved standalone high-resolution portraits at their full 1086 × 1448 pixel dimensions. Both embed full-dimension AVIF artwork; the 300 × 400 SVG viewBox controls game layout only. All other artwork is unchanged. '
exporter = re.sub(r'Four bamboo uses B — The Sleeping Landscape:.*?East wind uses', new_intro + 'East wind uses', exporter, count=1)
exporter = replace(exporter, 'extracted from the selected panel without its border or caption.', 'using its approved high-resolution standalone portrait.')
write(exporter_path, exporter)
preview_path = 'web/public/tiles/dali/preview.html'
preview = read(preview_path)
preview = re.sub(r'Four bamboo uses B — The Sleeping Landscape:.*?East wind uses', new_intro + 'East wind uses', preview, count=1)
preview = replace(preview, 'extracted from the selected panel without its border or caption.', 'using its approved high-resolution standalone portrait.')
write(preview_path, preview)
dump(manifest_path, manifest)

tests = replace(tests, 'the embedded WebP records its original', 'the embedded raster records its original')
tests = replace(tests, "assert.equal(metadata.rasterMimeType, 'image/webp');", "assert.equal(metadata.rasterMimeType, face.mime ?? 'image/webp');")
tests = replace(tests, "svg.match(/data:image\\/webp;base64,([^\"\\s]+)/)", "svg.match(/data:image\\/(?:webp|avif);base64,([^\"\\s]+)/)")
tests = replace(tests, "    assert.equal(raster.toString('ascii', 0, 4), 'RIFF');\n    assert.equal(raster.toString('ascii', 8, 12), 'WEBP');", """    if (face.mime === 'image/avif') {
      assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
      const ispe = raster.indexOf(Buffer.from('ispe'));
      assert.ok(ispe >= 0);
      assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [1086, 1448]);
      assert.deepEqual(raster, readFileSync(new URL(metadata.rasterSource, root)));
    } else {
      assert.equal(raster.toString('ascii', 0, 4), 'RIFF');
      assert.equal(raster.toString('ascii', 8, 12), 'WEBP');
    }""")
tests = replace(tests, '{ x: 11, y: 12, width: 482, height: 877 }', '{ x: 0, y: 0, width: 1086, height: 1448 }')
tests = replace(tests, '{ x: 782, y: 0, width: 754, height: 1024 }', '{ x: 0, y: 0, width: 1086, height: 1448 }')
for word, candidate in [('three', 'a'), ('four', 'b')]:
    tests = replace(tests, f"'{word}-bamboo-{candidate}-approved.svg', \\[0, 0, 300, 400\\]", f"'{word}-bamboo-{candidate}-approved.svg', \\[0, 0, 1086, 1448\\]")
tests = replace(tests, 'three bamboo preserves selected panel A', 'three bamboo preserves the approved high-resolution A portrait')
tests = replace(tests, 'four bamboo preserves the selected right panel', 'four bamboo preserves the approved high-resolution B portrait')
tests = replace(tests, '// Three bamboo remains the original selected A, not the regenerated diptych left panel.', '// Three bamboo uses the separately approved high-resolution A portrait.')
tests = replace(tests, "assert.equal(three.svgSha256, '43741c2f0f26ef70f89044f9a822ecfcffdb95fbdc2f130b7152eb3302363c95');", "assert.equal(three.svgSha256, FACES.find(entry => entry.tile === '3s').artwork);")
write(tests_path, tests)
faces_path = 'web/tests/tile-faces.test.js'
faces = read(faces_path)
old = "assert.match(svg, ['3s', '4s', '7s', '8s', '9s', '1z', '4z'].includes(tile) ? /data:image\\/webp;base64,/ : /data:image\\/png;base64,/);"
new = "assert.match(svg, ['3s', '4s'].includes(tile) ? /data:image\\/avif;base64,/\n        : ['7s', '8s', '9s', '1z', '4z'].includes(tile) ? /data:image\\/webp;base64,/ : /data:image\\/png;base64,/);"
write(faces_path, replace(faces, old, new))
readme_path = 'docs/design/dali/README.md'
readme = read(readme_path)
readme = replace(readme, 'The latest is four bamboo **The Sleeping Landscape — B**, using the approved right-hand painting with exactly four green stalks on a sleeping stone eyelid and a suspended green tear. The existing three-bamboo A artwork is unchanged.', 'Three bamboo **The Bamboo That Tied Itself — A** and four bamboo **The Sleeping Landscape — B** now use Carl\'s approved standalone high-resolution portraits at their full 1086 × 1448 pixel dimensions.')
for rank, word, candidate, original, original_hash, raster_hash, speed, indices in fixtures:
    paragraph = f'{word.capitalize()} bamboo uses [the approved high-resolution {candidate} portrait](studies/{word}-bamboo-{candidate.lower()}-approved.svg), from `{original}` (1086 × 1448; original PNG SHA-256 `{original_hash}`). The full portrait is encoded as AVIF quality 50, speed {speed}, **without resizing**, cropping, redrawing or a colour edit. The full-dimension AVIF is stored separately and embedded in byte-identical source/runtime SVGs. The 300 × 400 SVG viewBox, rounded clipping and 1% bleed preserve game geometry; they do not reduce the embedded raster to 300 × 400. Compression is lossy, not an assertion of original PNG-byte preservation. The original PNG remains in the conversation. See [provenance]({word}-bamboo.json) for the exact source/raster/SVG hashes and superseded low-resolution revision. All other tile identities and artwork are unchanged.'
    readme, count = re.subn(r'^' + word.capitalize() + r' bamboo uses .*$', lambda m: paragraph, readme, flags=re.M)
    assert count == 1
write(readme_path, readme)
assert {e['tile']: json.dumps(e, sort_keys=True) for e in manifest['tiles'] if e['tile'] not in ('3s', '4s')} == unchanged
assert all(hash_bytes(Path(p).read_bytes()) == h for p, h in old_art.items())
print(json.dumps({'replaced': summary, 'otherApprovedFacesUnchanged': len(unchanged)}, indent=2))
