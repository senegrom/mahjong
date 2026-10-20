"""One-time import of Carl's selected cupboard C. Check every byte and fail closed."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
TRANSFER = ROOT / '.dali-eight-cabinet-transfer'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Patch anchor changed: {old!r}')
    return text.replace(old, new, 1)

def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout

def main():
    manifest_path = ROOT / 'web/public/tiles/dali/manifest.json'
    before = json.loads(manifest_path.read_text())
    assert len(before['tiles']) == 25 and len(before['placeholders']) == 9
    assert not any(e['tile'] == '8p' for e in before['tiles']), 'Already installed: stop, do not overwrite'
    tracked = [p for p in run('git', 'ls-files', '-z').split('\0') if p]
    protected = {p: sha((ROOT / p).read_bytes()) for p in tracked if Path(p).suffix.lower() in ('.png', '.svg', '.avif', '.webp', '.jpg', '.jpeg', '.gif')}
    other_sets = {p: (ROOT / p).read_bytes() for p in ('web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/matisse/manifest.json', 'web/src/lib/van-gogh-faces.js', 'web/src/lib/matisse-faces.js')}
    preview_path = ROOT / 'web/public/tiles/dali/preview.html'
    hand = re.search(r'<div class="hand">(.*?)</div>', preview_path.read_text(), re.S).group(1)
    assert hand.count('<img ') == 14
    parts = json.loads((TRANSFER / 'parts.json').read_text())
    chunks = []
    for part in parts:
        data = (ROOT / part['path']).read_bytes()
        assert len(data) == part['bytes'] and sha(data) == part['sha256'], part['path']
        chunks.append(data)
    raster = b''.join(chunks)
    record = json.loads((TRANSFER / 'eight-disks.json').read_text())
    assert len(raster) == 42787
    assert sha(raster) == record['rasterSha256'] == 'e1c99935fa07713f8a2c37175090036dde471997573f78ee7dbcf713607b55e0'
    assert raster[4:12] == b'ftypavif'
    ispe = raster.index(b'ispe')
    assert [int.from_bytes(raster[ispe+8:ispe+12], 'big'), int.from_bytes(raster[ispe+12:ispe+16], 'big')] == [609, 812]
    metadata = {k: v for k, v in record.items() if k != 'svgSha256'}
    text = json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))
    assert not any(c in text for c in '<>&')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">8 disks — Dali — The Cabinet of Captured Moons — C</title><metadata>{text}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,{base64.b64encode(raster).decode()}"/></svg>\n'.encode()
    assert sha(svg) == record['svgSha256'] == 'c25c605794ae06fce15a758ce34840ef555d33af1558d58f43e017d5a448c724'

    exporter_path = ROOT / 'web/scripts/export-dali-tiles.mjs'
    exporter = exporter_path.read_text()
    pin7 = "  ['Pin7', '7p', '7 disks', 'The Shell That Holds the Sea — A', 'seven-disks-a-shell-approved.svg', [0, 0, 1086, 1448]],"
    pin8 = "  ['Pin8', '8p', '8 disks', 'The Cabinet of Captured Moons — C', 'eight-disks-c-cabinet-approved.svg', [0, 0, 609, 812]],"
    exporter = once(exporter, pin7, pin7 + '\n' + pin8)
    exporter = once(exporter, '${manifest.tiles.length} approved faces. White dragon', '${manifest.tiles.length} approved faces. Eight disks uses C — The Cabinet of Captured Moons: the selected cupboard with eight moon-disks in two columns of four. Its 609 × 812 painting crop removes only the triptych gutter and bottom panel letters, without resizing, stretching, repainting or recolouring. White dragon')
    exporter = once(exporter, 'Ranks 1–7 of disks are complete.', 'Ranks 1–8 of disks are complete.')
    changes = {exporter_path: exporter}

    # Update only coverage snapshots, not artwork/behavior assertions.
    tests = sorted((ROOT / 'web/tests').glob('dali-*.test.js')) + [ROOT / 'web/tests/tile-faces.test.js']
    for p in tests:
        original = p.read_text()
        content = original
        if p.name in ('dali-bamboo.test.js', 'tile-faces.test.js'):
            content = once(content, "'6p', '7p', '1s'", "'6p', '7p', '8p', '1s'")
        if p.name == 'tile-faces.test.js':
            content = once(content, "['2p', '4p', '6p', '7p', '3s', '4s', '5z'].includes(tile)", "['2p', '4p', '6p', '7p', '8p', '3s', '4s', '5z'].includes(tile)")
        for old, new in (
            ('twenty-five approved identities and nine placeholders', 'twenty-six approved identities and eight placeholders'),
            ('Dali resolves twenty-five approved images', 'Dali resolves twenty-six approved images'),
            ('twenty-five-face preview', 'twenty-six-face preview'),
            ('assert.equal(set.tiles.length, 25);', 'assert.equal(set.tiles.length, 26);'),
            ('assert.equal(set.placeholders.length, 9);', 'assert.equal(set.placeholders.length, 8);'),
            ('25 approved faces', '26 approved faces'),
            ('remaining 9 tiles', 'remaining 8 tiles'),
            ('Ranks 1–7 of disks are complete', 'Ranks 1–8 of disks are complete'),
        ):
            content = content.replace(old, new)
        if content != original:
            changes[p] = content

    doc_path = ROOT / 'docs/design/dali/README.md'
    docs = doc_path.read_text()
    docs = once(docs, "Twenty-five faces are approved, drawn from the original studies and Carl's later selections. The latest is white dragon", "Twenty-six faces are approved, drawn from the original studies and Carl's later selections. The latest is eight disks **The Cabinet of Captured Moons — C**, the selected right-hand cupboard with eight moon-disks in two columns of four. This completes disks 1–8. The previously selected white dragon")
    docs = once(docs, '| `Sou1.svg` |', '| `Pin8.svg` | 8 disks | The Cabinet of Captured Moons — C: eight moon-disks in two columns of four inside the melting cupboard over shallow water |\n| `Sou1.svg` |')
    docs = once(docs, 'The other 9 tiles have no approved artwork yet', 'The other 8 tiles have no approved artwork yet')
    details = "Eight disks uses [selected C — The Cabinet of Captured Moons](studies/eight-disks-c-cabinet-approved.svg), the right-hand cupboard from the approved triptych. The lossless crop at (1271, 0), 609 × 812 pixels, removes the white gutter, one right-edge pixel and 24 bottom pixels containing stray panel letters. All eight moon-disks and the entire cupboard remain. Its already-prepared quality-50 AVIF is copied exactly without resizing, stretching, repainting, recolouring or recompression. The source and runtime SVGs are byte-identical and retain the shared 300 × 400 layout, radius-26 clipping and 1% bleed. Compression is lossy; the untouched review board, full C panel and lossless crop remain preserved in `dali-eight-disks-C-originals.zip` supplied in chat, rather than claiming original PNG bytes are in the repository. See [provenance](eight-disks.json). The curtain and butterfly alternatives are not selected. All prior artwork and the fourteen-tile example hand remain unchanged. Reproduce this tile with `node web/scripts/export-dali-tiles.mjs --only=8p`.\n\n"
    changes[doc_path] = once(docs, '## Presentation\n\n', '## Presentation\n\n' + details)
    changes[ROOT / 'web/tests/dali-eight-disks.test.js'] = (TRANSFER / 'dali-eight-disks.test.js').read_text()

    for p, data in ((ROOT / record['source'], svg), (ROOT / record['rasterSource'], raster), (ROOT / 'docs/design/dali/eight-disks.json', (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode())):
        assert not p.exists(), f'Refuse existing source: {p}'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for p, text in changes.items():
        p.write_text(text, encoding='utf-8')
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=8p')
    after = json.loads(manifest_path.read_text())
    assert len(after['tiles']) == 26 and len(after['placeholders']) == 8
    assert [e for e in after['tiles'] if e['tile'] != '8p'] == before['tiles']
    assert after['placeholders'] == [e for e in before['placeholders'] if e['tile'] != '8p']
    for key in set(before) - {'tiles', 'placeholders'}:
        assert after[key] == before[key], key
    assert (ROOT / record['runtime']).read_bytes() == svg
    for p, digest in protected.items():
        assert (ROOT / p).is_file() and sha((ROOT / p).read_bytes()) == digest, f'Other artwork changed: {p}'
    for p, data in other_sets.items():
        assert (ROOT / p).read_bytes() == data, f'Other set changed: {p}'
    assert re.search(r'<div class="hand">(.*?)</div>', preview_path.read_text(), re.S).group(1) == hand
    generated = {p: p.read_bytes() for p in (manifest_path, preview_path, ROOT / 'web/src/lib/dali-faces.js', ROOT / record['runtime'])}
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=8p')
    for p, data in generated.items():
        assert p.read_bytes() == data, f'Non-deterministic export: {p}'
    changed_js = sorted(str(p.relative_to(ROOT / 'web')) for p in changes if p.suffix in ('.js', '.mjs')) + ['src/lib/dali-faces.js']
    (TRANSFER / 'lint-files.json').write_text(json.dumps(changed_js))
    print(f'PASS: exact approved cupboard C; {len(protected)} prior artworks and 25 prior Dali entries unchanged; 26 painted/8 named; identical 14-tile hand; deterministic selective export.')

if __name__ == '__main__':
    main()
