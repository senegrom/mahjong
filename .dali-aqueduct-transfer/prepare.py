"""Import the selected aqueduct A; stop on any unexpected input or artwork change."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path.cwd()
T = ROOT / '.dali-aqueduct-transfer'
sha = lambda b: hashlib.sha256(b).hexdigest()

def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout

def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Patch anchor changed: {old!r}')
    return text.replace(old, new, 1)

def verify():
    before = json.loads((T / 'baseline.json').read_text())
    for filename, digest in before['protected'].items():
        p = ROOT / filename
        assert p.is_file() and sha(p.read_bytes()) == digest, f'Existing artwork changed: {filename}'
    after = json.loads((ROOT / 'web/public/tiles/dali/manifest.json').read_text())
    assert len(after['tiles']) == 29 and len(after['placeholders']) == 5
    assert [e for e in after['tiles'] if e['tile'] != '1m'] == before['manifest']['tiles']
    assert after['placeholders'] == [e for e in before['manifest']['placeholders'] if e['tile'] != '1m']
    for key in set(before['manifest']) - {'tiles', 'placeholders'}:
        assert after[key] == before['manifest'][key], key
    preview = (ROOT / 'web/public/tiles/dali/preview.html').read_text()
    assert re.search(r'<div class="hand">(.*?)</div>', preview, re.S).group(1) == before['hand']
    record = json.loads((ROOT / 'docs/design/dali/one-character.json').read_text())
    assert sha((ROOT / record['source']).read_bytes()) == record['sourceSvgSha256']
    assert sha((ROOT / record['runtime']).read_bytes()) == record['runtimeSvgSha256']
    assert sha((ROOT / record['rasterSource']).read_bytes()) == record['rasterSha256']
    print(f"PASS: selected aqueduct A; {len(before['protected'])} protected files and all 28 prior Dali entries unchanged; 29 painted / 5 named; unchanged 14-tile hand.", flush=True)

def main():
    manifest = ROOT / 'web/public/tiles/dali/manifest.json'
    before = json.loads(manifest.read_text())
    assert len(before['tiles']) == 28 and len(before['placeholders']) == 6
    assert not any(e['tile'] == '1m' for e in before['tiles']), 'Stop rather than replace existing art'
    tracked = run('git', 'ls-files', '-z').split('\0')
    protected = {p: sha((ROOT / p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in ('.png', '.svg', '.avif', '.webp', '.jpg', '.jpeg', '.gif')}
    for p in ('web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/matisse/manifest.json', 'web/src/lib/van-gogh-faces.js', 'web/src/lib/matisse-faces.js'):
        protected[p] = sha((ROOT / p).read_bytes())
    preview_path = ROOT / 'web/public/tiles/dali/preview.html'
    hand = re.search(r'<div class="hand">(.*?)</div>', preview_path.read_text(), re.S).group(1)
    assert hand.count('<img ') == 14
    (T / 'baseline.json').write_text(json.dumps({'protected': protected, 'manifest': before, 'hand': hand}))
    parts = []
    for item in json.loads((T / 'parts.json').read_text()):
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], item['path']
        parts.append(data)
    raster = b''.join(parts)
    record = json.loads((T / 'one-character.json').read_text())
    assert len(raster) == 41959 and sha(raster) == record['rasterSha256'] == '7aba9e73d7754bed3274235ebc262d0b73579aa21c9a0acdabef40dfca82077f'
    assert raster[4:12] == b'ftypavif'
    i = raster.index(b'ispe')
    assert [int.from_bytes(raster[i+8:i+12], 'big'), int.from_bytes(raster[i+12:i+16], 'big')] == [454, 804]
    metadata = {k: v for k, v in record.items() if k not in ('sourceSvgSha256', 'runtimeSvgSha256')}
    text = json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))
    assert not any(c in text for c in '<>&')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">1 character — Dali — The Aqueduct That Flows Upward — A</title><metadata>{text}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,{base64.b64encode(raster).decode()}"/></svg>\n').encode()
    assert sha(svg) == record['sourceSvgSha256'] == 'a33eb1b7f79fe603f59d363b93650428f1b10b0ff159bbbabdf89367261c03a3'
    p = ROOT / 'web/scripts/export-dali-tiles.mjs'
    exporter = p.read_text()
    anchor = "  ['Man5', '5m', '5 characters', 'Theatre of the Impossible — A', 'five-characters-a-approved.png', [0, 0, 1086, 1448]],"
    addition = "  ['Man1', '1m', '1 character', 'The Aqueduct That Flows Upward — A', 'one-character-a-aqueduct-approved.svg', [0, 0, 454, 804]],"
    exporter = once(exporter, anchor, addition + '\n' + anchor)
    exporter = once(exporter, 'const fits = {', "const fits = {\n  // Preserve the full tall source; keep the bridge, red stone glyph and water in view.\n  '1m': { painting: [454, 804], stretch: 0.02, anchor: 2 / 3, bleedPixels: 0 },")
    exporter = once(exporter, '${manifest.tiles.length} approved faces. ', '${manifest.tiles.length} approved faces. One character uses A — The Aqueduct That Flows Upward: the selected ivory aqueduct above the red-stone character and water connection. The full 454 × 804 panel remains in the source, without resizing, repainting or recolouring. Its 3:4 game fit crops surrounding sky and foreground with 2% horizontal stretch and no side bleed. ')
    changes = {p: exporter}
    for p in sorted((ROOT / 'web/tests').glob('dali-*.test.js')) + [ROOT / 'web/tests/tile-faces.test.js']:
        old = p.read_text()
        text = old
        for a, b in (
            ('set.tiles.length, 28', 'set.tiles.length, 29'),
            ('set.placeholders.length, 6', 'set.placeholders.length, 5'),
            ('28 approved faces', '29 approved faces'),
            ('remaining 6 tiles', 'remaining 5 tiles'),
            ('twenty-eight approved identities and six placeholders', 'twenty-nine approved identities and five placeholders'),
            ('twenty-eight approved images', 'twenty-nine approved images'),
            ('twenty-eight-face preview', 'twenty-nine-face preview'),
        ):
            text = text.replace(a, b)
        if p.name in ('dali-bamboo.test.js', 'tile-faces.test.js'):
            text = once(text, "'9s', '5m'", "'9s', '1m', '5m'")
        if p.name == 'tile-faces.test.js':
            text = once(text, "['2p', '4p', '6p', '7p', '8p', '9p', '3s', '4s', '5z', '6z'].includes(tile)", "['1m', '2p', '4p', '6p', '7p', '8p', '9p', '3s', '4s', '5z', '6z'].includes(tile)")
        if text != old:
            changes[p] = text
    p = ROOT / 'docs/design/dali/README.md'
    docs = p.read_text()
    docs = once(docs, "Twenty-eight faces are approved, drawn from the original studies and Carl's later selections. The latest is green dragon", "Twenty-nine faces are approved, drawn from the original studies and Carl's later selections. The latest is one character **The Aqueduct That Flows Upward — A**, the selected left-hand painting with an ivory aqueduct, red-stone character and water connection. The previously selected green dragon")
    docs = once(docs, '| `Man5.svg` |', '| `Man1.svg` | 1 character | The Aqueduct That Flows Upward — A: a single ivory aqueduct above a monumental red-stone character, with a river-shaped shadow and water joining a small pool |\n| `Man5.svg` |')
    docs = once(docs, 'The other 6 tiles have no approved artwork yet', 'The other 5 tiles have no approved artwork yet')
    details = 'One character uses [selected A — The Aqueduct That Flows Upward](studies/one-character-a-aqueduct-approved.svg), extracted from the left panel of `dalí_inspired_mahjong_three_surreal_horizons.png` at (24, 190), 454 × 804 pixels. The full artwork-only panel is retained at these dimensions in quality-50 AVIF, without resizing, repainting or recolouring. The existing face-fit helper crops surrounding sky and foreground, using 2% horizontal stretch, anchor 2/3 and no side bleed to keep the aqueduct, red-stone character and water connection visible. Source and runtime differ only in image placement. Compression is lossy; the untouched PNG board and lossless panel are preserved separately in `dali-one-character-A-originals.zip` supplied in chat. See [provenance](one-character.json). The zipper B and brush C are not used. All previous artwork and the fourteen-tile hand remain unchanged. Reproduce only this tile with `node web/scripts/export-dali-tiles.mjs --only=1m`.\n\n'
    docs = once(docs, '## Presentation\n\n', '## Presentation\n\n' + details)
    changes[p] = docs
    changes[ROOT / 'web/tests/dali-one-character.test.js'] = (T / 'dali-one-character.test.js').read_text()
    for p, data in ((ROOT / record['source'], svg), (ROOT / record['rasterSource'], raster), (ROOT / 'docs/design/dali/one-character.json', (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode())):
        assert not p.exists(), f'Refuse to overwrite source {p}'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for p, text in changes.items():
        p.write_text(text, encoding='utf-8')
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=1m')
    verify()
    generated = {p: p.read_bytes() for p in (manifest, preview_path, ROOT / 'web/src/lib/dali-faces.js', ROOT / record['runtime'])}
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=1m')
    for p, data in generated.items():
        assert p.read_bytes() == data, f'Non-deterministic export: {p}'
    verify()

if __name__ == '__main__':
    verify() if '--verify' in sys.argv else main()
