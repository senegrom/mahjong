"""Branch-only import of Carl's approved smaller-Wan poppy painting."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path('.')
work = root / '.art-import-poppies'
hash_bytes = lambda data: hashlib.sha256(data).hexdigest()
record = json.loads((work / 'provenance.json').read_text())
manifest_path = root / 'web/public/tiles/van-gogh/manifest.json'
previous = json.loads(manifest_path.read_text())
assert not any(t['tile'] == '9m' for t in previous['tiles']), '9m is already painted; stop rather than replace it'
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
images = {p: hash_bytes(Path(p).read_bytes()) for p in tracked if p and (p.startswith('docs/design/') or p.startswith('web/public/tiles/')) and Path(p).suffix.lower() in {'.svg', '.png', '.webp', '.avif', '.jpg', '.jpeg'}}
raster = b''.join((work / f'chunk{i}.bin').read_bytes() for i in range(4))
assert hash_bytes(raster) == record['rasterSha256']
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Nine characters — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert hash_bytes(svg) == record['sourceSha256']
source = root / record['source']
source.parent.mkdir(parents=True, exist_ok=True)
source.write_bytes(svg)
shutil.copyfile(work / 'provenance.json', root / 'docs/design/van-gogh/nine-characters-poppy-field.json')
shutil.copyfile(work / 'nine-characters.test.js', root / 'web/tests/van-gogh-nine-characters.test.js')

def once(text, old, new):
    assert text.count(old) == 1, f'Expected one patch anchor: {old!r}'
    return text.replace(old, new, 1)

def pattern_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text, flags=re.M | re.S)
    assert count == 1, f'Expected one pattern: {pattern}'
    return result

exporter = root / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
s = once(s, 'const fits = {\n', "const fits = {\n  // The revised poppy portrait is wider than 3:4; keep the glyphs' exact proportions.\n  '9m': { painting: [1122, 1402], stretch: 0, anchor: 0.5, bleedPixels: 0 },\n")
s = once(s, 'const sources = {\n', "const sources = {\n  ninePoppyField: '" + record['source'] + "',\n")
s = pattern_once(s, r'(const definitions = \[\n)(.*?)(^\];)', lambda m: m[1] + m[2] + "  ['Poppies B (smaller Wan)', '9m', 'Nine characters', 'ninePoppyField', [0, 0, 300, 400]],\n" + m[3])
s = pattern_once(s, r"(^  approval: ')(.*)(',\n  fallback:)", lambda m: m[1] + m[2] + ' ' + record['approval'] + m[3])
s = once(s, r'/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/', r'/href="data:(image\/(?:webp|avif));base64,([A-Za-z0-9+/=]+)"/')
s = once(s, 'Missing embedded WebP in ${source}', 'Missing embedded WebP or AVIF in ${source}')
s = once(s, "const raster = Buffer.from(match[1], 'base64');", "const raster = Buffer.from(match[2], 'base64');")
s = once(s, "rasterMimeType: 'image/webp', rasterSha256: hash(raster)", "rasterMimeType: match[1], rasterSha256: hash(raster)")
s = once(s, '. Includes Seven Apples (7p)', '. Includes Poppy Field with smaller Wan (9m), Seven Apples (7p)')
s = once(s, 'const featured = [', "const featured = ['9m', ")
s = once(s, 'approved faces, including Seven Apples', 'approved faces, including Poppy Field with smaller 萬 for 9 characters, Seven Apples')
exporter.write_text(s)

tests = root / 'web/tests/tile-faces.test.js'
s = tests.read_text()
s = pattern_once(s, r"(test\('Van Gogh preserves[\s\S]*?const approved = \[)([^\]]*)(\])", lambda m: m[1] + m[2] + ", '9m'" + m[3])
s = pattern_once(s, r'(assert\.deepEqual\(set\.tiles\.map\(tile => tile\.candidate\), \[)([^\]]*)(\])', lambda m: m[1] + m[2] + ", 'Poppies B (smaller Wan)'" + m[3])
s = once(s, "assert.equal(entry.rasterMimeType, 'image/webp');", "assert.equal(entry.rasterMimeType, tile === '9m' ? 'image/avif' : 'image/webp');\n      assert.ok(svg.includes(`data:${entry.rasterMimeType};base64,`));")
s = once(s, r'/href="data:image\/webp;base64,([A-Za-z0-9+/=]+)"/', r'/href="data:image\/(?:webp|avif);base64,([A-Za-z0-9+/=]+)"/')
tests.write_text(s)

subprocess.run(['node', str(exporter), '--only=9m'], check=True)
current = json.loads(manifest_path.read_text())
entry = next(t for t in current['tiles'] if t['tile'] == '9m')
assert entry['svgSha256'] == record['svgSha256']
assert hash_bytes(Path(record['runtime']).read_bytes()) == record['svgSha256']
assert [t for t in current['tiles'] if t['tile'] != '9m'] == previous['tiles']
assert current['remaining'] == [t for t in previous['remaining'] if t != '9m']
assert current['superseded'] == previous['superseded']
assert current['rejected'] == previous['rejected']
assert [t for t in current['sources'] if t['source'] != record['source']] == previous['sources']
outputs = [manifest_path, Path(record['runtime']), root / 'web/public/tiles/van-gogh/preview.html', root / 'web/src/lib/van-gogh-faces.js']
first = {str(p): p.read_bytes() for p in outputs}
subprocess.run(['node', str(exporter), '--only=9m'], check=True)
assert all(Path(p).read_bytes() == data for p, data in first.items())
assert all(Path(p).exists() and hash_bytes(Path(p).read_bytes()) == sha for p, sha in images.items())
painted, remaining = len(current['tiles']), len(current['remaining'])
notes = f'''\n\n## Poppy Field — 9 characters, smaller 萬\n\nCarl selected the middle poppy-field idea, requested a smaller characters sign to sit more organically among the flowers, and approved the resulting revised image as `9m` / `Man9` (九萬). The earlier large-sign version, wheat and harbour are not used. The approved floral nine and smaller golden 萬 are retained without repainting or recolouring.\n\nThe source SVG embeds the entire approved 1122 × 1402 portrait in a 300 × 400 AVIF (quality 50) game export. The existing face-fit helper restores the original proportions with zero stretch, clipping a small centered strip from each side to fill the 3:4 face. Source and runtime differ only in image placement. This is a lossy game-sized export; the untouched PNG is preserved in `Van_Gogh_9_Characters_Poppy_Field_Approved.zip` supplied in chat. Original, raster, source and runtime hashes and the fit are recorded in `docs/design/van-gogh/nine-characters-poppy-field.json`.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=9m` to reproduce this face without rewriting other artwork. The set now has **{painted} painted faces and {remaining} unpainted identities showing their names**. Characters 2–9 are covered; only 1 characters remains in that suit. The fourteen-tile hand retains the previous thirteen featured paintings.\n'''
for relative in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = root / relative
    s = p.read_text()
    s = re.sub(r'\*\*Current set:\*\*[^\n]*', f'**Current set:** {painted} painted faces and {remaining} unpainted identities showing their names. Poppy Field with smaller 萬 is active for 9 characters; characters 2–9 and all nine bamboo tiles are painted. Earlier sections retain historical counts.', s, count=1)
    s = re.sub(r'(Choose \*\*Options → Tile face → Van Gogh\*\*\. The )[^.\n]+?( approved faces)', lambda m: m[1] + str(painted) + m[2], s, count=1)
    s = re.sub(r'The other \d+ identities awaiting Van Gogh artwork[^.]*\.', f'The other {remaining} identities awaiting Van Gogh artwork display their names.', s, count=1)
    p.write_text(s.rstrip() + notes)
print(f'PRESERVED {len(images)} existing source/runtime artwork files and {len(previous["tiles"])} prior manifest entries. Repeated export is deterministic.')
