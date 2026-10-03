from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
TRANSFER = ROOT / '.potters-table-transfer'
OUT = ROOT / 'web/public/tiles/van-gogh'
PROVENANCE = 'docs/design/van-gogh/nine-disks-potters-table-c.json'
EXPECTED_PARTS = [
    '632c19268195ffcfe9684bffdb5729450ed4ada8',
    'b3873f564960421f0f7f36c47e17c1498d37dc9c',
    '5fbec4159b02d928532693fa5b1fa94dafe7913d',
    '9bf3dfdc7979bc7d9c6eaf758c51c03feeae13cd',
    '1fb7fa7e8e8bb4f9fd0df8dd3a95f831bb57c6ec',
    '556715c9517a34b97eefcbc8240406054a0ef3f6',
    '8a4b39d0f9cdf78b77a094cc899170d8c8620540',
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Unexpected repository drift; review anchor: {old!r}')
    return text.replace(old, new, 1)


def put_new(relative, data):
    p = ROOT / relative
    if p.exists() or p.is_symlink():
        raise ValueError(f'Refusing to replace an existing source or archive: {relative}')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


metadata = json.loads((TRANSFER / 'provenance.json').read_text())
parts = []
for index, expected in enumerate(EXPECTED_PARTS):
    data = (TRANSFER / f'part{index:02d}.bin').read_bytes()
    # Recover one independently identified transport typo, before any artwork writes.
    if index == 3 and blob(data) == '334e3cbdbf222085b7d0d1303812ac7aa9644ea6':
        encoded = base64.b64encode(data).decode('ascii')
        if encoded.count('DndM5stJJ6oi8') != 1:
            raise ValueError('Unexpected transport correction input')
        data = base64.b64decode(encoded.replace('DndM5stJJ6oi8', 'DndM5stJ6oi8') + 'u', validate=True)
    if blob(data) != expected:
        raise ValueError(f'Chunk {index} checksum mismatch')
    parts.append(data)
raster = b''.join(parts)
if len(raster) != metadata['rasterBytes'] or sha(raster) != metadata['rasterSha256']:
    raise ValueError('The assembled image does not match the selected artwork')
if raster[:4] != b'RIFF' or raster[8:16] != b'WEBPVP8 ':
    raise ValueError('Expected a VP8 WebP source')
if [int.from_bytes(raster[26:28], 'little') & 0x3fff, int.from_bytes(raster[28:30], 'little') & 0x3fff] != [300, 400]:
    raise ValueError('Wrong source dimensions')
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Nine disks — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,' + base64.b64encode(raster).decode('ascii') + '"/></svg>\n').encode('utf-8')
if sha(svg) != metadata['svgSha256']:
    raise ValueError('Unexpected SVG wrapper')

before = json.loads((OUT / 'manifest.json').read_text())
old = next(entry for entry in before['tiles'] if entry['tile'] == '9p')
if old['candidate'] != 'I' or old['source'] != metadata['supersedes']['source']:
    raise ValueError('9 disks has changed since approval; refusing to overwrite a different selection')
old_png = (OUT / old['png']).read_bytes()
old_svg = (OUT / old['svg']).read_bytes()
if sha(old_png) != old['pngSha256']:
    raise ValueError('Existing Nine Stars PNG is not valid')
metadata['supersedes']['pngSha256'] = sha(old_png)
metadata['supersedes']['svgSha256'] = sha(old_svg)

excluded = {OUT / 'approved/Pin9.png', OUT / 'approved/Pin9.svg'}
artwork = {}
for parent in [ROOT / 'docs/design', ROOT / 'web/public/tiles']:
    for p in parent.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.png', '.svg', '.webp', '.jpg', '.jpeg'} and p not in excluded:
            artwork[p] = sha(p.read_bytes())

script_path = ROOT / 'web/scripts/export-van-gogh-tiles.mjs'
script = script_path.read_text()
script = once(script, '\n};\nconst definitions = [',
              "\n  pottersTable: '" + metadata['source'] + "',\n};\nconst definitions = [")
script = once(script,
              "  ['I', '9p', 'Nine disks', 'second', [838, 100, 380, 471]],",
              "  ['Disks C', '9p', 'Nine disks', 'pottersTable', [0, 0, 300, 400]],")
script = once(script, "',\n  fallback: 'classic',",
              " Carl selected C — The Potter’s Table to replace Nine Stars I for 9p and explicitly requested deployment. The nine patterned plates retain the selected painting; the old face is archived.',\n  fallback: 'classic',")
anchor = "    { candidate: 'J', tile: '6s', activeCandidate: 'Six B (green)', source: sources.second },"
script = once(script, anchor, anchor + "\n    { candidate: 'I', tile: '9p', activeCandidate: 'Disks C', source: sources.second, archivedPng: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png', archivedSvg: 'docs/design/van-gogh/superseded/nine-stars-Pin9.svg' },")
script = once(script, '// Generated by web/scripts/export-van-gogh-tiles.mjs. Includes ',
              '// Generated by web/scripts/export-van-gogh-tiles.mjs. Includes The Potter’s Table C (9p), replacing Nine Stars I, ')
script = once(script, "const featured = ['9s',", "const featured = ['9p', '9s',")
script = once(script, '${approved.length} approved faces, including ',
              '${approved.length} approved faces, including The Potter’s Table C for 9 disks, ')
script = once(script, 'Earlier selected art is preserved in lossless PNG crops.',
              'Earlier selected art is preserved in lossless PNG crops. The Potter’s Table C uses a documented quality-95 WebP crop; its former Nine Stars face is archived.')

tests_path = ROOT / 'web/tests/tile-faces.test.js'
tests = once(tests_path.read_text(), "'H', 'I', 'Six B (green)'", "'H', 'Disks C', 'Six B (green)'")
new_test_path = 'web/tests/van-gogh-nine-disks.test.js'
new_tests = (TRANSFER / 'nine-disks.test.js').read_bytes()
readmes = {}
section = f'''\n## The Potter’s Table C — 9 disks replacement\n\nCarl selected **C — The Potter’s Table** from the final artwork-only comparison and requested deployment. It replaces **I — Nine Stars** for `9p` / `Pin9`: nine blue-and-cream patterned plates in a three-by-three arrangement, with a wooden table, sunflower corners and swirling sky. This is the selected painting, not a regeneration.\n\nOnly the right-hand painting is cropped from the original composite at `[551, 56, 729, 1093]`; its inaccurate checklist and exterior margin are excluded. The complete selected crop is resized to a 300 × 400 quality-95 WebP and embedded in the standard SVG. This game export is not the lossless full-resolution original. Original, crop, embedded-raster and SVG checksums are recorded in `docs/design/van-gogh/nine-disks-potters-table-c.json`. The untouched original and lossless selected crop are preserved in the downloadable `van-gogh-nine-disks-potters-table-C-originals.zip` supplied in chat.\n\nThe original Nine Stars PNG and SVG are preserved byte-for-byte under `docs/design/van-gogh/superseded/nine-stars-Pin9.*`; the earlier study board is untouched. The stale Nine Stars PNG is removed from the playable directory after archiving. All other artwork is unchanged. This replacement retains **{len(before['tiles'])} approved faces and {len(before['remaining'])} Classic fallbacks**. Regenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=9p`.\n'''
for relative in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = ROOT / relative
    readmes[p] = p.read_text().rstrip() + '\n' + section

# Validate every new destination and patch anchor before writing anything.
new_paths = [metadata['source'], metadata['supersedes']['archivedPng'], metadata['supersedes']['archivedSvg'], PROVENANCE, new_test_path]
if any((ROOT / p).exists() or (ROOT / p).is_symlink() for p in new_paths):
    raise ValueError('An intended new source, archive or regression file already exists')
put_new(metadata['source'], svg)
put_new(metadata['supersedes']['archivedPng'], old_png)
put_new(metadata['supersedes']['archivedSvg'], old_svg)
put_new(PROVENANCE, (json.dumps(metadata, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
put_new(new_test_path, new_tests)
script_path.write_text(script, encoding='utf-8')
tests_path.write_text(tests, encoding='utf-8')
for p, text in readmes.items():
    p.write_text(text, encoding='utf-8')
subprocess.run(['node', '--check', str(script_path)], check=True)
command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=9p']
subprocess.run(command, check=True)
(OUT / 'approved/Pin9.png').unlink()

now = json.loads((OUT / 'manifest.json').read_text())
entry = next(e for e in now['tiles'] if e['tile'] == '9p')
if entry['candidate'] != 'Disks C' or entry['source'] != metadata['source'] or entry.get('png'):
    raise ValueError('Wrong runtime registration')
if (ROOT / metadata['runtime']).read_bytes() != svg or entry['svgSha256'] != sha(svg) or entry['rasterSha256'] != sha(raster):
    raise ValueError('Runtime differs from approved source')
if [e for e in before['tiles'] if e['tile'] != '9p'] != [e for e in now['tiles'] if e['tile'] != '9p']:
    raise ValueError('Unrelated manifest entry changed')
if before['remaining'] != now['remaining'] or len(before['tiles']) != len(now['tiles']):
    raise ValueError('A replacement must not change set coverage')
if now['sources'][:-1] != before['sources']:
    raise ValueError('Existing sources changed')
for p, expected in artwork.items():
    if not p.exists() or sha(p.read_bytes()) != expected:
        raise ValueError(f'Unrelated artwork changed: {p}')
if (ROOT / metadata['supersedes']['archivedPng']).read_bytes() != old_png or (ROOT / metadata['supersedes']['archivedSvg']).read_bytes() != old_svg:
    raise ValueError('Old Nine Stars archives changed')
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', (OUT / 'preview.html').read_text())
if not rack or rack.group(1).count('<img ') != 14 or 'approved/Pin9.svg' not in rack.group(1):
    raise ValueError('Preview must include the new face in a 14-tile hand')
outputs = {p: sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
outputs[ROOT / 'web/src/lib/van-gogh-faces.js'] = sha((ROOT / 'web/src/lib/van-gogh-faces.js').read_bytes())
subprocess.run(command, check=True)
if any(sha(p.read_bytes()) != expected for p, expected in outputs.items()) or (OUT / 'approved/Pin9.png').exists():
    raise ValueError('Isolated re-export is not deterministic')
print(f'PASS: exact approved C raster and SVG; {len(artwork)} unrelated artwork files unchanged; old Nine Stars preserved; {len(now["tiles"])} approved / {len(now["remaining"])} fallback; 14-tile preview; deterministic export.')
