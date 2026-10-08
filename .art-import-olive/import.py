"""Branch-only import of Carl's approved Starry Olive Grove eight characters."""
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path.cwd()
transport = root / '.art-import-olive'
manifest_path = root / 'web/public/tiles/van-gogh/manifest.json'
old = json.loads(manifest_path.read_text())
assert '8m' in old['remaining'] and not any(t['tile'] == '8m' for t in old['tiles'])
hash_bytes = lambda data: hashlib.sha256(data).hexdigest()
before = {}
for directory in ['docs/design', 'web/public/tiles']:
    for file in (root / directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in {'.png', '.svg', '.webp', '.avif', '.jpg', '.jpeg'}:
            before[file.relative_to(root).as_posix()] = hash_bytes(file.read_bytes())
record = json.loads((transport / 'provenance.json').read_text())
raster = b''.join((transport / f'chunk{i}.bin').read_bytes() for i in range(4))
assert len(raster) == record['rasterBytes']
assert hash_bytes(raster) == record['rasterSha256']
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Eight characters — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/webp;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n').encode()
assert hash_bytes(svg) == record['svgSha256']
source = root / record['source']
assert not source.exists() and not (root / record['runtime']).exists()
source.write_bytes(svg)
(root / 'docs/design/van-gogh/eight-characters-olive-grove.json').write_bytes((transport / 'provenance.json').read_bytes())
(root / 'web/tests/van-gogh-eight-characters.test.js').write_bytes((transport / 'eight-characters.test.js').read_bytes())

def replace_once(text, old_text, new_text):
    if text.count(old_text) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old_text!r}')
    return text.replace(old_text, new_text, 1)

exporter = root / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
s = replace_once(s, '};\nconst definitions = [', "  eightOliveGrove: '" + record['source'] + "',\n};\nconst definitions = [")
s = replace_once(s, '];\nconst onlyArgument', "  ['Starry Olive Grove', '8m', 'Eight characters', 'eightOliveGrove', [0, 0, 300, 400]],\n];\nconst onlyArgument")
lines = s.splitlines(keepends=True)
indices = [i for i, line in enumerate(lines) if line.startswith('  approval: ')]
assert len(indices) == 1
idx = indices[0]
assert lines[idx].endswith("',\n")
lines[idx] = lines[idx][:-3] + " Carl approved the generated Starry Olive Grove painting as 8 characters (8m / 八萬) and explicitly requested GitHub deployment. The complete olive-branch composition, colours and proportions are preserved; the older seven-characters olive study is not used.',\n"
s = ''.join(lines)
s = replace_once(s, 'Includes Four Oranges A (4p),', 'Includes Starry Olive Grove (8m), Four Oranges A (4p),')
s, count = re.subn(r'^const featured = .*;$', "const featured = ['8m', '4p', '7m', '6m', '5m', '2m', '3m', '4m', '9p', '3p', '9s', '6s'];", s, flags=re.M)
assert count == 1
s = replace_once(s, 'approved faces, including ', 'approved faces, including Starry Olive Grove for 8 characters, ')
exporter.write_text(s)

shared = root / 'web/tests/tile-faces.test.js'
s = shared.read_text()
s = replace_once(s, "'7m', '4p'];", "'7m', '4p', '8m'];")
s = replace_once(s, "'Irises C', 'Oranges A']);", "'Irises C', 'Oranges A', 'Starry Olive Grove']);")
s = replace_once(s, "'7m', '4p']) {", "'7m', '4p', '8m']) {")
shared.write_text(s)

note = '\n## Starry Olive Grove — 8 characters (八萬)\n\nCarl approved the olive-branch painting for `8m` / `Man8` and explicitly requested GitHub deployment. Two spreading olive branches form 八 above copper-red 萬 in a swirling blue-and-gold night landscape. This is the newly approved eight-characters painting, not the earlier olive seven study.\n\nThe entire 1086 × 1448 portrait is resized proportionally to a 300 × 400 quality-80 WebP using Lanczos, then embedded in the standard self-contained SVG with rounded clipping and 1% bleed. No repainting, recolouring or selective crop is applied. The source and runtime SVGs are identical; original, raster and SVG hashes are recorded in `docs/design/van-gogh/eight-characters-olive-grove.json`. The untouched full-resolution source PNG is preserved in `Van_Gogh_8_Characters_Starry_Olive_Grove_Approved.zip` supplied in chat. The game export is not the lossless original.\n\nRun `node web/scripts/export-van-gogh-tiles.mjs --only=8m` to copy this face exactly without rewriting any other artwork. This addition brings the set to **26 painted faces and 8 Classic fallbacks**, with characters **2–8** and all nine bamboo faces covered. All previous artwork, including Irises at Dusk C, Four Oranges A and The Potter’s Table C, remains unchanged.\n'
public_readme = root / 'web/public/tiles/van-gogh/README.md'
s = public_readme.read_text()
s = replace_once(s, 'The twenty-five approved faces appear', 'The twenty-six approved faces appear')
s = replace_once(s, 'The other 9 identities awaiting Van Gogh artwork', 'The other 8 identities awaiting Van Gogh artwork')
s = replace_once(s, 'plus **The Potter’s Table C** for 9 disks', 'plus **Starry Olive Grove** for 8 characters, **The Potter’s Table C** for 9 disks')
public_readme.write_text(s.rstrip() + '\n' + note)
design_readme = root / 'docs/design/van-gogh/README.md'
s = design_readme.read_text()
s = replace_once(s, '**Current set:** 25 painted faces and 9 Classic fallbacks;', '**Current set:** 26 painted faces and 8 Classic fallbacks;')
s = replace_once(s, 'Four Oranges A is active for 4 disks, Irises at Dusk C for 7 characters', 'Starry Olive Grove is active for 8 characters, Four Oranges A for 4 disks, Irises at Dusk C for 7 characters')
design_readme.write_text(s.rstrip() + '\n' + note)

command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=8m']
subprocess.run(command, check=True)
new = json.loads(manifest_path.read_text())
assert len(new['tiles']) == len(old['tiles']) + 1
assert new['remaining'] == [tile for tile in old['remaining'] if tile != '8m']
assert [t for t in new['tiles'] if t['tile'] != '8m'] == old['tiles']
assert new['sources'][:-1] == old['sources']
assert new['superseded'] == old['superseded'] and new['rejected'] == old['rejected']
assert (root / record['runtime']).read_bytes() == svg
for relative, checksum in before.items():
    assert hash_bytes((root / relative).read_bytes()) == checksum, relative
outputs = [record['runtime'], 'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js']
first = {f: (root / f).read_bytes() for f in outputs}
subprocess.run(command, check=True)
for f, data in first.items():
    assert (root / f).read_bytes() == data, f
print(f'PRESERVED {len(before)} existing source/runtime artwork files byte-for-byte.')
print('Verified exact approved raster/SVG, unchanged earlier manifest entries and repeatable --only=8m export.')
