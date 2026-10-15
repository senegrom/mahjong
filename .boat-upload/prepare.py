from pathlib import Path
import hashlib, base64, json, re, shutil, subprocess

ROOT = Path.cwd()
STUDY = 'docs/design/van-gogh/studies/26-one-characters-red-sail-c-approved.svg'
PROVENANCE = 'docs/design/van-gogh/one-characters-red-sail.json'
RASTER_HASH = '0a6b2b96aaebe241576e94d6cb1d80acf5fc862a6b7a33f8a5323b2c4bbbcc70'
ORIGINAL_HASH = '81570fc132f12f48fceb262ecc0f7e6a4f1e9109198d38281a149dea0a8e4d2a'
sha = lambda b: hashlib.sha256(b).hexdigest()
def replace_once(text, old, new):
    assert text.count(old) == 1, ('Expected exactly one anchor', old, text.count(old))
    return text.replace(old, new, 1)

tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
protected = {p: sha(Path(p).read_bytes()) for p in tracked if p and p.endswith(('.png', '.svg', '.webp', '.avif', '.jpg'))}
manifest_path = Path('web/public/tiles/van-gogh/manifest.json')
before = json.loads(manifest_path.read_text())
assert '1m' in before['remaining'] and not any(t['tile'] == '1m' for t in before['tiles'])
assert len(before['tiles']) == 31 and before['remaining'] == ['1m', '2z', '3z']
assert not Path(STUDY).exists() and not Path('web/public/tiles/van-gogh/approved/Man1.svg').exists()
raster = b''.join(Path(f'.boat-upload/part-{i}').read_bytes() for i in range(1, 4))
assert len(raster) == 36049 and sha(raster) == RASTER_HASH
svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title">'
       '<title id="title">One character — Van Gogh — The Red Sail</title>'
       '<defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs>'
       '<image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,'
       + base64.b64encode(raster).decode() + '"/></svg>\n')
Path(STUDY).write_text(svg)
provenance = {
  'tile': '1m', 'name': 'Man1', 'label': 'One character', 'candidate': 'Red Sail C',
  'title': 'The Red Sail', 'approvalDate': '2026-10-15',
  'approval': 'Carl selected the third option, the red-sail boat, and explicitly requested GitHub deployment as 1 characters (1m / 一萬). The sower and window alternatives are not selected.',
  'source': STUDY, 'sourceSha256': sha(svg.encode()),
  'originalFilename': 'crimson_sail_in_a_mediterranean_harbour.png',
  'originalDimensions': [1086, 1448], 'originalSha256': ORIGINAL_HASH,
  'originalBackup': 'Van_Gogh_1_Characters_Red_Sail_Approved.zip supplied in chat; the untouched PNG is not the game-sized source SVG.',
  'processing': 'Use the complete 1086 x 1448 RGB canvas (exactly 3:4). Uniformly downsample with Pillow LANCZOS to 300 x 400 and encode AVIF at quality 60, speed 6. No crop, stretch, repainting, recolouring, lettering changes or additions. The SVG uses the existing rounded-corner clip with zero image bleed, preserving the full composition apart from normal corner clipping.',
  'rasterDimensions': [300, 400], 'rasterMimeType': 'image/avif', 'rasterBytes': len(raster), 'rasterSha256': RASTER_HASH,
  'fit': {'painting': [1086, 1448], 'stretch': 0, 'anchor': 0.5, 'bleedPixels': 0},
  'runtimeSvg': 'web/public/tiles/van-gogh/approved/Man1.svg',
  'runtimeSvgSha256': '85a214007ce059ed61aa47f2257db9ce2c6499ed58c95954d926d88fd2dbcfea',
}
Path(PROVENANCE).write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
export = Path('web/scripts/export-van-gogh-tiles.mjs')
s = export.read_text()
s = replace_once(s, 'const fits = {', "const fits = {\n  // The approved boat is exactly 3:4: keep the complete painting with no stretch or bleed.\n  '1m': { painting: [1086, 1448], stretch: 0, anchor: 0.5, bleedPixels: 0 },")
s = replace_once(s, 'const sources = {', "const sources = {\n  oneRedSail: '" + STUDY + "',")
s = replace_once(s, '\n];\nconst onlyArgument', "\n  ['Red Sail C', '1m', 'One character', 'oneRedSail', [0, 0, 300, 400]],\n];\nconst onlyArgument")
lines = s.splitlines(keepends=True)
approvals = [i for i, line in enumerate(lines) if line.startswith("  approval: '")]
assert len(approvals) == 1
idx = approvals[0]
assert lines[idx].endswith("',\n")
lines[idx] = lines[idx][:-3] + " Carl selected C, The Red Sail, for 1 characters (1m / 一萬) and explicitly requested GitHub deployment. Its small hull lettering and complete 3:4 boat painting are preserved without stretching, cropping, repainting or recolouring. The sower and open-window alternatives are not selected. This completes the characters suit.',\n"
s = ''.join(lines)
s = replace_once(s, 'Includes Green Dragon at the Vineyard C', 'Includes The Red Sail C (1m), completing the characters suit, Green Dragon at the Vineyard C')
s = replace_once(s, 'approved faces, including Green Dragon at the Vineyard C', 'approved faces, including The Red Sail for 1 character, completing the characters suit, Green Dragon at the Vineyard C')
featured = next(line for line in s.splitlines() if line.startswith('const featured = '))
assert "'4m', " in featured
s = replace_once(s, featured, featured.replace("['9m'", "['1m', '9m'").replace("'4m', ", ''))
s = replace_once(s, '<p class="notes">Earlier selected art', '<p class="notes">The Red Sail (1 character) retains the complete approved 1086 × 1448 composition in a 300 × 400 quality-60 AVIF game export, with zero stretch and zero bleed; the untouched original PNG is backed up separately. Earlier selected art')
export.write_text(s)
tests = Path('web/tests/tile-faces.test.js')
s = tests.read_text()
s = replace_once(s, "'7p', '9m', '8p', '6z'];", "'7p', '9m', '8p', '6z', '1m'];")
s = replace_once(s, "'Eight Lemons', 'Vineyard Dragon C']);", "'Eight Lemons', 'Vineyard Dragon C', 'Red Sail C']);")
s = replace_once(s, "['9m', '6z'].includes(tile)", "['9m', '6z', '1m'].includes(tile)")
tests.write_text(s)
shutil.copyfile('.boat-upload/red-sail.test.js', 'web/tests/van-gogh-one-red-sail.test.js')
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=1m'], check=True)
after = json.loads(manifest_path.read_text())
assert len(after['tiles']) == 32 and after['remaining'] == ['2z', '3z']
assert [t for t in after['tiles'] if t['tile'] != '1m'] == before['tiles']
assert [t for t in after['sources'] if t['id'] != 'oneRedSail'] == before['sources']
assert Path('web/public/tiles/van-gogh/approved/Man1.svg').read_bytes() == svg.replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"').encode()
for p, h in protected.items():
    assert sha(Path(p).read_bytes()) == h, ('Existing artwork changed', p)

for file in ['docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md']:
    p = Path(file)
    text = p.read_text()
    if file.startswith('docs/'):
        old = next(line for line in text.splitlines() if line.startswith('**Current set:**'))
        new = '**Current set:** 32 painted faces and 2 unpainted identities showing their names: South and West winds. The Red Sail C is active for 1 character (1m / 一萬), completing the characters suit. All three numbered suits are now complete. Green Dragon at the Vineyard C and all prior artwork are unchanged. Earlier sections retain historical counts.'
        text = replace_once(text, old, new)
    else:
        text = replace_once(text, 'The 31 approved faces appear', 'The 32 approved faces appear')
        text = replace_once(text, '**Green Dragon at the Vineyard C** is active', '**The Red Sail C** is active for 1 character (1m / 一萬), completing the characters suit. All three numbered suits are complete. **Green Dragon at the Vineyard C** is active')
        text = replace_once(text, 'The other 3 identities awaiting Van Gogh artwork display their names.', 'The other 2 identities awaiting Van Gogh artwork, South and West winds, display their names.')
    text += '\n\n## The Red Sail C — 1 character\n\nCarl selected the third boat painting, preserving the small 一萬 on its cream hull, and requested GitHub deployment. The complete 1086 × 1448 portrait is exactly 3:4; it is uniformly downsampled to a 300 × 400 quality-60 AVIF game export with no crop, stretching, repainting or recolouring. The source SVG embeds the approved game raster in the shared study box. The runtime changes only that image box to zero bleed, with no stretching and the normal 26-unit rounded-corner clip. The game export is not the full-resolution PNG. The untouched original is supplied in `Van_Gogh_1_Characters_Red_Sail_Approved.zip`; its hash and processing details are recorded in `docs/design/van-gogh/one-characters-red-sail.json`.\n\nRegenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=1m`. Regression tests pin the source and embedded-raster hashes, check the full-frame geometry, verify registration and hidden-tile privacy, and ensure selective exports are repeatable without changing other artwork. All 31 previous faces remain byte-for-byte unchanged.\n'
    p.write_text(text)
print(json.dumps({'status': 'prepared', 'approved': len(after['tiles']), 'remaining': after['remaining'], 'protectedArtFiles': len(protected), 'sourceSha256': sha(svg.encode()), 'rasterSha256': RASTER_HASH}, indent=2))
