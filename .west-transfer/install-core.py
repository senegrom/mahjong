"""Install only the approved West painting and completion-aware registration."""
from pathlib import Path
import base64, hashlib, json, re, shutil, subprocess

root = Path.cwd()
transfer = root / '.west-transfer'
sha = lambda b: hashlib.sha256(b).hexdigest()
prov = json.loads((transfer / 'provenance.json').read_text())
parts = [transfer / f'part-{i}.bin' for i in range(5)]
expected = ['c278ddf7b5c7433eb520d83be5d5711ecfbbd066f65ba90383d063e4b690b71e', '7ffbe38d193367804db75bf3d68c336f06a7ffff173de73f82794202bae9f7a2', '08ce5d24eb075ecdb17969373d2f538e733edda643b768470c2b073e5ae2c63a', 'a0a1331500386e1b3d09c0d4c352e4e922ee457c425180281f98c0c3f8350ead', '984a5c5d867cef99833fb3fe453d1c64f96f35a4007dad5823d1411e878b1542']
for p, digest in zip(parts, expected):
    assert sha(p.read_bytes()) == digest, p
raster = b''.join(p.read_bytes() for p in parts)
assert sha(raster) == prov['rasterSha256']
manifest_path = root / 'web/public/tiles/van-gogh/manifest.json'
before = json.loads(manifest_path.read_text())
assert before['remaining'] == ['3z'], before['remaining']
assert len(before['tiles']) == 33
tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
art = {p: sha((root / p).read_bytes()) for p in tracked if p.startswith(('docs/design/', 'web/public/tiles/')) and Path(p).suffix.lower() in {'.png', '.svg', '.webp', '.avif', '.jpg', '.jpeg'}}
preview_path = root / 'web/public/tiles/van-gogh/preview.html'
old_hand = re.search(r'<div class="rack" id="rack">(.*?)</div>', preview_path.read_text(), re.S).group(1)
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">West wind — Van Gogh</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="0" y="0" width="300" height="400" preserveAspectRatio="xMidYMid meet" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n'
assert sha(svg.encode()) == prov['svgSha256']
for name in [prov['source'], prov['runtime']]:
    target = root / name
    assert not target.exists(), name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg)
(root / 'docs/design/van-gogh/west-wind-clouds.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2) + '\n')

def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)

exporter = root / 'web/scripts/export-van-gogh-tiles.mjs'
s = exporter.read_text()
s = replace_once(s, 'const sources = {\n', "const sources = {\n  westClouds: '" + prov['source'] + "',\n")
s = replace_once(s, '];\nconst onlyArgument', "  ['West B (cloud strokes)', '3z', 'West wind', 'westClouds', [0, 0, 300, 400]],\n];\nconst onlyArgument")
s = replace_once(s, "',\n  fallback: 'text',", " Carl approved the latest standalone lighthouse painting stormy_sunset_lighthouse_seascape.png for West wind (3z / Shaa), with cloud-shaped 西. Its complete 1086 x 1448 composition is uniformly downsampled, without cropping, stretching, repainting or recolouring during deployment. This completes all 34 Van Gogh identities.',\n  fallback: 'text',")
s = replace_once(s, 'Includes Snowbound North B', 'Includes Cloudborne West (3z), completing all 34 faces, Snowbound North B')
s = replace_once(s, 'approved faces, including ', 'approved faces, including Cloudborne West above the sunset lighthouse for West wind, ')
exporter.write_text(s)

p = root / 'web/tests/tile-faces.test.js'
s = p.read_text()
s = replace_once(s, "'6z', '1m', '2z'];", "'6z', '1m', '2z', '3z'];")
s = replace_once(s, "'Red Sail C', 'South A']);", "'Red Sail C', 'South A', 'West B (cloud strokes)']);")
s = replace_once(s, "['9m', '6z', '1m', '2z', '4z'].includes(tile)", "['9m', '6z', '1m', '2z', '4z', '3z'].includes(tile)")
needle = next(line for line in s.splitlines() if "assert.match(preview('van-gogh'), new RegExp(`remaining" in line)
s = replace_once(s, needle, "  if (vanGogh.remaining.length) {\n  " + needle + "\n  } else {\n    assert.match(preview('van-gogh'), /Every tile type now has approved artwork\\./);\n    assert.doesNotMatch(preview('van-gogh'), /remaining 0 tiles/);\n  }")
s = replace_once(s, '      assert.ok(unpainted.length, face);', "      if (face === 'van-gogh') assert.deepEqual(unpainted, [], 'the Van Gogh set is complete');\n      else assert.ok(unpainted.length, face);")
p.write_text(s)
shutil.copyfile(transfer / 'west.test.js', root / 'web/tests/van-gogh-west-wind.test.js')

note = '\n\n## Cloudborne West — final tile\n\nCarl approved the latest standalone `stormy_sunset_lighthouse_seascape.png` for West wind (`3z` / `Shaa`): a billowing cloud-shaped 西 above a lighthouse, sunset and breaking waves. The earlier re-uploaded JPEG and alternative concept sheets are not used.\n\nThe entire **1086 × 1448** approved PNG is exactly **3:4**. It is uniformly Lanczos-downsampled to **300 × 400**, encoded as quality-65, 4:4:4 AVIF, and embedded in the shared radius-26 SVG. There is **zero crop, zero stretch and zero bleed**; only the rounded corners are clipped. No image regeneration, repainting or recolouring occurs during deployment. Source and runtime SVGs are byte-identical.\n\n`west-wind-clouds.json` records the original and export checksums and processing. The untouched original and game export are preserved in `Van_Gogh_West_Wind_Clouds_Approved.zip`, supplied in chat; the repository source SVG embeds the game-sized image. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=3z`.\n\n**All 34 tile identities now have approved Van Gogh artwork.** All 33 earlier paintings, their proportions and the fourteen-tile example hand remain unchanged.\n'
p = root / 'docs/design/van-gogh/README.md'
s = p.read_text()
s, n = re.subn(r'\*\*Current set:\*\*[^\n]*', '**Current set:** All 34 tile identities are painted. Cloudborne West completes the four winds and the entire Van Gogh set. Snowbound North B, The Billowing Courtyard A and every earlier selection remain unchanged. Earlier sections retain historical counts.', s, count=1)
assert n == 1
p.write_text(s.rstrip() + note)
p = root / 'web/public/tiles/van-gogh/README.md'
s = p.read_text()
s = replace_once(s, 'The 33 approved faces', 'The 34 approved faces')
s = replace_once(s, 'West wind is the only identity awaiting Van Gogh artwork and displays its name.', '**Cloudborne West** is active for West wind (3z / Shaa), completing all 34 identities; no tile requires an unpainted fallback.')
s = replace_once(s, '**Wind Ribbons (North B)**', '**Snowbound North (Winter B)**')
p.write_text(s.rstrip() + note.replace('`west-wind-clouds.json`', '`docs/design/van-gogh/west-wind-clouds.json`'))

subprocess.run(['node', str(exporter), '--only=3z'], check=True)
after = json.loads(manifest_path.read_text())
assert len(after['tiles']) == 34 and after['remaining'] == []
assert after['tiles'][:-1] == before['tiles'], 'an existing tile entry changed'
for old in before['sources']:
    assert old in after['sources'], old
for key in ['rejected','superseded','canvas','facePresentation','fallback']:
    assert after[key] == before[key], key
for p, digest in art.items():
    assert sha((root / p).read_bytes()) == digest, p
assert re.search(r'<div class="rack" id="rack">(.*?)</div>', preview_path.read_text(), re.S).group(1) == old_hand
outputs = [manifest_path, exporter, root/'web/src/lib/van-gogh-faces.js', preview_path, root/prov['runtime']]
checks = {p:sha(p.read_bytes()) for p in outputs}
subprocess.run(['node', str(exporter), '--only=3z'], check=True)
assert all(sha(p.read_bytes()) == digest for p,digest in checks.items())
print(f'PRESERVED {len(art)} existing artwork files and all 33 prior tile entries. Full set: 34/34. Export reproducible. Original PNG SHA256: {prov["originalSha256"]}')
