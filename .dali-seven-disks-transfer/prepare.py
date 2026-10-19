"""One-time GitHub-only import of the approved seven-pearl shell. Fail closed."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
TRANSFER = ROOT / '.dali-seven-disks-transfer'
sha = lambda b: hashlib.sha256(b).hexdigest()
def blob(b):
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Patch anchor changed: {old!r}')
    return text.replace(old, new, 1)
def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout

def main():
    manifest_path = ROOT / 'web/public/tiles/dali/manifest.json'
    before = json.loads(manifest_path.read_text())
    assert len(before['tiles']) == 23 and len(before['placeholders']) == 11
    assert not any(e['tile'] == '7p' for e in before['tiles']), 'Already installed; stop rather than overwrite'
    tracked = run('git', 'ls-files', '-z').split('\0')
    protected = {p: sha((ROOT / p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in ('.png','.svg','.avif','.webp','.jpg','.jpeg','.gif')}
    other_sets = {p: (ROOT / p).read_bytes() for p in ('web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/matisse/manifest.json', 'web/src/lib/van-gogh-faces.js', 'web/src/lib/matisse-faces.js')}
    preview_path = ROOT / 'web/public/tiles/dali/preview.html'
    hand = re.search(r'<div class="hand">(.*?)</div>', preview_path.read_text(), re.S).group(1)
    assert hand.count('<img ') == 14

    chunks = []
    for part in json.loads((TRANSFER / 'parts.json').read_text()):
        data = (ROOT / part['path']).read_bytes()
        if part['path'].endswith('part-01') and blob(data) == '7118aecc943819c308c55972fc0157bb39c4f2fa':
            # Repair the known transport-only base64 typo; both the corrected
            # chunk and the assembled asset are checked against original hashes.
            encoded = base64.b64encode(data).decode()
            assert encoded.count('B9HdFd4j') == 1
            data = base64.b64decode(encoded.replace('B9HdFd4j', 'B9HdF4j') + '9', validate=True)
        assert len(data) == part['bytes'] and blob(data) == part['sha'], f'Wrong chunk: {part["path"]}'
        chunks.append(data)
    raster = b''.join(chunks)
    record = json.loads((TRANSFER / 'seven-disks.json').read_text())
    assert len(raster) == 112979 and sha(raster) == record['rasterSha256'] == '021e5788c3e11ed089ed79981498c97d575a60a04eb439427143e5b0b2fe3664'
    assert raster[4:12] == b'ftypavif'
    ispe = raster.index(b'ispe')
    assert [int.from_bytes(raster[ispe+8:ispe+12], 'big'), int.from_bytes(raster[ispe+12:ispe+16], 'big')] == [1086,1448]
    metadata = {k:v for k,v in record.items() if k != 'svgSha256'}
    text = json.dumps(metadata, ensure_ascii=False, separators=(',',':'))
    assert not any(c in text for c in '<>&')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">7 disks — Dali — The Shell That Holds the Sea — A</title><metadata>{text}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,{base64.b64encode(raster).decode()}"/></svg>\n'.encode()
    assert sha(svg) == record['svgSha256'] == '1d6bb184b46ad23b34c2454904287a56e11f991b1866645a194238848ed00fa2'

    exporter_path = ROOT / 'web/scripts/export-dali-tiles.mjs'
    exporter = exporter_path.read_text()
    pin6 = "  ['Pin6', '6p', '6 disks', 'The Spiral Staircase — C', 'six-disks-c-spiral-approved.svg', [0, 0, 1086, 1448]],"
    pin7 = "  ['Pin7', '7p', '7 disks', 'The Shell That Holds the Sea — A', 'seven-disks-a-shell-approved.svg', [0, 0, 1086, 1448]],"
    exporter = once(exporter, pin6, pin6 + '\n' + pin7)
    guard = """const onlyArgument = process.argv.find(argument => argument.startsWith('--only='));
const onlyTiles = onlyArgument ? new Set(onlyArgument.slice(7).split(',')) : null;
if (onlyTiles && [...onlyTiles].some(tile => !definitions.some(definition => definition[1] === tile))) {
  throw new Error('Unknown tile in --only selection');
}
"""
    exporter = once(exporter, 'const facePresentation = ', guard + 'const facePresentation = ')
    exporter = once(exporter, '  const source = `${studies}/${sourceName}`;', '  const selected = !onlyTiles || onlyTiles.has(tile);\n  const source = `${studies}/${sourceName}`;')
    exporter = once(exporter, '    writeFileSync(path.join(out, svg), face);', '    if (selected) writeFileSync(path.join(out, svg), face);')
    exporter = once(exporter, "  const raster = sourceName !== board ? sourceBytes : execFileSync('convert', [", "  const raster = !selected ? readFileSync(path.join(out, `approved/${name}.png`))\n    : sourceName !== board ? sourceBytes : execFileSync('convert', [")
    exporter = once(exporter, '  writeFileSync(path.join(out, png), raster);', '  if (selected) writeFileSync(path.join(out, png), raster);')
    exporter = once(exporter, '  writeFileSync(path.join(out, svg), markup);', '  if (selected) writeFileSync(path.join(out, svg), markup);')
    exporter = once(exporter, '${manifest.tiles.length} approved faces. Six disks', '${manifest.tiles.length} approved faces. Seven disks uses A — The Shell That Holds the Sea: seven luminous pearls in a 3–1–3 arrangement inside a sea-filled shell, with a sailing-ship shadow. Its complete 1086 × 1448 portrait is retained without resizing or compositional cropping. Six disks')
    exporter = once(exporter, 'Ranks 1–6 of disks are complete.', 'Ranks 1–7 of disks are complete.')

    changes = {exporter_path: exporter}
    for filename in ('dali-bamboo.test.js', 'dali-six-disks.test.js', 'tile-faces.test.js'):
        p = ROOT / 'web/tests' / filename
        content = p.read_text()
        if filename != 'dali-six-disks.test.js':
            content = once(content, "'4p', '5p', '6p', '1s'", "'4p', '5p', '6p', '7p', '1s'")
        if filename == 'tile-faces.test.js':
            content = once(content, "['2p', '4p', '6p', '3s', '4s'].includes(tile)", "['2p', '4p', '6p', '7p', '3s', '4s'].includes(tile)")
        for old,new in (
            ('twenty-three approved identities and eleven placeholders', 'twenty-four approved identities and ten placeholders'),
            ('Dali resolves twenty-three approved images', 'Dali resolves twenty-four approved images'),
            ('assert.equal(set.tiles.length, 23);', 'assert.equal(set.tiles.length, 24);'),
            ('assert.equal(set.placeholders.length, 11);', 'assert.equal(set.placeholders.length, 10);'),
            ("'23 approved faces'", "'24 approved faces'"),
            ("'remaining 11 tiles", "'remaining 10 tiles"),
        ):
            content = content.replace(old,new)
        changes[p] = content

    doc_path = ROOT / 'docs/design/dali/README.md'
    docs = doc_path.read_text()
    docs = once(docs, "Twenty-three faces are approved, drawn from the original studies and Carl's later selections. The latest is six disks", "Twenty-four faces are approved, drawn from the original studies and Carl's later selections. The latest is seven disks **The Shell That Holds the Sea — A**: seven luminous pearls in a 3–1–3 arrangement inside a sea-filled shell, casting a sailing-ship shadow on the salt plain. This completes disks 1–7. The previously selected six disks")
    docs = once(docs, '| `Sou1.svg` |', '| `Pin7.svg` | 7 disks | The Shell That Holds the Sea — A: seven luminous pearls in a sea-filled shell, a sailing-ship shadow and water evaporating above the salt plain |\n| `Sou1.svg` |')
    docs = once(docs, 'The other 11 tiles have no approved artwork yet', 'The other 10 tiles have no approved artwork yet')
    details = 'Seven disks uses [selected A — The Shell That Holds the Sea](studies/seven-disks-a-shell-approved.svg), from `seven_pearls_in_a_dreaming_seashell.png`. The entire 1086 × 1448 painting is encoded as AVIF quality 50, speed 0 at its original pixel dimensions, with no resizing, compositional crop, repainting, recolouring or added tile frame. The repository retains the full-dimension AVIF and identical source/runtime SVGs. Compression is lossy; the untouched original PNG is preserved separately in `dali-seven-disks-A-originals.zip` supplied in chat. See [provenance](seven-disks.json) for original, raster and SVG hashes. The buttoned-sky and ivory-instrument alternatives are not selected. All prior artwork and fits are unchanged. Export this tile alone with `node web/scripts/export-dali-tiles.mjs --only=7p`.\n\n'
    docs = once(docs, '## Presentation\n\n', '## Presentation\n\n' + details)
    changes[doc_path] = docs
    changes[ROOT / 'web/tests/dali-seven-disks.test.js'] = (TRANSFER / 'dali-seven-disks.test.js').read_text()

    # Every input and patch has now been validated; only new files are created.
    for p,data in ((ROOT/record['source'],svg),(ROOT/record['rasterSource'],raster),(ROOT/'docs/design/dali/seven-disks.json',(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())):
        assert not p.exists(), f'Refuse existing source: {p}'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for p,text in changes.items():
        p.write_text(text, encoding='utf-8')
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=7p')
    after = json.loads(manifest_path.read_text())
    assert len(after['tiles']) == 24 and len(after['placeholders']) == 10
    assert [e for e in after['tiles'] if e['tile'] != '7p'] == before['tiles'], 'Existing tile entry changed'
    assert after['placeholders'] == [e for e in before['placeholders'] if e['tile'] != '7p']
    for key in set(before) - {'tiles', 'placeholders'}:
        assert after[key] == before[key], key
    assert (ROOT/record['runtime']).read_bytes() == svg
    for p,digest in protected.items():
        assert (ROOT/p).is_file() and sha((ROOT/p).read_bytes()) == digest, f'Other artwork changed: {p}'
    for p,data in other_sets.items():
        assert (ROOT/p).read_bytes() == data, f'Other set changed: {p}'
    assert re.search(r'<div class="hand">(.*?)</div>', preview_path.read_text(), re.S).group(1) == hand
    generated = {p: p.read_bytes() for p in (manifest_path,preview_path,ROOT/'web/src/lib/dali-faces.js',ROOT/record['runtime'])}
    run('node', 'web/scripts/export-dali-tiles.mjs', '--only=7p')
    for p,data in generated.items():
        assert p.read_bytes() == data, f'Non-deterministic export: {p}'
    print(f'PASS: exact full-resolution A; {len(protected)} prior artwork files and all 23 prior Dali entries unchanged; 24 painted/10 named; identical 14-tile hand; deterministic selective export.')

if __name__ == '__main__':
    main()
