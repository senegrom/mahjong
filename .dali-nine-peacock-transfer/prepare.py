"""One-time approved peacock import; checksums and existing artwork fail closed."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path.cwd()
TRANSFER = ROOT / '.dali-nine-peacock-transfer'
def sha(data):
    return hashlib.sha256(data).hexdigest()
def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout
def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Changed patch anchor: {old!r}')
    return text.replace(old, new, 1)
def read(relative):
    return (ROOT / relative).read_text(encoding='utf-8')
def verify():
    saved = json.loads((TRANSFER / 'preservation.json').read_text())
    for path, digest in saved['protected'].items():
        assert (ROOT / path).is_file() and sha((ROOT / path).read_bytes()) == digest, f'Prior file changed: {path}'
    after = json.loads(read('web/public/tiles/dali/manifest.json'))
    before = saved['before']
    assert len(after['tiles']) == 27 and len(after['placeholders']) == 7
    assert [e for e in after['tiles'] if e['tile'] != '9p'] == before['tiles'], 'Prior tile entry changed'
    assert after['placeholders'] == [e for e in before['placeholders'] if e['tile'] != '9p']
    for key in set(before) - {'tiles', 'placeholders'}:
        assert before[key] == after[key], key
    hand = re.search(r'<div class="hand">(.*?)</div>', read('web/public/tiles/dali/preview.html'), re.S).group(1)
    assert hand == saved['hand'] and hand.count('<img ') == 14
    record = json.loads(read('docs/design/dali/nine-disks.json'))
    for key, hash_key in [('source','sourceSha256'),('runtime','svgSha256'),('rasterSource','rasterSha256')]:
        assert sha((ROOT / record[key]).read_bytes()) == record[hash_key]
    assert all(any(e['tile'] == f'{rank}p' for e in after['tiles']) for rank in range(1,10))
    print(f'PASS: {len(saved["protected"])} prior files and all 26 prior Dali entries unchanged; nine disks complete; 27 painted / 7 named; unchanged 14-tile hand.')

def main():
    before = json.loads(read('web/public/tiles/dali/manifest.json'))
    assert len(before['tiles']) == 26 and len(before['placeholders']) == 8
    assert not any(e['tile'] == '9p' for e in before['tiles']), 'Already installed: refuse overwrite'
    tracked = run('git', 'ls-files', '-z').split('\0')
    protected = {p: sha((ROOT/p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in ('.png','.svg','.avif','.webp','.jpg','.jpeg','.gif')}
    for set_name in ('van-gogh','matisse'):
        for path in (f'web/public/tiles/{set_name}/manifest.json', f'web/src/lib/{set_name}-faces.js'):
            protected[path] = sha((ROOT/path).read_bytes())
    hand = re.search(r'<div class="hand">(.*?)</div>', read('web/public/tiles/dali/preview.html'), re.S).group(1)
    (TRANSFER/'preservation.json').write_text(json.dumps({'protected':protected,'before':before,'hand':hand}))
    chunks = []
    for part in json.loads((TRANSFER/'parts.json').read_text()):
        data = (ROOT/part['path']).read_bytes()
        assert len(data) == part['bytes'] and git_blob(data) == part['sha'], f'Invalid image chunk: {part["path"]}'
        chunks.append(data)
    raster = b''.join(chunks)
    record = json.loads((TRANSFER/'nine-disks.json').read_text())
    assert len(raster) == 40634
    assert sha(raster) == record['rasterSha256'] == '4e0cc2a4398d4132936dd9d8371c4114403a8c2ee7ae343e1f7fbcd5479aec33'
    assert raster[4:12] == b'ftypavif'
    ispe = raster.index(b'ispe')
    assert [int.from_bytes(raster[ispe+8:ispe+12],'big'),int.from_bytes(raster[ispe+12:ispe+16],'big')] == [462,847]
    metadata = {k:v for k,v in record.items() if k not in ('sourceSha256','svgSha256')}
    text = json.dumps(metadata,ensure_ascii=False,separators=(',',':'))
    assert not any(c in text for c in '<>&')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">9 disks — Dali — The Peacock of Small Universes — C</title><metadata>{text}</metadata><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,{base64.b64encode(raster).decode()}"/></svg>\n'.encode()
    assert sha(svg) == record['sourceSha256'] == '540a965b379aca2f7d3c2f423fac12bc0ae211f8b547629565f1422f19a47b4f'
    exporter = read('web/scripts/export-dali-tiles.mjs')
    pin8 = "  ['Pin8', '8p', '8 disks', 'The Cabinet of Captured Moons — C', 'eight-disks-c-cabinet-approved.svg', [0, 0, 609, 812]],"
    pin9 = "  ['Pin9', '9p', '9 disks', 'The Peacock of Small Universes — C', 'nine-disks-c-peacock-approved.svg', [0, 0, 462, 847]],"
    exporter = once(exporter,pin8,pin8+'\n'+pin9)
    exporter = once(exporter,'const fits = {', 'const fits = {\n  // Keep all nine tail windows and the bird at their exact painted proportions.\n  // The complete tall panel, including its reflection, remains in the source.\n  \'9p\': { painting: [462, 847], stretch: 0, anchor: 1 / 11, bleedPixels: 0 },')
    exporter = once(exporter,'${manifest.tiles.length} approved faces. Eight disks', '${manifest.tiles.length} approved faces. Nine disks uses C — The Peacock of Small Universes: the selected ivory peacock with nine circular landscape windows in its tail. The full 462 × 847 panel and reflection remain in the source; the zero-stretch 3:4 game framing keeps all nine disks and the bird visible, while most of the reflection lies outside the face. Eight disks')
    exporter = once(exporter,'Ranks 1–8 of disks are complete.', 'Ranks 1–9 of disks are complete.')
    changes = {'web/scripts/export-dali-tiles.mjs':exporter}
    # Only coverage/preview expectations change in existing tests. Art hashes stay pinned.
    replacements = [
        ("'7p', '8p', '1s'", "'7p', '8p', '9p', '1s'"),
        ("'7p', '8p', '3s'", "'7p', '8p', '9p', '3s'"),
        ('set.tiles.length, 26', 'set.tiles.length, 27'),
        ('set.placeholders.length, 8', 'set.placeholders.length, 7'),
        ('26 approved faces','27 approved faces'),
        ('remaining 8 tiles','remaining 7 tiles'),
        ('twenty-six approved identities and eight placeholders','twenty-seven approved identities and seven placeholders'),
        ('Dali resolves twenty-six approved images','Dali resolves twenty-seven approved images'),
        ('twenty-six-face preview','twenty-seven-face preview'),
        ('Ranks 1–8 of disks are complete','Ranks 1–9 of disks are complete'),
    ]
    paths = sorted((ROOT/'web/tests').glob('dali-*.test.js')) + [ROOT/'web/tests/tile-faces.test.js']
    for path in paths:
        old = path.read_text()
        new = old
        for a,b in replacements:
            new = new.replace(a,b)
        if new != old:
            changes[path.relative_to(ROOT).as_posix()] = new
    docs = read('docs/design/dali/README.md')
    docs = once(docs, "Twenty-six faces are approved, drawn from the original studies and Carl's later selections. The latest is eight disks", "Twenty-seven faces are approved, drawn from the original studies and Carl's later selections. The latest is nine disks **The Peacock of Small Universes — C**, the selected ivory peacock with nine circular landscape windows in its tail. This completes all nine disks. The previously selected eight disks")
    docs = once(docs,'| `Sou1.svg` |','| `Pin9.svg` | 9 disks | The Peacock of Small Universes — C: nine circular landscape windows in an ivory peacock tail over mirror water |\n| `Sou1.svg` |')
    docs = once(docs,'The other 8 tiles have no approved artwork yet','The other 7 tiles have no approved artwork yet')
    details = 'Nine disks uses [selected C — The Peacock of Small Universes](studies/nine-disks-c-peacock-approved.svg), from the right panel of `nine_disks_a_surreal_concept_board.png`. The artwork-only panel at (973, 116), 462 × 847 pixels, removes the review-board labels, gutters and frame. The complete panel, including the reflected bird, remains at those pixel dimensions in its quality-50 AVIF and self-contained source SVG. There is no resizing, repainting or recolouring; compression is lossy. The existing face-fit helper uses zero stretch and zero bleed to show a 3:4 window retaining all nine disks and the bird. Most of the reflection is outside that playable window, not deleted from the source. Runtime and study differ only in image placement. The original board and lossless panel are preserved in `dali-nine-disks-C-originals.zip` supplied in chat. See [provenance](nine-disks.json) for all hashes and framing details. The carpet and orchard alternatives are not selected. All prior artwork and the fourteen-tile sample hand remain unchanged. Reproduce only this tile with `node web/scripts/export-dali-tiles.mjs --only=9p`.\n\n'
    docs = once(docs,'## Presentation\n\n','## Presentation\n\n'+details)
    changes['docs/design/dali/README.md'] = docs
    changes['web/tests/dali-nine-disks.test.js'] = (TRANSFER/'dali-nine-disks.test.js').read_text()
    for path, data in [(record['source'],svg),(record['rasterSource'],raster),('docs/design/dali/nine-disks.json',(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())]:
        target = ROOT/path
        assert not target.exists(), f'Refuse to replace source: {path}'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    for path, text in changes.items():
        (ROOT/path).write_text(text,encoding='utf-8')
    run('node','web/scripts/export-dali-tiles.mjs','--only=9p')
    verify()
    generated = {p:(ROOT/p).read_bytes() for p in [record['runtime'],'web/public/tiles/dali/manifest.json','web/public/tiles/dali/preview.html','web/src/lib/dali-faces.js']}
    run('node','web/scripts/export-dali-tiles.mjs','--only=9p')
    for path,data in generated.items():
        assert (ROOT/path).read_bytes() == data, f'Non-deterministic export: {path}'
    verify()
    print('PASS: checksum-verified peacock C, exact source/runtime/raster, deterministic repeated selective export.')

if __name__ == '__main__':
    verify() if '--verify' in sys.argv else main()
