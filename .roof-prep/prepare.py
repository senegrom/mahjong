import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

BASE = os.environ['BASE_SHA']
TARGET = os.environ['TARGET_BRANCH']
PREP = Path('.roof-prep')
DESIGN = 'docs/design/van-gogh'
OUT = 'web/public/tiles/van-gogh'
SOURCE = f'{DESIGN}/studies/37-two-characters-roof-garden-c-approved.svg'
RUNTIME = f'{OUT}/approved/Man2.svg'
OLD_PNG = f'{OUT}/approved/Man2.png'
ARCHIVE = f'{DESIGN}/superseded/two-characters-night-cafe-Man2'
RECORD = f'{DESIGN}/two-characters-roof-garden-c.json'
RASTER_HASH = '097c1b200cf2603502818fd04249308cb5f8c2317b82ef20664f29f8bc71df64'
SOURCE_HASH = '3a1a10ccba933fc27b5b267d67b9c3f394ef0f6c1e83fd648be38ccc4420059e'
RUNTIME_HASH = 'd3aa6d977455047b42716dec3069174e86b99171d04517de05544f3aa4cf72c6'
ALLOWED = [f'{DESIGN}/README.md', SOURCE, ARCHIVE+'.png', ARCHIVE+'.svg', RECORD,
           f'{OUT}/README.md', RUNTIME, OLD_PNG, f'{OUT}/manifest.json', f'{OUT}/preview.html',
           'web/scripts/export-van-gogh-tiles.mjs', 'web/src/lib/van-gogh-faces.js',
           'web/tests/tile-faces.test.js', 'web/tests/van-gogh-two-roofs.test.js']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def run(*args, **kwargs):
    return subprocess.check_output(args, **kwargs).decode().strip()

def read(p):
    return Path(p).read_text()

def write(p, text):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(text)

def replace(text, old, new):
    assert text.count(old) == 1, f'Expected exactly one match: {old[:100]}'
    return text.replace(old, new, 1)

TESTS = r'''import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VAN_GOGH_APPROVED } from '../src/lib/van-gogh-faces.js';
import { TILE_TYPES } from '../src/lib/tiles.js';
import { TILE_IMAGE_URLS, tileImage } from '../src/lib/tile-faces.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = relative => readFileSync(path.join(root, relative));
const hash = data => createHash('sha256').update(data).digest('hex');
const set = JSON.parse(read('web/public/tiles/van-gogh/manifest.json'));
const record = JSON.parse(read('docs/design/van-gogh/two-characters-roof-garden-c.json'));
const entry = set.tiles.find(tile => tile.tile === '2m');
const sourceHash = '3a1a10ccba933fc27b5b267d67b9c3f394ef0f6c1e83fd648be38ccc4420059e';
const runtimeHash = 'd3aa6d977455047b42716dec3069174e86b99171d04517de05544f3aa4cf72c6';
const rasterHash = '097c1b200cf2603502818fd04249308cb5f8c2317b82ef20664f29f8bc71df64';

test('two characters uses the exact right-hand roof-and-flower-pots C selection', () => {
  assert.equal(entry.candidate, 'Roof Garden C');
  assert.equal(entry.name, 'Man2');
  assert.equal(entry.source, 'docs/design/van-gogh/studies/37-two-characters-roof-garden-c-approved.svg');
  assert.equal(record.originalFilename, 'impasto_lanternlit_temple_gate.png');
  assert.equal(record.originalSha256, '41985fa1d76b04bbdface9459132c3b1f8379e7cf857854edc3334fec68237f7');
  assert.deepEqual(record.originalDimensions, [1086, 1448]);
  assert.deepEqual(record.originalCrop, { x: 0, y: 0, width: 1086, height: 1448 });
  assert.equal(hash(read(record.source)), sourceHash);
  assert.equal(hash(read(record.runtime)), runtimeHash);
  assert.equal(record.sourceSha256, sourceHash);
  assert.equal(record.runtimeSvgSha256, runtimeHash);
  assert.equal(entry.svgSha256, runtimeHash);
  assert.equal(set.sources.find(source => source.source === entry.source).sha256, sourceHash);
});

test('the two roofs and flowers retain their entire native portrait with zero stretch or bleed', () => {
  assert.deepEqual(entry.fit, {
    painting: { width: 1086, height: 1448 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
    image: { x: 0, y: 0, width: 300, height: 400 },
  });
  const svg = read(record.runtime).toString();
  assert.equal(read(record.source).toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"'), svg);
  assert.match(svg, /viewBox="0 0 300 400"/);
  assert.match(svg, /rx="26"/);
  assert.doesNotMatch(svg, /<script|<foreignObject|<text|href="https?:/i);
  const embedded = svg.match(/href="data:image\/avif;base64,([A-Za-z0-9+/=]+)"/);
  assert.ok(embedded);
  const raster = Buffer.from(embedded[1], 'base64');
  assert.equal(raster.length, 40984);
  assert.equal(hash(raster), rasterHash);
  assert.equal(entry.rasterSha256, rasterHash);
  assert.equal(record.rasterSha256, rasterHash);
  assert.equal(raster.toString('ascii', 4, 12), 'ftypavif');
  const ispe = raster.indexOf(Buffer.from('ispe'));
  assert.ok(ispe >= 0);
  assert.deepEqual([raster.readUInt32BE(ispe + 8), raster.readUInt32BE(ispe + 12)], [300, 400]);
});

test('original Night Cafe stays recoverable byte-for-byte without an obsolete active PNG', () => {
  const previous = set.superseded.find(tile => tile.tile === '2m' && tile.candidate === 'Characters B');
  assert.equal(previous.activeCandidate, 'Roof Garden C');
  assert.equal(previous.source, 'docs/design/van-gogh/studies/07-two-characters-night-cafe.png');
  assert.equal(previous.archivedSvg, record.archivedSvg);
  assert.equal(previous.archivedPng, record.archivedPng);
  assert.equal(hash(read(record.archivedSvg)), record.archivedSvgSha256);
  assert.equal(hash(read(record.archivedPng)), record.archivedPngSha256);
  const original = read(previous.source);
  assert.equal(createHash('sha1').update(Buffer.from(`blob ${original.length}\0`)).update(original).digest('hex'), '5ce8e6822ece3d11b0e33b21a666b6272717adc5');
  assert.deepEqual(read(record.archivedPng), original);
  assert.equal(existsSync(path.join(root, 'web/public/tiles/van-gogh/approved/Man2.png')), false);
});

test('Roof Garden C preloads once and keeps all 34 identities, other sets and hidden tiles intact', () => {
  assert.equal(set.tiles.length, 34);
  assert.equal(set.remaining.length, 0);
  assert.deepEqual(set.tiles.map(tile => tile.tile), VAN_GOGH_APPROVED);
  assert.deepEqual([...VAN_GOGH_APPROVED].sort(), [...TILE_TYPES].sort());
  assert.equal(VAN_GOGH_APPROVED.filter(tile => tile === '2m').length, 1);
  assert.equal(tileImage('2m', 'van-gogh'), 'tiles/van-gogh/approved/Man2.svg');
  assert.equal(TILE_IMAGE_URLS.filter(url => url === tileImage('2m', 'van-gogh')).length, 1);
  assert.equal(tileImage('2m', 'classic'), 'tiles/Man2.svg');
  assert.equal(tileImage('2m', 'van-gogh', true), 'tiles/Back.svg');
  for (const [tile, candidate] of [['2s', 'Flutes B'], ['3s', 'Vase A'], ['4s', 'Sunlit Garden B'], ['6m', 'Scarlet Loom C'], ['9m', 'Poppy Hillside B'], ['5z', 'Almond Blossom C'], ['7z', 'Painter C']]) {
    assert.equal(set.tiles.find(entry => entry.tile === tile).candidate, candidate);
  }
});

test('Roof Garden C appears in the gallery and the unchanged fourteen-tile sample hand', () => {
  const preview = read('web/public/tiles/van-gogh/preview.html').toString();
  assert.match(preview, /Roof Garden C · Two characters/);
  const hand = preview.match(/<div class="rack" id="rack">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(hand);
  assert.equal((hand.match(/<img /g) ?? []).length, 14);
  assert.ok(hand.includes('approved/Man2.svg'));
});

test('--only=2m is repeatable and preserves every other playable image', t => {
  const temporary = mkdtempSync(path.join(tmpdir(), 'van-gogh-two-roofs-'));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));
  const put = (relative, bytes) => {
    const target = path.join(temporary, relative);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, bytes);
  };
  put('package.json', '{"type":"module"}');
  for (const file of ['web/scripts/export-van-gogh-tiles.mjs', 'web/scripts/face-fit.mjs', 'web/src/lib/tiles.js']) put(file, read(file));
  for (const source of set.sources) put(source.source, source.source.endsWith('.svg') ? read(source.source) : Buffer.from(`source fixture: ${source.id}`));
  const unchanged = [];
  for (const tile of set.tiles.filter(tile => tile.tile !== '2m')) {
    for (const file of [tile.png, tile.svg].filter(Boolean)) {
      const relative = `web/public/tiles/van-gogh/${file}`;
      const bytes = Buffer.from(`unchanged fixture: ${file}`);
      put(relative, bytes);
      unchanged.push([relative, bytes]);
    }
  }
  const run = argument => execFileSync(process.execPath, [path.join(temporary, 'web/scripts/export-van-gogh-tiles.mjs'), argument], { encoding: 'utf8', stdio: 'pipe' });
  for (let i = 0; i < 2; i++) {
    run('--only=2m');
    const generated = JSON.parse(readFileSync(path.join(temporary, 'web/public/tiles/van-gogh/manifest.json')));
    assert.deepEqual(generated.tiles.find(tile => tile.tile === '2m'), entry);
    assert.deepEqual(generated.superseded, set.superseded);
    assert.equal(hash(readFileSync(path.join(temporary, record.runtime))), runtimeHash);
    for (const [relative, bytes] of unchanged) assert.deepEqual(readFileSync(path.join(temporary, relative)), bytes);
  }
  assert.throws(() => run('--only=not-a-tile'), /Unknown tile/);
});
'''

def install():
    before = json.loads(read(f'{OUT}/manifest.json'))
    old = next(t for t in before['tiles'] if t['tile'] == '2m')
    assert len(before['tiles']) == 34 and old['candidate'] == 'Characters B'
    paths = run('git', 'ls-tree', '-r', '--name-only', BASE, '--', 'web/public/tiles', 'docs/design').splitlines()
    artwork = {p: sha(Path(p).read_bytes()) for p in paths if Path(p).suffix in {'.svg', '.png', '.webp', '.avif'}}
    write(PREP/'before.json', json.dumps({'manifest': before, 'artwork': artwork}))
    for suffix in ('.svg', '.png'):
        Path(ARCHIVE+suffix).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f'{OUT}/approved/Man2{suffix}', ARCHIVE+suffix)
    Path(OLD_PNG).unlink()
    raster = b''.join((PREP/f'part{n}').read_bytes() for n in range(4))
    assert len(raster) == 40984 and sha(raster) == RASTER_HASH
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" role="img" aria-labelledby="title"><title id="title">Two characters — Van Gogh — Night Cafe Roof Garden C</title><defs><clipPath id="face"><rect width="300" height="400" rx="26"/></clipPath></defs><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" preserveAspectRatio="none" href="data:image/avif;base64,' + base64.b64encode(raster).decode() + '"/></svg>\n'
    assert sha(svg.encode()) == SOURCE_HASH
    write(SOURCE, svg)
    record = {
        'tile': '2m', 'name': 'Man2', 'candidate': 'Roof Garden C', 'title': 'Night Cafe — Roofs and Flower Pots',
        'source': SOURCE, 'runtime': RUNTIME,
        'approval': 'Carl selected the right-hand standalone painting with flower pots after explicitly clarifying that the two strokes of the numeral must remain roofs. Use the exact final third image, not the earlier triptych or hanging-sign cafe alternatives.',
        'originalFilename': 'impasto_lanternlit_temple_gate.png',
        'originalSha256': '41985fa1d76b04bbdface9459132c3b1f8379e7cf857854edc3334fec68237f7',
        'originalDimensions': [1086, 1448], 'originalCrop': {'x': 0, 'y': 0, 'width': 1086, 'height': 1448},
        'originalRetention': 'Full-resolution original retained in the approval conversation and supplied download archive. Repository stores the optimized approved artwork and provenance, not the full-resolution PNG.',
        'processing': 'Uniformly resize the entire exact 3:4 RGB painting with Pillow Lanczos to 300 x 400 and encode AVIF quality 70, speed 0, 4:2:0. No cropping, stretching, repainting or recolouring; zero extra bleed. The two roof strokes, structural Wan, flower pots, furniture and lantern are retained. Only standard rounded tile corners are clipped.',
        'rasterDimensions': [300, 400], 'rasterMimeType': 'image/avif', 'rasterBytes': len(raster),
        'rasterSha256': RASTER_HASH, 'sourceSha256': SOURCE_HASH, 'svgSha256': RUNTIME_HASH, 'runtimeSvgSha256': RUNTIME_HASH,
        'fit': {'painting': [1086, 1448], 'stretch': 0, 'anchor': 0.5, 'bleedPixels': 0},
        'supersededCandidate': old['candidate'], 'supersededSource': old['source'],
        'archivedSvg': ARCHIVE+'.svg', 'archivedSvgSha256': artwork[RUNTIME],
        'archivedPng': ARCHIVE+'.png', 'archivedPngSha256': artwork[OLD_PNG],
    }
    write(RECORD, json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    p = 'web/scripts/export-van-gogh-tiles.mjs'
    s = read(p)
    s = replace(s, "  // A centred cut would leave the upper roof of 二 against the top edge; this\n  // one gives it and the hook of 萬 equal room.\n  '2m': { stretch: 0.02, anchor: 0.466 },", "  // Roof Garden C is exactly 3:4: retain both roof strokes and the flower pots.\n  '2m': { painting: [1086, 1448], stretch: 0, anchor: 0.5, bleedPixels: 0 },")
    s = replace(s, 'const sources = {', "const sources = {\n  twoRoofGarden: '"+SOURCE+"',")
    s = replace(s, "['Characters B', '2m', 'Two characters', 'nightCafe', [0, 0, 1022, 1539]]", "['Roof Garden C', '2m', 'Two characters', 'twoRoofGarden', [0, 0, 300, 400]]")
    s = replace(s, '  superseded: [', "  superseded: [\n    { candidate: 'Characters B', tile: '2m', activeCandidate: 'Roof Garden C', source: sources.nightCafe, archivedPng: '"+ARCHIVE+".png', archivedSvg: '"+ARCHIVE+".svg' },")
    s = replace(s, "its source and provenance are unchanged.',", "its source and provenance are unchanged. Carl selected the final right-hand Roof Garden C for Two Characters, with flower pots, and explicitly required the two numeral strokes to remain roofs. The complete 1086 x 1448 painting is proportionally downsampled with zero stretch and zero extra bleed. The former Night Cafe PNG and SVG remain archived byte-for-byte.',")
    s = replace(s, 'Includes Sunlit Garden B', 'Includes Roof Garden C (2m), replacing Night Cafe, Sunlit Garden B')
    s = replace(s, 'Night Cafe for 2 of characters', 'Night Cafe Roof Garden C for 2 of characters')
    s = replace(s, 'Earlier selected art is preserved in lossless PNG crops.', 'Roof Garden C retains the complete selected 1086 × 1448 portrait in a 300 × 400 quality-70 AVIF, with zero stretch and zero extra bleed. Both numeral strokes remain roofs; the structural Wan, flower pots, furniture and lantern are retained. The former Night Cafe playable PNG and SVG are archived unchanged. Earlier selected art is preserved in lossless PNG crops.')
    write(p, s)
    p = 'web/tests/tile-faces.test.js'
    s = read(p)
    s = replace(s, "'Characters B', 'Characters C'", "'Roof Garden C', 'Characters C'")
    s = replace(s, "entry.rasterMimeType, ['2s'", "entry.rasterMimeType, ['2m', '2s'")
    s = replace(s, 'Van Gogh 2 and 4 characters preserve their exact approved source images', 'Van Gogh 4 characters preserves its exact approved source image')
    s, count = re.subn(r"^    \['2m', '07-two-characters-night-cafe\.png', '[0-9a-f]{40}'\],\n", '', s, flags=re.M)
    assert count == 1
    write(p, s)
    write('web/tests/van-gogh-two-roofs.test.js', TESTS)
    section = '\n\n## Two Characters — Night Cafe Roof Garden C (2m / Man2)\n\nCarl selected the final right-hand standalone painting with the flower pots after clarifying that both strokes of 二 must remain roofs. The two golden roof tiers, structural 萬, flower pots, cafe furniture, lantern and dark violet/rust-red setting are preserved. This is not the earlier triptych or hanging-sign alternative.\n\nThe entire 1086 × 1448 portrait is uniformly downsampled to 300 × 400 and encoded as quality-70 AVIF, with zero stretch, zero extra bleed and no cropping, repainting or recolouring. Only normal rounded tile clipping applies. The old playable PNG and SVG are archived byte-for-byte under `docs/design/van-gogh/superseded/two-characters-night-cafe-Man2.*`; the original Night Cafe study remains unchanged. The full-resolution selected original is retained with the approval conversation and supplied download archive; this repository stores the optimized artwork and provenance. See `docs/design/van-gogh/two-characters-roof-garden-c.json`. Re-export only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=2m`. All 33 other Van Gogh faces and other sets remain unchanged.\n'
    for p in (f'{DESIGN}/README.md', f'{OUT}/README.md'):
        write(p, read(p)+section)
    subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=2m'], check=True)
    assert sha(Path(RUNTIME).read_bytes()) == RUNTIME_HASH
    print('ASSEMBLED ROOF GARDEN C', RUNTIME_HASH)

def publish():
    before = json.loads(read(PREP/'before.json'))
    outputs = [RUNTIME, f'{OUT}/manifest.json', f'{OUT}/preview.html', 'web/src/lib/van-gogh-faces.js']
    first = {p: Path(p).read_bytes() for p in outputs}
    subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=2m'], check=True)
    assert all(Path(p).read_bytes() == data for p, data in first.items()), 'Nondeterministic export'
    current = json.loads(read(f'{OUT}/manifest.json'))
    assert [t for t in current['tiles'] if t['tile'] != '2m'] == [t for t in before['manifest']['tiles'] if t['tile'] != '2m']
    assert len(current['tiles']) == 34 and not current['remaining']
    for p, expected in before['artwork'].items():
        if p not in (RUNTIME, OLD_PNG):
            assert sha(Path(p).read_bytes()) == expected, f'Unrelated artwork changed: {p}'
    assert sha(Path(ARCHIVE+'.svg').read_bytes()) == before['artwork'][RUNTIME]
    assert sha(Path(ARCHIVE+'.png').read_bytes()) == before['artwork'][OLD_PNG]
    assert not Path(OLD_PNG).exists()
    assert sha(Path(RUNTIME).read_bytes()) == RUNTIME_HASH
    changed = set(run('git', 'diff', '--name-only', BASE).splitlines()) | set(run('git', 'ls-files', '--others', '--exclude-standard').splitlines())
    assert all(p in ALLOWED or p.startswith('.roof-prep/') or p == '.github/workflows/prepare-two-roofs-art.yml' for p in changed)
    subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], check=True)
    subprocess.run(['git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], check=True)
    with tempfile.TemporaryDirectory() as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary)/'index'))
        subprocess.run(['git', 'read-tree', BASE], env=env, check=True)
        subprocess.run(['git', 'add', '-A', '--', *ALLOWED], env=env, check=True)
        diff = set(run('git', 'diff', '--cached', '--name-only', '--no-renames', BASE, env=env).splitlines())
        assert diff == set(ALLOWED), f'Unexpected final diff: {diff.symmetric_difference(ALLOWED)}'
        tree = run('git', 'write-tree', env=env)
        message = 'Deploy Van Gogh Two Characters — Roof Garden C\n\nInstall the exact selected final right-hand painting with flower pots. Preserve both numeral strokes as roofs and the complete native 3:4 composition. Archive the previous Night Cafe PNG and SVG byte-for-byte. All 33 other Van Gogh faces and unrelated artwork are unchanged.\n\nFocused tile regressions, scoped ESLint, deterministic export and checksum-preservation checks passed. No workflow, dependency, gameplay or preparation files in this clean artwork commit. Publish through the existing full CI and Pages gates.'
        commit = run('git', 'commit-tree', tree, '-p', BASE, '-m', message, env=env)
    subprocess.run(['git', 'push', 'origin', f'{commit}:refs/heads/{TARGET}'], check=True)
    print('PRESERVATION PASS: all 33 other faces and unrelated artwork unchanged')
    print('CLEAN_ARTWORK_COMMIT='+commit)
    print('RUNTIME_SHA256='+RUNTIME_HASH)

if __name__ == '__main__':
    {'install': install, 'publish': publish}[sys.argv[1]]()
