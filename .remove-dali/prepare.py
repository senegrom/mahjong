"""Prepare only the requested Dalí removal. Never touch main or other artwork."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = Path.cwd()
STATE = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'remove-dali-preservation.json'
TEMP = {'.remove-dali/prepare.py', '.github/workflows/prepare-remove-dali.yml'}
MODIFIED = {
    'README.md', 'docs/UI.md', 'web/src/lib/tile-faces.js',
    'web/src/lib/Tile.svelte', 'web/src/lib/app/types.ts',
    'web/scripts/component-check.mjs', 'web/scripts/network-picker-check.mjs',
    'web/scripts/trained-default-check.mjs', 'web/scripts/tile-effects-check.mjs',
    'web/tests/tile-faces.test.js', 'web/tests/tile-proportions.test.js',
    'web/tests/van-gogh-three-lanterns.test.js',
}
NEW = 'web/tests/retired-tile-face.test.js'

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

def tracked():
    return [p for p in subprocess.check_output(['git','ls-files','-z']).decode().split('\0') if p]

def removed(name):
    return name.startswith(('docs/design/dali/', 'web/public/tiles/dali/')) or name in (
        'web/src/lib/dali-faces.js', 'web/scripts/export-dali-tiles.mjs'
    ) or (name.startswith('web/tests/dali-') and name.endswith('.test.js'))

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def once(text, old, new, count=1):
    assert text.count(old) == count, f'Changed patch anchor ({text.count(old)} != {count}): {old!r}'
    return text.replace(old, new)

def patch(name, edits):
    p = Path(name)
    text = p.read_text()
    for old, new, *count in edits:
        text = once(text, old, new, *(count or [1]))
    p.write_text(text)

def verify():
    state = json.loads(STATE.read_text())
    for p, expected in state['protected'].items():
        assert Path(p).is_file() and digest(p) == expected, f'Unrelated file changed: {p}'
    for p in state['removed']:
        assert not Path(p).exists(), f'Dali file remains: {p}'
    allowed = set(state['removed']) | MODIFIED | {NEW} | TEMP
    changed = git('diff','--name-only').splitlines()
    assert set(changed) <= allowed, f'Unexpected changes: {set(changed) - allowed}'
    assert not any(removed(p) for p in tracked() if Path(p).exists())
    references = []
    for name in tracked():
        if name in TEMP or name == NEW or name in state['removed'] or not Path(name).is_file(): continue
        try: text = Path(name).read_text()
        except UnicodeDecodeError: continue
        if re.search(r'\bdal[ií]\b|DALI_', text, re.I): references.append(name)
    assert not references, f'Unremoved references: {references}'
    assert not any('dali' in p.lower() for p in tracked() if Path(p).exists() and p not in TEMP), 'A dedicated path remains'
    print(f"PASS: removed {len(state['removed'])} dedicated files ({state['removedBytes']} bytes); "
          f"preserved {len(state['protected'])} unrelated tracked files exactly, including {state['protectedImages']} images; "
          'no active Dali references; main and other sets untouched.', flush=True)

if len(sys.argv) > 1 and sys.argv[1] == '--verify':
    verify()
    raise SystemExit

names = tracked()
assert not any(Path(p).name == 'AGENTS.md' for p in names), 'Read new repository instructions before proceeding'
assert not Path(NEW).exists()
removals = [p for p in names if removed(p)]
assert len(removals) > 50, 'Unexpected removal scope'
protected = {p: digest(p) for p in names if p not in TEMP and p not in MODIFIED and not removed(p)}
STATE.write_text(json.dumps({
    'removed': removals, 'removedBytes': sum(Path(p).stat().st_size for p in removals),
    'protected': protected,
    'protectedImages': sum(Path(p).suffix.lower() in {'.png','.svg','.avif','.webp','.jpg','.jpeg','.gif'} for p in protected),
}, indent=2))

patch('web/src/lib/tile-faces.js', [
    ("import { DALI_APPROVED } from './dali-faces.js';\n", ''),
    ("  { value: 'dali', label: 'Dalí' },\n", ''),
    ("['dali', DALI_APPROVED], ", ''),
])
patch('web/src/lib/app/types.ts', [(" | 'dali'", '')])
patch('web/src/lib/Tile.svelte', [
    ("    class:dali={tileFace === 'dali' && !facedown && Boolean(tile)}\n", '', 2),
    ('.tile.matisse, .tile.dali, .tile.van-gogh', '.tile.matisse, .tile.van-gogh'),
    ('.rotated:is(.matisse, .dali, .van-gogh)', '.rotated:is(.matisse, .van-gogh)'),
])
patch('web/scripts/component-check.mjs', [
    ("import { DALI_APPROVED } from '../src/lib/dali-faces.js';\n", ''),
    (", ['dali', DALI_APPROVED]", ''),
    ("    await page.select('select[aria-label=\"Tile face\"]', 'matisse');", "    assert.deepEqual(await page.$$eval('select[aria-label=\"Tile face\"] option', options => options.map(option => option.value)), ['classic', 'matisse', 'van-gogh']);\n    await page.select('select[aria-label=\"Tile face\"]', 'matisse');"),
])
patch('web/scripts/network-picker-check.mjs', [("tileFace: 'dali'", "tileFace: 'matisse'")])
patch('web/scripts/trained-default-check.mjs', [("'tiles/dali/', ", '')])
patch('web/tests/tile-proportions.test.js', [("'matisse', 'dali', 'van-gogh'", "'matisse', 'van-gogh'")])
patch('web/tests/van-gogh-three-lanterns.test.js', [
    ("'classic', 'van-gogh', 'matisse', 'dali'", "'classic', 'van-gogh', 'matisse'"),
    ("  assert.equal(tileImage('3p', 'dali'), 'tiles/dali/approved/Pin3.svg');\n", ''),
])
patch('README.md', [('Classic, Matisse, Dalí, and Van Gogh', 'Classic, Matisse, and Van Gogh')])
patch('docs/UI.md', [
    ("Classic and Matisse have a picture for every tile. Dalí and Van Gogh are still\nbeing painted: a tile the chosen set has no approved artwork for shows its name\n", "Classic, Matisse and Van Gogh have a picture for every tile. Unavailable saved\ntile-face choices fall back to Classic without resetting other preferences.\nThe shared renderer also supports future incomplete sets: a tile without approved artwork shows its name\n"),
])

p = Path('web/tests/tile-faces.test.js')
t = p.read_text()
t = once(t, "import { DALI_APPROVED } from '../src/lib/dali-faces.js';\n", '')
t, n = re.subn(r"\ntest\('Dali resolves[^\n]*\n.*?\n\}\);\n", '\n', t, flags=re.S)
assert n == 1
for old, new, *count in [
    ("  assert.ok(TILE_IMAGE_URLS.includes('tiles/dali/approved/Pin1.svg'));\n", ''),
    ("['dali', DALI_APPROVED], ", '', 2),
    ("  const dali = read('dali'), vanGogh = read('van-gogh');", "  const vanGogh = read('van-gogh');"),
    ("  for (const set of [dali, vanGogh]) assert.equal(set.fallback, 'text');", "  assert.equal(vanGogh.fallback, 'text');"),
    ("  assert.deepEqual(dali.placeholders, TILE_TYPES.filter(tile => !DALI_APPROVED.includes(tile)).map(tile => ({ tile })));\n", ''),
    ("['dali', 'matisse']", "['matisse', 'van-gogh']"),
    ("  assert.ok(preview('dali').includes(`remaining ${dali.placeholders.length} tiles show their names until their artwork is approved`));\n", ''),
    ("['dali', 'van-gogh']", "['van-gogh']"),
    ("'export-dali-tiles', ", ''),
    ("'dali'", "'matisse'", 6),
    ("['matisse', 'matisse', 'van-gogh']", "['matisse', 'van-gogh']"),
    (r'dali\/|', ''),
]:
    t = once(t, old, new, *(count or [1]))
# Remove the retired set's incomplete-art fixture. The browser regression below
# now supplies an isolated incomplete registry, independent of shipped sets.
a = t.index("    // A tile its set has not painted keeps the set's corners and every")
b = t.index('\n  }\n});', a)
t = t[:a] + t[b:]
p.write_text(t)

# Retain name/foil/claimed/rotated regression coverage in a temporary Vite build.
# It supplies an incomplete Van Gogh registry only inside this fixture; no
# production registry or art is changed, and all green-face cases remain painted.
p = Path('web/scripts/tile-effects-check.mjs')
t = p.read_text()
t = once(t, "import { DALI_APPROVED } from '../src/lib/dali-faces.js';\n", '', 2)
t = once(t, 'const writtenTiles = [DALI_APPROVED, VAN_GOGH_APPROVED].flatMap(painted => TILE_TYPES.filter(tile => !painted.includes(tile)));', "// Supply missing-art cases only in this fixture, now that shipped sets are complete.\nconst FIXTURE_APPROVED = VAN_GOGH_APPROVED.filter(tile => !['2m', '2z', '5z'].includes(tile));\nconst writtenTiles = TILE_TYPES.filter(tile => !FIXTURE_APPROVED.includes(tile));\nconst unpaintedFixture = {\n  name: 'unpainted-tile-fixture',\n  enforce: 'pre',\n  transform(code, id) {\n    if (id === resolve(web, 'src/lib/van-gogh-faces.js')) {\n      assert.match(code, /export const VAN_GOGH_APPROVED/);\n      return { code: 'export const VAN_GOGH_APPROVED = Object.freeze(' + JSON.stringify(FIXTURE_APPROVED) + ');', map: null };\n    }\n  },\n};")
t = once(t, '// Tiles Dalí has not painted yet, whichever they are when this runs, at the', '// Artificially unpainted faces in this temporary fixture, shown at the')
t = once(t, 'const unpainted = TILE_TYPES.filter(tile => !DALI_APPROVED.includes(tile));', 'const unpainted = TILE_TYPES.filter(tile => !VAN_GOGH_APPROVED.includes(tile));')
t = once(t, 'matisse:MATISSE_APPROVED, dali:DALI_APPROVED,', 'matisse:MATISSE_APPROVED,')
t = once(t, "['dali',['2s','4s','8s']],", '')
t = once(t, "['dali','van-gogh']", "['van-gogh']")
t = t.replace('DALI_APPROVED', 'FIXTURE_APPROVED').replace("'dali'", "'van-gogh'")
t = once(t, 'the fixture shows tiles Dalí has not painted', 'the fixture shows deliberately unpainted tiles')
t = once(t, 'plugins:[svelte()]', 'plugins:[unpaintedFixture,svelte()]')
assert not re.search(r'\bdal[ií]\b|DALI_', t, re.I)
p.write_text(t)

Path(NEW).write_text("""import assert from 'node:assert/strict';
import test from 'node:test';
import { existsSync, readdirSync } from 'node:fs';
import { TILE_TYPES, tileFile } from '../src/lib/tiles.js';
import { TILE_FACE_OPTIONS, TILE_IMAGE_URLS, normalizeTileFace, tileImage } from '../src/lib/tile-faces.js';
import { readSettings } from '../src/lib/session.js';

const root = new URL('../../', import.meta.url);

test('the selectable faces are Classic, Matisse and Van Gogh with every tile available', () => {
  assert.deepEqual(TILE_FACE_OPTIONS.map(face => face.value), ['classic', 'matisse', 'van-gogh']);
  for (const { value } of TILE_FACE_OPTIONS) {
    for (const tile of TILE_TYPES) {
      const url = tileImage(tile, value);
      assert.ok(url, `${value} ${tile}`);
      assert.ok(existsSync(new URL(`web/public/${url}`, root)), url);
      assert.equal(TILE_IMAGE_URLS.filter(item => item === url).length, 1);
    }
  }
});

test('retired Dali preferences become Classic without resetting other supported settings', () => {
  assert.equal(normalizeTileFace('dali'), 'classic');
  for (const version of [1, 2, 3, 4]) {
    const saved = { version, tileFace: 'dali', hints: false, confirmDiscards: false, shortcuts: false };
    const read = settings => readSettings({ getItem: () => JSON.stringify(settings) });
    assert.deepEqual(read(saved), read({ ...saved, tileFace: 'classic' }));
    assert.equal(read(saved).tileFace, 'classic');
  }
  for (const tile of TILE_TYPES) {
    assert.equal(tileImage(tile, 'dali'), `tiles/${tileFile(tile)}.svg`);
    assert.equal(tileImage(tile, 'dali', true), 'tiles/Back.svg');
  }
  assert.equal(tileImage(null, 'dali'), 'tiles/Back.svg');
});

test('retired artwork, design sources and tooling are absent from the checkout and preload inventory', () => {
  for (const name of ['docs/design/dali', 'web/public/tiles/dali', 'web/src/lib/dali-faces.js', 'web/scripts/export-dali-tiles.mjs']) {
    assert.equal(existsSync(new URL(name, root)), false, name);
  }
  assert.equal(readdirSync(new URL('web/tests/', root)).some(name => name.startsWith('dali-')), false);
  assert.equal(TILE_IMAGE_URLS.some(url => url.includes('/dali/')), false);
  assert.equal(new Set(TILE_IMAGE_URLS).size, TILE_IMAGE_URLS.length);
});
""")

for name in removals:
    Path(name).unlink()
# Remove empty directories as well, so the absence tests match a fresh checkout.
for name in ('docs/design/dali', 'web/public/tiles/dali'):
    folder = Path(name)
    for child in sorted((p for p in folder.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True): child.rmdir()
    folder.rmdir()
verify()
