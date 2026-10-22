import assert from 'node:assert/strict';
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
