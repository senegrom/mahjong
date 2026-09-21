import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { copyRuntimeAssets } from '../scripts/runtime-assets.mjs';
import { buildOffline } from '../scripts/offline-build.mjs';
import { TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-assets-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, 'public'), output = join(root, 'dist');
  const names = [...TILE_IMAGE_URLS, ...RUNTIME_FILES.map(name => `ort/${name}`), 'apple-touch-icon.png',
    'favicon-16x16.png', 'favicon-32x32.png', 'favicon.ico', 'favicon.svg', 'manifest.webmanifest',
    'icons/mahjong-192.png', 'tiles/LICENSE.md', 'tiles/matisse/approved/Man1.png',
    'tiles/matisse/README.md', 'tiles/matisse/preview.html', 'tiles/matisse/manifest.json'];
  for (const name of names) {
    await mkdir(dirname(join(source, name)), { recursive: true });
    await writeFile(join(source, name), name);
  }
  return { source, output };
}
test('packaging retains every exact tile body and licence but excludes duplicate art crops', async t => {
  const { source, output } = await fixture(t);
  const report = await copyRuntimeAssets(source, output);
  for (const name of TILE_IMAGE_URLS) assert.deepEqual(await readFile(join(output, name)), await readFile(join(source, name)));
  assert.ok(await readFile(join(output, 'tiles/LICENSE.md')));
  assert.ok(report.excludedBytes > 0);
  for (const name of ['approved/Man1.png', 'README.md', 'preview.html', 'manifest.json']) {
    await assert.rejects(readFile(join(output, 'tiles/matisse', name)), { code: 'ENOENT' });
    assert.ok(await readFile(join(source, 'tiles/matisse', name)), 'source workspace is untouched');
  }
});
test('missing required artwork fails before a partial package is copied', async t => {
  const { source, output } = await fixture(t);
  await rm(join(source, TILE_IMAGE_URLS.at(-1)));
  await assert.rejects(copyRuntimeAssets(source, output), /Missing runtime asset/);
  await assert.rejects(readFile(join(output, 'favicon.svg')), { code: 'ENOENT' });
});
test('production manifest reports sizes, scoped cache version and deterministic content identity', async t => {
  const { source, output } = await fixture(t);
  await copyRuntimeAssets(source, output);
  await writeFile(join(output, 'index.html'), 'game');
  const first = await buildOffline(output), second = await buildOffline(output);
  assert.deepEqual(second, first);
  assert.equal(first.networkCacheVersion, 2);
  assert.ok(first.bytes.core > 0 && first.bytes.runtime > 0);
  assert.equal(first.bytes.network, first.network.bytes);
  assert.ok((await readFile(join(output, 'sw.js'), 'utf8')).includes('pruneNetworkCache'));
});
