import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir, stat } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RUNTIME_FILES } from '../src/lib/model-package.js';
import { TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';

const web = dirname(dirname(fileURLToPath(import.meta.url)));
async function walk(dir) {
  return (await Promise.all((await readdir(dir, { withFileTypes: true })).map(entry =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : join(dir, entry.name)))).flat();
}

test('production includes the reduced runtime and all tile faces, not model files or art crops', async () => {
  const dist = join(web, 'dist');
  const names = (await walk(dist)).map(file => relative(dist, file).split(sep).join('/'));
  assert.equal(names.filter(file => /^assets\/ort-.*\.wasm$/.test(file)).length, 0,
    'Vite must not bundle the generic ONNX Runtime WASM');
  for (const file of RUNTIME_FILES) assert.ok(names.includes(`ort/${file}`));
  for (const file of TILE_IMAGE_URLS) assert.ok(names.includes(file), `Missing playable artwork: ${file}`);
  assert.equal(names.some(name => /\.onnx$/i.test(name)), false, 'the only model is remote');
  assert.equal(names.some(name => /^tiles\/.*\.png$/.test(name)), false, 'PNG crops duplicate the unchanged embedded SVG pixels');
  assert.equal(names.some(name => /^tiles\/.*(?:README\.md|preview\.html|manifest\.json)$/.test(name)), false);
  const reduced = (await stat(join(web, 'runtime/ort-wasm-simd-threaded.wasm'))).size;
  const generic = (await stat(join(web, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'))).size;
  assert.ok(reduced < generic, `reduced runtime ${reduced} must be smaller than generic ${generic}`);
});
