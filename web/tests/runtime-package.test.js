import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdir, readFile, stat } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = dirname(dirname(fileURLToPath(import.meta.url)));
async function walk(dir) {
  return (await Promise.all((await readdir(dir, { withFileTypes: true })).map(entry =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : join(dir, entry.name)))).flat();
}

test('production uses the reduced external ONNX runtime for the current model', async () => {
  const dist = join(web, 'dist');
  const files = await walk(dist);
  const names = files.map(file => relative(dist, file).split(sep).join('/'));
  assert.equal(names.filter(file => /^assets\/ort-.*\.wasm$/.test(file)).length, 0,
    'Vite must not bundle the generic ONNX Runtime WASM');
  assert.ok(names.includes('ort/ort-wasm-simd-threaded.wasm'));
  assert.ok(names.includes('ort/ort-wasm-simd-threaded.mjs'));

  const reduced = (await stat(join(web, 'runtime', 'ort-wasm-simd-threaded.wasm'))).size;
  const generic = (await stat(join(web, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.wasm'))).size;
  assert.ok(reduced < generic, `reduced runtime ${reduced} must be smaller than generic ${generic}`);

  const expected = (await readFile(join(web, 'runtime', 'model.sha256'), 'utf8')).trim();
  const actual = createHash('sha256').update(await readFile(join(web, 'public', 'model.onnx'))).digest('hex');
  assert.equal(expected, actual, 'reduced runtime must be rebuilt when model.onnx changes');
});
