import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile, stat } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RUNTIME_FILES, WEBGPU_RUNTIME_FILES } from '../src/lib/model-package.js';
import { TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { runtimeDirectory, webgpuDirectory } from '../scripts/copy-runtime.mjs';

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
  // One runtime, in the folder named by its contents, which the page was built to load.
  const runtime = await runtimeDirectory(web);
  assert.deepEqual(names.filter(name => name.startsWith('ort/')).sort(), RUNTIME_FILES.map(file => `${runtime}${file}`).sort());
  // The WebGPU runtime, onnxruntime-web's own files, in a content-named folder of its own.
  const webgpu = await webgpuDirectory(web);
  assert.deepEqual(names.filter(name => name.startsWith('ort-webgpu/')).sort(), WEBGPU_RUNTIME_FILES.map(file => `${webgpu}${file}`).sort());
  for (const file of WEBGPU_RUNTIME_FILES) {
    assert.deepEqual(await readFile(join(dist, webgpu, file)), await readFile(join(web, 'node_modules/onnxruntime-web/dist', file)), file);
  }
  // The module the worker imports loads the very files shipped beside it,
  // and the worker imports that module.
  const [module, ...beside] = WEBGPU_RUNTIME_FILES;
  const loads = new Set((await readFile(join(dist, webgpu, module), 'utf8')).match(/ort-wasm-simd-threaded[.a-z]*\.(?:mjs|wasm)/g));
  assert.deepEqual([...loads].filter(name => name.endsWith('.mjs')), beside.filter(name => name.endsWith('.mjs')));
  assert.ok(beside.every(name => name.endsWith('.mjs') || loads.has(name) || loads.has(name.replace(/\.wasm$/, '.mjs'))));
  // The card's worker imports that module; the WebAssembly worker never does.
  const chunk = pattern => names.filter(name => pattern.test(name));
  const [cardChunk] = chunk(/^assets\/gpu-batch\.worker-[^/]+\.js$/), [playChunk] = chunk(/^assets\/policy\.worker-[^/]+\.js$/);
  assert.ok((await readFile(join(dist, cardChunk), 'utf8')).includes(module), 'the card\'s worker imports the shipped module');
  assert.equal((await readFile(join(dist, playChunk), 'utf8')).includes(module), false, 'play never loads it');
  const page = names.filter(name => /^assets\/index-[^/]+\.js$/.test(name));
  assert.equal(page.length, 1);
  assert.ok((await readFile(join(dist, page[0]), 'utf8')).includes(webgpu), 'the page names the shipped WebGPU runtime folder');
  // The AI package every player saves for offline play leaves the WebGPU
  // runtime out; the service worker saves its files as a page fetches them.
  const offline = JSON.parse(await readFile(join(dist, 'offline-manifest.json'), 'utf8'));
  assert.deepEqual(offline.entries.filter(entry => entry.url.startsWith('ort-webgpu/')).map(entry => entry.group), WEBGPU_RUNTIME_FILES.map(() => 'webgpu'));
  assert.equal(offline.entries.filter(entry => entry.group === 'ai').some(entry => entry.url.startsWith('ort-webgpu/')), false);
  assert.ok(offline.bytes.webgpu > 20e6 && offline.bytes.runtime < 10e6, JSON.stringify(offline.bytes));
  assert.ok((await readFile(join(dist, page[0]), 'utf8')).includes(runtime), 'the page names the shipped runtime folder');
  for (const file of TILE_IMAGE_URLS) assert.ok(names.includes(file), `Missing playable artwork: ${file}`);
  assert.equal(names.some(name => /\.onnx$/i.test(name)), false, 'the only model is remote');
  assert.equal(names.some(name => /^tiles\/.*\.png$/.test(name)), false, 'PNG crops duplicate the unchanged embedded SVG pixels');
  assert.equal(names.some(name => /^tiles\/.*(?:README\.md|preview\.html|manifest\.json)$/.test(name)), false);
  const reduced = (await stat(join(web, 'runtime/ort-wasm-simd-threaded.wasm'))).size;
  const generic = (await stat(join(web, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'))).size;
  assert.ok(reduced < generic, `reduced runtime ${reduced} must be smaller than generic ${generic}`);
});
