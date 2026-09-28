import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { copyRuntime, runtimeDirectory, runtimeRelease } from '../scripts/copy-runtime.mjs';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-runtime-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const dir of ['runtime', 'public', 'src/lib']) await mkdir(join(root, dir), { recursive: true });
  for (const file of RUNTIME_FILES.filter(file => file !== 'memory-budget.mjs')) await writeFile(join(root, 'runtime', file), file);
  await writeFile(join(root, 'src/lib/memory-budget.js'), 'memory');
  await release(root, '1.29.0', '1.29.0');
  return root;
}
// A runtime built from ONNX Runtime `built`, beside onnxruntime-web `installed`.
async function release(root, built, installed) {
  await writeFile(join(root, 'runtime/ort-wasm-simd-threaded.wasm'), `\0asm\0${built}\0kernels`);
  await mkdir(join(root, 'node_modules/onnxruntime-web'), { recursive: true });
  await writeFile(join(root, 'node_modules/onnxruntime-web/package.json'), JSON.stringify({ version: installed }));
}

test('a clean checkout copies the complete runtime without a local-model registry', async t => {
  const root = await fixture(t);
  const { directory } = await copyRuntime(root);
  assert.match(directory, /^ort\/[0-9a-f]{16}\/$/);
  for (const name of RUNTIME_FILES) assert.ok((await readFile(join(root, 'public', directory, name))).length);
});
test('the runtime folder is named by its contents, and only the current one is published', async t => {
  const root = await fixture(t);
  const first = await copyRuntime(root);
  assert.equal(first.directory, await runtimeDirectory(root));
  assert.deepEqual(await copyRuntime(root), first, 'the same files keep their address');
  // Any of the three files changing moves all three: the loader imports the
  // allocator controls from beside itself.
  await writeFile(join(root, 'src/lib/memory-budget.js'), 'memory, revised');
  const second = await copyRuntime(root);
  assert.notEqual(second.directory, first.directory);
  assert.deepEqual(await readdir(join(root, 'public/ort')), [second.directory.split('/')[1]]);
  assert.equal(await readFile(join(root, 'public', second.directory, 'memory-budget.mjs'), 'utf8'), 'memory, revised');
});
test('incomplete runtime sources never partly replace an existing copy', async t => {
  const root = await fixture(t);
  const { directory } = await copyRuntime(root);
  const target = join(root, 'public', directory, RUNTIME_FILES[0]);
  await writeFile(target, 'previous runtime');
  await rm(join(root, 'src/lib/memory-budget.js'));
  await assert.rejects(copyRuntime(root), /ENOENT/);
  assert.equal(await readFile(target, 'utf8'), 'previous runtime');
});
test('unexpected ONNX files at any public path fail before copying', async t => {
  const root = await fixture(t);
  for (const name of ['model.onnx', 'old/model-full.onnx', 'old/MODEL.ONNX']) {
    const path = join(root, 'public', name);
    await mkdir(join(path, '..'), { recursive: true }); await writeFile(path, 'legacy');
    await assert.rejects(copyRuntime(root), /Unexpected local ONNX/);
    await assert.rejects(readdir(join(root, 'public/ort')), { code: 'ENOENT' });
    await rm(path);
  }
});
test('the installed JavaScript API must be the release the runtime was built from', async t => {
  const root = await fixture(t);
  await release(root, '1.29.0', '1.30.0');
  await assert.rejects(copyRuntime(root), /onnxruntime-web 1\.30\.0 is installed, but web\/runtime was built from ONNX Runtime 1\.29\.0/);
  await assert.rejects(readdir(join(root, 'public/ort')), { code: 'ENOENT' });
  await release(root, '', '1.29.0');
  await assert.rejects(copyRuntime(root), /built from ONNX Runtime an unknown release/);
});
test('the committed runtime was built from the exactly pinned onnxruntime-web', async () => {
  const web = fileURLToPath(new URL('../', import.meta.url));
  const pinned = JSON.parse(await readFile(join(web, 'package.json'), 'utf8')).dependencies['onnxruntime-web'];
  assert.match(pinned, /^\d+\.\d+\.\d+$/, 'onnxruntime-web moves only with a runtime rebuild');
  assert.equal(await runtimeRelease(web), pinned);
});
