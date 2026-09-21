import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { copyRuntime } from '../scripts/copy-runtime.mjs';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-runtime-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const dir of ['runtime', 'public', 'src/lib']) await mkdir(join(root, dir), { recursive: true });
  for (const file of RUNTIME_FILES.filter(file => file !== 'memory-budget.mjs')) await writeFile(join(root, 'runtime', file), file);
  await writeFile(join(root, 'src/lib/memory-budget.js'), 'memory');
  return root;
}

test('a clean checkout copies the complete runtime without a local-model registry', async t => {
  const root = await fixture(t);
  await copyRuntime(root);
  for (const name of RUNTIME_FILES) assert.ok((await readFile(join(root, 'public/ort', name))).length);
});
test('incomplete runtime sources never partly replace an existing copy', async t => {
  const root = await fixture(t);
  await copyRuntime(root);
  const target = join(root, 'public/ort', RUNTIME_FILES[0]);
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
    await assert.rejects(readFile(join(root, 'public/ort', RUNTIME_FILES[0])), { code: 'ENOENT' });
    await rm(path);
  }
});
