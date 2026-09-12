import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { copyRuntime } from '../scripts/copy-runtime.mjs';
import { MODEL_FILES, RUNTIME_FILES } from '../src/lib/model-package.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-model-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const dir of ['runtime', 'public', 'src/lib']) await mkdir(join(root, dir), { recursive: true });
  const hash = createHash('sha256').update('network').digest('hex');
  await writeFile(join(root, 'public', MODEL_FILES.full), 'network');
  await writeFile(join(root, 'runtime', 'models.sha256'), `${hash}  ${MODEL_FILES.full}\n`);
  for (const file of RUNTIME_FILES.filter(file => file !== 'memory-budget.mjs')) {
    await writeFile(join(root, 'runtime', file), file);
  }
  await writeFile(join(root, 'src/lib/memory-budget.js'), 'memory');
  return { root, hash };
}

test('current checksum manifest copies the complete runtime on a clean checkout', async t => {
  const { root } = await fixture(t);
  await copyRuntime(root);
  for (const name of RUNTIME_FILES) assert.ok((await readFile(join(root, 'public/ort', name))).length);
});
test('changed models and unknown or duplicate manifest entries fail before copying', async t => {
  const { root, hash } = await fixture(t);
  await writeFile(join(root, 'public', MODEL_FILES.full), 'changed');
  await assert.rejects(copyRuntime(root), /changed/);
  await assert.rejects(readFile(join(root, 'public/ort', RUNTIME_FILES[0])), { code: 'ENOENT' });
  await writeFile(join(root, 'public', MODEL_FILES.full), 'network');
  for (const text of [
    `${hash}  ${MODEL_FILES.full}\n${hash}  ${MODEL_FILES.full}\n`,
    `${hash}  ../escape.onnx\n`, `${hash}  model.onnx\n`, '',
  ]) {
    await writeFile(join(root, 'runtime/models.sha256'), text);
    await assert.rejects(copyRuntime(root), /Invalid|exactly/);
  }
});
test('unlisted model files cannot slip into the site', async t => {
  const { root } = await fixture(t);
  await writeFile(join(root, 'public/model.onnx'), 'legacy');
  await assert.rejects(copyRuntime(root), /exactly/);
});
