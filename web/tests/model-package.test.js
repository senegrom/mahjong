import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { copyRuntime } from '../scripts/copy-runtime.mjs';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'mahjong-model-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const dir of ['runtime', 'public', 'src/lib']) await mkdir(join(root, dir), { recursive: true });
  // The trained network is not a file of this site: it comes from its bucket
  // and is kept in Cache Storage (src/lib/network-store.js). What ships here
  // is the runtime, and the record of checked networks is therefore empty.
  await writeFile(join(root, 'runtime', 'models.sha256'), '');
  for (const file of RUNTIME_FILES.filter(file => file !== 'memory-budget.mjs')) {
    await writeFile(join(root, 'runtime', file), file);
  }
  await writeFile(join(root, 'src/lib/memory-budget.js'), 'memory');
  return { root, hash: createHash('sha256').update('network').digest('hex') };
}

test('a site that ships no network copies the complete runtime on a clean checkout', async t => {
  const { root } = await fixture(t);
  await copyRuntime(root);
  for (const name of RUNTIME_FILES) assert.ok((await readFile(join(root, 'public/ort', name))).length);
});
test('a record naming a network no longer shipped fails before copying', async t => {
  const { root, hash } = await fixture(t);
  for (const text of [
    `${hash}  model-full.onnx
`,
    `${hash}  model-full.onnx
${hash}  model-full.onnx
`,
    `${hash}  ../escape.onnx
`,
  ]) {
    await writeFile(join(root, 'runtime/models.sha256'), text);
    await assert.rejects(copyRuntime(root), /Invalid|exactly/);
    await assert.rejects(readFile(join(root, 'public/ort', RUNTIME_FILES[0])), { code: 'ENOENT' });
  }
});
test('a network cannot slip into the site unchecked', async t => {
  const { root } = await fixture(t);
  await writeFile(join(root, 'public/model.onnx'), 'legacy');
  await assert.rejects(copyRuntime(root), /exactly/);
});
