import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { publish, atomicManifest } from '../scripts/publish-model-r2.mjs';

async function folder(t) {
  const dir = await fs.mkdtemp(join(tmpdir(), 'mahjong-publish-test-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  return dir;
}
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

test('interleaved publishers upload only their own immutable staged bytes', async t => {
  const dir = await folder(t), paths = [join(dir, 'a.onnx'), join(dir, 'b.onnx')];
  await Promise.all(paths.map((path, index) => fs.writeFile(path, `model ${index}`)));
  let release, entered;
  const paused = new Promise(resolve => entered = resolve), gate = new Promise(resolve => release = resolve);
  const sent = [], staged = [];
  const options = { generation: 36, source: 'test checkpoint', origin: 'https://model.invalid' };
  const a = publish({ ...options, modelPath: paths[0], manifestPath: join(dir, 'a.js'),
    upload: async (_bucket, key, file) => {
      staged.push(file); entered(); await gate;
      sent.push({ key, bytes: gunzipSync(await fs.readFile(file)) });
    } });
  await paused;
  const b = await publish({ ...options, modelPath: paths[1], manifestPath: join(dir, 'b.js'),
    upload: async (_bucket, key, file) => {
      staged.push(file); sent.push({ key, bytes: gunzipSync(await fs.readFile(file)) });
    } });
  release(); const first = await a;
  for (const item of sent) assert.ok(item.key.endsWith(sha(item.bytes)));
  assert.notEqual(first.sha256, b.sha256);
  assert.notEqual(staged[0], staged[1]);
  for (const file of staged) await assert.rejects(fs.stat(file), { code: 'ENOENT' });
});

test('invalid publication requests fail before uploading or changing the manifest', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js');
  await fs.writeFile(manifestPath, 'previous manifest');
  let uploads = 0;
  const good = { modelPath: join(dir, 'missing.onnx'), generation: 36, source: 'actor',
    origin: 'https://model.invalid', manifestPath, upload: () => uploads++ };
  for (const bad of [{ generation: undefined }, { generation: 'g36' }, { generation: null },
    { generation: true }, { generation: 1.5 }, { generation: -1 }, { generation: 2 ** 53 },
    { origin: undefined }, { origin: 'http://model.invalid' }, { origin: 'https://model.invalid/path' },
    { origin: 'https://user:password@model.invalid' }, { source: '' }]) {
    await assert.rejects(publish({ ...good, ...bad }), /generation|origin|checkpoint/);
    assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
  }
  assert.equal(uploads, 0);
});

test('upload and pre-rename failures preserve the previous manifest and clean up', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  await fs.writeFile(manifestPath, 'previous manifest'); await fs.writeFile(modelPath, 'model');
  let staged;
  await assert.rejects(publish({ modelPath, manifestPath, generation: 36, source: 'actor', origin: 'https://model.invalid',
    upload: (_bucket, _key, file) => { staged = file; throw new Error('upload failed'); } }), /upload failed/);
  await assert.rejects(fs.stat(staged), { code: 'ENOENT' });
  await assert.rejects(atomicManifest(manifestPath, 'complete new manifest', {
    ...fs, rename: () => { throw new Error('rename failed'); },
  }), /rename failed/);
  assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
  assert.deepEqual((await fs.readdir(dir)).sort(), ['manifest.js', 'model.onnx']);
  await atomicManifest(manifestPath, 'complete new manifest');
  assert.equal(await fs.readFile(manifestPath, 'utf8'), 'complete new manifest');
});
