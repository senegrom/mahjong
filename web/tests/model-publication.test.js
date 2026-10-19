import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { publish, atomicManifest, listedNetworks, manifestSource } from '../scripts/publish-model-r2.mjs';
import { NETWORKS } from '../src/lib/model-manifest.js';

async function folder(t) {
  const dir = await fs.mkdtemp(join(tmpdir(), 'mahjong-publish-test-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  return dir;
}
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
// The smallest ONNX model the publisher reads: a graph holding one initializer
// of ONNX element type `type`, whose stored bytes are `data`.
function network(type, data) {
  const message = (field, body) => Buffer.concat([Buffer.from([field * 8 + 2, body.length]), body]);
  return message(7, message(5, Buffer.concat([Buffer.from([2 * 8, type]), message(9, Buffer.from(data))])));
}

test('interleaved publishers upload only their own immutable staged bytes', async t => {
  const dir = await folder(t), paths = [join(dir, 'a.onnx'), join(dir, 'b.onnx')];
  await Promise.all(paths.map((path, index) => fs.writeFile(path, network(1, `model ${index}`))));
  let release, entered;
  const paused = new Promise(resolve => entered = resolve), gate = new Promise(resolve => release = resolve);
  const sent = [], staged = [];
  const options = { generation: 36, source: 'test checkpoint', origin: 'https://model.invalid', reward: 'hybrid' };
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
  const good = { modelPath: join(dir, 'missing.onnx'), generation: 36, source: 'actor', reward: 'hybrid',
    origin: 'https://model.invalid', manifestPath, upload: () => uploads++ };
  for (const bad of [{ generation: undefined }, { generation: 'g36' }, { generation: null },
    { generation: true }, { generation: 1.5 }, { generation: -1 }, { generation: 2 ** 53 },
    { origin: undefined }, { origin: 'http://model.invalid' }, { origin: 'https://model.invalid/path' },
    { origin: 'https://user:password@model.invalid' }, { source: '' }, { source: undefined }]) {
    await assert.rejects(publish({ ...good, ...bad }), /generation|origin|checkpoint/);
    assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
  }
  // A version is what a player picks a network by: three whole numbers.
  for (const version of ['', 'v0.1.36', '0.1', '0.1.36.1', '0.01.36', '0.1.-36', ' 0.1.36', 36]) {
    await assert.rejects(publish({ ...good, version }), /Invalid version/, String(version));
    assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
  }
  // The reward is named as the network's trainer was given it: hybrid, or
  // placement with its alpha, a number of 0 or more.
  for (const [reward, alpha, reason] of [[undefined, undefined, /Name the reward/], ['mixed', undefined, /Name the reward/],
    ['Hybrid', undefined, /Name the reward/], ['placement', undefined, /--alpha, the weight/],
    ...[-1, -0.5, NaN, Infinity, '', '-1', 'x', '1e-2', '.5', null, true, [0]].map(alpha => ['placement', alpha, /Invalid --alpha/]),
    ['hybrid', 0, /Only the placement reward/], ['hybrid', '0', /Only the placement reward/]]) {
    await assert.rejects(publish({ ...good, reward, alpha }), reason, `${reward} ${String(alpha)}`);
    assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
  }
  assert.equal(uploads, 0);
});

test('upload and pre-rename failures preserve the previous manifest and clean up', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  await fs.writeFile(manifestPath, 'previous manifest'); await fs.writeFile(modelPath, network(1, 'model'));
  let staged;
  await assert.rejects(publish({ modelPath, manifestPath, generation: 36, source: 'actor', reward: 'hybrid', origin: 'https://model.invalid',
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

test('the manifest records the precision the weights are stored in, or nothing is published', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  let uploads = 0;
  const request = { modelPath, manifestPath, generation: 36, source: 'actor', reward: 'hybrid', origin: 'https://model.invalid',
    upload: () => { uploads++; } };
  for (const [type, precision] of [[1, 'float32'], [3, 'int8'], [10, 'float16']]) {
    await fs.writeFile(modelPath, network(type, `weights of type ${type}`));
    assert.equal((await publish(request)).precision, precision);
    assert.ok((await fs.readFile(manifestPath, 'utf8')).includes(`"precision": "${precision}"`));
  }
  assert.equal(uploads, 3);
  await fs.writeFile(manifestPath, 'previous manifest');
  for (const [bytes, reason] of [['not a network', /not a readable ONNX file/],
    [network(7, 'shape only'), /precision of the network's weights/]]) {
    await fs.writeFile(modelPath, bytes);
    await assert.rejects(publish(request), reason);
  }
  assert.equal(uploads, 3);
  assert.equal(await fs.readFile(manifestPath, 'utf8'), 'previous manifest');
});

/** The manifest at path as the page imports it. */
async function imported(path) {
  return import(`data:text/javascript;base64,${Buffer.from(await fs.readFile(path)).toString('base64')}`);
}

test('a published network becomes the default and every network listed before it stays listed', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  const sent = [];
  const request = { modelPath, manifestPath, source: 'actor', reward: 'hybrid', origin: 'https://model.invalid',
    upload: (_bucket, key) => { sent.push(key); } };
  const published = [];
  for (const generation of [36, 410, 810]) {
    await fs.writeFile(modelPath, network(1, `weights of generation ${generation}`));
    published.push(await publish({ ...request, generation, source: `run/gen-${generation}` }));
  }
  assert.deepEqual(published.map(entry => entry.version), ['0.1.36', '0.1.410', '0.1.810']);
  const { NETWORKS: listed, MANIFEST } = await imported(manifestPath);
  assert.deepEqual(listed.map(entry => entry.version), ['0.1.810', '0.1.410', '0.1.36'], 'the default first, then newest first');
  assert.equal(MANIFEST, listed[0]);
  assert.deepEqual(listed.map(entry => ({ ...entry })), [...published].reverse(), 'each keeps its whole identity');
  assert.deepEqual(sent, published.map(entry => entry.object));
  for (const entry of listed) {
    assert.deepEqual(Object.keys(entry), ['version', 'source', 'generation', 'reward', 'precision', 'bytes', 'storedBytes', 'origin', 'object', 'sha256']);
    assert.equal(entry.object, `models/g${entry.generation}/${entry.sha256}`);
  }
  assert.deepEqual(await listedNetworks(manifestPath), [...published].reverse());
});

test('each network records the reward it learned from, as its trainer was given it, and keeps it on every later publication', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  const request = { modelPath, manifestPath, origin: 'https://model.invalid', upload: () => {} };
  const rewards = async () => (await listedNetworks(manifestPath)).map(entry => [entry.version, entry.reward]);
  // As the command line gives them, --alpha as text, and as a script would.
  for (const [generation, reward, alpha] of [[410, 'hybrid'], [810, 'hybrid'], [1110, 'placement', '0'], [1210, 'placement', 0.05]]) {
    await fs.writeFile(modelPath, network(1, `weights of generation ${generation}`));
    await publish({ ...request, generation, source: `run/gen-${generation}`, reward, alpha });
  }
  const placement = alpha => ({ kind: 'placement', alpha }), hybrid = { kind: 'hybrid' };
  assert.deepEqual(await rewards(), [['0.1.1210', placement(0.05)], ['0.1.1110', placement(0)], ['0.1.810', hybrid], ['0.1.410', hybrid]]);
  // As the page imports them, each reward frozen with its network.
  const { NETWORKS: listed } = await imported(manifestPath);
  for (const entry of listed) assert.ok(Object.isFrozen(entry) && Object.isFrozen(entry.reward), entry.version);
  // Publishing a listed network again as the default moves it first and
  // changes no network's reward.
  await fs.writeFile(modelPath, network(1, 'weights of generation 810'));
  await publish({ ...request, generation: 810, source: 'run/gen-810', reward: 'hybrid' });
  assert.deepEqual(await rewards(), [['0.1.810', hybrid], ['0.1.1210', placement(0.05)], ['0.1.1110', placement(0)], ['0.1.410', hybrid]]);
});

test('a listed version is replaced, listed bytes published again become the default, and a version may be named', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  const request = { modelPath, manifestPath, source: 'actor', reward: 'hybrid', origin: 'https://model.invalid', upload: () => {} };
  const versions = async () => (await listedNetworks(manifestPath)).map(entry => [entry.version, entry.sha256]);
  await fs.writeFile(modelPath, network(1, 'generation 410'));
  const older = await publish({ ...request, generation: 410 });
  await fs.writeFile(modelPath, network(1, 'generation 810'));
  const newer = await publish({ ...request, generation: 810 });
  // Publishing the older network again puts it first, once.
  await fs.writeFile(modelPath, network(1, 'generation 410'));
  await publish({ ...request, generation: 410 });
  assert.deepEqual(await versions(), [['0.1.410', older.sha256], ['0.1.810', newer.sha256]]);
  // Another export of a listed version replaces that entry.
  await fs.writeFile(modelPath, network(10, 'generation 810 in float16'));
  const reexport = await publish({ ...request, generation: 810 });
  assert.equal(reexport.precision, 'float16');
  assert.deepEqual(await versions(), [['0.1.810', reexport.sha256], ['0.1.410', older.sha256]]);
  // A version need not follow the generation.
  await fs.writeFile(modelPath, network(1, 'another line'));
  assert.equal((await publish({ ...request, generation: 12, version: '0.2.12' })).version, '0.2.12');
  assert.deepEqual((await versions()).map(([version]) => version), ['0.2.12', '0.1.810', '0.1.410']);
});

test('the same network under a second version, or a manifest that is not a list, changes nothing', async t => {
  const dir = await folder(t), manifestPath = join(dir, 'manifest.js'), modelPath = join(dir, 'model.onnx');
  const request = { modelPath, manifestPath, generation: 410, source: 'actor', reward: 'hybrid', origin: 'https://model.invalid', upload: () => {} };
  await fs.writeFile(modelPath, network(1, 'generation 410'));
  await publish(request);
  const before = await fs.readFile(manifestPath, 'utf8');
  await assert.rejects(publish({ ...request, version: '0.2.410' }), /listed already, as 0\.1\.410/);
  assert.equal(await fs.readFile(manifestPath, 'utf8'), before);
  // A manifest it cannot read as whole networks would lose every network
  // players picked if it were written over.
  const [listed] = await listedNetworks(manifestPath);
  for (const [text, reason] of [['previous manifest', /could not be read/], ['export const NETWORKS = [];', /lists no networks/],
    ['export const NETWORKS = [{ "version": "0.1.1" }];', /network whose source is not valid/],
    [manifestSource([{ ...listed, object: 'models/g1/elsewhere' }]), /network whose object is not valid/],
    // A network listed without the reward it learned from could not be
    // described to players, so its entry is not whole.
    ...[undefined, 'hybrid', { kind: 'hybrid', alpha: 0 }, { kind: 'placement' }, { kind: 'placement', alpha: '0' },
      { kind: 'placement', alpha: -1 }, { kind: 'hybrid', note: 'mixed' }].map(reward =>
      [manifestSource([{ ...listed, reward }]), /network whose reward is not valid/])]) {
    await fs.writeFile(manifestPath, text);
    await assert.rejects(publish(request), reason);
    assert.equal(await fs.readFile(manifestPath, 'utf8'), text);
  }
  assert.deepEqual(await listedNetworks(join(dir, 'missing.js')), [], 'before the first publication nothing is listed');
});

test('the checked-in manifest is what the publisher writes, and reads back whole', async () => {
  const path = fileURLToPath(new URL('../src/lib/model-manifest.js', import.meta.url));
  // A Windows checkout may end its lines with CR LF; the module is the same.
  assert.equal((await fs.readFile(path, 'utf8')).replaceAll('\r\n', '\n'), manifestSource(NETWORKS.map(entry => ({ ...entry }))));
  assert.deepEqual(await listedNetworks(path), NETWORKS.map(entry => ({ ...entry })));
});
