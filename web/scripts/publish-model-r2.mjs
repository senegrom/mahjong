// Publish a checked ONNX export under its digest, then atomically publish the
// manifest that lists it as the default network, with every network listed
// before it kept for players to pick by version. Uploading does not certify
// export parity or playing strength.
//
//   node web/scripts/publish-model-r2.mjs network.onnx --generation <n> \
//     --source <checkpoint> --origin https://<model host> \
//     --reward hybrid | --reward placement --alpha <weight> [--version <a.b.c>]
//
// The version is 0.1.<generation> unless --version names another. --reward
// and --alpha name the reward the network learned from, as its trainer was
// given them (neural/train_combined.py), which the page says beside the
// network's value. Publishing a version that is listed already replaces that
// entry, as a new export of the same network; publishing the listed bytes
// again makes them the default. Publish one network at a time: each
// publication reads the list the last one wrote.
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, open, rename, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { gzip } from 'node:zlib';
import { dirname, join, resolve, win32 } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual, promisify } from 'node:util';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const MANIFEST = join(ROOT, 'web/src/lib/model-manifest.js');
const compress = promisify(gzip);
const run = promisify(execFile);
// The Cloudflare CLI runs with the account's credentials, so it is pinned to
// an exact release rather than whatever is newest when it runs. Keep it in
// step with workers/model-cdn/wrangler.jsonc.
const WRANGLER = 'wrangler@4.131.2';
// ONNX element types (TensorProto.DataType) named by the precision they store.
const PRECISIONS = new Map([[1, 'float32'], [10, 'float16'], [16, 'bfloat16'], [3, 'int8'], [2, 'uint8']]);
// TensorProto fields that hold a tensor's values rather than its description.
const TENSOR_DATA = new Set([4, 5, 6, 7, 9, 10, 11]);

function option(name, fallback) {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 && at + 1 < process.argv.length ? process.argv[at + 1] : fallback;
}

const unreadable = () => new Error('The model is not a readable ONNX file.');

function varint(bytes, at) {
  let value = 0, scale = 1;
  for (;;) {
    if (at >= bytes.length) throw unreadable();
    const byte = bytes[at++];
    value += (byte & 0x7f) * scale;
    if (byte < 0x80) return [value, at];
    scale *= 128;
  }
}

/** The fields of the protobuf message between start and end: a number for a
 * varint field, a start and end for a length-delimited one. Fixed-width
 * fields are skipped: ONNX packs its numeric arrays into length-delimited
 * fields. */
function* fields(bytes, start, end) {
  let at = start, key, value;
  while (at < end) {
    [key, at] = varint(bytes, at);
    const field = Math.floor(key / 8), wire = key % 8;
    if (wire === 0) {
      [value, at] = varint(bytes, at);
      yield { field, value };
    } else if (wire === 2) {
      [value, at] = varint(bytes, at);
      if (at + value > end) throw unreadable();
      yield { field, start: at, end: at + value };
      at += value;
    } else if (wire === 1 || wire === 5) at += wire === 1 ? 8 : 4;
    else throw unreadable();
  }
  if (at > end) throw unreadable();
}

/** The precision the network's weights are stored in, read from the model
 * rather than assumed: the element type of the initializers holding the most
 * bytes (ModelProto.graph is field 7, GraphProto.initializer field 5 and
 * TensorProto.data_type field 2). An int8 export keeps float scales and
 * biases beside its int8 weights; the weights outweigh them. */
export function weightPrecision(bytes) {
  const stored = new Map();
  for (const graph of fields(bytes, 0, bytes.length)) {
    if (graph.field !== 7 || graph.start === undefined) continue;
    for (const tensor of fields(bytes, graph.start, graph.end)) {
      if (tensor.field !== 5 || tensor.start === undefined) continue;
      let type, size = 0;
      for (const part of fields(bytes, tensor.start, tensor.end)) {
        if (part.field === 2 && part.value !== undefined) type = part.value;
        else if (TENSOR_DATA.has(part.field) && part.start !== undefined) size += part.end - part.start;
      }
      stored.set(type, (stored.get(type) ?? 0) + size);
    }
  }
  const [largest] = [...stored].sort((a, b) => b[1] - a[1]);
  if (!largest || !PRECISIONS.has(largest[0])) throw new Error("Could not tell the precision of the network's weights.");
  return PRECISIONS.get(largest[0]);
}

/** The program and arguments that run npx, never through a shell: a shell
 * joins the arguments unquoted, so a staged file under a temporary directory
 * with a space in its path would split. On Windows npx is a batch file, which
 * Node refuses to start without a shell (EINVAL), so node runs the script
 * that npx.cmd itself runs, which npm installs beside node. */
export function npxCommand(args, platform = process.platform, node = process.execPath) {
  if (platform !== 'win32') return ['npx', args];
  const script = win32.join(win32.dirname(node), 'node_modules', 'npm', 'bin', 'npx-cli.js');
  return [node, [script, ...args]];
}

async function uploadR2(bucket, key, file) {
  const [command, args] = npxCommand(['--yes', WRANGLER, 'r2', 'object', 'put',
    `${bucket}/${key}`, '--file', file, '--content-type', 'application/octet-stream',
    '--content-encoding', 'gzip', '--remote']);
  const { stdout, stderr } = await run(command, args, { cwd: ROOT, maxBuffer: 1 << 24 });
  process.stdout.write(stdout || stderr);
}

// A version players pick a network by: three whole numbers, such as 0.1.810.
const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;

/** The reward a network learned from, as the manifest records it, from the
 * options its trainer was given (neural/train_combined.py --reward, --alpha):
 * { kind: 'hybrid' }, paying a decision its hand's points over 4,000 plus the
 * game's placement bonus, or { kind: 'placement', alpha }, paying it the bonus
 * the scoreboard promises at the start of the next hand, and after the game's
 * last hand the bonus itself, plus alpha times the hand's points over 4,000.
 * The value the network answers is an estimate of that reward, and the page
 * says which in words (trained-network.js). Throws saying what is missing. */
export function rewardOf(kind, alpha) {
  if (kind === 'hybrid') {
    if (alpha !== undefined) throw new Error('Only the placement reward takes --alpha.');
    return { kind };
  }
  if (kind !== 'placement') throw new Error('Name the reward the network learned from: --reward hybrid, or --reward placement with its --alpha.');
  if (alpha === undefined) throw new Error("Name the placement reward's --alpha, the weight it gave the hand's points: 0 for placement alone.");
  const weight = typeof alpha === 'string' && /^[0-9]+(\.[0-9]+)?$/.test(alpha) ? Number(alpha) : alpha;
  if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0) {
    throw new Error('Invalid --alpha: name it as a number of 0 or more, such as 0 or 0.05.');
  }
  return { kind, alpha: weight };
}

/** Whether a manifest entry is a whole network identity, as publish writes
 * one. Throws naming the first field that is not. */
function checkNetwork(network) {
  const bad = field => new Error(`The model manifest lists a network whose ${field} is not valid.`);
  if (!network || typeof network !== 'object') throw bad('entry');
  if (typeof network.version !== 'string' || !VERSION.test(network.version)) throw bad('version');
  if (typeof network.source !== 'string' || !network.source.trim()) throw bad('source');
  if (!Number.isSafeInteger(network.generation) || network.generation < 0) throw bad('generation');
  // A network listed without its reward could not be described to players,
  // so a manifest that lacks one is not written over either.
  let reward;
  try { reward = rewardOf(network.reward?.kind, network.reward?.alpha); } catch { throw bad('reward'); }
  if (!isDeepStrictEqual({ ...network.reward }, reward)) throw bad('reward');
  if (![...PRECISIONS.values()].includes(network.precision)) throw bad('precision');
  for (const field of ['bytes', 'storedBytes']) {
    if (!Number.isSafeInteger(network[field]) || network[field] <= 0) throw bad(field);
  }
  if (typeof network.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(network.sha256)) throw bad('sha256');
  if (network.object !== `models/g${network.generation}/${network.sha256}`) throw bad('object');
  let origin;
  try { origin = new URL(network.origin); } catch { throw bad('origin'); }
  if (origin.protocol !== 'https:' || origin.origin !== network.origin) throw bad('origin');
}

/** The manifest module for these networks, the default first: what publish
 * writes, and reads back when it publishes the next one. */
export function manifestSource(networks) {
  return `/** Which trained networks this page can play, and where their bytes are.
 *
 * Written by \`web/scripts/publish-model-r2.mjs\` when a network is published:
 * the network published last comes first and is the default, and the others
 * stay listed, newest first, for a player to pick by version. Each object key
 * carries the network's own digest, so the response may be immutable and a
 * different export is a different address; \`network-store.js\` checks the
 * digest again before anything runs. Each names the reward it learned from,
 * which the page says beside the network's value.
 */
export const NETWORKS = Object.freeze(${JSON.stringify(networks, null, 2)}.map(network => {
  Object.freeze(network.reward);
  return Object.freeze(network);
}));

/** The default network: the one a player who has picked none plays. */
export const MANIFEST = NETWORKS[0];
`;
}

/** The networks a manifest lists, the default first; none before the first
 * publication. A manifest that cannot be read as a list of whole networks
 * stops the publication: writing over it would drop networks players picked. */
export async function listedNetworks(path) {
  let text;
  try { text = await readFile(path, 'utf8'); }
  catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
  let networks;
  try {
    // Its own module, evaluated by itself: the manifest imports nothing.
    ({ NETWORKS: networks } = await import(`data:text/javascript;base64,${Buffer.from(text).toString('base64')}`));
  } catch (error) { throw new Error(`The model manifest could not be read: ${error.message}`, { cause: error }); }
  if (!Array.isArray(networks) || !networks.length) throw new Error('The model manifest lists no networks.');
  for (const network of networks) checkNetwork(network);
  for (const field of ['version', 'sha256']) {
    if (new Set(networks.map(network => network[field])).size !== networks.length) {
      throw new Error(`The model manifest lists one ${field} twice.`);
    }
  }
  return networks.map(network => ({ ...network }));
}

/** A complete, synced manifest is the publication boundary. Concurrent writers
 * can finish in either order, but every visible manifest is a complete snapshot. */
export async function atomicManifest(path, body, io = { open, rename, mkdir, mkdtemp, rm }) {
  path = resolve(path);
  await io.mkdir(dirname(path), { recursive: true });
  const dir = await io.mkdtemp(join(dirname(path), '.model-manifest-'));
  try {
    const staged = join(dir, 'manifest.js');
    const file = await io.open(staged, 'wx');
    try { await file.writeFile(body, 'utf8'); await file.sync(); }
    finally { await file.close(); }
    await io.rename(staged, path);
    if (process.platform !== 'win32') {
      const parent = await io.open(dirname(path), 'r');
      try { await parent.sync(); } finally { await parent.close(); }
    }
  } finally { await io.rm(dir, { recursive: true, force: true }); }
}

export async function publish({ modelPath, generation, source, origin, version, reward, alpha,
  bucket = process.env.R2_BUCKET ?? 'mahjong-models', manifestPath = MANIFEST,
  upload = uploadR2 } = {}) {
  // Validate before reading/staging/uploading anything. Never coerce undefined
  // into a key or NaN into a manifest's null generation.
  if (typeof bucket !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$/.test(bucket)) throw new Error('Invalid R2 bucket name.');
  if ((typeof generation !== 'number' && !(typeof generation === 'string' && /^[0-9]+$/.test(generation)))
      || !Number.isSafeInteger(Number(generation)) || Number(generation) < 0) throw new Error('Invalid generation.');
  generation = Number(generation);
  version ??= `0.1.${generation}`;
  if (typeof version !== 'string' || !VERSION.test(version)) throw new Error('Invalid version: name it as three numbers, such as 0.1.810.');
  if (typeof modelPath !== 'string' || !modelPath) throw new Error('Name the exported network to publish.');
  if (typeof source !== 'string' || !source.trim()) throw new Error('Name the source checkpoint.');
  const learned = rewardOf(reward, alpha);
  let address;
  try { address = new URL(origin); } catch { throw new Error('A valid HTTPS model origin is required.'); }
  if (address.protocol !== 'https:' || address.username || address.password
      || address.pathname !== '/' || address.search || address.hash) throw new Error('A bare HTTPS model origin is required.');
  origin = address.origin;
  const bytes = await readFile(resolve(modelPath));
  if (!bytes.length || bytes.length > 512 * 1024 * 1024) throw new Error('Unsupported model byte size.');
  const precision = weightPrecision(bytes);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const key = `models/g${generation}/${sha256}`;
  const packed = await compress(bytes, { level: 9 });
  const scratch = await mkdtemp(join(tmpdir(), 'mahjong-model-upload-'));
  try {
    const staged = join(scratch, 'network.gz');
    const file = await open(staged, 'wx');
    try { await file.writeFile(packed); await file.sync(); } finally { await file.close(); }
    await upload(bucket, key, staged);
    const network = { version, source, generation, reward: learned, precision, bytes: bytes.length,
      storedBytes: packed.length, origin, object: key, sha256 };
    // Read just before the write that replaces it. The upload is immutable
    // and named by its digest, so one that ends up unlisted costs nothing.
    const listed = await listedNetworks(manifestPath);
    const twin = listed.find(entry => entry.sha256 === sha256 && entry.version !== version);
    if (twin) throw new Error(`This network is listed already, as ${twin.version}; publish it as that version to make it the default.`);
    await atomicManifest(manifestPath, manifestSource([network, ...listed.filter(entry => entry.version !== version)]));
    return network;
  } finally { await rm(scratch, { recursive: true, force: true }); }
}

const invoked = process.argv[1] ? process.argv[1].split(/[/\\]/).pop() : '';
if (invoked === 'publish-model-r2.mjs') {
  const network = await publish({ modelPath: process.argv[2], generation: option('generation'),
    source: option('source'), origin: option('origin'), version: option('version'),
    reward: option('reward'), alpha: option('alpha') });
  process.stdout.write(`${JSON.stringify(network, null, 2)}\n`);
  const listed = await listedNetworks(MANIFEST);
  process.stdout.write(`Listed: ${listed.map((entry, index) => index ? entry.version : `${entry.version} (default)`).join(', ')}\n`);
}
