// Publish a checked ONNX export under its digest, then atomically publish its
// manifest. Uploading does not certify export parity or playing strength.
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, open, rename, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { gzip } from 'node:zlib';
import { dirname, join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const MANIFEST = join(ROOT, 'web/src/lib/model-manifest.js');
const compress = promisify(gzip);
const run = promisify(execFile);

function option(name, fallback) {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 && at + 1 < process.argv.length ? process.argv[at + 1] : fallback;
}

async function uploadR2(bucket, key, file) {
  const { stdout, stderr } = await run('npx', ['--yes', 'wrangler', 'r2', 'object', 'put',
    `${bucket}/${key}`, '--file', file, '--content-type', 'application/octet-stream',
    '--content-encoding', 'gzip', '--remote',
  ], { cwd: ROOT, shell: process.platform === 'win32', maxBuffer: 1 << 24 });
  process.stdout.write(stdout || stderr);
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

export async function publish({ modelPath, generation, source, origin,
  bucket = process.env.R2_BUCKET ?? 'mahjong-models', manifestPath = MANIFEST,
  upload = uploadR2 } = {}) {
  // Validate before reading/staging/uploading anything. Never coerce undefined
  // into a key or NaN into a manifest's null generation.
  if (typeof bucket !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$/.test(bucket)) throw new Error('Invalid R2 bucket name.');
  if ((typeof generation !== 'number' && !(typeof generation === 'string' && /^[0-9]+$/.test(generation)))
      || !Number.isSafeInteger(Number(generation)) || Number(generation) < 0) throw new Error('Invalid generation.');
  generation = Number(generation);
  if (typeof modelPath !== 'string' || !modelPath) throw new Error('Name the exported network to publish.');
  if (typeof source !== 'string' || !source.trim()) throw new Error('Name the source checkpoint.');
  let address;
  try { address = new URL(origin); } catch { throw new Error('A valid HTTPS model origin is required.'); }
  if (address.protocol !== 'https:' || address.username || address.password
      || address.pathname !== '/' || address.search || address.hash) throw new Error('A bare HTTPS model origin is required.');
  origin = address.origin;
  const bytes = await readFile(resolve(modelPath));
  if (!bytes.length || bytes.length > 512 * 1024 * 1024) throw new Error('Unsupported model byte size.');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const key = `models/g${generation}/${sha256}`;
  const packed = await compress(bytes, { level: 9 });
  const scratch = await mkdtemp(join(tmpdir(), 'mahjong-model-upload-'));
  try {
    const staged = join(scratch, 'network.gz');
    const file = await open(staged, 'wx');
    try { await file.writeFile(packed); await file.sync(); } finally { await file.close(); }
    await upload(bucket, key, staged);
    const manifest = { source, generation, precision: 'float32', bytes: bytes.length,
      storedBytes: packed.length, origin, object: key, sha256 };
    await atomicManifest(manifestPath,
      `/** Published model identity; verify bytes before use. */\nexport const MANIFEST = Object.freeze(${JSON.stringify(manifest, null, 2)});\n`);
    return manifest;
  } finally { await rm(scratch, { recursive: true, force: true }); }
}

const invoked = process.argv[1] ? process.argv[1].split(/[/\\]/).pop() : '';
if (invoked === 'publish-model-r2.mjs') {
  const manifest = await publish({ modelPath: process.argv[2], generation: option('generation'),
    source: option('source', 'unknown'), origin: option('origin') });
  process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
}
