// Publishes an exported network to the R2 bucket the page reads it from, under
// a content-addressed key, and writes the manifest the page ships.
//
//   node web/scripts/publish-model-r2.mjs fusion.onnx --generation 36 \
//     --source leashed-run/latest --origin https://mahjong-model.<account>.workers.dev
//
// The bytes are hashed before anything is uploaded and the key carries that
// digest, so one URL can only ever answer with one network and a response may
// be immutable. The object is stored gzipped because R2 serves exactly the
// bytes it holds and compresses nothing on the way out.
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { gzip } from 'node:zlib';
import { dirname, join, resolve } from 'node:path';
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

export async function publish({ modelPath, generation, source, origin,
  bucket = process.env.R2_BUCKET ?? 'mahjong-models', manifestPath = MANIFEST,
  upload = uploadR2 } = {}) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$/.test(bucket)) throw new Error('Invalid R2 bucket name.');
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(String(generation))) throw new Error('Invalid generation.');
  const bytes = await readFile(resolve(modelPath));
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const key = `models/g${generation}/${sha256}`;
  const packed = await compress(bytes, { level: 9 });
  const staged = join(ROOT, 'web', 'model-upload.gz');
  await writeFile(staged, packed);
  await upload(bucket, key, staged);
  const manifest = {
    source, generation: Number(generation),
    precision: 'float32', bytes: bytes.length, storedBytes: packed.length,
    origin, object: key, sha256,
  };
  // A module rather than JSON: every tool here reads one without an import
  // attribute, and the page's bundler inlines it.
  const note = `/** Which trained network this page plays, and where its bytes are.
 *
 * Written by web/scripts/publish-model-r2.mjs when a network is published. The
 * key carries the network's own digest, so the response may be immutable and a
 * different export is a different address; network-store.js checks that digest
 * again before anything runs.
 */
`;
  const body = `${note}export const MANIFEST = Object.freeze(${JSON.stringify(manifest, null, 2)});
`;
  await writeFile(manifestPath, body, 'utf8');
  return manifest;
}

const invoked = process.argv[1] ? process.argv[1].split(/[/\\]/).pop() : '';
if (invoked === 'publish-model-r2.mjs') {
  const modelPath = process.argv[2];
  if (!modelPath) throw new Error('Name the exported network to publish.');
  const manifest = await publish({
    modelPath,
    generation: option('generation'),
    source: option('source', 'unknown'),
    origin: option('origin'),
  });
  process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
}
