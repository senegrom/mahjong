/** Verify every shipped model before copying the matching reduced runtime. */
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODEL_FILES, RUNTIME_FILES } from '../src/lib/model-package.js';

export async function copyRuntime(web = fileURLToPath(new URL('../', import.meta.url))) {
  const from = join(web, 'runtime'), to = join(web, 'public', 'ort');
  const expected = new Map();
  for (const line of (await readFile(join(from, 'models.sha256'), 'utf8')).split(/\r?\n/)) {
    if (!line.trim()) continue;
    const match = /^([a-f0-9]{64})\s+([A-Za-z0-9_-]+\.onnx)$/.exec(line.trim());
    if (!match || expected.has(match[2])) throw new Error('Invalid or duplicate model checksum entry');
    expected.set(match[2], match[1]);
  }
  const supported = Object.values(MODEL_FILES).sort();
  const shipped = (await readdir(join(web, 'public'))).filter(name => name.endsWith('.onnx')).sort();
  if (JSON.stringify([...expected.keys()].sort()) !== JSON.stringify(supported)
      || JSON.stringify(shipped) !== JSON.stringify(supported)) {
    throw new Error('Shipped models, models.sha256 and MODEL_FILES must name exactly the same networks');
  }
  for (const name of supported) {
    const actual = createHash('sha256').update(await readFile(join(web, 'public', name))).digest('hex');
    if (expected.get(name) !== actual) throw new Error(`${name} changed; rebuild web/runtime before publishing`);
  }
  // Validate every source first, so an incomplete package cannot partly replace the runtime.
  const sources = RUNTIME_FILES.map(file => [file, file === 'memory-budget.mjs'
    ? join(web, 'src', 'lib', 'memory-budget.js') : join(from, file)]);
  for (const [, source] of sources) await stat(source);
  await mkdir(to, { recursive: true });
  for (const [file, source] of sources) await copyFile(source, join(to, file));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await copyRuntime();
  console.log('Verified trained model checksums and copied the reduced runtime');
}
