/** Copy the model-specific ONNX Runtime build next to the site.
 *
 * The runtime is built from ONNX Runtime 1.29.0 with only the operators named
 * in runtime/reduced-ops.config. Keep the models themselves as ONNX: this is
 * reduced operators, not the ORT-only minimal model format.
 *
 * A network asking for an operator the runtime lacks does not degrade in the
 * browser, it fails to load, and the opponent simply never moves. So no
 * network ships unless `web/scripts/check-model-operators.py` has passed on
 * exactly this file: that script writes runtime/models.sha256, and this
 * refuses to publish anything that does not match it.
 */
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readdir, readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, '..', 'runtime');
const to = join(here, '..', 'public', 'ort');
const publicDir = join(here, '..', 'public');
const files = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

const checked = new Map((await readFile(join(from, 'models.sha256'), 'utf8'))
  .split('\n').map(line => line.trim()).filter(Boolean)
  .map(line => { const [hash, name] = line.split(/\s+/); return [name, hash]; }));
const shipped = (await readdir(publicDir)).filter(name => name.endsWith('.onnx')).sort();
const missing = shipped.filter(name => !checked.has(name));
if (missing.length) {
  throw new Error(`${missing.join(', ')} was never checked against the runtime's operators; `
    + 'run web/scripts/check-model-operators.py');
}
const absent = [...checked.keys()].filter(name => !shipped.includes(name));
if (absent.length) throw new Error(`${absent.join(', ')} is recorded as checked but is not here`);
for (const name of shipped) {
  const actual = createHash('sha256').update(await readFile(join(publicDir, name))).digest('hex');
  if (actual !== checked.get(name)) {
    throw new Error(`${name} changed since it was checked; run web/scripts/check-model-operators.py`);
  }
}

await mkdir(to, { recursive: true });
for (const file of files) {
  await copyFile(join(from, file), join(to, file));
  const { size } = await stat(join(to, file));
  console.log(`copied reduced ${file} (${(size / 1e6).toFixed(2)} MB)`);
}
await copyFile(join(here, '..', 'src', 'lib', 'memory-budget.js'), join(to, 'memory-budget.mjs'));
