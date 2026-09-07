/** Copy the model-specific ONNX Runtime build next to the site.
 *
 * The runtime is built from ONNX Runtime 1.29.0 with only the kernels used by
 * public/model.onnx. Keep the model itself as ONNX: this is reduced operators,
 * not the ORT-only minimal model format. A hash binds runtime and model so a
 * new network cannot accidentally ship with missing operator kernels.
 */
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, '..', 'runtime');
const to = join(here, '..', 'public', 'ort');
const model = join(here, '..', 'public', 'model.onnx');
const files = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

const expected = (await readFile(join(from, 'model.sha256'), 'utf8')).trim();
const actual = createHash('sha256').update(await readFile(model)).digest('hex');
if (expected !== actual) {
  throw new Error('model.onnx changed; rebuild web/runtime from the new model before publishing');
}

await mkdir(to, { recursive: true });
for (const file of files) {
  await copyFile(join(from, file), join(to, file));
  const { size } = await stat(join(to, file));
  console.log(`copied reduced ${file} (${(size / 1e6).toFixed(2)} MB)`);
}
