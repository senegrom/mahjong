/**
 * Puts the inference runtime's WebAssembly next to the site.
 *
 * The bundler emits it under a hashed name, which its loader cannot then
 * find; serving it from a known folder and telling the runtime where to look
 * is the arrangement that works in a module worker.
 */
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, '..', 'node_modules', 'onnxruntime-web', 'dist');
const to = join(here, '..', 'public', 'ort');

const files = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

await mkdir(to, { recursive: true });
for (const file of files) {
  await copyFile(join(from, file), join(to, file));
  const { size } = await stat(join(to, file));
  console.log(`copied ${file} (${(size / 1e6).toFixed(1)} MB)`);
}
