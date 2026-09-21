/** Validate the runtime package before copying any of its files. */
import { copyFile, mkdir, readdir, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

export async function publicFiles(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const name = prefix + entry.name;
    if (entry.isSymbolicLink()) throw new Error(`Public assets cannot be symlinks: ${name}`);
    if (entry.isDirectory()) result.push(...await publicFiles(join(directory, entry.name), `${name}/`));
    else if (entry.isFile()) {
      if (/\.onnx$/i.test(name)) throw new Error(`Unexpected local ONNX network: ${name}. Publish through the remote model manifest.`);
      result.push(name);
    }
  }
  return result;
}

export async function copyRuntime(web = fileURLToPath(new URL('../', import.meta.url))) {
  const from = join(web, 'runtime'), to = join(web, 'public', 'ort');
  await publicFiles(join(web, 'public'));
  const sources = RUNTIME_FILES.map(file => [file, file === 'memory-budget.mjs'
    ? join(web, 'src', 'lib', 'memory-budget.js') : join(from, file)]);
  // An incomplete package must not partly replace a working generated copy.
  for (const [, source] of sources) {
    const info = await stat(source);
    if (!info.isFile() || !info.size) throw new Error(`Missing or empty runtime file: ${source}`);
  }
  await mkdir(to, { recursive: true });
  for (const [file, source] of sources) await copyFile(source, join(to, file));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await copyRuntime();
  console.log('Validated and copied the reduced execution runtime; model bytes are verified when loaded.');
}
