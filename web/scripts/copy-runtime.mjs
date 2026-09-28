/** Validate the runtime package before copying any of its files. */
import { copyFile, mkdir, readdir, readFile, rm, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RUNTIME_FILES } from '../src/lib/model-package.js';

const WEB = fileURLToPath(new URL('../', import.meta.url));

/** The installed onnxruntime-web must be the ONNX Runtime release the reduced
 * WASM was built from: the calls its JavaScript makes into the WASM change
 * between releases. The build leaves its release number in the binary as a
 * NUL-terminated string. */
export async function runtimeRelease(web) {
  const { version } = JSON.parse(await readFile(join(web, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8'));
  const binary = (await readFile(join(web, 'runtime', 'ort-wasm-simd-threaded.wasm'))).toString('latin1');
  const built = [...new Set(binary.match(/(?<=\0)\d+\.\d+\.\d+(?=\0)/g))];
  if (!built.includes(version)) {
    throw new Error(`onnxruntime-web ${version} is installed, but web/runtime was built from ONNX Runtime `
      + `${built.join(' or ') || 'an unknown release'}. They move together: run the reduced ONNX runtime `
      + 'workflow with the release to build, which pins onnxruntime-web to it in the same commit.');
  }
  return version;
}

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

/** Each shipped runtime file and its source. memory-budget.mjs is the page's
 * own allocator control, which the patched loader imports from beside it. */
const runtimeSources = web => RUNTIME_FILES.map(file => [file, file === 'memory-budget.mjs'
  ? join(web, 'src', 'lib', 'memory-budget.js') : join(web, 'runtime', file)]);

/** The runtime is served from a folder named by its contents, so an address
 * never holds two versions' bytes: not in a CDN or an HTTP cache, and not for
 * another version's service worker, which checks every file it saves against
 * its own manifest. One name covers the three files, which load one another. */
export async function runtimeDirectory(web = WEB) {
  const hash = createHash('sha256');
  for (const [file, source] of runtimeSources(web)) hash.update(`${file}\0`).update(await readFile(source)).update('\0');
  return `ort/${hash.digest('hex').slice(0, 16)}/`;
}

export async function copyRuntime(web = WEB) {
  await publicFiles(join(web, 'public'));
  const release = await runtimeRelease(web);
  const sources = runtimeSources(web);
  // An incomplete package must not partly replace a working generated copy.
  for (const [, source] of sources) {
    const info = await stat(source);
    if (!info.isFile() || !info.size) throw new Error(`Missing or empty runtime file: ${source}`);
  }
  const directory = await runtimeDirectory(web), root = join(web, 'public', 'ort'), to = join(web, 'public', directory);
  await mkdir(to, { recursive: true });
  for (const [file, source] of sources) await copyFile(source, join(to, file));
  // Only the current runtime is published; earlier generated copies go.
  for (const entry of await readdir(root)) {
    if (`ort/${entry}/` !== directory) await rm(join(root, entry), { recursive: true, force: true });
  }
  return { release, directory };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { release, directory } = await copyRuntime();
  console.log(`Validated and copied the reduced ONNX Runtime ${release} to ${directory}; model bytes are verified when loaded.`);
}
