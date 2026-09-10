/** Content-addressed offline inventory of the actual production output.
 * No hand-maintained chunk names and no service worker in the dev server. */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, basename } from 'node:path';

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map(entry => entry.isDirectory()
    ? walk(resolve(directory, entry.name)) : resolve(directory, entry.name)))).flat();
}
export async function buildOffline(root) {
  const files = (await walk(root)).filter(file => !['sw.js', 'offline-manifest.json'].includes(basename(file)));
  const entries = await Promise.all(files.sort().map(async file => {
    const data = await readFile(file), url = relative(root, file).split('\\').join('/');
    // ORT may emit a second, hashed copy of its WASM. Equal hashes share one
    // stored body and one download, including across application upgrades.
    // The network and its runtime are a group of their own, so a player who
    // only wants to play the rules is never made to download them.
    const ai = url.endsWith('.onnx') || url.startsWith('ort/') || /ort-.*\.wasm$/.test(url);
    return {
      url, hash: createHash('sha256').update(data).digest('hex'), bytes: data.length,
      group: ai ? 'ai' : 'core',
    };
  }));
  for (const required of ['index.html', 'tiles/Back.svg', 'tiles/Haku.svg']) {
    if (!entries.some(entry => entry.url === required)) throw new Error(`Offline build is missing ${required}`);
  }
  const hasModel = entries.some(entry => entry.url.endsWith('.onnx'));
  if (hasModel) for (const name of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 'memory-budget.mjs']) {
    if (!entries.some(entry => entry.url === `ort/${name}`)) throw new Error(`Missing AI runtime ${name}`);
  }
  const version = createHash('sha256').update(JSON.stringify(entries)).digest('hex').slice(0, 20);
  const manifest = { version, hasModel, entries };
  const template = await readFile(new URL('../src/offline/service-worker.js', import.meta.url), 'utf8');
  await writeFile(resolve(root, 'sw.js'), template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(manifest)));
  await writeFile(resolve(root, 'offline-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}
export function offlineBuild() {
  let output;
  return {
    name: 'mahjong-offline', apply: 'build',
    configResolved(config) { output = resolve(config.root, config.build.outDir); },
    async closeBundle() { await buildOffline(output); },
  };
}
