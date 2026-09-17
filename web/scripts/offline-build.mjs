/** Content-addressed offline inventory of the actual production output.
 * No hand-maintained chunk names and no service worker in the dev server. */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, basename } from 'node:path';
import { MODEL_FILES, RUNTIME_FILES } from '../src/lib/model-package.js';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { validateNetwork } from '../src/lib/network-transfer.js';

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map(entry => entry.isDirectory()
    ? walk(resolve(directory, entry.name)) : resolve(directory, entry.name)))).flat();
}
export async function serviceWorkerTemplate() {
  const [template, transfer] = await Promise.all([
    readFile(new URL('../src/offline/service-worker.js', import.meta.url), 'utf8'),
    readFile(new URL('../src/lib/network-transfer.js', import.meta.url), 'utf8'),
  ]);
  return template.replace('/* NETWORK_TRANSFER */ null', `(() => {\n${transfer.replace(/^export /gm, '')}\nreturn { storageError, verifiedNetworkIsStored, verifiedNetworkBytes };\n})()`);
}
export async function buildOffline(root, { network = MANIFEST } = {}) {
  validateNetwork(network);
  const files = (await walk(root)).filter(file => !['sw.js', 'offline-manifest.json'].includes(basename(file)));
  const entries = await Promise.all(files.sort().map(async file => {
    const data = await readFile(file), url = relative(root, file).split('\\').join('/');
    // ORT may emit a second, hashed copy of its WASM. Equal hashes share one
    // stored body and one download, including across application upgrades.
    const ai = Object.values(MODEL_FILES).includes(url) || url.startsWith('ort/') || /ort-.*\.wasm$/.test(url);
    const group = ai ? 'ai' : 'core';
    return { url, hash: createHash('sha256').update(data).digest('hex'), bytes: data.length, group };
  }));
  for (const required of ['index.html', 'tiles/Back.svg', 'tiles/Haku.svg']) {
    if (!entries.some(entry => entry.url === required)) throw new Error(`Offline build is missing ${required}`);
  }
  const supported = Object.values(MODEL_FILES);
  if (entries.some(entry => entry.url.endsWith('.onnx') && !supported.includes(entry.url))) {
    throw new Error('Offline build contains an unsupported network');
  }
  // The trained network is fetched from its bucket and kept in Cache Storage
  // (web/src/lib/network-store.js), so what this group carries is the runtime
  // that runs it. Without that runtime there is nothing to save for the AI.
  const hasModel = RUNTIME_FILES.every(name => entries.some(entry => entry.url === `ort/${name}`));
  if (!hasModel) throw new Error('Offline build is missing the AI runtime');
  const version = createHash('sha256').update(JSON.stringify({ entries, network })).digest('hex').slice(0, 20);
  const manifest = { version, hasModel, entries, network };
  const template = await serviceWorkerTemplate();
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
