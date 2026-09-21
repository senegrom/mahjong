/** Content-addressed inventory of the actual production output. */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, basename } from 'node:path';
import { RUNTIME_FILES } from '../src/lib/model-package.js';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { validateNetwork } from '../src/lib/network-transfer.js';
import { copyRuntimeAssets } from './runtime-assets.mjs';

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
  return template.replace('/* NETWORK_TRANSFER */ null', `(() => {\n${transfer.replace(/^export /gm, '')}\nreturn { storageError, verifiedNetworkIsStored, verifiedNetworkBytes, pruneNetworkCache };\n})()`);
}
export async function buildOffline(root, { network = MANIFEST } = {}) {
  validateNetwork(network);
  const files = (await walk(root)).filter(file => !['sw.js', 'offline-manifest.json'].includes(basename(file)));
  const entries = await Promise.all(files.sort().map(async file => {
    const data = await readFile(file), url = relative(root, file).split('\\').join('/');
    if (/\.onnx$/i.test(url)) throw new Error('Offline build contains an unsupported network; use the remote model manifest');
    const group = url.startsWith('ort/') || /ort-.*\.wasm$/.test(url) ? 'ai' : 'core';
    return { url, hash: createHash('sha256').update(data).digest('hex'), bytes: data.length, group };
  }));
  for (const required of ['index.html', 'tiles/Back.svg', 'tiles/Haku.svg']) {
    if (!entries.some(entry => entry.url === required)) throw new Error(`Offline build is missing ${required}`);
  }
  const hasModel = RUNTIME_FILES.every(name => entries.some(entry => entry.url === `ort/${name}`));
  if (!hasModel) throw new Error('Offline build is missing the AI runtime');
  const networkCacheVersion = 2;
  const version = createHash('sha256').update(JSON.stringify({ entries, network, networkCacheVersion })).digest('hex').slice(0, 20);
  const unique = [...new Map(entries.map(entry => [entry.hash, entry])).values()];
  const bytes = {
    core: unique.filter(entry => entry.group === 'core').reduce((sum, entry) => sum + entry.bytes, 0),
    runtime: unique.filter(entry => entry.group === 'ai').reduce((sum, entry) => sum + entry.bytes, 0),
    network: network.bytes,
  };
  const manifest = { version, hasModel, networkCacheVersion, entries, network, bytes };
  const template = await serviceWorkerTemplate();
  await writeFile(resolve(root, 'sw.js'), template.replace('/* OFFLINE_CONFIG */ null', JSON.stringify(manifest)));
  await writeFile(resolve(root, 'offline-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}
export function offlineBuild() {
  let output, source;
  return {
    name: 'mahjong-offline', apply: 'build',
    configResolved(config) {
      output = resolve(config.root, config.build.outDir);
      source = resolve(config.root, 'public');
    },
    async closeBundle() {
      const assets = await copyRuntimeAssets(source, output);
      const { bytes } = await buildOffline(output);
      console.log(`Offline bytes: core=${bytes.core}, AI runtime=${bytes.runtime}, remote network=${bytes.network}; excluded art-workspace bytes=${assets.excludedBytes}`);
    },
  };
}
