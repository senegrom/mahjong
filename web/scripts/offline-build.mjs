/** Content-addressed offline inventory of the actual production output.
 * No hand-maintained chunk names and no service worker in the dev server. */
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, relative, basename } from 'node:path';
import { MODEL_FILES, RUNTIME_FILES } from '../src/lib/model-package.js';

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
  const hasModel = supported.every(file => entries.some(entry => entry.url === file));
  if (hasModel) for (const name of RUNTIME_FILES) {
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
