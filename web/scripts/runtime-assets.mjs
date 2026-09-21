/** Publish game assets, not the art workspace. SVGs are copied byte-for-byte;
 * source PNG crops, design notes and galleries remain available in the repo. */
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';
import { RUNTIME_FILES } from '../src/lib/model-package.js';
import { publicFiles } from './copy-runtime.mjs';

const SHELL = ['apple-touch-icon.png', 'favicon-16x16.png', 'favicon-32x32.png',
  'favicon.ico', 'favicon.svg', 'manifest.webmanifest'];

export async function copyRuntimeAssets(source, output) {
  const files = await publicFiles(source);
  const selected = new Set([...SHELL, ...TILE_IMAGE_URLS, ...RUNTIME_FILES.map(name => `ort/${name}`),
    ...files.filter(name => /^icons\/[^/]+\.(png|svg)$/.test(name) || /(^|\/)LICENSE(?:\.[^/]*)?$/i.test(name))]);
  // Check the whole inventory before copying; missing approved artwork is an error.
  for (const name of selected) if (!files.includes(name)) throw new Error(`Missing runtime asset: ${name}`);
  let copiedBytes = 0, excludedBytes = 0;
  for (const name of files) {
    const input = join(source, name), bytes = (await stat(input)).size;
    if (!selected.has(name)) { excludedBytes += bytes; continue; }
    const target = join(output, name);
    await mkdir(dirname(target), { recursive: true });
    await copyFile(input, target);
    copiedBytes += bytes;
  }
  return { files: selected.size, copiedBytes, excludedBytes };
}
