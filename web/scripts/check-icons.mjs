// Check both source and Vite output: node scripts/check-icons.mjs [--built]
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

assert.ok(process.argv.length <= 3 && (!process.argv[2] || process.argv[2] === '--built'));
const web = new URL('../', import.meta.url);
const built = process.argv[2] === '--built';
const directory = new URL(built ? 'dist/' : 'public/', web);
const html = readFileSync(new URL(built ? 'dist/index.html' : 'index.html', web), 'utf8');
const project = new URL('https://example.test/mahjong/');
const head = html.match(/<head\b[^>]*>([\s\S]*?)<\/head>/i)?.[1];
assert.ok(head, 'Missing document head');
if (built) assert.ok(!head.includes('%BASE_URL%'), 'Vite did not resolve the public base');
const links = [...head.matchAll(/<link\b[^>]*>/gi)].map(([tag]) => Object.fromEntries(
  [...tag.matchAll(/([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)]
    .map(([, key, a, b]) => [key, a ?? b]),
));
const oneLink = (rel) => {
  const matches = links.filter((link) => link.rel?.split(/\s+/).includes(rel));
  assert.equal(matches.length, 1, `Expected exactly one ${rel}`);
  return matches[0];
};
function pathFor(href, base = project) {
  assert.equal(typeof href, 'string');
  const url = new URL(href.replaceAll('%BASE_URL%', './'), base);
  assert.equal(url.origin, project.origin, 'Icon resources must be same-origin');
  assert.ok(url.pathname.startsWith(project.pathname), `Resource escaped /mahjong/: ${href}`);
  return url.pathname.slice(project.pathname.length);
}
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function png(bytes, expected, label) {
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', `${label}: PNG signature`);
  assert.equal(bytes.toString('ascii', 12, 16), 'IHDR', `${label}: PNG header`);
  assert.equal(bytes.readUInt32BE(16), expected, `${label}: width`);
  assert.equal(bytes.readUInt32BE(20), expected, `${label}: height`);
  assert.deepEqual([...bytes.subarray(24, 29)], [8, 2, 0, 0, 0], `${label}: opaque, non-interlaced RGB`);
  const data = [];
  let offset = 8;
  let ended = false;
  while (offset < bytes.length) {
    assert.ok(offset + 12 <= bytes.length, `${label}: truncated chunk`);
    const length = bytes.readUInt32BE(offset);
    const end = offset + 12 + length;
    assert.ok(end <= bytes.length, `${label}: truncated payload`);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    assert.equal(crc32(bytes.subarray(offset + 4, end - 4)), bytes.readUInt32BE(end - 4), `${label}: ${type} checksum`);
    if (type === 'IDAT') data.push(bytes.subarray(offset + 8, end - 4));
    if (type === 'tRNS') assert.fail(`${label}: unexpected transparency`);
    offset = end;
    if (type === 'IEND') {
      assert.equal(length, 0);
      ended = true;
      break;
    }
  }
  assert.ok(ended && offset === bytes.length, `${label}: incomplete PNG or trailing data`);
  const stride = expected * 3 + 1;
  const pixels = inflateSync(Buffer.concat(data), { maxOutputLength: stride * expected });
  assert.equal(pixels.length, stride * expected, `${label}: incomplete pixels`);
  for (let row = 0; row < expected; row++) assert.ok(pixels[row * stride] <= 4, `${label}: invalid row filter`);
}
const checkPng = (path, size) => png(readFileSync(new URL(path, directory)), size, path);
const apple = oneLink('apple-touch-icon');
assert.equal(apple.sizes, '180x180');
checkPng(pathFor(apple.href), 180);
for (const size of [16, 32]) {
  const icon = links.find((entry) => entry.rel === 'icon' && entry.type === 'image/png' && entry.sizes === `${size}x${size}`);
  assert.ok(icon, `Missing ${size}px browser favicon`);
  checkPng(pathFor(icon.href), size);
}
const icoLink = links.find((entry) => entry.rel === 'icon' && entry.type === 'image/x-icon');
assert.ok(icoLink, 'Missing ICO fallback');
const ico = readFileSync(new URL(pathFor(icoLink.href), directory));
assert.equal(ico.readUInt16LE(0), 0);
assert.equal(ico.readUInt16LE(2), 1);
assert.equal(ico.readUInt16LE(4), 3);
const icoSizes = [];
for (let n = 0; n < 3; n++) {
  const entry = 6 + n * 16;
  const width = ico[entry] || 256;
  assert.equal(ico[entry + 1] || 256, width);
  const length = ico.readUInt32LE(entry + 8);
  const start = ico.readUInt32LE(entry + 12);
  assert.ok(start >= 54 && start + length <= ico.length, 'Invalid ICO frame offset');
  png(ico.subarray(start, start + length), width, `ICO ${width}px`);
  icoSizes.push(width);
}
assert.deepEqual(icoSizes.sort((a, b) => a - b), [16, 32, 48]);
const manifestLink = oneLink('manifest');
const manifestPath = pathFor(manifestLink.href);
const manifestUrl = new URL(manifestPath, project);
const manifest = JSON.parse(readFileSync(new URL(manifestPath, directory), 'utf8'));
assert.equal(manifest.name, 'Riichi Mahjong');
assert.equal(manifest.short_name, 'Riichi');
assert.equal(manifest.display, 'standalone');
for (const field of ['start_url', 'scope']) assert.equal(new URL(manifest[field], manifestUrl).href, project.href);
const startUrl = new URL(manifest.start_url, manifestUrl);
assert.equal(new URL(manifest.id, `${startUrl.origin}/`).href, project.href, 'App ID collides with another Pages app');
const available = new Set();
for (const icon of manifest.icons) {
  assert.equal(icon.type, 'image/png');
  const dimensions = icon.sizes.match(/^(\d+)x\1$/);
  assert.ok(dimensions, 'Expected square PNG sizes');
  checkPng(pathFor(icon.src, manifestUrl), Number(dimensions[1]));
  for (const purpose of (icon.purpose ?? 'any').split(/\s+/)) available.add(`${icon.sizes}:${purpose}`);
}
for (const requirement of ['192x192:any', '512x512:any', '512x512:maskable']) assert.ok(available.has(requirement), requirement);
console.log(`${built ? 'Built site' : 'Source'}: PNG checksums/pixels, ICO frames, Apple icon, manifest and /mahjong/ scope verified.`);
