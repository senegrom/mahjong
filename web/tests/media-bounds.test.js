import test from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

// A window or container whose width is not a whole number of CSS pixels, as at
// a browser zoom of 110 or 125 per cent (951 device pixels at 125% are 760.8),
// matched neither (max-width: 760px) nor (min-width: 761px), and the table,
// the seats and the hand fell back to styles meant for no size. An upper bound
// whose partner is the next whole pixel ends at .98 of the pixel before it.
const src = fileURLToPath(new URL('../src/', import.meta.url));

async function sources(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) { if (entry.name !== 'wasm') found.push(...await sources(path)); }
    else if (/\.(svelte|css|js)$/.test(entry.name)) found.push(path);
  }
  return found;
}

test('no width or height falls between a max- bound and the min- bound of the next whole pixel', async () => {
  const bounds = [];
  for (const file of await sources(src)) {
    const text = await readFile(file, 'utf8');
    // In a stylesheet's media and container queries, and in the strings
    // handed to matchMedia, a bound is written in parentheses.
    for (const [, side, axis, value] of text.matchAll(/\((max|min)-(width|height):\s*([\d.]+)px\)/g)) {
      bounds.push({ file: relative(src, file).replaceAll('\\', '/'), side, axis, value: Number(value) });
    }
  }
  assert.ok(bounds.some(bound => bound.side === 'min' && bound.axis === 'width' && bound.value === 761), 'the bounds are read');
  const mins = new Set(bounds.filter(bound => bound.side === 'min').map(bound => `${bound.axis} ${bound.value}`));
  const gaps = bounds.filter(bound => bound.side === 'max' && Number.isInteger(bound.value) && mins.has(`${bound.axis} ${bound.value + 1}`));
  assert.deepEqual(gaps.map(bound => `${bound.file}: (max-${bound.axis}: ${bound.value}px)`), []);
});
