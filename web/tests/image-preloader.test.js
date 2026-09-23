import test from 'node:test';
import assert from 'node:assert/strict';
import { createImagePreloader } from '../src/lib/image-preloader.js';
const turn = () => new Promise(resolve => setImmediate(resolve));
function fixture(options = {}) {
  const images = [], progress = [];
  let live = 0, peak = 0;
  const loader = createImagePreloader({ concurrency: 2, timeoutMs: 2000, ...options, createImage: () => {
    let resolveDecode, rejectDecode;
    const decoded = new Promise((resolve, reject) => { resolveDecode = resolve; rejectDecode = reject; });
    // Rejection may happen before onload; always observe it.
    decoded.catch(() => {});
    const image = {
      onload: null, onerror: null, stopped: false, finished: false,
      set src(url) { this.url = url; live++; peak = Math.max(peak, live); },
      removeAttribute() { this.stopped = true; this.end(); rejectDecode(new Error('cancelled decode')); },
      decode: () => decoded,
      end() { if (!this.finished) { this.finished = true; live--; } },
      good() { this.onload?.(); this.end(); resolveDecode(); },
      bad() { this.onerror?.(); },
      loaded() { this.onload?.(); },
    };
    images.push(image); return image;
  } });
  return { loader, images, progress, get peak() { return peak; }, get live() { return live; } };
}
test('deduplicates URLs and successes, reports one completion each', async () => {
  const f = fixture();
  const job = f.loader.load(['a', 'b', 'a'], { onProgress: (...p) => f.progress.push(p) });
  await turn(); f.images.forEach(image => image.good()); await job;
  await f.loader.load(['b', 'a']);
  assert.equal(f.images.length, 2);
  assert.deepEqual(f.progress, [[1, 2], [2, 2]]);
});
test('partial failure cancels siblings; retry reuses successes without stale progress', async () => {
  const f = fixture();
  const first = f.loader.load(['a', 'b', 'c', 'd'], { onProgress: n => f.progress.push(n) });
  const failed = assert.rejects(first, /could not load/);
  await turn(); f.images[0].good(); await turn();
  f.images[1].bad(); await failed;
  assert.equal(f.live, 0); assert.equal(f.images[2].stopped, true);
  const stale = f.progress.length;
  f.images.slice(0, 3).forEach(image => image.good()); await turn();
  assert.equal(f.progress.length, stale);
  const retry = f.loader.load(['a', 'b', 'c', 'd']); await turn();
  assert.equal(f.images.filter(image => image.url === 'a').length, 1);
  f.images.slice(3).forEach(image => image.good()); await turn();
  f.images.slice(3).forEach(image => image.good()); await retry;
  assert.ok(f.peak <= 2);
});
test('superseding an attempt drains its pool before starting the replacement', async () => {
  const f = fixture(); const first = f.loader.load(['a', 'b', 'c']);
  const rejected = assert.rejects(first, { name: 'AbortError' }); await turn();
  const next = f.loader.load(['d', 'e']); await turn();
  assert.equal(f.images[0].stopped, true); assert.equal(f.images[1].stopped, true);
  f.images.slice(2).forEach(image => image.good()); await Promise.all([rejected, next]);
  assert.ok(f.peak <= 2);
});
test('caller cancellation removes handlers and ignores callbacks after unmount', async () => {
  const f = fixture(); const controller = new AbortController();
  const job = f.loader.load(['a', 'b'], { signal: controller.signal, onProgress: n => f.progress.push(n) });
  const rejected = assert.rejects(job, { name: 'AbortError' }); await turn(); controller.abort(); await rejected;
  for (const image of f.images) { assert.equal(image.onload, null); assert.equal(image.onerror, null); assert.equal(image.stopped, true); }
  assert.deepEqual(f.progress, []);
});
test('timeout includes a decode that never settles and leaves retry possible', async () => {
  const f = fixture({ timeoutMs: 20 });
  const job = f.loader.load(['a']); const rejected = assert.rejects(job, /timed out/);
  await turn(); f.images[0].loaded(); await rejected;
  assert.equal(f.images[0].stopped, true);
  const next = f.loader.load(['a']); await turn(); f.images[1].good(); await next;
});
test('completed switches release old sets, while failed switches keep the displayed set', async () => {
  const f = fixture(); let job = f.loader.load(['a']); await turn(); f.images[0].good(); await job;
  job = f.loader.load(['b']); const rejected = assert.rejects(job); await turn(); f.images[1].bad(); await rejected;
  await f.loader.load(['a']); assert.equal(f.images.length, 2);
  job = f.loader.load(['b']); await turn(); f.images[2].good(); await job;
  job = f.loader.load(['a']); await turn(); assert.equal(f.images.length, 4); f.images[3].good(); await job;
});
test('an already cancelled request does not cancel the current attempt', async () => {
  const f = fixture(); const job = f.loader.load(['a']); await turn();
  const controller = new AbortController(); controller.abort();
  await assert.rejects(f.loader.load(['b'], { signal: controller.signal }), { name: 'AbortError' });
  f.images[0].good(); await job; assert.equal(f.images.length, 1);
});
