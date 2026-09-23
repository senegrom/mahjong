/** Latest-request-wins image decoding with a bounded pool and reusable successes.
 * A replacement drains the cancelled pool before starting. Timeouts cover both
 * transport and decode; no failed attempt can publish progress or retain images.
 */
export function createImagePreloader({ createImage = () => new Image(), concurrency = 6, timeoutMs = 45000 } = {}) {
  if (!Number.isInteger(concurrency) || concurrency < 1 || (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) throw new RangeError('Invalid image loading limits');
  const cached = new Map();
  let retained = new Set();
  let active = null;
  const cancelled = () => new DOMException('Image loading cancelled', 'AbortError');

  function decode(url, signal) {
    return new Promise((resolve, reject) => {
      const image = createImage();
      let finished = false;
      let timer;
      const finish = error => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        signal.removeEventListener('abort', abort);
        image.onload = null;
        image.onerror = null;
        if (error) { image.removeAttribute('src'); reject(error); }
        else resolve(image);
      };
      const abort = () => finish(signal.reason ?? cancelled());
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) { abort(); return; }
      timer = setTimeout(() => finish(new Error(`Tile graphic timed out: ${url}`)), timeoutMs);
      image.decoding = 'async';
      image.onerror = () => finish(new Error(`Tile graphic could not load: ${url}`));
      image.onload = () => Promise.resolve().then(() => image.decode()).then(() => finish(), finish);
      try { image.src = url; } catch (error) { finish(error); }
    });
  }

  function load(urls, { onProgress = () => {}, signal } = {}) {
    const wanted = [...new Set(urls)];
    if (wanted.some(url => typeof url !== 'string' || !url)) return Promise.reject(new TypeError('Image URLs must be nonempty strings'));
    // Do not cancel useful work for a request that was already withdrawn.
    if (signal?.aborted) return Promise.reject(signal.reason ?? cancelled());
    const previous = active;
    previous?.controller.abort(cancelled());
    const controller = new AbortController();
    const job = { controller, settled: null };
    active = job;
    const abort = () => controller.abort(signal.reason ?? cancelled());
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    job.settled = (async () => {
      await previous?.settled.catch(() => {});
      const current = () => active === job && !controller.signal.aborted;
      if (!current()) throw controller.signal.reason ?? cancelled();
      // Retain the displayed set during a switch, and only useful partial
      // successes from retries. Repeated failed switches cannot grow this map.
      for (const url of cached.keys()) if (!retained.has(url) && !wanted.includes(url)) cached.delete(url);
      let next = 0, complete = 0;
      const workers = Array.from({ length: Math.min(concurrency, wanted.length) }, async () => {
        try {
          while (current() && next < wanted.length) {
            const url = wanted[next++];
            const image = cached.get(url) ?? await decode(url, controller.signal);
            if (!current()) throw controller.signal.reason ?? cancelled();
            cached.set(url, image);
            onProgress(++complete, wanted.length);
          }
        } catch (error) {
          controller.abort(error);
          throw error;
        }
      });
      await Promise.allSettled(workers);
      if (!current()) throw controller.signal.reason ?? cancelled();
      retained = new Set(wanted);
      for (const url of cached.keys()) if (!retained.has(url)) cached.delete(url);
    })().finally(() => {
      signal?.removeEventListener('abort', abort);
      if (active === job) active = null;
    });
    return job.settled;
  }

  return { load, clear() { active?.controller.abort(cancelled()); cached.clear(); retained.clear(); } };
}
