/** Where the trained network comes from, and where it is kept once fetched.
 *
 * The network is the one asset this site cannot host: the fusion is 116 MB in
 * float, GitHub stores no file over 100 MB and Pages serves no Git LFS. It
 * lives in an R2 bucket behind the Worker in `workers/model-cdn/`, under a key
 * that carries its own digest, so the response is immutable and a different
 * export is a different URL.
 *
 * The bytes are kept in Cache Storage rather than left to the HTTP cache,
 * which refuses a single entry this large and would fetch the whole network
 * again on every visit. What comes back is hashed before it is stored or run:
 * a truncated or swapped download must not reach the runtime.
 */
import { MANIFEST as manifest } from './model-manifest.js';

export const NETWORK = Object.freeze(manifest);
export const NETWORK_URL = `${manifest.origin}/${manifest.object}`;
const STORE = 'mahjong-network-v1';

async function store() {
  if (typeof caches === 'undefined') return null;
  try { return await caches.open(STORE); } catch { return null; }
}

/** Whether this browser already holds the network. */
export async function networkIsStored(url = NETWORK_URL) {
  const cache = await store();
  return Boolean(cache && await cache.match(url));
}

async function digestOf(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
}

/** The network's bytes, from this browser's store or from the bucket.
 *
 * `onProgress` is told how many of the network's own bytes have arrived. The
 * length in the headers is the compressed size, so the fraction is taken
 * against the manifest's own figure instead. */
export async function networkBytes({ url = NETWORK_URL, expect = NETWORK, onProgress, signal } = {}) {
  const cache = await store();
  const held = cache && await cache.match(url);
  if (held) {
    const bytes = new Uint8Array(await held.arrayBuffer());
    if (bytes.length === expect.bytes) return bytes;
    // Something else wrote this key, or it was cut short. Fetch it again.
    await cache.delete(url);
  }
  const response = await fetch(url, { mode: 'cors', signal });
  if (!response.ok) throw new Error(`The network could not be fetched: ${response.status}`);
  const chunks = [];
  let seen = 0;
  const reader = response.body?.getReader();
  if (reader) {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      seen += value.length;
      onProgress?.({ bytes: seen, total: expect.bytes });
    }
  } else {
    chunks.push(new Uint8Array(await response.arrayBuffer()));
  }
  const bytes = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  if (bytes.length !== expect.bytes) {
    throw new Error(`The network arrived as ${bytes.length} bytes, not ${expect.bytes}`);
  }
  if (await digestOf(bytes) !== expect.sha256) {
    throw new Error('The network that arrived is not the one this page was built for');
  }
  // A copy, because a Response takes the buffer and the caller runs the bytes.
  await cache?.put(url, new Response(bytes.slice().buffer, {
    headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.length) },
  }));
  return bytes;
}
