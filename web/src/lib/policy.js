/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
/* global __RUNTIME_DIRECTORY__ */
import { prepareOfflineAi, refreshOffline } from './offline.js';
import { networkIsStored } from './network-store.js';
import { networkUrl, playedNetwork } from './trained-network.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';

/** The trained opponent: the network itself, not a small copy taught to
 * imitate it. It reads Mortal's 1012 planes, which the engine in this page
 * builds, and answers in Mortal's forty-six moves, which the engine turns
 * back into moves it can play. A page plays one network, the one its player
 * picked, so every request names that one.
 */
// The runtime's folder is named by its contents (scripts/copy-runtime.mjs), so
// its address never serves another version's bytes, from any cache.
const runtimeBase = () => new URL(__RUNTIME_DIRECTORY__, document.baseURI).href;
// Two phases, two deadlines. Loading the network into a fresh worker can mean
// downloading 116 MB, which takes minutes on a slow link, so it is bounded by
// silence: a worker that reports nothing for this long has stalled. Only a
// worker holding the network is given a decision's deadline, and only from
// when it begins the decision. It answers its requests in turn, so a decision
// sent behind another, such as a review's batch, waits its turn untimed: that
// wait is the batch's, which the batch's own deadline bounds (backstop).
const DECISION_MS = 20000;
const LOADING_SILENCE_MS = 120000;
let worker = null;
let loaded = false;
let loading;
// Preparation belongs to the runtime lifetime, not to every decision. Even
// when durability is unavailable, a verified resident network can keep playing.
// Explicit downloads and foreground status checks still verify offline storage.
let preparation = null;
// Verified network bytes that offline storage refused, for the next worker.
let unstored = null;
let generation = 0;
let memoryLimitMiB = MEMORY_LIMITS_MIB[0];
let nextId = 1;
const waiting = new Map();
let onProgress = null;

export function reportProgress(callback) { onProgress = callback; }

export function resetPolicy(reason = new DOMException('Match changed', 'AbortError')) {
  generation++;
  preparation = null;
  unstored = null;
  const old = worker;
  worker = null;
  loaded = false;
  clearTimeout(loading);
  old?.terminate();
  for (const pending of waiting.values()) pending.reject(reason);
  waiting.clear();
}

/** Restarts the loading phase's deadline for this worker. */
function stillLoading(current) {
  clearTimeout(loading);
  loading = setTimeout(() => {
    if (worker === current) resetPolicy(new Error('The trained network stopped loading. Check the connection and retry.'));
  }, LOADING_SILENCE_MS);
}

/** The network is in the worker: decisions waiting for it are queued there
 * from now, each timed by its backstop until the worker begins it. */
function networkLoaded() {
  loaded = true;
  clearTimeout(loading);
  for (const request of waiting.values()) request.queued();
}

/** How long the decision `id` may wait in a worker holding the network and
 * then run, should the worker never say that it has begun it: its own
 * deadline and those of the requests ahead of it, every one still waiting
 * that was sent before it. The worker runs those in turn, and each ends the
 * worker if it runs past its own deadline, so a worker that keeps to them
 * begins this one in time. One that falls silent, as a worker that dies
 * without an error does, is ended here, and every request still waiting
 * fails with it, as when a request runs out of time. */
function backstop(id) {
  let total = 0;
  for (const [other, request] of waiting) {
    total += request.timeout;
    if (other === id) break;
  }
  return total;
}

function preparePolicy() {
  if (!preparation) {
    const held = prepareOfflineAi().then(prepared => {
      if (preparation === held) unstored = prepared?.unstoredNetwork ?? null;
    }, error => {
      if (preparation === held) preparation = null;
      throw error;
    });
    preparation = held;
  }
  return preparation;
}

function ensureWorker() {
  if (worker) return worker;
  const current = new Worker(new URL('./policy.worker.js', import.meta.url), { type: 'module' });
  worker = current;
  loaded = false;
  stillLoading(current);
  current.onmessage = ({ data }) => {
    if (worker !== current) return;
    const { id, action, analysis, analyses, error, progress, ready, memory } = data;
    const pending = waiting.get(id);
    // Initialization and memory failures poison ORT inside this worker. All
    // modes need a fresh runtime on retry, including after a cancelled call.
    if (error) {
      const larger = pending && memory?.limitMiB === memoryLimitMiB ? nextMemoryLimit(memoryLimitMiB, memory) : null;
      if (larger) {
        // Keep unresolved decisions and their original position. Release
        // the old reservation before asking the browser for bounded headroom.
        worker = null;
        current.terminate();
        memoryLimitMiB = larger;
        onProgress?.('retrying the agent with more memory');
        try { for (const request of waiting.values()) request.dispatch(); }
        catch (failure) { resetPolicy(failure); }
      } else {
        // A memory failure nothing larger can fix starts the next runtime
        // small again; any other failure keeps the ceiling that was found to
        // work, so a retry does not repeat the whole fail-and-grow cycle.
        if (memory) memoryLimitMiB = MEMORY_LIMITS_MIB[0];
        resetPolicy(new Error(error));
      }
      return;
    }
    if (progress) {
      // While the network loads, a note about any request is the worker's
      // sign of life: a cancelled request's download serves the next one.
      // Once it is loaded, the worker's notes only say again that the
      // resident network is ready. Passing them on would replace the table's
      // "… is thinking" with "network ready" on every later move, so they
      // stop here; a worker replaced after a memory failure starts unloaded.
      if (!loaded) {
        if (ready) networkLoaded(); else stillLoading(current);
        onProgress?.(progress);
      }
      // The worker says the network is ready as it begins each request, so
      // that request's own deadline runs from now: until now it waited for
      // the requests ahead of it.
      if (ready) pending?.begun();
      return;
    }
    // An answer, from any request, means the network ran: it is loaded
    // whether or not the worker said so first.
    if (!loaded) networkLoaded();
    pending?.resolve(analyses ?? analysis ?? action);
  };
  current.onerror = (event) => {
    if (worker === current) resetPolicy(new Error(event.message || 'The opponent worker failed'));
  };
  current.onmessageerror = () => {
    if (worker === current) resetPolicy(new Error('The opponent returned an unreadable response'));
  };
  return current;
}

export async function modelIsAvailable() {
  try {
    // Offline, the runtime must be saved too, and only the service worker's
    // aiReady covers both. The opponent's worker can store the network by
    // itself before a worker controls the page, and nothing then saves the
    // runtime: offering Trained would fail at its first move.
    if (!navigator.onLine) return Boolean((await refreshOffline())?.aiReady);
    // Online, held here already, or the bucket answers; the server has the
    // runtime.
    if (await networkIsStored()) return true;
    const response = await fetch(networkUrl(playedNetwork()), { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    return response.ok;
  } catch { return false; }
}

/** The network's best legal move for this position; it is never sampled.
 * The timeout bounds the decision from when the worker begins it, not the
 * load, nor its wait behind the worker's other requests. */
export function chooseAction(planes, mask, signal, timeout = DECISION_MS) {
  return requestPolicy(planes, mask, timeout, signal, false);
}

export function analyzePolicy(planes, mask, signal) {
  return requestPolicy(planes, mask, DECISION_MS, signal, true);
}

/** The best legal move and weights of several positions, asked in one
 * forward pass, in the order given: what a whole-game review asks of the
 * network. Each position is `{ planes, mask }`. The deadline grows with the
 * batch, since a slow phone takes a good part of a second a position. */
export function analyzeBatch(positions, signal) {
  if (!positions.length) return Promise.resolve([]);
  const width = positions[0].planes.length;
  const planes = new Float32Array(positions.length * width);
  positions.forEach((position, index) => planes.set(position.planes, index * width));
  const mask = positions.flatMap(position => Array.from(position.mask));
  return requestPolicy(planes, mask, DECISION_MS * Math.ceil(positions.length / 4), signal, true, positions.length);
}

async function requestPolicy(planes, mask, timeout, signal, details, count) {
  // Prepare once before any deadline starts. Cache eviction or a failed
  // persistence retry must not gate a network already resident in ORT.
  if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
  // A transferred observation cannot be replayed after a memory-limit retry.
  // Retain this small snapshot until the decision settles; send a copy below.
  planes = planes.slice();
  mask = Array.from(mask);
  const owner = generation;
  const prepared = preparePolicy();
  if (signal) {
    await new Promise((resolve, reject) => {
      const abort = () => { signal.removeEventListener('abort', abort); reject(new DOMException('Match changed', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
      prepared.then(value => { signal.removeEventListener('abort', abort); resolve(value); },
        error => { signal.removeEventListener('abort', abort); reject(error); });
      if (signal.aborted) abort();
    });
  } else await prepared;
  if (signal?.aborted || owner !== generation) throw new DOMException('Match changed', 'AbortError');
  return new Promise((resolve, reject) => {
    const id = nextId++;
    // A match that ends drops its own request and no more: the worker,
    // with the runtime and the model loaded, stays for the next match.
    const abort = () => {
      finish(reject, new DOMException('Match changed', 'AbortError'));
      // Drop queued work for an edited physical position or a closed match.
      // Active inference can finish; its tensors are still disposed normally.
      try { worker?.postMessage({ cancel: id }); } catch { /* A failed worker is handled by its error event. */ }
    };
    let timer;
    const finish = (callback, value) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      waiting.delete(id);
      callback(value);
    };
    // Either clock, run out, ends the worker and every request waiting on it.
    const time = ms => {
      clearTimeout(timer);
      timer = setTimeout(() => resetPolicy(new Error('The trained opponent did not answer in time')), ms);
    };
    // Queued in a worker holding the network, this decision is timed by its
    // backstop, and once the worker begins it by its own deadline. A worker
    // still loading the network queues it when it is ready; a replacement
    // worker loads it again first.
    const dispatch = () => {
      clearTimeout(timer);
      const copy = planes.slice();
      const target = ensureWorker();
      const message = { id, url: networkUrl(playedNetwork()), runtimeBase: runtimeBase(),
        scope: new URL('./', document.baseURI).href, planes: copy, mask, details, memoryLimitMiB };
      if (count) message.count = count;
      const transfer = [copy.buffer];
      // A fresh worker is handed bytes storage refused rather than fetch them.
      if (unstored && !loaded) {
        message.network = unstored;
        transfer.push(unstored.buffer);
        unstored = null;
      }
      target.postMessage(message, transfer);
      if (loaded) time(backstop(id));
    };
    waiting.set(id, { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error),
      dispatch, timeout, queued: () => time(backstop(id)), begun: () => time(timeout) });
    signal?.addEventListener('abort', abort, { once: true });
    try {
      dispatch();
    } catch (error) {
      resetPolicy(error);
    }
  });
}
