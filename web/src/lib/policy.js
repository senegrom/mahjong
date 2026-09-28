/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
/* global __RUNTIME_DIRECTORY__ */
import { prepareOfflineAi } from './offline.js';
import { NETWORK_URL, networkIsStored } from './network-store.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';

/** The trained opponent: the network itself, not a small copy taught to
 * imitate it. It reads Mortal's 1012 planes, which the engine in this page
 * builds, and answers in Mortal's forty-six moves, which the engine turns
 * back into moves it can play. One network ships, so every request names it.
 */
// The runtime's folder is named by its contents (scripts/copy-runtime.mjs), so
// its address never serves another version's bytes, from any cache.
const runtimeBase = () => new URL(__RUNTIME_DIRECTORY__, document.baseURI).href;
// Two phases, two deadlines. Loading the network into a fresh worker can mean
// downloading 116 MB, which takes minutes on a slow link, so it is bounded by
// silence: a worker that reports nothing for this long has stalled. Only a
// worker holding the network is given a decision's deadline.
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

/** The network is in the worker: decisions waiting for it are timed from now. */
function networkLoaded() {
  loaded = true;
  clearTimeout(loading);
  for (const request of waiting.values()) request.startDeadline();
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
    const { id, action, analysis, error, progress, ready, memory } = data;
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
      if (!loaded) {
        if (ready) networkLoaded(); else stillLoading(current);
        onProgress?.(progress);
      } else if (pending) onProgress?.(progress);
      return;
    }
    pending?.resolve(analysis ?? action);
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
    // Held here already, or the bucket answers. The runtime that runs it is
    // the service worker's business; this is about the network itself.
    if (await networkIsStored(NETWORK_URL)) return true;
    const response = await fetch(NETWORK_URL, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    return response.ok;
  } catch { return false; }
}

/** The network's best legal move for this position; it is never sampled.
 * The timeout bounds the decision once the network is loaded, not the load. */
export function chooseAction(planes, mask, signal, timeout = DECISION_MS) {
  return requestPolicy(planes, mask, timeout, signal, false);
}

export function analyzePolicy(planes, mask, signal) {
  return requestPolicy(planes, mask, DECISION_MS, signal, true);
}

async function requestPolicy(planes, mask, timeout, signal, details) {
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
    const startDeadline = () => {
      clearTimeout(timer);
      timer = setTimeout(() => resetPolicy(new Error('The trained opponent did not answer in time')), timeout);
    };
    // A worker still loading the network starts this decision's deadline
    // when it is ready; a replacement worker loads it again first.
    const dispatch = () => {
      clearTimeout(timer);
      const copy = planes.slice();
      const target = ensureWorker();
      const message = { id, url: NETWORK_URL, runtimeBase: runtimeBase(),
        scope: new URL('./', document.baseURI).href, planes: copy, mask, details, memoryLimitMiB };
      const transfer = [copy.buffer];
      // A fresh worker is handed bytes storage refused rather than fetch them.
      if (unstored && !loaded) {
        message.network = unstored;
        transfer.push(unstored.buffer);
        unstored = null;
      }
      target.postMessage(message, transfer);
      if (loaded) startDeadline();
    };
    waiting.set(id, { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error),
      dispatch, startDeadline });
    signal?.addEventListener('abort', abort, { once: true });
    try {
      dispatch();
    } catch (error) {
      resetPolicy(error);
    }
  });
}
