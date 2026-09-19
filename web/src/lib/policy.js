/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
import { prepareOfflineAi } from './offline.js';
import { NETWORK, NETWORK_URL, networkIsStored } from './network-store.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';

/** The trained opponent: the network itself, not a small copy taught to
 * imitate it. It reads Mortal's 1012 planes, which the engine in this page
 * builds, and answers in Mortal's forty-six moves, which the engine turns
 * back into moves it can play. */
export const MODEL_URLS = Object.freeze({ full: NETWORK_URL });
export const MODEL_GENERATION = NETWORK.generation;
export const MODEL_CHOICES = Object.freeze(Object.keys(MODEL_URLS));
const RUNTIME_BASE = new URL('ort/', document.baseURI).href;
let chosen = 'full';
let worker = null;
// Preparation belongs to the runtime lifetime, not to every decision. Even
// when durability is unavailable, a verified resident network can keep playing.
// Explicit downloads and foreground status checks still verify offline storage.
let preparation = null;
let generation = 0;
let memoryLimitMiB = MEMORY_LIMITS_MIB[0];
let nextId = 1;
const waiting = new Map();
let onProgress = null;

export function reportProgress(callback) { onProgress = callback; }

export function resetPolicy(reason = new DOMException('Match changed', 'AbortError')) {
  generation++;
  preparation = null;
  const old = worker;
  worker = null;
  old?.terminate();
  for (const pending of waiting.values()) pending.reject(reason);
  waiting.clear();
}

function preparePolicy(model) {
  if (!preparation) {
    const job = prepareOfflineAi(model);
    const held = job.catch(error => {
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
  current.onmessage = ({ data }) => {
    if (worker !== current) return;
    const { id, action, analysis, error, progress, memory } = data;
    const pending = waiting.get(id);
    // Initialization and memory failures poison ORT inside this worker. All
    // modes need a fresh runtime on retry, including after a cancelled call.
    if (error) {
      const larger = pending && memory?.limitMiB === memoryLimitMiB ? nextMemoryLimit(memoryLimitMiB, memory) : null;
      if (larger) {
        // Keep unresolved decisions and their original model/position. Release
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
    if (!pending) return;
    if (progress) {
      onProgress?.(progress);
      return;
    }
    pending.resolve(analysis ?? action);
  };
  current.onerror = (event) => {
    if (worker === current) resetPolicy(new Error(event.message || 'The opponent worker failed'));
  };
  current.onmessageerror = () => {
    if (worker === current) resetPolicy(new Error('The opponent returned an unreadable response'));
  };
  return current;
}

/** Default for subsequent decisions. Requests capture their own model and
 * the worker finishes each turn before changing its loaded network, so this
 * does not interrupt in-flight turns or differently configured Watch agents. */
export function useModel(which) {
  if (!MODEL_CHOICES.includes(which) || which === chosen) return chosen;
  chosen = which;
  return chosen;
}

export function chosenModel() { return chosen; }

export async function modelIsAvailable(which = chosen) {
  try {
    // Held here already, or the bucket answers. The runtime that runs it is
    // the service worker's business; this is about the network itself.
    if (await networkIsStored(MODEL_URLS[which])) return true;
    const response = await fetch(MODEL_URLS[which], { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    return response.ok;
  } catch { return false; }
}

export function chooseAction(planes, mask, temperature = 0, timeout = 20000, signal, model = chosen) {
  return requestPolicy(planes, mask, temperature, timeout, signal, model, false);
}

export function analyzePolicy(planes, mask, signal, model = chosen) {
  return requestPolicy(planes, mask, 0, 20000, signal, model, true);
}

async function requestPolicy(planes, mask, temperature, timeout, signal, model, details) {
  if (!MODEL_CHOICES.includes(model)) throw new Error('Unknown trained agent');
  // Prepare once before the inference deadline starts. Cache eviction or a
  // failed persistence retry must not gate a network already resident in ORT.
  if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
  // A transferred observation cannot be replayed after a memory-limit retry.
  // Retain this small snapshot until the decision settles; send a copy below.
  planes = planes.slice();
  mask = Array.from(mask);
  const owner = generation;
  const prepared = preparePolicy(model);
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
    const dispatch = () => {
      clearTimeout(timer);
      timer = setTimeout(() => resetPolicy(new Error('The trained opponent did not answer in time')), timeout);
      const copy = planes.slice();
      ensureWorker().postMessage({ id, url: MODEL_URLS[model], runtimeBase: RUNTIME_BASE,
        planes: copy, mask, temperature, details, memoryLimitMiB }, [copy.buffer]);
    };
    waiting.set(id, { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error), dispatch });
    signal?.addEventListener('abort', abort, { once: true });
    try {
      dispatch();
    } catch (error) {
      resetPolicy(error);
    }
  });
}
