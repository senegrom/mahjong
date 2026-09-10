/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
import { startOffline, prepareOfflineAi } from './offline.js';
import { MODEL_FILES } from './model-package.js';
import { MEMORY_LIMITS_MIB, nextMemoryLimit } from './memory-budget.js';

/** The trained opponent: the network itself, not a small copy taught to
 * imitate it. It reads Mortal's 1012 planes, which the engine in this page
 * builds, and answers in Mortal's forty-six moves, which the engine turns
 * back into moves it can play. */
export const MODEL_URLS = Object.freeze(Object.fromEntries(
  Object.entries(MODEL_FILES).map(([name, file]) => [name, new URL(file, document.baseURI).href]),
));
export const MODEL_CHOICES = Object.freeze(Object.keys(MODEL_URLS));
const RUNTIME_BASE = new URL('ort/', document.baseURI).href;
let chosen = 'full';
let worker = null;
let memoryLimitMiB = MEMORY_LIMITS_MIB[0];
let nextId = 1;
const waiting = new Map();
let onProgress = null;

export function reportProgress(callback) { onProgress = callback; }

export function resetPolicy(reason = new DOMException('Match changed', 'AbortError')) {
  const old = worker;
  worker = null;
  old?.terminate();
  for (const pending of waiting.values()) pending.reject(reason);
  waiting.clear();
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
    const offline = await startOffline();
    if (offline) return Boolean(offline.hasModel);
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
  // Download and durably save the model AND runtime before the inference
  // timeout starts. Recreating a worker or reopening offline uses these bytes.
  if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
  // A transferred observation cannot be replayed after a memory-limit retry.
  // Retain this small snapshot until the decision settles; send a copy below.
  planes = planes.slice();
  mask = Array.from(mask);
  const preparation = prepareOfflineAi(model);
  if (signal) {
    await new Promise((resolve, reject) => {
      const abort = () => { signal.removeEventListener('abort', abort); reject(new DOMException('Match changed', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
      preparation.then(value => { signal.removeEventListener('abort', abort); resolve(value); },
        error => { signal.removeEventListener('abort', abort); reject(error); });
      if (signal.aborted) abort();
    });
  } else await preparation;
  if (signal?.aborted) return Promise.reject(new DOMException('Match changed', 'AbortError'));
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
