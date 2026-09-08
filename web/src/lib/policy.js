/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
import { startOffline, prepareOfflineAi } from './offline.js';

/** The two trained opponents: the small network the game has always
 * carried, and a larger one distilled from the lineage being trained now. */
export const MODEL_URLS = Object.freeze({
  quick: new URL('model.onnx', document.baseURI).href,
  strong: new URL('model-strong.onnx', document.baseURI).href,
});
export const MODEL_CHOICES = Object.freeze(Object.keys(MODEL_URLS));
const RUNTIME_BASE = new URL('ort/', document.baseURI).href;
let chosen = 'quick';
let worker = null;
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
    const { id, action, error, progress } = data;
    const pending = waiting.get(id);
    if (!pending) return;
    if (progress) {
      onProgress?.(progress);
      return;
    }
    if (error) pending.reject(new Error(error));
    else pending.resolve(action);
  };
  current.onerror = (event) => {
    if (worker === current) resetPolicy(new Error(event.message || 'The opponent worker failed'));
  };
  current.onmessageerror = () => {
    if (worker === current) resetPolicy(new Error('The opponent returned an unreadable response'));
  };
  return current;
}

/** Which network the opponents play with from here on. Changing it drops
 * the worker, since the one that is running has the other network loaded. */
export function useModel(which) {
  if (!MODEL_CHOICES.includes(which) || which === chosen) return chosen;
  chosen = which;
  resetPolicy(new DOMException('Opponent changed', 'AbortError'));
  return chosen;
}

export function chosenModel() { return chosen; }

export async function modelIsAvailable(which = chosen) {
  try {
    const offline = await startOffline();
    if (offline) return which === 'strong' ? Boolean(offline.hasStrongModel) : Boolean(offline.hasModel);
    const response = await fetch(MODEL_URLS[which], { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    return response.ok;
  } catch { return false; }
}

export async function chooseAction(planes, mask, temperature = 0, timeout = 20000, signal) {
  // Download and durably save the model AND runtime before the inference
  // timeout starts. Recreating a worker or reopening offline uses these bytes.
  if (signal?.aborted) throw new DOMException('Match changed', 'AbortError');
  const preparation = prepareOfflineAi(chosen);
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
    const abort = () => finish(reject, new DOMException('Match changed', 'AbortError'));
    const timer = setTimeout(() => resetPolicy(new Error('The trained opponent did not answer in time')), timeout);
    const finish = (callback, value) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      waiting.delete(id);
      callback(value);
    };
    waiting.set(id, { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error) });
    signal?.addEventListener('abort', abort, { once: true });
    try {
      ensureWorker().postMessage({ id, url: MODEL_URLS[chosen], runtimeBase: RUNTIME_BASE, planes, mask, temperature }, [planes.buffer]);
    } catch (error) {
      resetPolicy(error);
    }
  });
}
