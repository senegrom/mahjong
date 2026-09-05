/** Worker ownership, bounded requests and explicit retry. A broken worker is
 * discarded; retry never reuses a rejected loading promise or a hung process. */
const MODEL_URL = new URL('model.onnx', document.baseURI).href;
const RUNTIME_BASE = new URL('ort/', document.baseURI).href;
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

export async function modelIsAvailable() {
  try {
    const response = await fetch(MODEL_URL, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
    return response.ok;
  } catch { return false; }
}

export function chooseAction(planes, mask, temperature = 0, timeout = 20000, signal) {
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
      ensureWorker().postMessage({ id, url: MODEL_URL, runtimeBase: RUNTIME_BASE, planes, mask, temperature }, [planes.buffer]);
    } catch (error) {
      resetPolicy(error);
    }
  });
}
