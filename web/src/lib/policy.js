/**
 * The page's side of the trained opponent: starts the worker, asks it for a
 * move, and reports honestly when there is no model to ask.
 */

// Both are relative to wherever the site is published, which is the only
// thing the page knows and the worker does not.
const MODEL_URL = new URL('model.onnx', document.baseURI).href;
const RUNTIME_BASE = new URL('ort/', document.baseURI).href;

let worker = null;
let nextId = 1;
const waiting = new Map();
let onProgress = null;

/** Watches the loading of the runtime and the network. */
export function reportProgress(callback) {
  onProgress = callback;
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./policy.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (event) => {
    const { id, action, error, progress } = event.data;
    if (progress) {
      // Loading the runtime and the network takes a moment on a first
      // visit; saying so beats a table that appears to have stopped.
      if (onProgress) onProgress(progress);
      return;
    }
    const pending = waiting.get(id);
    if (!pending) return;
    waiting.delete(id);
    if (error) pending.reject(new Error(error));
    else pending.resolve(action);
  };
  worker.onerror = (event) => {
    for (const pending of waiting.values()) pending.reject(new Error(event.message));
    waiting.clear();
  };
  return worker;
}

/** Whether a trained opponent has been published alongside the game. */
export async function modelIsAvailable() {
  try {
    const response = await fetch(MODEL_URL, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Asks the trained opponent for a move.
 *
 * `planes` is the observation the engine produced for that seat and `mask`
 * says which entries of the action space the rules allow, so the answer is
 * always one the engine will accept.
 */
export function chooseAction(planes, mask, temperature = 0, timeout = 20000) {
  const id = nextId;
  nextId += 1;
  return new Promise((resolve, reject) => {
    // A worker that never answers must say so rather than leave the table
    // waiting: the game falls back to the heuristic opponents.
    const timer = setTimeout(() => {
      waiting.delete(id);
      reject(new Error('the trained opponent did not answer in time'));
    }, timeout);
    waiting.set(id, {
      resolve: (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      },
    });
    ensureWorker().postMessage(
      { id, url: MODEL_URL, runtimeBase: RUNTIME_BASE, planes, mask, temperature },
      [planes.buffer],
    );
  });
}
