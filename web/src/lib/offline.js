/** Offline preparation and honest, cache-backed download status. No localStorage
 * flag is accepted as proof that a model or its runtime is actually present.
 *
 * The service worker saves the game and the few megabytes of runtime. The
 * trained network is saved here, by the page: a worker's install, activation
 * or message event is abandoned after a few minutes, and a slow link needs
 * longer than that for 100 MB. A page has no such deadline. */
import { NETWORK, NETWORK_URL, networkBytes, networkIsStored } from './network-store.js';
const base = new URL('./', document.baseURI);
const script = new URL('sw.js', base).href;
let worker = null;
let boot = null;
let aiJob = null;
let coreJob = null;
// A player who asked for Trained offline keeps it across upgrades: each
// version's page saves its own network, automatically once per page.
let filled = false;
const updatesSaved = new WeakSet();
let state = { supported: null, coreReady: false, aiReady: false, hasModel: false,
  phase: 'checking', progress: 0, warning: '', coreWarning: '', coreLoading: false,
  installing: false, persistent: false, updateReady: false };
const listeners = new Set();
function update(values) {
  state = { ...state, ...values };
  for (const listener of listeners) listener(state);
}
export function watchOffline(listener) {
  listeners.add(listener);
  listener(state);
  return () => listeners.delete(listener);
}
function request(type, progress, target = worker) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    // The worker's longest job is a few megabytes of game or runtime.
    const timer = setTimeout(() => finish(new Error('Offline preparation timed out. Please reconnect and retry.')), 180000);
    const finish = (error, value) => {
      clearTimeout(timer); channel.port1.close();
      if (error) reject(error); else resolve(value);
    };
    channel.port1.onmessage = ({ data }) => {
      if (data.progress) progress?.(data.progress);
      else {
        const error = data.error ? new Error(data.error) : null;
        if (error && data.storage) error.name = 'NetworkStorageError';
        finish(error, data.value);
      }
    };
    try { target.postMessage({ type }, [channel.port2]); }
    catch (error) { finish(error); }
  });
}
/** Whether storage says it can take this many more bytes. An unknown answer
 * counts as room: the save is tried, and a refusal is reported as one. */
async function hasRoom(bytes) {
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return !(quota - usage < bytes);
  } catch { return true; }
}
/** Saves a network durably from this page. The verified bytes are dropped once
 * stored: the runtime reads, and hashes, its own copy when it starts. A
 * network storage has no room for is not downloaded only to be refused. */
async function saveNetwork(url, expect, onProgress) {
  if (await networkIsStored(url, expect)) return;
  if (!await hasRoom(expect.bytes)) {
    const error = new Error('The network is not saved offline: this device has too little free storage for it.');
    error.name = 'NetworkStorageError';
    throw error;
  }
  await networkBytes({ url, expect, requireStored: true, onProgress });
}
function activated(registration) {
  if (registration.active) return Promise.resolve();
  const installing = registration.installing ?? registration.waiting;
  if (!installing) return Promise.reject(new Error('Offline installation did not start.'));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('Offline installation is incomplete. Reconnect and retry.')), 120000);
    const finish = error => {
      clearTimeout(timer); installing.removeEventListener('statechange', changed);
      if (error) reject(error); else resolve();
    };
    const changed = () => {
      if (installing.state === 'activated') finish();
      else if (installing.state === 'redundant') finish(new Error('Game download incomplete. Reconnect and retry.'));
    };
    installing.addEventListener('statechange', changed); changed();
  });
}
async function persistentStorage() {
  try {
    const storage = navigator.storage;
    const persistent = await storage?.persisted?.() || await storage?.persist?.() || false;
    update({ persistent });
  } catch { /* Some browsers decline persistence; never claim it was granted. */ }
}
/** An update installs with the game alone and waits for this version's
 * windows to close. When Trained was requested, save the update's runtime and
 * network now, while this version runs, so the update does not start without
 * them on a plane. Should this not finish, the update's own page saves them. */
async function saveUpdate(registration) {
  const next = registration.waiting;
  if (!next || updatesSaved.has(next) || !navigator.onLine) return;
  updatesSaved.add(next);
  try {
    // One network download at a time: this version's own comes first.
    await aiJob?.catch(() => {});
    const info = await request('MAHJONG_STATUS', null, next);
    if (!info?.hasModel || !info.aiRequested || info.aiReady || !info.network) return;
    await request('MAHJONG_PREPARE_AI', null, next);
    await saveNetwork(info.network.url, info.network);
  } catch { /* The update's own page retries after activation. */ }
}
function observeUpdates(registration) {
  const check = () => {
    update({ updateReady: Boolean(registration.waiting) });
    void saveUpdate(registration);
  };
  check();
  registration.addEventListener('updatefound', () => {
    registration.installing?.addEventListener('statechange', check);
  });
}
/** The worker reports the network it found stored. Match that identity to
 * this page's manifest; a reply about another version's network is checked
 * against this page's own stored record instead. Neither reads the model:
 * never hash the same model again just to repeat status. */
async function withNetwork(info) {
  if (!info || !info.aiReady) return info;
  return { ...info, aiReady: await networkIsStored(undefined, undefined, info.network) };
}
/** Once per page, for a player who asked for Trained offline: a new
 * version's network (and runtime) is saved by its page after activation. */
function fillRequested(info) {
  if (filled || !info?.aiRequested || info.aiReady || !info.hasModel || !info.coreReady || !navigator.onLine) return;
  filled = true;
  void prepareOfflineAi().catch(() => {});
}

// The game and graphics are mandatory, not an optional offline pack. Repair
// an evicted or interrupted core cache automatically, without requesting AI.
async function prepareCore(info) {
  update(info);
  if (info.coreReady) {
    update({ coreWarning: '' });
    return info;
  }
  if (coreJob) return coreJob;
  coreJob = (async () => {
    update({ coreLoading: true, coreWarning: '' });
    try {
      const ready = await request('MAHJONG_PREPARE_CORE');
      update({ ...ready, coreWarning: '' });
      return ready;
    } catch (error) {
      update({ coreReady: false, coreWarning: `Game and tile download incomplete: ${error.message} It will retry automatically when you reconnect or reopen the app.` });
      return null;
    } finally { update({ coreLoading: false }); }
  })().finally(() => { coreJob = null; });
  return coreJob;
}
export function startOffline() {
  if (boot) return boot;
  const attempt = (async () => {
    if (import.meta.env.DEV || !('serviceWorker' in navigator) || !('caches' in globalThis) || !isSecureContext) {
      update({ supported: false, phase: 'unavailable', warning: 'Offline saving is unavailable here. Keep a connection for this session.' });
      return null;
    }
    try {
      // Existing registration is local. Never make a successful online check
      // a prerequisite for launching an already downloaded app on a plane.
      let registration = await navigator.serviceWorker.getRegistration(base.href);
      if (registration?.scope !== base.href || registration?.active?.scriptURL !== script) registration = null;
      if (!registration) registration = await navigator.serviceWorker.register(script, { scope: base.href, updateViaCache: 'none' });
      // A first visit's worker saves the whole game and every tile graphic
      // before it activates, and a Trained player's network waits for that.
      // The install reports no progress, so say that it is running.
      const installing = !registration.active;
      if (installing) update({ installing: true });
      try { await activated(registration); }
      finally { if (installing) update({ installing: false }); }
      worker = registration.active;
      // clients.claim() takes control on the first visit, before tile/ORT loads.
      if (navigator.serviceWorker.controller?.scriptURL !== script) {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => finish(new Error('Reload once to enable offline play.')), 15000);
          const finish = error => {
            clearTimeout(timer); navigator.serviceWorker.removeEventListener('controllerchange', changed);
            if (error) reject(error); else resolve();
          };
          const changed = () => { if (navigator.serviceWorker.controller?.scriptURL === script) finish(); };
          navigator.serviceWorker.addEventListener('controllerchange', changed); changed();
        });
      }
      update({ supported: true, phase: 'ready', warning: '' });
      const info = await prepareCore(await withNetwork(await request('MAHJONG_STATUS')));
      // This version's own network first; a waiting update's queues behind it.
      fillRequested(info);
      observeUpdates(registration);
      void persistentStorage();
      // Background updates never block this version or replace it mid-match.
      // They install with the game alone, so a slow link still completes one.
      if (navigator.onLine) void registration.update().catch(() => {});
      return state;
    } catch (error) {
      worker = null;
      update({ supported: false, coreReady: false, aiReady: false, phase: 'unavailable',
        warning: `Offline saving is not ready: ${error.message}` });
      return null;
    }
  })();
  // Only a boot that reached the service worker is kept. One that did not (a
  // first visit that lost its connection, an install still running, a hard
  // reload) is tried again by the next caller: a reconnect or return to the
  // app, or a Trained opponent's retry.
  boot = attempt;
  void attempt.then(result => { if (!result && boot === attempt) boot = null; });
  return attempt;
}
export async function refreshOffline() {
  await startOffline();
  if (!worker) return null;
  const info = await prepareCore(await withNetwork(await request('MAHJONG_STATUS')));
  // A connection that returns may still owe a requested network its save.
  fillRequested(info);
  return info;
}
/** Saves the one trained network the game carries, with its runtime. It is
 * downloaded only when a Trained player is chosen (or was, in an earlier
 * version), and one job runs at a time. */
export function prepareOfflineAi() {
  if (aiJob) return aiJob;
  aiJob = (async () => {
    if (!(await startOffline())) return null; // Online-only browsers still work, with a visible warning.
    // This action adds only the trained network/runtime. Core preparation has
    // its own automatic startup/reconnect path and is not opt-in.
    const info = await withNetwork(await request('MAHJONG_STATUS'));
    update(info);
    if (info.aiReady) return info;
    update({ phase: 'ai', progress: 0, warning: '' });
    try {
      // The runtime first, by the worker that serves it: a runtime that
      // cannot be saved spends none of the network's bytes.
      await request('MAHJONG_PREPARE_AI', ({ bytes, total }) => {
        update({ progress: total ? Math.floor(20 * bytes / total) : 0 });
      });
      await saveNetwork(NETWORK_URL, NETWORK, ({ bytes, total }) => {
        update({ progress: 20 + (total ? Math.floor(80 * bytes / total) : 0) });
      });
      // The worker now keeps this network and the previous one only.
      const verified = await withNetwork(await request('MAHJONG_NETWORK_SAVED'));
      update({ ...verified, phase: 'ready', progress: 100, warning: '' });
      void persistentStorage();
      return verified;
    } catch (error) {
      if (error.name === 'NetworkStorageError') {
        update({ aiReady: false, phase: 'incomplete', warning: `${error.message} Online play is still available; reconnect before playing offline.` });
        // Bytes verified but refused by storage go to the opponent's worker,
        // which would otherwise download the same network again.
        return error.bytes ? { aiReady: false, unstoredNetwork: error.bytes } : null;
      }
      update({ aiReady: false, phase: 'incomplete', warning: `AI download incomplete: ${error.message}` });
      throw error;
    }
  })().finally(() => { aiJob = null; });
  return aiJob;
}
