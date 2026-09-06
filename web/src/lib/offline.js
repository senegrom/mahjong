/** Offline preparation and honest, cache-backed download status. No localStorage
 * flag is accepted as proof that a model or its runtime is actually present. */
const base = new URL('./', document.baseURI);
const script = new URL('sw.js', base).href;
let worker = null;
let boot = null;
let aiJob = null;
let coreJob = null;
let state = { supported: null, coreReady: false, aiReady: false, hasModel: false,
  phase: 'checking', progress: 0, warning: '', coreWarning: '', coreLoading: false,
  persistent: false, updateReady: false };
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
function request(type, progress) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => finish(new Error('Offline preparation timed out. Please reconnect and retry.')), 180000);
    const finish = (error, value) => {
      clearTimeout(timer); channel.port1.close();
      if (error) reject(error); else resolve(value);
    };
    channel.port1.onmessage = ({ data }) => {
      if (data.progress) progress?.(data.progress);
      else finish(data.error ? new Error(data.error) : null, data.value);
    };
    try { worker.postMessage({ type }, [channel.port2]); }
    catch (error) { finish(error); }
  });
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
function observeUpdates(registration) {
  const check = () => update({ updateReady: Boolean(registration.waiting) });
  check();
  registration.addEventListener('updatefound', () => {
    registration.installing?.addEventListener('statechange', check);
  });
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
  boot = (async () => {
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
      await activated(registration);
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
      await prepareCore(await request('MAHJONG_STATUS'));
      observeUpdates(registration);
      void persistentStorage();
      // Background updates never block this version or replace it mid-match.
      if (navigator.onLine) void registration.update().catch(() => {});
      return state;
    } catch (error) {
      worker = null;
      update({ supported: false, coreReady: false, aiReady: false, phase: 'unavailable',
        warning: `Offline saving is not ready: ${error.message}` });
      return null;
    }
  })();
  return boot;
}
export async function refreshOffline() {
  // A first visit may have lost its connection during installation. The
  // reconnect/foreground event retries automatically, never via the AI button.
  if (!worker && state.phase === 'unavailable') boot = null;
  await startOffline();
  if (!worker) return null;
  return prepareCore(await request('MAHJONG_STATUS'));
}
export function prepareOfflineAi() {
  if (aiJob) return aiJob;
  aiJob = (async () => {
    if (!(await startOffline())) return null; // Online-only browsers still work, with a visible warning.
    // This action adds only the trained network/runtime. Core preparation has
    // its own automatic startup/reconnect path and is not opt-in.
    const info = await request('MAHJONG_STATUS');
    update(info);
    if (info.aiReady) return info;
    update({ phase: 'ai', progress: 0, warning: '' });
    try {
      const ready = await request('MAHJONG_PREPARE_AI', ({ bytes, total }) => {
        update({ progress: total ? Math.floor(100 * bytes / total) : 0 });
      });
      update({ ...ready, phase: 'ready', progress: 100, warning: '' });
      void persistentStorage();
      return ready;
    } catch (error) {
      update({ aiReady: false, phase: 'incomplete', warning: `AI download incomplete: ${error.message}` });
      throw error;
    }
  })().finally(() => { aiJob = null; });
  return aiJob;
}
