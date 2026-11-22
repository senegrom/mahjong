/** Offline preparation and honest, cache-backed download status. No localStorage
 * flag is accepted as proof that a model or its runtime is actually present.
 *
 * The service worker saves the game and the few megabytes of runtime. The
 * trained network is saved here, by the page: a worker's install, activation
 * or message event is abandoned after a few minutes, and a slow link needs
 * longer than that for 100 MB. A page has no such deadline, and it waits for
 * no worker either: it begins the network the moment a Trained player is
 * chosen, beside a first visit's install, and the runtime follows once the
 * worker is there (prepareOfflineAi).
 *
 * A build lists several networks and a page plays one of them, the one its
 * player picked (trained-network.js). Every message names that pick, so the
 * worker reports on, and keeps, the network the page plays. */
import { networkBytes, networkIsStored } from './network-store.js';
import { networkUrl, pickedVersion, playedNetwork } from './trained-network.js';
const base = new URL('./', document.baseURI);
const script = new URL('sw.js', base).href;
let worker = null;
let boot = null;
let aiJob = null;
// The network's part of that job, which is all the opponent's worker waits
// for (prepareNetwork).
let networkJob = null;
let coreJob = null;
// A player who asked for Trained offline keeps it across upgrades: each
// version's page saves its own network, automatically once per page. So does
// one who asked on this page before its worker was there to save the runtime
// (wanted).
let filled = false;
let wanted = false;
const updatesSaved = new WeakSet();
/** The state before anything is known, which App also renders until its
 * first report. Each update replaces the state; none is changed in place. */
export const INITIAL_OFFLINE = Object.freeze({ supported: null, coreReady: false, aiReady: false, hasModel: false,
  phase: 'checking', progress: 0, warning: '', coreWarning: '', installing: false, persistent: false, updateReady: false });
let state = INITIAL_OFFLINE;
// The phase and its warning have two sources, which run side by side: the
// start says checking, ready or unavailable, and a trained save says ai while
// it runs and incomplete once it has failed. The save's word stands over the
// start's until the save is whole.
let started = { phase: state.phase, warning: state.warning };
let trained = null;
const listeners = new Set();
function update(values) {
  state = { ...state, ...values, ...(trained ?? started) };
  for (const listener of listeners) listener(state);
}
function startSays(phase, warning, values) {
  started = { phase, warning };
  update(values);
}
/** A trained save's word on the phase, or none once it is whole. */
function trainedSays(phase, warning, values) {
  trained = phase ? { phase, warning } : null;
  update(values);
}
/** Whether nothing can be saved here at all: a development server, a browser
 * without service workers or Cache Storage, or a page not served securely. */
const unsupported = () => import.meta.env.DEV || !('serviceWorker' in navigator) || !('caches' in globalThis) || !isSecureContext;
export function watchOffline(listener) {
  listeners.add(listener);
  listener(state);
  return () => listeners.delete(listener);
}
function request(type, progress, target = worker) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    // The worker's longest jobs are the runtime, 4 MB, and a repair of the
    // game and tile pictures, about 7 MB at most. The deadline is the whole
    // job's: the progress it reports does not extend it.
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
    // The version the player picked, not the network this page resolved it
    // to: a waiting update reads it against its own list, as its page will,
    // and one that no longer lists it answers with its own default. An update
    // that forgets earlier picks plays its default all the same (saveUpdate).
    try { target.postMessage({ type, version: pickedVersion() }, [channel.port2]); }
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
 * network storage has no room for is not downloaded only to be refused.
 * Saves run one at a time, page-wide: each transfer holds a whole network
 * until it is stored, and two at once would also split the link. */
let networkSave = Promise.resolve();
function saveNetwork(url, expect, onProgress) {
  networkSave = networkSave.catch(() => {}).then(() => storeNetwork(url, expect, onProgress));
  return networkSave;
}
async function storeNetwork(url, expect, onProgress) {
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
  const ended = installEnd(installing);
  return new Promise((resolve, reject) => {
    // The install goes on after the page stops waiting for it, and the page
    // goes on listening (installEnd). One that fails later is said and tried
    // again, where the page used to go on asking the player to stay
    // connected; one that finishes later takes control of the page
    // (refreshWhenControlled).
    let waiting = true;
    const timer = setTimeout(() => {
      waiting = false;
      reject(new Error('Still saving for offline play. Stay connected until this status says ready.'));
    }, 120000);
    ended.then(() => { clearTimeout(timer); resolve(); }, failed => {
      clearTimeout(timer);
      if (waiting) reject(failed); else update({ coreWarning: failed.message });
    });
  });
}
// Why a first install failed, as its worker said before it went
// (MAHJONG_INSTALL_FAILED in the service worker), and whether one has failed
// with nothing said yet: the word and the worker's end reach the page in
// either order.
let installSaid = null, installUnexplained = false, hearing = false;
function hearInstalls() {
  if (hearing) return;
  hearing = true;
  navigator.serviceWorker.addEventListener('message', ({ data, source }) => {
    if (data?.type !== 'MAHJONG_INSTALL_FAILED' || source?.scriptURL !== script) return;
    installSaid = { storage: Boolean(data.storage), error: String(data.error ?? '') };
    if (installUnexplained) update({ coreWarning: installFailure() });
  });
}
/** How an installing worker ends: activated, or failed with what the game's
 * line then says (installFailure). One listener a worker, however many
 * starts wait on it, so that its failure is judged once. */
const installs = new WeakMap();
function installEnd(installing) {
  if (!installs.has(installing)) {
    // What a worker before this one said of its failure is over with it.
    installSaid = null;
    installUnexplained = false;
    const ended = new Promise((resolve, reject) => {
      const changed = () => {
        if (!['activated', 'redundant'].includes(installing.state)) return;
        installing.removeEventListener('statechange', changed);
        if (installing.state === 'activated') resolve(); else reject(new Error(installFailure()));
      };
      installing.addEventListener('statechange', changed); changed();
    });
    ended.catch(() => {});
    installs.set(installing, ended);
  }
  return installs.get(installing);
}
/** What the game's line says of a first install that failed, which is tried
 * again by itself (retryInstall) unless storage refused it: with no room, or
 * storage that refuses, another attempt helps as little as a reconnect, and
 * the line says what storage said instead. */
function installFailure() {
  const said = installSaid;
  installSaid = null;
  installUnexplained = !said;
  if (said?.storage) {
    clearTimeout(retryTimer);
    retryTimer = null;
    return said.error;
  }
  retryInstall();
  return 'Game download incomplete. It will be tried again; stay connected.';
}
/** A first install that failed is tried again by itself while the page stays
 * open: after five seconds, then twice as long each time, up to five minutes.
 * Each attempt keeps the files it finished, so an install a slow link cuts
 * short (Chrome abandons one after five minutes) gets further every time.
 * Offline, nothing is tried: a reconnect starts it (model-availability.js). */
const RETRY_INSTALL_MS = Object.freeze({ first: 5000, most: 300000 });
let retryDelay = RETRY_INSTALL_MS.first, retryTimer = null;
function retryInstall() {
  if (retryTimer !== null) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    if (navigator.onLine) void startOffline();
  }, retryDelay);
  retryDelay = Math.min(retryDelay * 2, RETRY_INSTALL_MS.most);
}
/** Whether the browser keeps this site's storage when the device runs short.
 * The start only reads the answer. Asking for it (`ask`) is left until a
 * trained AI has been saved, the download a player asks for by choosing a
 * Trained player: Firefox puts the question to the player, and asked at
 * every start it came before they had done anything, and again each visit
 * until answered. The other browsers decide by themselves. */
async function persistentStorage(ask = false) {
  try {
    const storage = navigator.storage;
    const persistent = await storage?.persisted?.() || (ask && await storage?.persist?.()) || false;
    update({ persistent });
  } catch { /* Some browsers decline persistence; never claim it was granted. */ }
}
/** An update installs with the game alone and waits for this version's
 * windows to close. When Trained was requested, save the update's runtime and
 * network now, while this version runs, so the update does not start without
 * them on a plane: the network its worker names for this player's pick, which
 * is the one its page will play unless the update makes a new network the
 * default. Such an update's page forgets every earlier pick and plays its
 * default, which its worker cannot tell, so for a player who had picked
 * another network nothing new is saved here. Should this not finish, or save
 * nothing, the update's own page saves what it plays once it runs online. */
async function saveUpdate(registration) {
  const next = registration.waiting;
  if (!next || updatesSaved.has(next) || !navigator.onLine) return;
  updatesSaved.add(next);
  try {
    // This version's own preparation, when under way, comes first; its
    // network is then ahead of the update's in saveNetwork's queue.
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
  // An update the browser found by itself, before the page began to listen,
  // is installing already: its updatefound has been and gone, and no other
  // comes for it.
  const watch = () => registration.installing?.addEventListener('statechange', check);
  check();
  watch();
  registration.addEventListener('updatefound', watch);
}
/** The worker reports on the network it resolved the player's pick to. Match
 * that identity to the network this page plays; a reply about another
 * version's network is checked against this page's own stored record
 * instead. Neither reads the model: never hash the same model again just to
 * repeat status. */
async function withNetwork(info) {
  if (!info || !info.aiReady) return info;
  return { ...info, aiReady: await networkIsStored(undefined, undefined, info.network) };
}
/** Once per page, for a player who asked for Trained offline: a new
 * version's network (and runtime) is saved by its page after activation. So
 * is the runtime of one who asked on this page while its worker was not
 * there yet: a first install that outlasted the save's wait for it. */
function fillRequested(info) {
  if (filled || !info || !(info.aiRequested || wanted) || info.aiReady || !info.hasModel || !info.coreReady || !navigator.onLine) return;
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
    update({ coreWarning: '' });
    try {
      const ready = await request('MAHJONG_PREPARE_CORE');
      update({ ...ready, coreWarning: '' });
      return ready;
    } catch (error) {
      // Storage that refuses says so itself, by what it refused: a reconnect
      // would not help it.
      update({ coreReady: false, coreWarning: error.name === 'NetworkStorageError' ? error.message
        : `Game and tile download incomplete: ${error.message} It will retry automatically when you reconnect or reopen the app.` });
      return null;
    }
  })().finally(() => { coreJob = null; });
  return coreJob;
}
/** An install that outlasts the page's wait finishes later by itself, and
 * its worker then takes control of the page. Nothing else would look again
 * until a reconnect or a return to the app, so this does, once. */
let refreshOnControl = false;
function refreshWhenControlled() {
  if (refreshOnControl) return;
  refreshOnControl = true;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    refreshOnControl = false;
    void refreshOffline().catch(() => {});
  }, { once: true });
}
export function startOffline() {
  if (boot) return boot;
  const attempt = (async () => {
    if (unsupported()) {
      startSays('unavailable', 'Offline saving is unavailable here. Keep a connection for this session.', { supported: false });
      return null;
    }
    let registration = null;
    try {
      // Existing registration is local. Never make a successful online check
      // a prerequisite for launching an already downloaded app on a plane.
      hearInstalls();
      registration = await navigator.serviceWorker.getRegistration(base.href);
      if (registration?.scope !== base.href || registration?.active?.scriptURL !== script) registration = null;
      if (!registration) registration = await navigator.serviceWorker.register(script, { scope: base.href, updateViaCache: 'none' });
      // A first visit's worker saves the whole game and every tile graphic
      // before it activates. The install reports no progress, so say that it
      // is running; a Trained player's network is saved beside it.
      // What an attempt before this one said of its failure is over with it.
      const installing = !registration.active;
      if (installing) update({ installing: true, coreWarning: '' });
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
      startSays('ready', '', { supported: true });
      const info = await prepareCore(await withNetwork(await request('MAHJONG_STATUS')));
      // This version's own network first; a waiting update's queues behind it.
      fillRequested(info);
      observeUpdates(registration);
      void persistentStorage();
      // Background updates never block this version or replace it mid-match.
      // They install with the game alone, and a slow link resumes one over
      // several attempts.
      if (navigator.onLine) void registration.update().catch(() => {});
      return state;
    } catch (error) {
      worker = null;
      if (registration) {
        // Saving works here and has not finished: an install still running
        // or one that failed, or a page a hard reload loaded past its
        // worker. Say which, beside the game's own status, and drop what an
        // earlier attempt that never reached a worker said.
        startSays('checking', '', { supported: true, coreReady: false, aiReady: false, coreWarning: error.message });
        refreshWhenControlled();
      } else {
        startSays('unavailable', `Offline saving is not ready: ${error.message}`, { supported: false, coreReady: false, aiReady: false });
      }
      return null;
    }
  })();
  // Only a boot that reached the service worker is kept. One that did not (a
  // first visit that lost its connection, an install still running, a hard
  // reload) is tried again by the next caller: a reconnect or return to the
  // app, a Trained opponent's retry, or a worker taking control.
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
/** The runtime's share of a trained save's progress, in whole percent: its
 * few megabytes beside the network's hundred and more. */
const RUNTIME_SHARE = 5;
/** Saves the trained network this page plays, with its runtime. It is
 * downloaded only when a Trained player is chosen (or was, in an earlier
 * version), and one job runs at a time.
 *
 * Its two parts run side by side, and the save is whole when both are done,
 * in either order, and the worker has been told. The page begins the network
 * at once. It waits for no service worker: a first visit's worker is still
 * saving the game then, and the first Trained move waits for the network
 * alone (prepareNetwork). The runtime follows, by the worker that serves it,
 * once that worker is there. A network that cannot be had ends the job there
 * and then, so that a retry starts another; a runtime that cannot be saved
 * is said once the network has been. */
export function prepareOfflineAi() {
  if (aiJob) return aiJob;
  wanted = true;
  let handOver, over = false;
  const inHand = new Promise(resolve => { handOver = resolve; });
  const job = saveTrained(handOver, () => !over).finally(() => { over = true; aiJob = null; });
  // Settled as soon as the network is in hand, or with the job where it ends
  // without one: after the job has been let go, so that whoever hears that
  // the network failed can ask again at once and start another.
  networkJob = Promise.race([inHand, job.then(() => null)]);
  networkJob.catch(() => {});
  aiJob = job;
  return job;
}
/** What the opponent's worker waits for before it loads the network, and no
 * more: the network's part of prepareOfflineAi, which this starts. It
 * answers as soon as the page has the network in hand: null where it is
 * stored, or where the page leaves it to that worker to fetch (nothing can
 * be saved here, or storage has no room for it), and `{ unstoredNetwork }`,
 * verified bytes storage refused, for that worker to take over. It fails
 * where the network could not be had. Whether the runtime is saved yet is
 * not its business: the worker loads it from the site either way. */
export function prepareNetwork() {
  void prepareOfflineAi().catch(() => {});
  return networkJob;
}
/** One trained save (prepareOfflineAi). `handOver(handed)` is told the moment
 * the network is in hand, and `live()` says whether the job still runs: its
 * runtime part can outlast it, and then says nothing more. */
async function saveTrained(handOver, live) {
  // Nothing can be saved here. The start says so, and the opponent's worker
  // fetches the network for itself: online-only browsers still work.
  if (unsupported()) { await startOffline(); return null; }
  const network = playedNetwork(), url = networkUrl(network);
  // How much of each part is done, from 0 to 1, and the progress they make
  // between them.
  const done = { network: 0, runtime: 0 };
  const saving = () => {
    if (live()) trainedSays('ai', '', { progress: Math.floor((100 - RUNTIME_SHARE) * done.network + RUNTIME_SHARE * done.runtime) });
  };
  // The runtime, by the worker that serves it, once that worker is there:
  // on a first visit after its install, which the network does not wait for.
  // This adds only the runtime. Core preparation has its own automatic
  // startup/reconnect path and is not opt-in.
  const runtime = (async () => {
    if (!(await startOffline())) return null;
    const info = await withNetwork(await request('MAHJONG_STATUS'));
    if (live()) update(info);
    if (info.aiReady) return info;
    saving();
    await request('MAHJONG_PREPARE_AI', ({ bytes, total }) => {
      done.runtime = total ? bytes / total : 0;
      saving();
    });
    done.runtime = 1;
    saving();
    return info;
  })();
  // Judged below, once the network is in hand.
  runtime.catch(() => {});
  try {
    // Storage that refuses, the network or the runtime, leaves online play
    // as it was: said as a warning at the end, never as a failed download.
    let handed = null, refused = null;
    const stored = await networkIsStored(url, network);
    if (stored) done.network = 1;
    else {
      saving();
      try {
        await saveNetwork(url, network, ({ bytes, total }) => {
          done.network = total ? bytes / total : 0;
          saving();
        });
        done.network = 1;
        saving();
      } catch (error) {
        if (error.name !== 'NetworkStorageError') throw error;
        refused = error;
        // Bytes verified but refused by storage go to the opponent's worker,
        // which would otherwise download the same network again.
        if (error.bytes) handed = { unstoredNetwork: error.bytes };
      }
    }
    handOver(handed);
    let info = null;
    try { info = await runtime; } catch (error) {
      if (error.name !== 'NetworkStorageError') throw error;
      refused ??= error;
    }
    if (refused) {
      trainedSays('incomplete', `${refused.message} Online play is still available.`, { aiReady: false });
      return handed ? { aiReady: false, ...handed } : null;
    }
    // No worker here to keep a runtime: the start says why, and play goes on
    // online. Nor is there anything to tell a worker that found both saved.
    if (!info || (stored && info.aiReady)) { trainedSays(null); return info; }
    // The worker now keeps this network, the one saved before it, its
    // default and those the player picked, and no other.
    const verified = await withNetwork(await request('MAHJONG_NETWORK_SAVED'));
    trainedSays(null, '', { ...verified, progress: 100 });
    void persistentStorage(true);
    return verified;
  } catch (error) {
    trainedSays('incomplete', `AI download incomplete: ${error.message}`, { aiReady: false });
    throw error;
  }
}
