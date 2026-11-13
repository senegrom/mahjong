/** A review's batches on the graphics card, where the card has shown that it
 * answers them as WebAssembly does, at least twice as fast. Play never goes
 * there: single decisions, and every batch until the card is ready, go to
 * the WebAssembly worker exactly as before (policy.js).
 *
 * The card goes through four states: untried, starting, ready and declined.
 *  - Its bring-up never delays anything. It starts only once WebAssembly has
 *    answered a batch of two or more positions, the second such batch of the
 *    page (the first may carry the network's load, or a wait behind a move),
 *    and only where webgpuWanted holds and no decline stands. That batch, its
 *    WebAssembly answers and the time WebAssembly took for it are the sample.
 *  - Its own worker (gpu-batch.worker.js) opens the network stored on this
 *    device and runs the timed batch, the sample's positions again and again
 *    to a hand's worth (timedOrder), once to warm up and TIMED_RUNS times
 *    more. Every run must agree with WebAssembly's answers (agreement), and
 *    the runs' median time must be at most half of what WebAssembly takes
 *    for as many positions (fastEnough), at the least time a position it
 *    has shown, over the sample and the batches it answers meanwhile. The
 *    whole bring-up has BRING_UP_MS. Anything else declines the card.
 *  - Ready, the card answers every batch of two or more positions, one at a
 *    time, each within CARD_BATCH_MS, and policyRuntime() says 'webgpu', so
 *    a review asks bigger batches (batchMostFor). A batch that fails or runs
 *    out of time there declines the card, ends its worker and is answered on
 *    WebAssembly instead: the caller gets an answer, or its own abort.
 *  - A decline about the device (no adapter or a software one, slow,
 *    disagreeing, a session or run that failed, a bring-up out of time) is
 *    remembered in localStorage for this runtime folder and this network's
 *    digest, so a later visit does not try again until either changes or a
 *    week has passed (DECLINE_LASTS_MS); `?gpu=on` forgets it. A runtime the
 *    browser could not fetch, or a network not saved on this device,
 *    declines the card for this visit only. `?gpu=off` never tries. Why the
 *    card was declined is said in the console once.
 *
 * While the card runs a batch the page is told (`onWork`), and holds still
 * whatever it paints again every frame, a dora's sheen (Tile.svelte). The
 * browser paints the page and runs the card's work in one process, on one
 * thread of it; a sheen in view had that thread paint its part of the page
 * again, every picture in it, for every frame, and kept it so busy that a
 * batch the card answers in 70 ms took 4.5 s (Chrome on a desktop with six
 * dora in view, 12 November 2026).
 */
/* global __WEBGPU_RUNTIME_DIRECTORY__ */
import { networkUrl, playedNetwork } from './trained-network.js';
import { BRING_UP_MS, CARD_BATCH_MS, DECLINE_LASTS_MS, TIMED_RUNS, agreement, fastEnough, median, timedOrder, webgpuWanted } from './webgpu.js';

/** A request the page dropped, as resetPolicy drops the WebAssembly worker's:
 * its caller hears `reason`, and the card is not blamed. */
class Dropped extends Error {
  constructor(reason) { super('dropped'); this.reason = reason; }
}
/** A failure of the card, and whether it is the device's (`remember`). */
class CardFailure extends Error {
  constructor(message, remember = true) { super(message); this.remember = remember; }
}
const abortError = () => new DOMException('Match changed', 'AbortError');

/** Positions as one request, as policy.js lays them out for its worker. */
function laid(positions) {
  const width = positions[0].planes.length;
  const planes = new Float32Array(positions.length * width);
  positions.forEach((position, index) => planes.set(position.planes, index * width));
  return { planes, mask: positions.flatMap(position => Array.from(position.mask)), count: positions.length };
}

/** The card's batches for one page. Every outside thing comes in, for the
 * tests: `createWorker()` starts the card's worker, `storage` is
 * localStorage (or null), `search` the address's query, `navigator` the
 * browser's, `network()` the manifest entry the page plays, `runtime` the
 * WebGPU runtime's folder, `scope` the page's folder, `now` and the timers,
 * `today()` the date in milliseconds, `warn` the console's,
 * `onRuntime(runtime)` hears each change of runtime(), and
 * `onWork(working)` hears the card begin a run and end its last. */
export function cardBatches({ createWorker, storage = null, search = '', navigator, network, runtime: folder, scope,
  now = () => performance.now(), setTimeout: later = setTimeout, clearTimeout: cancel = clearTimeout,
  today = () => Date.now(), warn = () => {}, onRuntime = () => {}, onWork = () => {} }) {
  const query = new URLSearchParams(search).get('gpu');
  const key = `mahjong-webgpu-declined:${new URL('./', scope).pathname}`;
  let state = 'untried', worker = null, opened = null, answeredBatches = 0, nextId = 0, generation = 0;
  let bringUp = null, cardTurn = Promise.resolve(), running = 0, least = Infinity;
  // Requests with the card's worker, and callers waiting on the card, whom a reset drops.
  const pending = new Map(), dropping = new Set();

  // A decline kept without its date, or with one yet to come, has lapsed too.
  const remembered = () => {
    try {
      const record = JSON.parse(storage?.getItem(key) ?? 'null');
      const age = today() - record?.at;
      return record?.runtime === folder && record?.network === network().sha256 && age >= 0 && age < DECLINE_LASTS_MS ? record : null;
    } catch { return null; }
  };
  const remember = reason => {
    try { storage?.setItem(key, JSON.stringify({ runtime: folder, network: network().sha256, reason, at: today() })); } catch { /* Storage refused: this visit only. */ }
  };
  if (query === 'on') { try { storage?.removeItem(key); } catch { /* Nothing to forget. */ } }

  function endWorker() {
    const old = worker;
    worker = null;
    opened = null;
    old?.terminate();
    for (const request of pending.values()) request.reject(new CardFailure('the card\'s worker was ended'));
    pending.clear();
  }

  function decline(reason, remembers) {
    if (state === 'declined') return;
    const was = state;
    state = 'declined';
    if (bringUp) { cancel(bringUp); bringUp = null; }
    endWorker();
    if (remembers) remember(reason);
    warn(`Trained AI reviews on WebAssembly: ${reason}`);
    if (was === 'ready') onRuntime('wasm');
  }

  /** Sends one request to the card's worker and waits for its answer. */
  function call(message) {
    const current = worker;
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      try { current.postMessage({ id, ...message }); } catch (error) { pending.delete(id); reject(new CardFailure(String(error?.message ?? error))); }
    });
  }

  /** One batch run on the card: the page hears that the card works from the
   * first run sent until the last is answered, or its worker ended. */
  function run(batch) {
    if (running++ === 0) onWork(true);
    return call(batch).finally(() => { if (--running === 0) onWork(false); });
  }

  function startWorker() {
    const current = createWorker();
    worker = current;
    current.onmessage = ({ data }) => {
      if (worker !== current) return;
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      if (data.error) request.reject(new CardFailure(data.error, data.remember !== false));
      else request.resolve(data);
    };
    current.onerror = event => { if (worker === current) decline(`the card's worker failed: ${event?.message ?? 'no reason given'}`, true); };
    current.onmessageerror = () => { if (worker === current) decline('the card\'s worker sent an unreadable answer', true); };
  }

  async function start(sample) {
    state = 'starting';
    bringUp = later(() => { bringUp = null; decline(`bringing the card up took longer than ${BRING_UP_MS / 1000} s`, true); }, BRING_UP_MS);
    let mine = null;
    try {
      const played = network();
      startWorker();
      mine = worker;
      await call({ open: { url: networkUrl(played), expect: { bytes: played.bytes, sha256: played.sha256 }, scope, webgpu: new URL(folder, scope).href } });
      opened = networkUrl(played);
      const order = timedOrder(sample.positions.length);
      const batch = laid(order.map(index => sample.positions[index])), expected = order.map(index => sample.answers[index]);
      const times = [];
      for (let turn = 0; turn <= TIMED_RUNS; turn++) {
        const began = now();
        const { analyses } = await run(batch);
        if (turn) times.push(now() - began);
        const differs = agreement(expected, analyses);
        if (differs) throw new CardFailure(`the card disagrees with WebAssembly: ${differs}`);
      }
      const ms = median(times), wasmMs = least * order.length;
      if (!fastEnough(ms, wasmMs)) {
        throw new CardFailure(`the card took ${Math.round(ms)} ms for ${order.length} positions, which WebAssembly runs in ${Math.round(wasmMs)} ms`);
      }
      if (state !== 'starting') return;
      cancel(bringUp);
      bringUp = null;
      state = 'ready';
      onRuntime('webgpu');
    } catch (error) {
      // A bring-up whose worker was ended for another network's is not this card's failure.
      if (state === 'starting' && worker === mine) decline(error.message, error.remember !== false);
    }
  }

  /** Considers the card after WebAssembly answered `positions` in `ms`. Until
   * the card serves or is declined, each batch of two or more after the
   * page's first also says how long WebAssembly takes a position, and the
   * least of those times is kept (`least`): a batch that waited its turn in
   * the worker behind another request took longer through no fault of
   * WebAssembly's, and its time alone, taken eight times over for the timed
   * batch, would let a card slower than WebAssembly pass. */
  function consider(positions, answers, ms) {
    if (positions.length < 2 || state === 'ready' || state === 'declined' || ++answeredBatches < 2) return;
    least = Math.min(least, ms / positions.length);
    if (state !== 'untried') return;
    if (!webgpuWanted(network(), { search, navigator })) { state = 'declined'; return; }
    if (query !== 'on' && remembered()) { state = 'declined'; return; }
    void start({ positions, answers });
  }

  /** One batch on the card, after the card's last batch: answered within
   * CARD_BATCH_MS, or a CardFailure; the caller's abort ends its wait at
   * once, though the card runs the batch to its end. A batch out of time
   * declines the card there and then, whether or not its caller still
   * waits: the next batch does not wait a turn behind it, and the page is
   * not left told that the card works. */
  function onCard(positions, signal) {
    const owner = generation;
    const turnTaken = cardTurn.then(() => new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(abortError());
      if (owner !== generation) return reject(new Dropped(abortError()));
      if (state !== 'ready') return reject(new CardFailure('the card was declined meanwhile'));
      const timer = later(() => {
        const failure = new CardFailure(`a batch took longer than ${CARD_BATCH_MS / 1000} s on the card`);
        decline(failure.message, failure.remember);
        reject(failure);
      }, CARD_BATCH_MS);
      run(laid(positions)).then(({ analyses }) => {
        cancel(timer);
        if (owner !== generation) reject(new Dropped(abortError()));
        else if (!Array.isArray(analyses) || analyses.length !== positions.length) reject(new CardFailure('the card answered the wrong number of positions'));
        else resolve(analyses);
      }, error => { cancel(timer); reject(error); });
    }));
    cardTurn = turnTaken.catch(() => {});
    return new Promise((resolve, reject) => {
      const abort = () => reject(abortError());
      signal?.addEventListener('abort', abort, { once: true });
      dropping.add(reject);
      turnTaken.then(resolve, reject).finally(() => { signal?.removeEventListener('abort', abort); dropping.delete(reject); });
    });
  }

  return {
    /** The answers to `positions`, as policy.js's analyzeBatch gives them,
     * `wasm(positions, signal)` being the WebAssembly worker's path. */
    async analyze(positions, signal, wasm) {
      if (opened && opened !== networkUrl(network())) {
        // Another network is played: the card starts over for it. The old
        // bring-up's time stops with its worker, a batch still on the card
        // is answered on WebAssembly without blame (below), and the page
        // hears that batches run there again.
        const was = state;
        if (bringUp) { cancel(bringUp); bringUp = null; }
        endWorker();
        state = 'untried';
        answeredBatches = 0;
        least = Infinity;
        if (was === 'ready') onRuntime('wasm');
      }
      if (state === 'ready' && positions.length >= 2) {
        try { return await onCard(positions, signal); } catch (error) {
          if (error instanceof Dropped) throw error.reason;
          if (!(error instanceof CardFailure)) throw error;
          // Only a card that still serves is declined: one already declined
          // is, and one started over for another network did not fail.
          if (state === 'ready') decline(error.message, error.remember);
        }
      }
      const began = now();
      const answers = await wasm(positions, signal);
      consider(positions, answers, now() - began);
      return answers;
    },
    /** What resetPolicy does here: every caller waiting on the card hears
     * `reason`, and the card's worker stays. */
    reset(reason) {
      generation++;
      for (const reject of dropping) reject(reason);
      dropping.clear();
    },
    runtime: () => (state === 'ready' ? 'webgpu' : 'wasm'),
    state: () => state,
  };
}

/** The attribute the page's root carries while the card runs a batch. */
export const CARD_AT_WORK = 'data-card-at-work';

let page = null;
let listener = null;
/** The page's own card, made when its first batch is asked. */
function card() {
  page ??= cardBatches({
    createWorker: () => new Worker(new URL('./gpu-batch.worker.js', import.meta.url), { type: 'module' }),
    storage: (() => { try { return globalThis.localStorage ?? null; } catch { return null; } })(),
    search: globalThis.location?.search ?? '', navigator: globalThis.navigator, network: playedNetwork,
    runtime: __WEBGPU_RUNTIME_DIRECTORY__, scope: new URL('./', document.baseURI).href,
    warn: message => globalThis.console?.warn(message), onRuntime: value => listener?.(value),
    // What paints again every frame holds still meanwhile (Tile.svelte).
    onWork: working => document.documentElement.toggleAttribute(CARD_AT_WORK, working),
  });
  return page;
}

/** policy.js's analyzeBatch: on the card once it is ready, else `wasm`. */
export const batchesOnCard = (positions, signal, wasm) => card().analyze(positions, signal, wasm);
/** policy.js's resetPolicy: callers waiting on the card hear `reason`. */
export const resetCard = reason => page?.reset(reason);
/** Where a review's batches run: 'webgpu' once the card is ready, 'wasm'
 * otherwise. The review's cap follows it (batchMostFor). */
export const policyRuntime = () => page?.runtime() ?? 'wasm';
/** `callback(runtime)` hears each change of policyRuntime(). */
export function reportRuntime(callback) { listener = callback; }
