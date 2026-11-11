/** Whether a review's batches may run on the graphics card, through ONNX
 * Runtime Web's WebGPU provider, and the tests the card must pass first.
 * Play never runs there: every single decision goes to the WebAssembly
 * worker, as it always did (policy.worker.js).
 *
 * The page considers the card only where its network's manifest entry says
 * the graph runs whole on WebGPU (`webgpu`, written by
 * scripts/publish-model-r2.mjs), the browser has WebGPU at all, the device is
 * neither a phone nor a tablet, and the player has not turned it off with
 * `?gpu=off` (webgpuWanted). Its own worker (gpu-batch.worker.js) then needs
 * a real adapter (webgpuAdapter), and the card must answer a batch the
 * WebAssembly worker has already answered as that worker did (agreement)
 * and in at most half the time (fastEnough) before it is given any batch
 * (gpu-batches.js).
 */

/** Runs of the sample timed after the warm-up, whose median is compared
 * with WebAssembly's time for the same batch. */
export const TIMED_RUNS = 3;
/** The most a move's weight on the card may differ from its weight on
 * WebAssembly, at any position of the sample. The card's float arithmetic
 * differs in the last bits: on a GeForce RTX 5070 Ti weights differed by
 * 6.5e-6 at most over 368 positions (10 November 2026). A thousandth is
 * generous against that, and far below anything a review shows (weights
 * are shown in whole percent) or a wrong answer would give. */
export const WEIGHT_BOUND = 0.001;
/** How many times faster than WebAssembly the card must run the sample. */
export const SPEEDUP = 2;
/** How long the card's bring-up may take in all, from creating its worker
 * to the last timed run, before the card is declined. */
export const BRING_UP_MS = 60000;
/** How long one batch may take on the card once it serves them. Sixty-four
 * positions took 61 ms on a desktop card; one that takes longer than this
 * has failed, and the batch is asked of WebAssembly instead. */
export const CARD_BATCH_MS = 10000;

/** Whether a phone or tablet is asking: its graphics card is the one its
 * battery runs, and its browsers' WebGPU is the least tried. Chromium says so
 * itself (`userAgentData.mobile`, false on an Android tablet, so its platform
 * is read too); elsewhere the user agent, where iPadOS asking for desktop
 * pages calls itself a Mac, which has no touch screen. */
export function mobileDevice(nav) {
  const data = nav?.userAgentData;
  if (typeof data?.mobile === 'boolean') return data.mobile || /^(Android|iOS|iPadOS)$/i.test(data.platform ?? '');
  const agent = String(nav?.userAgent ?? '');
  if (/Android|iPhone|iPad|iPod|Mobile|Silk|Kindle/i.test(agent)) return true;
  return /Macintosh/.test(agent) && (nav?.maxTouchPoints ?? 0) > 1;
}

/** Whether the page considers the graphics card for `network` (a manifest
 * entry): only where the entry says its graph runs whole on WebGPU, the
 * address does not say `?gpu=off`, the browser has WebGPU and the device is
 * not a phone or tablet. Older entries, without the field, never do. */
export function webgpuWanted(network, { search = '', navigator: nav } = {}) {
  if (network?.webgpu !== true) return false;
  if (new URLSearchParams(search).get('gpu') === 'off') return false;
  if (!nav?.gpu) return false;
  return !mobileDevice(nav);
}

/** Adapters that are the processor drawing in software, whatever they say of
 * themselves: Chrome's SwiftShader, Mesa's llvmpipe and Windows's basic
 * renderer. */
const SOFTWARE = /swiftshader|llvmpipe|softpipe|basic render/i;
/** Whether an adapter's `info` names a software renderer (SOFTWARE). */
export const softwareAdapter = ({ vendor = '', architecture = '', device = '', description = '' } = {}) =>
  SOFTWARE.test(`${vendor} ${architecture} ${device} ${description}`);

/** The graphics card's adapter, or null: none where the browser gives none,
 * and none for a fallback adapter, or one that renders in software, which
 * run on the processor. */
export async function webgpuAdapter(gpu) {
  if (typeof gpu?.requestAdapter !== 'function') return null;
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) return null;
  return softwareAdapter(adapter.info) ? null : adapter;
}

/** Why the card's answers to a batch, `given`, do not agree with
 * WebAssembly's, `expected` (each a list of `{ action, weights }`, a
 * position each), or null where they do: every move's weight within `bound`
 * of WebAssembly's, and the same best move, unless WebAssembly's own two best
 * are closer than twice the bound, where the arithmetic may tip either way. */
export function agreement(expected, given, bound = WEIGHT_BOUND) {
  if (!Array.isArray(given) || given.length !== expected.length) return 'the card answered the wrong number of positions';
  for (let row = 0; row < expected.length; row++) {
    const plain = expected[row].weights, card = given[row]?.weights;
    if (!card || card.length !== plain.length) return `the card answered position ${row} with the wrong number of moves`;
    for (let move = 0; move < plain.length; move++) {
      if (!(Math.abs(card[move] - plain[move]) <= bound)) {
        return `position ${row}, move ${move}: weight ${card[move]} on the card against ${plain[move]}`;
      }
    }
    if (given[row].action !== expected[row].action) {
      const [first = 0, second = 0] = [...plain].sort((a, b) => b - a);
      if (first - second >= 2 * bound) return `position ${row}: best move ${given[row].action} on the card against ${expected[row].action}`;
    }
  }
  return null;
}

/** Whether the card's median time for the sample, `cardMs`, is at most
 * 1/SPEEDUP of WebAssembly's time for the same batch, `wasmMs`. */
export const fastEnough = (cardMs, wasmMs) => cardMs * SPEEDUP <= wasmMs;

/** The median of a list of times. */
export const median = times => [...times].sort((a, b) => a - b)[Math.floor(times.length / 2)];
