/** Whether the trained network plays on the graphics card, through ONNX
 * Runtime Web's WebGPU provider, or on the processor, through the reduced
 * WebAssembly runtime, which every device can run.
 *
 * The page asks for the graphics card only where its network's manifest
 * entry says the graph runs whole on WebGPU (`webgpu`, written by
 * scripts/publish-model-r2.mjs), the browser has WebGPU at all, the device is
 * neither a phone nor a tablet, and the player has not turned it off with
 * `?gpu=off` (webgpuWanted). The worker then asks for an adapter, refuses a
 * fallback one, which is the processor again, loads the WebGPU runtime and
 * the network, and times it on the first position it is asked (openOnGpu).
 * Anything that fails or is slow leaves the network to WebAssembly, exactly
 * as a page plays it without this.
 */

/** The longest a position may take on the graphics card, after one warm-up
 * run, for the page to play there: the median of TIMED_RUNS runs of the first
 * position asked. On a desktop with a GeForce RTX 5070 Ti one position takes
 * about 58 ms on WebGPU and 122 to 163 ms on WebAssembly with one thread
 * (10 November 2026), and 64 positions about 61 ms on WebGPU. A card under
 * this bound is never slower than WebAssembly on a fast desktop for a single
 * move, and is far faster in a review's batches. */
export const GPU_POSITION_MS = 100;
/** Runs timed after the warm-up, whose median is held to GPU_POSITION_MS. */
export const TIMED_RUNS = 3;

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

/** Whether the page asks its worker to try the graphics card for `network`
 * (a manifest entry): only where the entry says its graph runs whole on
 * WebGPU, the address does not say `?gpu=off`, the browser has WebGPU and the
 * device is not a phone or tablet. Older entries, without the field, play on
 * WebAssembly. */
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

/** The median time of `runs` calls of `run`, after one that is not timed:
 * the first run on a graphics card compiles its shaders and takes a second
 * or so. */
export async function timedRuns(run, { runs = TIMED_RUNS, now = () => performance.now() } = {}) {
  await run();
  const times = [];
  for (let index = 0; index < runs; index++) {
    const began = now();
    await run();
    times.push(now() - began);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

/** The network opened on the graphics card, `{ runtime, session, ms }`, or
 * why not, `{ reason }`, where `bytes` are the verified network, `gpu` is the
 * worker's `navigator.gpu`, `importRuntime()` loads the WebGPU runtime, and
 * `measure(runtime, session)` runs the first position asked once. Nothing
 * here throws: a failure is a reason, and the session it opened is released,
 * so the worker plays on WebAssembly instead. */
export async function openOnGpu(bytes, { gpu, importRuntime, measure, limitMs = GPU_POSITION_MS, now } = {}) {
  let session = null;
  try {
    const adapter = await webgpuAdapter(gpu);
    if (!adapter) return { reason: 'no graphics card adapter' };
    const runtime = await importRuntime();
    runtime.env.logLevel = 'error';
    runtime.env.wasm.numThreads = 1;
    runtime.env.webgpu.adapter = adapter;
    session = await runtime.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
    const ms = await timedRuns(() => measure(runtime, session), { now });
    if (!(ms <= limitMs)) {
      const slow = session;
      session = null;
      await slow.release();
      return { reason: `${Math.round(ms)} ms a position`, ms };
    }
    return { runtime, session, ms };
  } catch (error) {
    try { await session?.release(); } catch { /* Released or lost with its device. */ }
    return { reason: String(error?.message ?? error) };
  }
}
