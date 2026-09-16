import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';
import { MEMORY_LIMITS_MIB } from '../src/lib/memory-budget.js';

// Run the shipped models and reduced runtime, with the large shared-memory
// reservation rejected as it is on affected phones. A tiny model must not
// require permission to reserve the full 4 GB WASM address space.
test('the network infers repeatedly within a phone-sized WASM reservation', async (t) => {
  const NativeMemory = WebAssembly.Memory;
  const limit = MEMORY_LIMITS_MIB[0] * 1024 * 1024;
  const memories = [];
  WebAssembly.Memory = class extends NativeMemory {
    constructor(options) {
      if (options.shared && options.maximum * 65536 > limit) {
        throw new RangeError(`Out of memory: shared reservation exceeds ${MEMORY_LIMITS_MIB[0]} MiB`);
      }
      super(options);
      memories.push(this);
    }
  };
  t.after(() => { WebAssembly.Memory = NativeMemory; });
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  // The trained network is not a file of this site any more: it comes from
  // its bucket (src/lib/network-store.js). This measures whatever network it
  // is pointed at, so that the reservation a phone gets is checked against a
  // real one rather than against nothing.
  const network = process.env.MAHJONG_NETWORK;
  if (!network) return t.skip('set MAHJONG_NETWORK to an exported network to measure it');
  ort.env.wasm.wasmPaths = {
    mjs: new URL('../dist/ort/ort-wasm-simd-threaded.mjs', import.meta.url).href,
    wasm: new URL('../dist/ort/ort-wasm-simd-threaded.wasm', import.meta.url).href,
  };

  const baselines = new Map();
  for (const name of Array(3).fill(network)) {
    let session, input;
    try {
      const bytes = new Uint8Array(await readFile(name));
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      const shape = session.inputMetadata[0].shape.map(size => typeof size === 'number' ? size : 1);
      const data = Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, index) => index % 11 / 10);
      input = new ort.Tensor('float32', data, shape);
      // The fused network reads the rules' mask as well as the planes, and
      // answers a move the rules forbid with negative infinity, so what must
      // be a number is the moves it was allowed.
      const wants = session.inputNames.includes('legal');
      const allowed = Array.from({ length: 46 }, (_, index) => index < 14);
      const mask = wants
        ? new ort.Tensor('float32', Float32Array.from(allowed, can => (can ? 1 : 0)), [1, allowed.length])
        : null;
      for (let turn = 0; turn < 12; turn++) {
        const outputs = await session.run(mask ? { planes: input, legal: mask } : { planes: input });
        try {
          const logits = Array.from(outputs.policy.data);
          assert.ok(logits.length > 34);
          assert.ok(logits.filter((_, index) => !wants || allowed[index]).every(Number.isFinite));
          if (!baselines.has(name)) baselines.set(name, logits);
          else assert.deepEqual(logits, baselines.get(name), 'switching sessions must preserve model output');
        } finally {
          for (const output of Object.values(outputs)) output.dispose();
        }
      }
    } finally {
      input?.dispose();
      await session?.release();
    }
  }
  assert.ok(memories.length > 0, 'the actual reduced WASM runtime must be exercised');
  const peak = Math.max(...memories.map(memory => memory.buffer.byteLength));
  // Room to load the network again without the heap running into the
  // reservation. A fifth of it was a whole model when the page carried a
  // small one; with a 116 MB network it is the margin that matters.
  assert.ok(peak < limit * 0.8, `model switching needs headroom inside the reservation; heap reached ${peak} bytes`);
  t.diagnostic(`72 inferences across 6 model loads; WASM heap ${(peak / 1048576).toFixed(1)} MiB`);
});
