import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';
import { MEMORY_LIMITS_MIB } from '../src/lib/memory-budget.js';

// Run the published network and reduced runtime, with large shared-memory
// reservations rejected as they are on affected phones.
test('the network infers repeatedly within a phone-sized WASM reservation', async (t) => {
  const network = process.env.MAHJONG_NETWORK;
  if (!network) {
    assert.ok(!process.env.CI, 'CI must supply the verified published network through MAHJONG_NETWORK');
    return t.skip('set MAHJONG_NETWORK to an exported network to measure it');
  }
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
  ort.env.wasm.wasmPaths = {
    mjs: new URL('../dist/ort/ort-wasm-simd-threaded.mjs', import.meta.url).href,
    wasm: new URL('../dist/ort/ort-wasm-simd-threaded.wasm', import.meta.url).href,
  };

  const loads = 3, inferences = 12;
  const baselines = new Map();
  for (const name of Array(loads).fill(network)) {
    let session, input, mask;
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
      mask = wants
        ? new ort.Tensor('float32', Float32Array.from(allowed, can => (can ? 1 : 0)), [1, allowed.length])
        : null;
      for (let turn = 0; turn < inferences; turn++) {
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
      mask?.dispose();
      await session?.release();
    }
  }
  assert.ok(memories.length > 0, 'the actual reduced WASM runtime must be exercised');
  const peak = Math.max(...memories.map(memory => memory.buffer.byteLength));
  // Preserve headroom to load the network again within the reservation.
  assert.ok(peak < limit * 0.8, `model switching needs headroom inside the reservation; heap reached ${peak} bytes`);
  t.diagnostic(`${loads * inferences} inferences across ${loads} model loads; WASM heap ${(peak / 1048576).toFixed(1)} MiB`);
});
