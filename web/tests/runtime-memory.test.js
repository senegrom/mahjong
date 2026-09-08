import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';

// Run the shipped models and reduced runtime, with the large shared-memory
// reservation rejected as it is on affected phones. A tiny model must not
// require permission to reserve the full 4 GB WASM address space.
test('both published models infer repeatedly within a phone-sized WASM reservation', async (t) => {
  const NativeMemory = WebAssembly.Memory;
  const limit = 192 * 1024 * 1024;
  const memories = [];
  WebAssembly.Memory = class extends NativeMemory {
    constructor(options) {
      if (options.shared && options.maximum * 65536 > limit) {
        throw new RangeError('Out of memory: shared reservation exceeds 192 MiB');
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

  const baselines = new Map();
  for (const name of Array(3).fill(['model.onnx', 'model-strong.onnx']).flat()) {
    let session, input;
    try {
      const bytes = new Uint8Array(await readFile(new URL(`../dist/${name}`, import.meta.url)));
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      const shape = session.inputMetadata[0].shape.map(size => typeof size === 'number' ? size : 1);
      const data = Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, index) => index % 11 / 10);
      input = new ort.Tensor('float32', data, shape);
      for (let turn = 0; turn < 12; turn++) {
        const outputs = await session.run({ planes: input });
        try {
          const logits = Array.from(outputs.policy.data);
          assert.ok(logits.length > 34 && logits.every(Number.isFinite));
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
  assert.ok(peak < limit / 2, `model switching needs headroom inside the reservation; heap reached ${peak} bytes`);
  t.diagnostic(`72 inferences across 6 model loads; WASM heap ${(peak / 1048576).toFixed(1)} MiB`);
});
