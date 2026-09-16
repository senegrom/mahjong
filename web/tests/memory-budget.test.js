import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { MEMORY_LIMITS_MIB, MemoryBudget, nextMemoryLimit } from '../src/lib/memory-budget.js';

const MIB = 1048576, PAGE = 65536;

test('a browser refusing spare heap capacity can still grant exactly the needed pages', () => {
  const budget = new MemoryBudget(), attempts = [];
  const current = 100 * MIB, needed = current + PAGE;
  let refreshed = false;
  const memory = { buffer: { byteLength: current }, grow(pages) {
    attempts.push(pages);
    if (pages !== 1) throw new RangeError('Out of memory');
    this.buffer = { byteLength: needed };
  } };
  assert.equal(budget.grow(memory, needed, () => { refreshed = true; }), true);
  assert.ok(attempts[0] > 1);
  assert.equal(attempts.at(-1), 1);
  assert.equal(refreshed, true);
  assert.equal(budget.failure, null);
});

test('browser allocation failures never request a larger reservation', () => {
  const budget = new MemoryBudget(), needed = 17 * MIB;
  const memory = { buffer: { byteLength: 16 * MIB }, grow() { throw new RangeError('Out of memory'); } };
  assert.equal(budget.grow(memory, needed, () => assert.fail('No views to refresh')), false);
  assert.deepEqual(budget.failure, { kind: 'browser', requestedBytes: needed });
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[0], budget.failure), null);
  budget.beginRequest();
  assert.equal(budget.failure, null, 'a successful later request must not inherit an earlier failure');
});

test('only an exceeded application ceiling qualifies for bounded extra headroom', () => {
  const budget = new MemoryBudget(), needed = (MEMORY_LIMITS_MIB[0] + 8) * MIB;
  const memory = { buffer: { byteLength: 16 * MIB }, grow() { assert.fail('Do not exceed the configured ceiling'); } };
  assert.equal(budget.grow(memory, needed, () => {}), false);
  assert.deepEqual(budget.failure, { kind: 'limit', requestedBytes: needed });
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[0], budget.failure), MEMORY_LIMITS_MIB[1]);
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[0], { kind: 'limit', requestedBytes: (MEMORY_LIMITS_MIB[1] + 1) * MIB }), MEMORY_LIMITS_MIB[2]);
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[1], { kind: 'limit', requestedBytes: (MEMORY_LIMITS_MIB[1] + 1) * MIB }), MEMORY_LIMITS_MIB[2]);
  for (const requestedBytes of [0, 16 * MIB, NaN, Infinity, (MEMORY_LIMITS_MIB[2] + 1) * MIB]) {
    assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[0], { kind: 'limit', requestedBytes }), null);
  }
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[2], { kind: 'limit', requestedBytes: (MEMORY_LIMITS_MIB[2] + 16) * MIB }), null);
  assert.throws(() => budget.configure(4096), /Invalid/);
});

test('the shipped allocator reports its limit, grows after a fresh bounded restart, and respects browser refusal', () => {
  function probe(limit, browserLimit = null, wanted = MEMORY_LIMITS_MIB[0] + 32) {
    // Each run gets a fresh module and reservation, just as a replacement worker does.
    const source = `
      import { readFileSync } from 'node:fs';
      import { memoryBudget } from './dist/ort/memory-budget.mjs';
      import factory from './dist/ort/ort-wasm-simd-threaded.mjs';
      if (${browserLimit} !== null) {
        const NativeMemory = WebAssembly.Memory;
        WebAssembly.Memory = class extends NativeMemory {
          constructor(options) {
            if (options.maximum * 65536 > ${browserLimit} * 1048576) throw new RangeError('Out of memory');
            super(options);
          }
        };
      }
      memoryBudget.configure(${limit});
      try {
        const runtime = await factory({ numThreads: 1, wasmBinary: new Uint8Array(readFileSync('./dist/ort/ort-wasm-simd-threaded.wasm')) });
        const pointer = runtime._malloc(${wanted} * 1048576);
        console.log(JSON.stringify({ pointer, failure: memoryBudget.failure, heap: memoryBudget.memory.buffer.byteLength }));
        if (pointer) runtime._free(pointer);
      } catch (error) {
        console.log(JSON.stringify({ error: String(error), failure: memoryBudget.failure }));
      }
    `;
    return JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-'], {
      input: source, encoding: 'utf8', cwd: new URL('../', import.meta.url), timeout: 20000,
    }));
  }
  const limited = probe(MEMORY_LIMITS_MIB[0]);
  assert.equal(limited.pointer, 0);
  assert.equal(limited.failure.kind, 'limit');
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[0], limited.failure), MEMORY_LIMITS_MIB[1]);
  assert.equal(limited.heap, 16 * MIB, 'a rejected request does not eagerly allocate its maximum');
  const expanded = probe(MEMORY_LIMITS_MIB[1]);
  assert.ok(expanded.pointer > 0);
  assert.equal(expanded.failure, null);
  assert.ok(expanded.heap > MEMORY_LIMITS_MIB[0] * MIB && expanded.heap <= MEMORY_LIMITS_MIB[1] * MIB);
  const refused = probe(MEMORY_LIMITS_MIB[1], MEMORY_LIMITS_MIB[0]);
  assert.match(refused.error, /Out of memory/);
  assert.equal(refused.failure.kind, 'browser');
  assert.equal(nextMemoryLimit(MEMORY_LIMITS_MIB[1], refused.failure), null);
});
