// These are reservation ceilings, not allocations. Start at 16 MiB and grow
// only when malloc needs it. A fresh worker is required to raise the ceiling.
export const MEMORY_LIMITS_MIB = Object.freeze([192, 256, 384]);
const MIB = 1048576;
const PAGE = 65536;

export function nextMemoryLimit(current, failure) {
  if (!MEMORY_LIMITS_MIB.includes(current) || failure?.kind !== 'limit'
    || !Number.isSafeInteger(failure.requestedBytes) || failure.requestedBytes <= current * MIB) return null;
  return MEMORY_LIMITS_MIB.find(limit => limit > current && limit * MIB >= failure.requestedBytes) ?? null;
}

export function isMemoryError(error) {
  return /out of memory|bad_alloc|bad allocation|allocat|memory limit|\boom\b|aborted/i.test(String(error));
}

export class MemoryBudget {
  constructor() {
    this.limitMiB = MEMORY_LIMITS_MIB[0];
    this.memory = null;
    this.failure = null;
  }
  get maximumBytes() { return this.limitMiB * MIB; }
  configure(limitMiB) {
    if (!MEMORY_LIMITS_MIB.includes(limitMiB)) throw new Error('Invalid agent memory limit');
    if (this.memory && limitMiB !== this.limitMiB) throw new Error('Restart the agent before changing its memory limit');
    this.limitMiB = limitMiB;
  }
  beginRequest() { this.failure = null; }
  create() {
    try {
      this.memory = new WebAssembly.Memory({ initial: 256, maximum: this.maximumBytes / PAGE, shared: true });
      return this.memory;
    } catch (error) {
      this.failure = { kind: 'browser', requestedBytes: this.maximumBytes };
      throw error;
    }
  }
  grow(memory, requestedBytes, refreshViews) {
    const current = memory.buffer.byteLength;
    if (requestedBytes <= current) return false;
    if (requestedBytes > this.maximumBytes) {
      this.failure = { kind: 'limit', requestedBytes };
      return false;
    }
    // Preserve Emscripten's modest over-allocation for ordinary play. If the
    // browser refuses the spare space, try exactly what this allocation needs.
    const targets = [1, 2, 4].map(divisor => Math.min(this.maximumBytes,
      PAGE * Math.ceil(Math.max(requestedBytes, Math.min(current * (1 + 0.2 / divisor), requestedBytes + 96 * MIB)) / PAGE)));
    targets.push(PAGE * Math.ceil(requestedBytes / PAGE));
    for (const target of new Set(targets)) {
      try {
        memory.grow((target - current) / PAGE);
        refreshViews();
        this.failure = null;
        return true;
      } catch { /* Try less spare space before reporting a browser refusal. */ }
    }
    this.failure = { kind: 'browser', requestedBytes };
    return false;
  }
}

// This file is also copied beside the generated ORT loader. The worker imports
// that exact URL to configure and read the instance used by malloc's hooks.
export const memoryBudget = new MemoryBudget();
