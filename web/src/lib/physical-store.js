import { PHYSICAL_KEY, emptyPosition, parsePhysical } from './physical-position.js';

// Finish this window's pending saves before reopening the editor. Each write
// also holds a cross-window lock and compares the exact record it started from.
let pending = Promise.resolve();
const LOCK = 'riichi.physical-table.writer';
const WARNING = 'This browser could not save the physical table. Keep this window open to retain your edits.';
const CONFLICT = 'Another window changed the physical table. Reload its saved table before continuing.';
const STALLED = 'A previous save is still waiting for the browser lock. Edits may conflict until it finishes.';
// A save normally finishes in milliseconds. One stuck behind a lock another
// window never gives back must not leave the editor loading for ever.
const PATIENCE = 5000;

export class PhysicalStore {
  constructor(storage, locks, { onConflict = () => {}, onWarning = () => {}, patience = PATIENCE } = {}) {
    this.storage = storage;
    this.locks = locks;
    this.patience = patience;
    this.onConflict = onConflict;
    this.onWarning = onWarning;
    this.expected = null;
    this.queuedKey = null;
    this.lastSave = Promise.resolve(true);
    this.opened = false;
    this.closing = false;
    this.conflicted = false;
    this.unreadable = false;
    this.disabled = !storage || !locks?.request;
  }

  async read() {
    const stalled = await this.settled();
    if (this.closing) return emptyPosition();
    this.conflicted = false;
    this.disabled = !this.storage || !this.locks?.request;
    try {
      this.expected = this.storage?.getItem(PHYSICAL_KEY) ?? null;
      const position = parsePhysical(this.expected);
      this.unreadable = this.expected !== null && !position;
      this.queuedKey = JSON.stringify(position ?? emptyPosition());
      this.lastSave = Promise.resolve(true);
      this.opened = true;
      this.onWarning(this.disabled ? WARNING : stalled ? STALLED : '');
      return position ?? emptyPosition();
    } catch {
      this.disabled = true;
      this.opened = true;
      this.onWarning(WARNING);
      return emptyPosition();
    }
  }

  /** Waits for this window's queued saves, for as long as is reasonable.
   * True when one is still stuck when the wait runs out. */
  settled() {
    let timer;
    const done = pending.then(() => false, () => false);
    const late = new Promise(resolve => { timer = setTimeout(() => resolve(true), this.patience); });
    return Promise.race([done, late]).finally(() => clearTimeout(timer));
  }

  assertCurrent() {
    if (this.storage.getItem(PHYSICAL_KEY) === this.expected) return true;
    this.conflicted = true;
    if (!this.closing) this.onConflict(CONFLICT);
    return false;
  }

  save(position, { clearUnreadable = false } = {}) {
    if (!this.opened || this.closing || this.conflicted || (this.unreadable && !clearUnreadable)) return Promise.resolve(false);
    const positionKey = JSON.stringify(position);
    if (!clearUnreadable && positionKey === this.queuedKey) return this.lastSave;
    this.queuedKey = positionKey;
    const text = JSON.stringify({ version: 1, position });
    const write = async () => {
      if (this.conflicted) return false;
      if (this.disabled) { if (!this.closing) this.onWarning(WARNING); return false; }
      // The wait for the lock is bounded too: a window that never gives it
      // back should cost a warning here, not a save that never returns.
      const options = { mode: 'exclusive' };
      const signal = typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(this.patience) : null;
      if (signal) options.signal = signal;
      try {
        return await this.locks.request(LOCK, options, () => {
          if (this.conflicted || !this.assertCurrent()) return false;
          this.storage.setItem(PHYSICAL_KEY, text);
          this.expected = text;
          this.unreadable = false;
          if (!this.closing) this.onWarning('');
          return true;
        });
      } catch (error) {
        // A lock that did not come free in time is a stall, not a browser
        // that cannot save: the next edit tries again.
        if (error?.name === 'AbortError' || error?.name === 'TimeoutError') {
          if (!this.closing) this.onWarning(STALLED);
          return false;
        }
        this.disabled = true;
        if (!this.closing) this.onWarning(WARNING);
        return false;
      }
    };
    // Capture each submitted snapshot; later edits cannot mutate its queued
    // contents. Already submitted saves finish after the mode is closed.
    this.lastSave = pending.then(write, write);
    pending = this.lastSave;
    return this.lastSave;
  }

  changed(event) {
    if (!this.opened || this.closing || this.disabled || this.conflicted
      || (event.storageArea && event.storageArea !== this.storage)
      || (event.key !== null && event.key !== PHYSICAL_KEY)) return;
    try { this.assertCurrent(); }
    catch { this.disabled = true; this.onWarning(WARNING); }
  }

  close() { this.closing = true; }
}
