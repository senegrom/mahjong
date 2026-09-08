import { PHYSICAL_KEY, emptyPosition, parsePhysical } from './physical-position.js';

// Finish this window's pending saves before reopening the editor. Each write
// also holds a cross-window lock and compares the exact record it started from.
let pending = Promise.resolve();
const LOCK = 'riichi.physical-table.writer';
const WARNING = 'This browser could not save the physical table. Keep this window open to retain your edits.';
const CONFLICT = 'Another window changed the physical table. Reload its saved table before continuing.';

export class PhysicalStore {
  constructor(storage, locks, { onConflict = () => {}, onWarning = () => {} } = {}) {
    this.storage = storage;
    this.locks = locks;
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
    await pending;
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
      this.onWarning(this.disabled ? WARNING : '');
      return position ?? emptyPosition();
    } catch {
      this.disabled = true;
      this.opened = true;
      this.onWarning(WARNING);
      return emptyPosition();
    }
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
      try {
        return await this.locks.request(LOCK, { mode: 'exclusive' }, () => {
          if (this.conflicted || !this.assertCurrent()) return false;
          this.storage.setItem(PHYSICAL_KEY, text);
          this.expected = text;
          this.unreadable = false;
          if (!this.closing) this.onWarning('');
          return true;
        });
      } catch {
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
