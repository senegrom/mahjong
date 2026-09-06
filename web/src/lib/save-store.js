/** Cooperative, single-writer transactions for the saved match. The lock covers
 * each complete command/AI sequence, not the lifetime of a tab. Compare the exact
 * previous record as well: a tab restored from an older revision never wins by
 * merely being the next process to acquire the lock.
 */
import { SAVE_KEY } from './session.js';
export const LEGACY_SAVE_KEY = 'riichi.match.v1';
const LOCK = 'riichi.saved-match.writer';

export class SaveConflict extends Error {}

export class MatchStore {
  constructor(storage, locks, { onConflict = () => {}, onWarning = () => {}, id = () => crypto.randomUUID() } = {}) {
    this.storage = storage;
    this.locks = locks;
    this.onConflict = onConflict;
    this.onWarning = onWarning;
    this.id = id;
    this.expected = null;
    this.legacy = null;
    this.opened = false;
    this.closed = false;
    this.writing = false;
    this.matchId = null;
    this.queue = Promise.resolve();
    this.newIdentity = false;
    this.disabled = !storage || !locks?.request;
  }

  read() {
    try {
      this.expected = this.storage?.getItem(SAVE_KEY) ?? null;
      this.legacy = this.expected === null ? this.storage?.getItem(LEGACY_SAVE_KEY) ?? null : null;
      this.opened = true;
      const text = this.expected ?? this.legacy;
      try { this.matchId = JSON.parse(text)?._storage?.matchId ?? null; } catch { /* Restore reports corrupt records. */ }
      if (this.disabled) this.warn();
      return text;
    } catch {
      this.disabled = true;
      this.opened = true;
      this.warn();
      return null;
    }
  }

  warn() {
    this.onWarning('This browser cannot safely save shared matches. You can play in this window, but keep it open to avoid losing progress.');
  }

  conflict() {
    const message = 'Another window is playing or has saved a newer match. Reload the latest match here before continuing.';
    this.onConflict(message);
    throw new SaveConflict(message);
  }

  assertCurrent() {
    if (this.closed) throw new SaveConflict('This match window is no longer active.');
    if (!this.opened) throw new Error('Read the saved match before playing');
    if (this.disabled) return;
    try {
      if (this.storage.getItem(SAVE_KEY) !== this.expected) this.conflict();
      // Before migration, an old-version window may still advance the old copy.
      if (this.expected === null && this.storage.getItem(LEGACY_SAVE_KEY) !== this.legacy) this.conflict();
    } catch (error) {
      if (error instanceof SaveConflict) throw error;
      this.disabled = true;
      this.warn();
    }
  }

  async run(task) {
    if (this.closed) return false;
    if (this.disabled) return task();
    const result = this.queue.then(async () => {
      if (this.closed) return false;
      if (this.disabled) return task();
      let entered = false;
      try {
        return await this.locks.request(LOCK, { mode: 'exclusive', ifAvailable: true }, async (lock) => {
          if (!lock) this.conflict();
          entered = true;
          this.assertCurrent();
          this.writing = true;
          try { return await task(); }
          finally { this.writing = false; }
        });
      } catch (error) {
        if (entered || error instanceof SaveConflict) throw error;
        // Some private contexts expose the API but deny lock acquisition.
        this.disabled = true;
        this.warn();
        return this.closed ? false : task();
      }
    });
    this.queue = result.catch(() => {});
    return result;
  }

  save(snapshot) {
    if (this.disabled) { this.warn(); return; }
    if (!this.writing) throw new Error('A saved match must be written inside its writer transaction');
    this.assertCurrent();
    if (this.disabled) return;
    let previous = null;
    try { previous = JSON.parse(this.expected); } catch { /* Deliberate New game can replace a corrupt record. */ }
    const { _storage, ...oldSnapshot } = previous ?? {};
    if (!this.newIdentity && this.expected !== null && JSON.stringify(oldSnapshot) === JSON.stringify(snapshot)) return;
    const sameMatch = !this.newIdentity && previous?.seed === snapshot.seed && previous?.difficulty === snapshot.difficulty;
    const matchId = sameMatch && _storage?.matchId ? _storage.matchId : this.matchId ?? this.id();
    const revision = Number.isSafeInteger(_storage?.revision) && _storage.revision < Number.MAX_SAFE_INTEGER ? _storage.revision + 1 : 1;
    const text = JSON.stringify({ ...snapshot, _storage: { matchId, revision } });
    try {
      this.storage.setItem(SAVE_KEY, text);
      this.expected = text;
      this.legacy = null;
      this.matchId = matchId;
      this.newIdentity = false;
      this.onWarning('');
    } catch {
      this.disabled = true;
      this.warn();
    }
  }

  newMatch() { this.matchId = this.id(); this.newIdentity = true; }

  changed(event) {
    if (!this.opened || this.closed || this.disabled || (event.storageArea && event.storageArea !== this.storage)) return;
    if (event.key === SAVE_KEY || event.key === null || (this.expected === null && event.key === LEGACY_SAVE_KEY)) {
      try { this.assertCurrent(); } catch (error) { if (!(error instanceof SaveConflict)) throw error; }
    }
  }

  close() { this.closed = true; }
}
