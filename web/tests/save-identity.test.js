import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Run the actual store in an HTTP/private-context-shaped global environment.
// Only its imported key is injected; no store behavior or methods are mocked.
const SAVE_KEY = 'riichi.match.v2';
const source = readFileSync(new URL('../src/lib/save-store.js', import.meta.url), 'utf8')
  .replace(/^import .*\n/gm, '').replace(/^export /gm, '');
const load = crypto => vm.runInNewContext(`${source}\nMatchStore`, { SAVE_KEY, crypto });
const snapshot = { version: 1, format: 6, seed: 11, difficulty: 'club', opponents: ['club', 'club', 'club'], commands: [], state: 'test' };
function environment() {
  const data = new Map();
  return {
    storage: { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) },
    locks: { request: async (name, options, task) => task({ name }) },
  };
}

for (const crypto of [undefined, {}]) {
  test(`new games and moves work without Locks or ${crypto ? 'randomUUID' : 'crypto'}`, async () => {
    const { storage } = environment();
    const before = JSON.stringify(snapshot);
    storage.setItem(SAVE_KEY, before);
    let warnings = 0;
    const Store = load(crypto);
    const store = new Store(storage, null, { onWarning: () => warnings++ });
    assert.equal(store.read(), before);
    for (let game = 0; game < 2; game++) {
      assert.doesNotThrow(() => store.newMatch());
      let played = false;
      await store.run(() => { played = true; store.save(snapshot); });
      assert.equal(played, true);
    }
    assert.equal(storage.getItem(SAVE_KEY), before, 'unsaved play preserves the existing record');
    assert.ok(warnings > 0);
  });
}

test('an exposed but denied Locks API does not require randomUUID', async () => {
  const { storage } = environment();
  const Store = load({});
  const store = new Store(storage, { request: async () => { throw new Error('SecurityError'); } });
  store.read();
  store.newMatch();
  let played = 0;
  await store.run(() => { played++; store.save(snapshot); });
  assert.equal(played, 1);
  assert.equal(store.disabled, true);
  assert.equal(storage.getItem(SAVE_KEY), null);
});

test('identity failure inside a valid transaction degrades to unsaved play', async () => {
  const { storage, locks } = environment();
  const before = JSON.stringify(snapshot);
  storage.setItem(SAVE_KEY, before);
  let warnings = 0;
  const Store = load({});
  const store = new Store(storage, locks, { onWarning: () => warnings++ });
  store.read();
  store.newMatch();
  await store.run(() => store.save(snapshot));
  assert.equal(store.disabled, true);
  assert.equal(storage.getItem(SAVE_KEY), before);
  assert.ok(warnings > 0);
});

test('persistent matches allocate one identity per new game, not per move', async () => {
  const { storage, locks } = environment();
  let ids = 0;
  const Store = load({ randomUUID: () => `id-${++ids}` });
  const store = new Store(storage, locks);
  store.read();
  store.newMatch();
  assert.equal(ids, 0, 'allocation is deferred to the first actual save');
  await store.run(() => store.save(snapshot));
  assert.equal(JSON.parse(storage.getItem(SAVE_KEY))._storage.matchId, 'id-1');
  await store.run(() => store.save({ ...snapshot, state: 'next move' }));
  assert.equal(ids, 1);
  assert.equal(JSON.parse(storage.getItem(SAVE_KEY))._storage.matchId, 'id-1');
  store.newMatch();
  await store.run(() => store.save(snapshot));
  assert.equal(ids, 2);
  assert.equal(JSON.parse(storage.getItem(SAVE_KEY))._storage.matchId, 'id-2');
});
