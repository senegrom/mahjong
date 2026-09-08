import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { PhysicalStore } from '../src/lib/physical-store.js';
import { emptyPosition, PHYSICAL_KEY } from '../src/lib/physical-position.js';

const encoded = position => JSON.stringify({ version: 1, position });
function environment(initial = null) {
  const values = new Map(initial === null ? [] : [[PHYSICAL_KEY, initial]]);
  let writing = false;
  return { values,
    storage: { getItem: key => values.get(key) ?? null, setItem(key, value) {
      assert.equal(writing, true, 'every draft write must hold the browser lock');
      values.set(key, value);
    } },
    locks: { async request(name, options, callback) {
      assert.equal(name, 'riichi.physical-table.writer');
      assert.equal(options.mode, 'exclusive');
      writing = true;
      try { return await callback({ name }); } finally { writing = false; }
    } },
  };
}
const make = (env, options) => new PhysicalStore(env.storage, env.locks, options);

test('opening drafts never rewrites them and unreadable records require an explicit clear', async () => {
  for (const original of [null, '{broken', JSON.stringify({ version: 2, position: emptyPosition() }),
    encoded({ ...emptyPosition(), indicators: [{}] }),
    encoded({ ...emptyPosition(), players: emptyPosition().players.map(p => ({ ...p, score: { toLocaleString: 1 } })) }),
    encoded(emptyPosition())]) {
    const env = environment(original), store = make(env);
    const position = await store.read();
    await store.save(position);
    assert.equal(env.storage.getItem(PHYSICAL_KEY), original);
    if (store.unreadable) {
      assert.equal(await store.save({ ...position, wall: 66 }), false);
      assert.equal(env.storage.getItem(PHYSICAL_KEY), original);
      assert.equal(await store.save(emptyPosition(), { clearUnreadable: true }), true);
      assert.equal(store.unreadable, false);
      assert.deepEqual(JSON.parse(env.storage.getItem(PHYSICAL_KEY)).position, emptyPosition());
    }
    store.close();
  }
});

test('a stale physical editor cannot overwrite another window, and reloading resumes editing', async () => {
  const env = environment(encoded(emptyPosition())), conflicts = [];
  const a = make(env), b = make(env, { onConflict: message => conflicts.push(message) });
  const pa = await a.read(), pb = await b.read();
  pa.wall = 68; pb.wall = 67;
  assert.equal(await a.save(pa), true);
  assert.equal(await b.save(pb), false);
  assert.equal(conflicts.length, 1);
  assert.equal(JSON.parse(env.storage.getItem(PHYSICAL_KEY)).position.wall, 68);
  assert.equal((await b.read()).wall, 68);
  assert.equal(await b.save(pb), true);
  assert.equal(JSON.parse(env.storage.getItem(PHYSICAL_KEY)).position.wall, 67);
  a.close(); b.close();
});

test('queued edits and undo finish when leaving the mode, before the next editor reads', async () => {
  const env = environment(encoded(emptyPosition())), store = make(env);
  const original = await store.read(), edited = { ...original, wall: 68 };
  const first = store.save(edited), undo = store.save(original);
  edited.wall = 3; // Changing the object cannot rewrite an already queued snapshot.
  store.close();
  assert.equal(await store.save({ ...original, wall: 1 }), false);
  const reopened = make(env);
  assert.deepEqual(await reopened.read(), original);
  assert.deepEqual(await Promise.all([first, undo]), [true, true]);
  assert.equal(JSON.parse(env.storage.getItem(PHYSICAL_KEY)).position.wall, 69);
  reopened.close();
});

test('draft comparison happens after obtaining the lock, including another window clearing storage', async () => {
  const env = environment(encoded(emptyPosition()));
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const request = env.locks.request;
  env.locks.request = async (...args) => { await gate; return request(...args); };
  const conflicts = [], store = make(env, { onConflict: message => conflicts.push(message) });
  const position = await store.read(); position.wall = 68;
  const save = store.save(position);
  await setImmediate();
  env.values.delete(PHYSICAL_KEY);
  release();
  assert.equal(await save, false);
  assert.equal(env.storage.getItem(PHYSICAL_KEY), null);
  assert.equal(conflicts.length, 1);
  store.close();
});

test('storage notifications block stale edits but ignore unrelated records', async () => {
  const env = environment(encoded(emptyPosition())), store = make(env);
  await store.read();
  env.values.set(PHYSICAL_KEY, encoded({ ...emptyPosition(), wall: 66 }));
  store.changed({ key: 'riichi.settings.v1', storageArea: env.storage });
  assert.equal(store.conflicted, false);
  store.changed({ key: PHYSICAL_KEY, storageArea: env.storage });
  assert.equal(store.conflicted, true);
  store.close();
});

test('denied storage or locks never report a saved draft or replace its last copy', async () => {
  for (const fault of ['missing', 'denied', 'quota']) {
    const original = encoded(emptyPosition()), env = environment(original), warnings = [];
    if (fault === 'missing') env.locks = null;
    if (fault === 'denied') env.locks.request = async () => { throw new Error('Denied'); };
    if (fault === 'quota') env.storage.setItem = () => { throw new Error('Full'); };
    const store = make(env, { onWarning: message => warnings.push(message) });
    const position = await store.read(); position.wall = 68;
    assert.equal(await store.save(position), false);
    assert.equal(await store.save(position), false, 'repeating a failed save cannot report success');
    assert.equal(env.storage.getItem(PHYSICAL_KEY), original);
    assert.ok(warnings.some(message => message.includes('could not save')));
    store.close();
  }
});
