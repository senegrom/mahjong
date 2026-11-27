import { test } from 'node:test';
import assert from 'node:assert/strict';
import { networkHeld, networkInWorker, watchNetworkHeld } from '../src/lib/network-held.js';

// The page keeps what it hears as state of its own, so it must hear every
// change, and nothing when the same is said twice: a screen computes its
// waiting note again only as that state changes.
test('whoever watches is told what holds now, and again at each change and at no other time', () => {
  networkHeld(false);
  const heard = [];
  const stop = watchNetworkHeld(value => heard.push(value));
  assert.deepEqual(heard, [false], 'told at once what holds');
  networkHeld(false);
  networkHeld(true);
  networkHeld(1);
  assert.equal(networkInWorker(), true);
  networkHeld(false);
  assert.deepEqual(heard, [false, true, false]);
  stop();
  networkHeld(true);
  assert.deepEqual(heard, [false, true, false], 'and no more once it has stopped watching');
  networkHeld(false);
});
