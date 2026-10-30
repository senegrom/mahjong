import test from 'node:test';
import assert from 'node:assert/strict';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';
import { networkAt, networkFor, networkUrl, pickedVersion, playedNetwork, playNetwork, trainedName, valueWords } from '../src/lib/trained-network.js';
import { loadModule } from './fixtures/load-module.js';

const find = version => NETWORKS.find(network => network.version === version);
const IDENTITY = ['version', 'source', 'generation', 'reward', 'precision', 'bytes', 'storedBytes', 'origin', 'object', 'sha256'];

test('the manifest lists every network whole, the default first, and keeps exporting the default as MANIFEST', () => {
  assert.ok(NETWORKS.length >= 2);
  assert.equal(MANIFEST, NETWORKS[0], 'every script that imports MANIFEST gets the default');
  assert.ok(Object.isFrozen(NETWORKS));
  for (const network of NETWORKS) {
    assert.deepEqual(Object.keys(network), IDENTITY);
    assert.ok(Object.isFrozen(network));
    assert.ok(Object.isFrozen(network.reward));
    assert.match(network.version, /^\d+\.\d+\.\d+$/);
    // The address carries the digest, so it names these bytes and no others.
    assert.equal(network.object, `models/g${network.generation}/${network.sha256}`);
  }
  for (const field of ['version', 'sha256', 'object']) {
    assert.equal(new Set(NETWORKS.map(network => network[field])).size, NETWORKS.length, `one ${field} per network`);
  }
});

test('0.1.2610 is the default, and 0.1.1410, 0.1.1110, 0.1.810 and 0.1.410 stay listed as published, newest first', () => {
  // Generation 2610 of the placement fork that pays points at alpha 0.05, as
  // 1410 is, at the owner's word; 1410, 1110, 810 and 410 are the exports the
  // site played before it, byte for byte.
  const origin = 'https://mahjong-model.connect4-chaos.workers.dev';
  assert.deepEqual(NETWORKS.map(network => network.version), ['0.1.2610', '0.1.1410', '0.1.1110', '0.1.810', '0.1.410']);
  assert.equal(MANIFEST, find('0.1.2610'));
  assert.deepEqual({ ...find('0.1.2610') }, { version: '0.1.2610', source: 'fusion-place-a005/history/gen-02610', generation: 2610,
    reward: { kind: 'placement', alpha: 0.05 }, precision: 'float32', bytes: 116633861, storedBytes: 107861561, origin,
    object: 'models/g2610/422d3a87d1143f7b2bbe161e9920e99891b00ce213fb472813fc95be1487b721',
    sha256: '422d3a87d1143f7b2bbe161e9920e99891b00ce213fb472813fc95be1487b721' });
  assert.deepEqual({ ...find('0.1.1410') }, { version: '0.1.1410', source: 'fusion-place-a005/history/gen-01410', generation: 1410,
    reward: { kind: 'placement', alpha: 0.05 }, precision: 'float32', bytes: 116633861, storedBytes: 107858348, origin,
    object: 'models/g1410/5c34e472d19e61714d8b2e32320fcbc46415dff932ef4ec2c3da603fe55fbf26',
    sha256: '5c34e472d19e61714d8b2e32320fcbc46415dff932ef4ec2c3da603fe55fbf26' });
  assert.deepEqual({ ...find('0.1.1110') }, { version: '0.1.1110', source: 'fusion-place-a0/history/gen-01110', generation: 1110,
    reward: { kind: 'placement', alpha: 0 }, precision: 'float32', bytes: 116633861, storedBytes: 107857675, origin,
    object: 'models/g1110/f3ef362fa4a93c04677f62c0aa296c0f5d39a52d402d2e5bc1c4aca3da643136',
    sha256: 'f3ef362fa4a93c04677f62c0aa296c0f5d39a52d402d2e5bc1c4aca3da643136' });
  assert.deepEqual({ ...find('0.1.810') }, { version: '0.1.810', source: 'fusion-long/history/gen-00810', generation: 810,
    reward: { kind: 'hybrid' }, precision: 'float32', bytes: 116633861, storedBytes: 107852892, origin,
    object: 'models/g810/001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688',
    sha256: '001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688' });
  assert.deepEqual({ ...find('0.1.410') }, { version: '0.1.410', source: 'fusion-long/history/gen-00410', generation: 410,
    reward: { kind: 'hybrid' }, precision: 'float32', bytes: 116633861, storedBytes: 107849226, origin,
    object: 'models/g410/f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72',
    sha256: 'f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72' });
});

test("a network's value is named and described by the reward it learned from, as its trainer pays it", t => {
  t.after(() => playNetwork(undefined));
  const BONUS = '+1.5 for first, +0.5 for second, −0.5 for third and −1.5 for fourth';
  // 810 and 410: neural/train_combined.py's hybrid, a hand's points over
  // 4,000 plus the game's placement bonus.
  for (const version of ['0.1.810', '0.1.410']) {
    assert.deepEqual(valueWords(find(version)), {
      label: 'Estimated training reward',
      title: 'Estimated mixed training reward, not an expected finishing place or win probability',
      help: `Trained ${version} learned from a mixed reward: each decision earns its hand's point change ÷ 4,000 plus the game's final placement bonus (${BONUS}). This is not a predicted finishing place, place gain or win probability.`,
    });
  }
  // 1110: the placement reward at alpha 0, U at the next hand's start and
  // the bonus itself after the last hand, with no points term.
  const placement = valueWords(find('0.1.1110'));
  assert.deepEqual(placement, {
    label: 'Estimated placement reward',
    title: 'Estimated placement bonus, not a win probability',
    help: `Trained 0.1.1110 was last trained on the placement bonus (${BONUS}) the scoreboard promises as each hand ends, points counting only through the places they win. An estimate, not a win probability.`,
  });
  assert.doesNotMatch(placement.help, /4,000|training reward|finishing place/);
  // 2610 and 1410: the same at alpha 0.05, which pays the hand's points
  // beside the bonus and says how much they weigh, in the tooltip too.
  for (const version of ['0.1.2610', '0.1.1410']) {
    const tiebreak = valueWords(find(version));
    assert.deepEqual(tiebreak, {
      label: 'Estimated placement reward',
      title: 'Estimated placement bonus plus 0.05 × point change ÷ 4,000, not a win probability',
      help: `Trained ${version} was last trained on the placement bonus (${BONUS}) the scoreboard promises as each hand ends, plus 0.05 × the hand's point change ÷ 4,000. An estimate, not a win probability.`,
    });
    assert.doesNotMatch(tiebreak.help, /training reward|finishing place|only through the places/);
  }
  // Another weight is said as it was given.
  assert.match(valueWords({ version: '0.1.9', reward: { kind: 'placement', alpha: 0.4 } }).help, /, plus 0\.4 × the hand's point change ÷ 4,000\./);
  // A reward this build cannot describe is not described.
  for (const reward of [undefined, { kind: 'other' }]) {
    const unknown = valueWords({ version: '0.1.9', reward });
    assert.equal(unknown.label, 'Estimated training reward');
    assert.doesNotMatch(unknown.help, /placement|points|4,000/);
  }
  // Every listed network is described by its own reward, and the page
  // describes the one it plays.
  for (const network of NETWORKS) {
    assert.match(valueWords(network).help, /learned from a mixed reward|was last trained on the placement bonus/, network.version);
    playNetwork(network.version);
    assert.deepEqual(valueWords(), valueWords(network));
  }
});

test('a listed version picks its network, and anything else plays the default', t => {
  t.after(() => playNetwork(undefined));
  assert.equal(networkFor('0.1.410'), find('0.1.410'));
  for (const other of [undefined, null, '', '0.1.36', '0.1.4100', 'strong', 410, { version: '0.1.410' }]) {
    assert.equal(networkFor(other), MANIFEST, String(other));
  }
  assert.equal(networkAt(networkUrl(find('0.1.410'))), find('0.1.410'));
  assert.equal(networkAt(`${MANIFEST.origin}/models/g36/${'0'.repeat(64)}`), null);

  // The page has picked nothing until it plays a version.
  assert.equal(playedNetwork(), MANIFEST);
  assert.equal(trainedName(), `Trained ${MANIFEST.version}`);
  assert.equal(playNetwork('0.1.410'), find('0.1.410'));
  assert.equal(playedNetwork(), find('0.1.410'));
  assert.equal(pickedVersion(), '0.1.410');
  assert.equal(trainedName(), 'Trained 0.1.410');
  assert.equal(trainedName('Trained AI'), 'Trained AI 0.1.410');
  // A pick this build does not list plays the default and is no pick at all,
  // so a waiting update answers with its own default too.
  assert.equal(playNetwork('0.1.36'), MANIFEST);
  assert.equal(pickedVersion(), undefined);
  // Picking the default is a pick, and stays one when a newer network is published.
  playNetwork(MANIFEST.version);
  assert.equal(pickedVersion(), MANIFEST.version);
});

/** network-store.js with the real choice of network and the transfer stood
 * in for, to see which address and identity each call hands on. */
function store(version) {
  const transfers = [], lookups = [];
  let playing = networkFor(version);
  const api = loadModule('network-store.js', {
    URL, document: { baseURI: 'https://test.invalid/mahjong/' },
    networkAt, networkUrl, playedNetwork: () => playing,
    verifiedNetworkBytes: async request => { transfers.push(request); return new Uint8Array(1); },
    hasStoredNetwork: async (url, expect, options) => { lookups.push({ url, expect, options }); return false; },
  }, ['networkBytes', 'networkIsStored']);
  return { api, transfers, lookups, play: next => { playing = networkFor(next); } };
}

test('the chosen network is the one fetched, checked by its own digest and looked for in storage', async () => {
  const older = find('0.1.410');
  const s = store('0.1.410');
  await s.api.networkBytes({ requireStored: true });
  assert.equal(s.transfers[0].url, networkUrl(older));
  assert.equal(s.transfers[0].expect, older, 'the bytes must match 410\'s digest and size');
  assert.equal(s.transfers[0].scope, 'https://test.invalid/mahjong/');
  assert.equal(s.transfers[0].requireStored, true);
  await s.api.networkIsStored();
  assert.deepEqual(s.lookups.map(({ url, expect }) => [url, expect]), [[networkUrl(older), older]]);
  // Back on the default, the default is fetched and looked for.
  s.play(undefined);
  await s.api.networkBytes();
  await s.api.networkIsStored();
  assert.equal(s.transfers[1].url, networkUrl(MANIFEST));
  assert.equal(s.transfers[1].expect, MANIFEST);
  assert.equal(s.lookups[1].url, networkUrl(MANIFEST));
});

test('the opponent\'s worker names an address, and is held to the network listed there', async () => {
  const s = store(undefined);
  const older = find('0.1.410');
  // The worker's own request: an address and the page's folder, no identity.
  await s.api.networkBytes({ url: networkUrl(older), scope: 'https://test.invalid/mahjong/' });
  assert.equal(s.transfers[0].expect, older);
  // An address no listed network has is refused before anything is fetched.
  await assert.rejects(s.api.networkBytes({ url: `${MANIFEST.origin}/models/g36/${'0'.repeat(64)}` }),
    /not one of the trained networks this page can play/);
  assert.equal(s.transfers.length, 1);
  // A waiting update's network, which this build may not list, comes with its identity.
  const next = { url: 'https://model.invalid/models/g900/next', bytes: 5, sha256: '9'.repeat(64) };
  await s.api.networkBytes({ url: next.url, expect: next });
  assert.equal(s.transfers[1].expect, next);
});

test('the worker\'s word that a network is stored counts only for the network this page plays', async () => {
  const s = store('0.1.410');
  const older = find('0.1.410');
  const reply = network => ({ url: networkUrl(network), version: network.version, sha256: network.sha256, bytes: network.bytes });
  assert.equal(await s.api.networkIsStored(undefined, undefined, reply(older)), true);
  assert.equal(s.lookups.length, 0, 'an exact identity needs no second look');
  // A reply about the default, which this page does not play, is checked against storage.
  assert.equal(await s.api.networkIsStored(undefined, undefined, reply(MANIFEST)), false);
  assert.deepEqual(s.lookups.map(({ url }) => url), [networkUrl(older)]);
});
