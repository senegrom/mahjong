/** The game review's engine facts as the insight prototype computed them in
 * JavaScript (its facts.mjs, 21 October 2026), kept as the oracle the
 * engine's `review_facts` is held to. The prototype's explanations, and the
 * probes behind the recommendation to build "What would change its mind?",
 * were all read from these, and they matched the engine's own review notes
 * on 228 of 228 tiles and the table's safe tiles on 17 of 17 decisions of
 * the review check's game. The distance from waiting is ported line for line
 * from engine/riichi-core/src/shanten.rs; the rest is counting.
 *
 * Two things are written as the engine writes them rather than as the
 * prototype did, so that the answers compare field by field: a form an open
 * hand cannot make is null, where the prototype wrote 99, and a tile's
 * copies held are a count, where it gave single and pair-breaking flags.
 * One thing is counted as the prototype counted it: the dora held are the
 * dora in the concealed hand, where the engine also counts those in the
 * seat's sets. The review check's player never calls, so on its game the
 * two agree. */

export const KINDS = [
  ...['m', 'p', 's'].flatMap(suit => Array.from({ length: 9 }, (_, rank) => `${rank + 1}${suit}`)),
  ...Array.from({ length: 7 }, (_, rank) => `${rank + 1}z`),
];
const ID = Object.fromEntries(KINDS.map((tile, index) => [tile, index]));
const TERMINALS_AND_HONOURS = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

/** How many of each kind a list of tiles holds. */
function counts(tiles) {
  const held = new Array(34).fill(0);
  for (const tile of tiles) held[ID[tile]]++;
  return held;
}

/** Every reading of one suit's tiles as sets, partial sets and a pair, as
 * shanten.rs collects them: `[sets, partials, pair]` for each. */
function collect(held, start, size, honours, sets, partials, pair, found) {
  let at = start;
  while (at < size && held[at] === 0) at++;
  if (at === size || sets + partials > 5) { found.push(`${sets},${partials},${pair ? 1 : 0}`); return; }
  if (held[at] >= 3) { held[at] -= 3; collect(held, at, size, honours, sets + 1, partials, pair, found); held[at] += 3; }
  if (!honours && at + 2 < size && held[at + 1] > 0 && held[at + 2] > 0) {
    held[at]--; held[at + 1]--; held[at + 2]--;
    collect(held, at, size, honours, sets + 1, partials, pair, found);
    held[at]++; held[at + 1]++; held[at + 2]++;
  }
  if (held[at] >= 2) {
    held[at] -= 2;
    if (!pair) collect(held, at, size, honours, sets, partials + 1, true, found);
    collect(held, at, size, honours, sets, partials + 1, pair, found);
    held[at] += 2;
  }
  if (!honours) {
    if (at + 1 < size && held[at + 1] > 0) {
      held[at]--; held[at + 1]--; collect(held, at, size, honours, sets, partials + 1, pair, found); held[at]++; held[at + 1]++;
    }
    if (at + 2 < size && held[at + 2] > 0) {
      held[at]--; held[at + 2]--; collect(held, at, size, honours, sets, partials + 1, pair, found); held[at]++; held[at + 2]++;
    }
  }
  held[at]--; collect(held, at, size, honours, sets, partials, pair, found); held[at]++;
}

const PROFILES = new Map();
/** The distinct readings of one suit's pattern, remembered by pattern. */
function profiles(pattern, size, honours) {
  const key = `${honours ? 'h' : 's'}${pattern.join('')}`;
  if (!PROFILES.has(key)) {
    const found = [];
    collect([...pattern], 0, size, honours, 0, 0, false, found);
    const distinct = [...new Set(found)].map(reading => reading.split(',').map(Number));
    PROFILES.set(key, distinct.length ? distinct : [[0, 0, 0]]);
  }
  return PROFILES.get(key);
}

/** The ordinary form's distance: four sets and a pair, with `called` sets
 * already in front of the seat. */
function regular(held, called) {
  let states = new Set(['0,0,0']);
  for (let block = 0; block < 4; block++) {
    const [offset, size] = block < 3 ? [block * 9, 9] : [27, 7];
    const pattern = new Array(9).fill(0);
    for (let at = 0; at < size; at++) pattern[at] = held[offset + at];
    const next = new Set();
    for (const state of states) {
      const [sets, partials, pair] = state.split(',').map(Number);
      for (const [moreSets, moreParts, morePair] of profiles(pattern, size, block === 3)) {
        const allSets = Math.min(4, sets + moreSets), allParts = partials + moreParts;
        if (allParts > 5 || called + allSets + allParts > 5) continue;
        next.add(`${allSets},${allParts},${pair | morePair}`);
      }
    }
    states = next;
  }
  let best = 8 - 2 * called;
  for (const state of states) {
    const [sets, partials, pair] = state.split(',').map(Number);
    const all = sets + called;
    best = Math.min(best, 8 - 2 * all - partials + (all + partials === 5 && pair === 0 ? 1 : 0));
  }
  return best;
}

/** Seven different pairs, a concealed form. */
function sevenPairs(held, called) {
  if (called > 0) return null;
  return 6 - held.filter(count => count >= 2).length + Math.max(0, 7 - held.filter(count => count > 0).length);
}

/** One of each terminal and honour and a pair of one, a concealed form. */
function thirteenOrphans(held, called) {
  if (called > 0) return null;
  const kinds = TERMINALS_AND_HONOURS.filter(at => held[at] >= 1).length;
  return 13 - kinds - (TERMINALS_AND_HONOURS.some(at => held[at] >= 2) ? 1 : 0);
}

/** The nearest form's distance from waiting. */
function shanten(held, called) {
  return Math.min(regular(held, called), sevenPairs(held, called) ?? 99, thirteenOrphans(held, called) ?? 99);
}

const nextTile = tile => {
  const rank = +tile[0], suit = tile[1];
  if (suit !== 'z') return `${rank === 9 ? 1 : rank + 1}${suit}`;
  return rank <= 4 ? `${rank === 4 ? 1 : rank + 1}z` : `${rank === 7 ? 5 : rank + 1}z`;
};
const meldTiles = meld => (meld.kind === 'chii'
  ? [0, 1, 2].map(step => `${+meld.tile[0] + step}${meld.tile[1]}`)
  : Array(meld.kind === 'pon' ? 3 : 4).fill(meld.tile));

/** Copies of each kind the deciding seat can see: its hand, every set, every
 * discard not claimed for a set, and the indicators. */
function seen(position) {
  const held = new Array(34).fill(0);
  position.indicators.forEach(tile => held[ID[tile]]++);
  for (const player of position.players) {
    player.hand.forEach(tile => held[ID[tile]]++);
    player.discards.filter(discard => !discard.claimed).forEach(discard => held[ID[discard.tile]]++);
    player.melds.forEach(meld => meldTiles(meld).forEach(tile => held[ID[tile]]++));
  }
  return held;
}

/** What cannot deal into `seat`: its own discards, and once it has declared
 * everything thrown from its declaration on. */
function safeAgainst(position, seat) {
  const player = position.players[seat];
  const safe = new Set(player.discards.map(discard => discard.tile));
  const declared = player.discards.find(discard => discard.riichi);
  if (player.riichi !== 'none' && declared) {
    for (const other of position.players) {
      for (const discard of other.discards) if (discard.order >= declared.order) safe.add(discard.tile);
    }
  }
  return safe;
}

/** How exposed a tile is against one riichi: the facts study's tiers. */
function tier(tile, safe, sight) {
  if (safe.has(tile)) return 'safe';
  const rank = +tile[0], suit = tile[1];
  if (suit === 'z') {
    return ['honour, none seen', 'honour, 1 seen', 'honour, 2 seen', 'honour, 3 seen'][Math.min(3, Math.max(0, sight[ID[tile]] - 1))];
  }
  const low = rank - 3 >= 1 && safe.has(`${rank - 3}${suit}`), high = rank + 3 <= 9 && safe.has(`${rank + 3}${suit}`);
  if (rank === 1 || rank === 9) return (rank === 1 ? high : low) ? 'suji 1/9' : 'no-suji 1/9';
  if (rank === 2 || rank === 8) return (rank === 2 ? high : low) ? 'suji 2/8' : 'no-suji 2/8';
  if (rank === 3 || rank === 7) return (rank === 3 ? high : low) ? 'suji 3-7' : 'no-suji 3/7';
  if (low && high) return 'suji 3-7';
  return low || high ? 'half-suji 4-6' : 'no-suji 4-6';
}

/** The facts of every kind the deciding seat holds, in the engine's shape. */
export function prototypeFacts(position) {
  const me = position.seat, mine = position.players[me], called = mine.melds.length;
  const hand = counts(mine.hand), sight = seen(position);
  const dora = position.indicators.map(nextTile);
  const winds = ['1z', '2z', '3z', '4z'];
  const valued = new Set(['5z', '6z', '7z', winds[position.round], winds[me]]);
  const threats = position.players.flatMap((player, seat) => (seat !== me && player.riichi !== 'none' ? [seat] : []));
  const safe = Object.fromEntries(threats.map(seat => [seat, safeAgainst(position, seat)]));
  const ownRiver = new Set(mine.discards.map(discard => discard.tile));
  const discards = [];
  for (const [index, tile] of KINDS.entries()) {
    if (!hand[index]) continue;
    hand[index]--;
    const improving = [];
    let acceptance = 0;
    const before = shanten(hand, called);
    for (const [at, kind] of KINDS.entries()) {
      if (sight[at] >= 4) continue;
      hand[at]++;
      if (shanten(hand, called) < before) { improving.push(kind); acceptance += Math.max(0, 4 - sight[at]); }
      hand[at]--;
    }
    discards.push({
      tile, shanten: before,
      forms: { regular: regular(hand, called), seven_pairs: sevenPairs(hand, called), thirteen_orphans: thirteenOrphans(hand, called) },
      improving, acceptance,
      tiers: threats.map(seat => ({ seat, tier: tier(tile, safe[seat], sight) })),
      own_river: ownRiver.has(tile), unseen: Math.max(0, 4 - sight[index]), held: hand[index] + 1,
      dora: dora.filter(kind => kind === tile).length, valued: valued.has(tile),
    });
    hand[index]++;
  }
  return {
    shanten: shanten(hand, called), dora,
    dora_held: dora.reduce((sum, kind) => sum + hand[ID[kind]], 0),
    valued_pairs: KINDS.filter(tile => valued.has(tile) && hand[ID[tile]] >= 2),
    threats: threats.map(seat => ({ seat, safe: KINDS.filter(tile => safe[seat].has(tile)) })),
    discards,
  };
}
