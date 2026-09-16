/** What each yaku is, and which tiles in a winning hand make it.
 *
 * The engine scores a hand and says which yaku it found and what each was
 * worth; it does not say which tiles made them. A person reading the result
 * wants exactly that, so the shapes are found again here from the hand the
 * engine reported, under the EMA 2025 names the engine prints.
 *
 * A yaku about *how* a hand was won rather than what it holds -- a riichi, a
 * win off the last tile -- has no shape to point at, and says so by naming
 * no tiles rather than lighting the whole hand.
 */

/** One sentence each, in the terms the rules use. */
export const YAKU_NOTES = Object.freeze({
  'Riichi': 'Declared ready with a concealed hand, betting 1,000 points on it.',
  'Double Riichi': 'Riichi declared on the first turn, with no call having interrupted it.',
  'Ippatsu': 'Won within one go-around of declaring riichi, with no call in between.',
  'Fully Concealed Hand': 'Self-drawn, with a hand nobody has called a tile into.',
  'Pinfu': 'Four sequences, a pair worth no minipoints and a two-sided wait: the hand that adds nothing extra.',
  'Pure Double Sequence': 'Two identical sequences in the same suit, in a concealed hand.',
  'Twice Pure Double Sequence': 'Two separate pairs of identical sequences, in a concealed hand.',
  'All Simples': 'No terminals and no honours: every tile is a 2 to 8.',
  'Dragon Triplet': 'A triplet or quad of one dragon, which scores wherever it sits.',
  'Seat Wind Triplet': 'A triplet or quad of your own seat wind.',
  'Round Wind Triplet': 'A triplet or quad of the wind of the current round.',
  'After a Quad': 'Won on the replacement tile drawn after declaring a quad.',
  'Robbing a Quad': 'Won on the tile another player added to their own triplet to make a quad.',
  'Under the Sea': 'Won on the very last tile drawn from the wall.',
  'Under the River': 'Won on the very last discard of the hand.',
  'Seven Pairs': 'Seven different pairs, concealed, instead of four sets and a pair.',
  'Mixed Triple Sequence': 'The same three numbers as a sequence in each of the three suits.',
  'Pure Straight': 'One suit run through: 1-2-3, 4-5-6 and 7-8-9.',
  'Half Outside Hand': 'Every set and the pair holds a terminal or an honour, with at least one sequence.',
  'Full Outside Hand': 'Every set and the pair holds a terminal, with at least one sequence and no honours.',
  'Triple Triplet': 'The same number as a triplet in each of the three suits.',
  'Three Concealed Triplets': 'Three triplets or quads that were never called.',
  'Three Quads': 'Three quads declared in one hand.',
  'All Triplets': 'Four triplets or quads and a pair: no sequence anywhere.',
  'Little Three Dragons': 'Two dragon triplets and a pair of the third dragon.',
  'All Terminals and Honours': 'Every tile is a terminal or an honour.',
  'Half Flush': 'One suit and honours, and nothing else.',
  'Full Flush': 'One suit alone, with no honours at all.',
  'Blessing of Man': 'Won on the first go-around off a discard, before drawing.',
  'Thirteen Orphans': 'One of each terminal and honour, and a second of any of them.',
  'Nine Gates': 'One suit held as 1112345678999 plus any tile of that suit, concealed.',
  'Blessing of Heaven': 'The dealer wins on their very first draw.',
  'Blessing of Earth': 'A non-dealer wins on their first draw, uninterrupted by a call.',
  'Four Concealed Triplets': 'Four triplets or quads, none of them called.',
  'Four Quads': 'All four sets are quads.',
  'All Green': 'Only green tiles: 2, 3, 4, 6 and 8 of bamboo, and the green dragon.',
  'All Terminals': 'Every tile is a 1 or a 9.',
  'All Honours': 'Every tile is a wind or a dragon.',
  'Big Three Dragons': 'A triplet of each of the three dragons.',
  'Little Four Winds': 'Three wind triplets and a pair of the fourth wind.',
  'Big Four Winds': 'A triplet of each of the four winds.',
  'Dora': 'Bonus tiles the indicators point to. They add han but are not a yaku on their own.',
  'Ura-dora': 'The bonus tiles under the indicators, revealed only to a riichi winner.',
});

const DRAGONS = ['5z', '6z', '7z'];
const WINDS = ['1z', '2z', '3z', '4z'];

const rank = tile => Number(tile[0]);
const suit = tile => tile[1];
const isHonour = tile => suit(tile) === 'z';

/** Where a tile sits in what the result screen draws. */
export const AT_HAND = index => `hand:${index}`;
export const AT_WON = 'won';
export const AT_MELD = (meld, index) => `meld:${meld}:${index}`;

/** The concealed tiles and the winning tile, with where each is shown. */
function concealed(win) {
  const places = (win.hand ?? []).map((tile, index) => ({ tile, at: AT_HAND(index) }));
  if (win.winning_tile) places.push({ tile: win.winning_tile, at: AT_WON });
  return places;
}

/** The called sets, each as its tiles and where they are shown. */
function called(win) {
  return (win.melds ?? []).map((meld, index) => ({
    kind: meld.kind ?? 'pon',
    tiles: (meld.tiles ?? []).map((tile, slot) => ({ tile, at: AT_MELD(index, slot) })),
  }));
}

function setOfMeld(meld) {
  const tiles = meld.tiles.map(place => place.tile);
  const run = tiles.length === 3 && new Set(tiles).size === 3;
  return {
    kind: run ? 'run' : meld.kind.includes('kan') ? 'quad' : 'triplet',
    concealed: meld.kind === 'concealed-kan',
    tiles: meld.tiles,
    faces: run ? [...tiles].sort((a, b) => rank(a) - rank(b)) : [tiles[0], tiles[0], tiles[0]],
  };
}

/** Every way the concealed tiles split into sets and one pair.
 *
 * Fourteen tiles split only a handful of ways, and which one a yaku means
 * matters: two identical sequences are a double sequence only under the
 * reading that makes them sequences at all.
 */
export function splittings(places) {
  const byTile = new Map();
  for (const place of places) {
    if (!byTile.has(place.tile)) byTile.set(place.tile, []);
    byTile.get(place.tile).push(place);
  }
  const left = new Map([...byTile].map(([tile, list]) => [tile, list.length]));
  const found = [];
  const take = (tile, count) => {
    const list = byTile.get(tile);
    const from = list.length - left.get(tile);
    left.set(tile, left.get(tile) - count);
    return list.slice(from, from + count);
  };
  const give = (tile, count) => left.set(tile, (left.get(tile) ?? 0) + count);
  const order = [...byTile.keys()].sort();
  const first = () => order.find(tile => (left.get(tile) ?? 0) > 0) ?? null;

  const walk = (sets, pair) => {
    if (found.length >= 32) return;
    const tile = first();
    if (tile === null) {
      if (pair) found.push({ sets: [...sets], pair });
      return;
    }
    const count = left.get(tile);
    if (!pair && count >= 2) {
      const used = take(tile, 2);
      walk(sets, { kind: 'pair', concealed: true, tiles: used, faces: [tile, tile] });
      give(tile, 2);
    }
    if (count >= 3) {
      const used = take(tile, 3);
      sets.push({ kind: 'triplet', concealed: true, tiles: used, faces: [tile, tile, tile] });
      walk(sets, pair);
      sets.pop();
      give(tile, 3);
    }
    if (!isHonour(tile) && rank(tile) <= 7) {
      const second = `${rank(tile) + 1}${suit(tile)}`;
      const third = `${rank(tile) + 2}${suit(tile)}`;
      if ((left.get(second) ?? 0) > 0 && (left.get(third) ?? 0) > 0) {
        const used = [...take(tile, 1), ...take(second, 1), ...take(third, 1)];
        sets.push({ kind: 'run', concealed: true, tiles: used, faces: [tile, second, third] });
        walk(sets, pair);
        sets.pop();
        give(tile, 1); give(second, 1); give(third, 1);
      }
    }
  };
  walk([], null);
  return found;
}

/** Every reading of the whole hand: called sets are fixed, the rest splits. */
export function readings(win) {
  const melds = called(win).map(setOfMeld);
  return splittings(concealed(win)).map(({ sets, pair }) => ({ sets: [...melds, ...sets], pair }));
}

const everything = win => [...concealed(win), ...called(win).flatMap(meld => meld.tiles)];
const placesOf = parts => parts.flatMap(part => part.tiles).map(place => place.at);
const numbersOf = run => run.faces.map(rank).join('-');

/** Which tiles make this yaku, as the places the screen draws them in, or
 * nothing where the yaku is about how the hand was won. */
export function tilesFor(name, win) {
  if (!win) return [];
  const all = () => everything(win).map(place => place.at);
  const runs = reading => reading.sets.filter(set => set.kind === 'run');
  const blocks = reading => [...reading.sets, reading.pair].filter(Boolean);
  const pick = (choose) => {
    for (const reading of readings(win)) {
      const parts = choose(reading);
      if (parts && parts.length) return placesOf(parts);
    }
    return [];
  };
  switch (name) {
    case 'All Simples': case 'Half Flush': case 'Full Flush': case 'All Terminals and Honours':
    case 'All Honours': case 'All Terminals': case 'All Green': case 'Seven Pairs':
    case 'Thirteen Orphans': case 'Nine Gates': case 'Pinfu': case 'All Triplets':
    case 'Half Outside Hand': case 'Full Outside Hand':
      return all();
    case 'Pure Double Sequence':
      return pick(reading => {
        const seen = new Map();
        for (const run of runs(reading)) {
          const key = run.faces.join();
          if (seen.has(key)) return [seen.get(key), run];
          seen.set(key, run);
        }
        return null;
      });
    case 'Twice Pure Double Sequence':
      return pick(reading => {
        const counted = new Map();
        for (const run of runs(reading)) {
          const key = run.faces.join();
          counted.set(key, [...(counted.get(key) ?? []), run]);
        }
        const doubled = [...counted.values()].filter(list => list.length >= 2);
        return doubled.length >= 2 ? doubled.flat() : null;
      });
    case 'Mixed Triple Sequence':
      return pick(reading => {
        const byNumbers = new Map();
        for (const run of runs(reading)) {
          const key = numbersOf(run);
          byNumbers.set(key, [...(byNumbers.get(key) ?? []), run]);
        }
        for (const list of byNumbers.values()) {
          const kept = [];
          for (const run of list) {
            if (!kept.some(other => suit(other.faces[0]) === suit(run.faces[0]))) kept.push(run);
          }
          if (kept.length === 3) return kept;
        }
        return null;
      });
    case 'Pure Straight':
      return pick(reading => {
        for (const which of ['m', 'p', 's']) {
          const parts = ['1-2-3', '4-5-6', '7-8-9'].map(numbers => runs(reading).find(
            run => suit(run.faces[0]) === which && numbersOf(run) === numbers));
          if (parts.every(Boolean)) return parts;
        }
        return null;
      });
    case 'Triple Triplet':
      return pick(reading => {
        const byRank = new Map();
        for (const set of reading.sets) {
          if (set.kind === 'run' || isHonour(set.faces[0])) continue;
          const key = rank(set.faces[0]);
          byRank.set(key, [...(byRank.get(key) ?? []), set]);
        }
        for (const list of byRank.values()) {
          const kept = [];
          for (const set of list) {
            if (!kept.some(other => suit(other.faces[0]) === suit(set.faces[0]))) kept.push(set);
          }
          if (kept.length === 3) return kept;
        }
        return null;
      });
    case 'Three Concealed Triplets': case 'Four Concealed Triplets': {
      const wanted = name.startsWith('Three') ? 3 : 4;
      return pick(reading => {
        const hidden = reading.sets.filter(set => set.kind !== 'run' && set.concealed);
        return hidden.length >= wanted ? hidden.slice(0, wanted) : null;
      });
    }
    case 'Three Quads': case 'Four Quads':
      return pick(reading => reading.sets.filter(set => set.kind === 'quad'));
    case 'Dragon Triplet':
      return pick(reading => reading.sets.filter(
        set => set.kind !== 'run' && DRAGONS.includes(set.faces[0])));
    case 'Seat Wind Triplet': case 'Round Wind Triplet':
      return pick(reading => {
        const winds = reading.sets.filter(set => set.kind !== 'run' && WINDS.includes(set.faces[0]));
        return winds.length ? [winds[0]] : null;
      });
    case 'Little Three Dragons':
      return pick(reading => {
        const parts = blocks(reading).filter(part => DRAGONS.includes(part.faces[0]));
        return parts.length >= 3 ? parts : null;
      });
    case 'Big Three Dragons':
      return pick(reading => reading.sets.filter(set => DRAGONS.includes(set.faces[0])));
    case 'Little Four Winds': case 'Big Four Winds':
      return pick(reading => blocks(reading).filter(part => WINDS.includes(part.faces[0])));
    default:
      return [];
  }
}

export function noteFor(name) {
  return YAKU_NOTES[name] ?? '';
}
