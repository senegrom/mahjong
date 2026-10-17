/** What each yaku is, and which tiles in a winning hand make it.
 *
 * The engine scores a hand and reports the reading it scored, `scoring`:
 * the sets and the pair, with where each of their tiles is shown. A person
 * reading the result wants to see which of them made each yaku, so each yaku
 * here picks its sets from that reading, under the EMA 2025 names the engine
 * prints. Seven Pairs and Thirteen Orphans come without a reading and need
 * none, since every yaku they can hold lights the whole hand or nothing.
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
  'Three Concealed Triplets': 'Three concealed triplets or quads. A triplet completed by a claimed discard does not count.',
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
  'Four Concealed Triplets': 'Four concealed triplets or quads. A discard win must complete the pair, not a triplet.',
  'Four Quads': 'All four sets are quads.',
  'All Green': 'Only green tiles: 2, 3, 4, 6 and 8 of bamboo, and the green dragon.',
  'All Terminals': 'Every tile is a 1 or a 9.',
  'All Honours': 'Every tile is a wind or a dragon.',
  'Big Three Dragons': 'A triplet of each of the three dragons.',
  'Little Four Winds': 'Three wind triplets and a pair of the fourth wind.',
  'Big Four Winds': 'A triplet of each of the four winds.',
});

const DRAGONS = ['5z', '6z', '7z'];
const WINDS = ['1z', '2z', '3z', '4z'];

const rank = tile => Number(tile[0]);
const suit = tile => tile[1];
const isHonour = tile => suit(tile) === 'z';

/** Where a tile sits in what the result screen draws. */
const AT_HAND = index => `hand:${index}`;
const AT_WON = 'won';
const AT_MELD = (meld, index) => `meld:${meld}:${index}`;

/** The concealed tiles and the winning tile, with where each is shown. */
function concealed(win) {
  const places = (win.hand ?? []).map((tile, index) => ({ tile, at: AT_HAND(index) }));
  if (win.winning_tile) places.push({ tile: win.winning_tile, at: AT_WON });
  return places;
}

/** The tiles of the called sets, with where each is shown. */
function called(win) {
  return (win.melds ?? []).flatMap((meld, index) =>
    (meld.tiles ?? []).map((tile, slot) => ({ tile, at: AT_MELD(index, slot) })));
}

const everything = win => [...concealed(win), ...called(win)];
const placesOf = parts => parts.flatMap(part => part.tiles).map(place => place.at);
const numbersOf = run => run.faces.map(rank).join('-');

/** Which tiles make this yaku, as the places the screen draws them in, or
 * nothing where the yaku is about how the hand was won. */
export function tilesFor(yaku, win) {
  const { name, tile: valueTile } = yaku;
  if (!win) return [];
  const all = () => everything(win).map(place => place.at);
  const runs = reading => reading.sets.filter(set => set.kind === 'run');
  const blocks = reading => [...reading.sets, reading.pair].filter(Boolean);
  // The sets the yaku is made of, picked from the reading the scorer scored.
  const pick = (choose) => {
    const parts = win.scoring ? choose(win.scoring) : null;
    return parts?.length ? placesOf(parts) : [];
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
        const paired = [...counted.values()].flatMap(list => list.slice(0, 2 * Math.floor(list.length / 2)));
        return paired.length >= 4 ? paired.slice(0, 4) : null;
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
    // The scorer names the tile each of these is for, so two dragon
    // triplets light apart, and a wind lights only the triplet it counts.
    case 'Dragon Triplet': case 'Seat Wind Triplet': case 'Round Wind Triplet':
      return pick(reading => reading.sets.filter(set => set.kind !== 'run' && set.faces[0] === valueTile));
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

export function isCircumstance(name) {
  return ['Riichi', 'Double Riichi', 'Ippatsu', 'Fully Concealed Hand', 'After a Quad',
    'Robbing a Quad', 'Under the Sea', 'Under the River', 'Blessing of Man',
    'Blessing of Heaven', 'Blessing of Earth'].includes(name);
}
