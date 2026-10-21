/** The game review's reasoning about one of the player's discards where
 * Trained AI chose another tile: why it chose its own, said from the engine's
 * facts and checked against its weights, and what would have changed its
 * mind, found by asking it again about the table changed in one thing the
 * player could see.
 *
 * Every changed table is a typed-in position, as Game.review_position writes
 * the decision's own, so the engine checks it by the rules a physical table
 * is checked by, and every number said is Trained AI's answer on a table the
 * engine accepted. What a change did is read from the engine's facts about
 * the table before and after it (review_facts), never from the change, so a
 * change that is true but not the reason is said with what it really
 * changed. Nothing here asks the network: `ask(positions, signal)` answers
 * each position with Trained AI's weights, `{ moves: [{ kind, tile, weight }] }`,
 * or with why the engine refused it, `{ refused }`, and `facts(position)` is
 * review_facts. The page asks the policy worker (review-insight-policy.js);
 * the tests answer from a recorded run.
 *
 * Version 1 keeps to what the prototype measured (insight/recommendation.md):
 * only the player's own discard against Trained AI's is explained; a table is
 * changed only in ways a game reaches, so the wall, the dealer and calls taken
 * back are left alone; a choice Trained AI made by more than e^7 to one is not
 * searched, since no small change moves it; and pairs of changes are tried
 * only when no single change flips the choice.
 */
import { TILE_TYPES, WINDS, tileWords } from './tiles.js';
import { percent } from './ui.js';

/** Below this margin, the log of the player's weight over the best other
 * move's, the search is not offered: in the prototype nothing small ever
 * flipped such a choice, and the pairs that did read as contrived. */
export const MARGIN_FLOOR = -7;
/** Positions asked in one forward pass at most: the review's own batch (BATCH
 * in game-review.js). The worker runs a pass it has begun to its end, so
 * whatever is sent to it meanwhile, a question of the hand's notes or
 * another decision's Why line, waits behind no more than it would behind the
 * review. Passes of 32 cost only about 3% less a position in the page's
 * runtime, too little for a wait four times as long. */
export const PASS = 8;
/** How far the unedited table's weights may stand from the review's own: a
 * forward pass of other positions beside it may round its last digits
 * otherwise, while a table that replayed differently moves them far more. */
export const SAME_WEIGHT = 1e-4;
/** The single changes nearest to flipping it that are paired, two at a time. */
const PAIRED = 8;
/** Trained AI's weight a fact must carry for the fact to be the reason. */
const MASS = 0.95;

const the = tile => `the ${tileWords(tile)}`;
const your = tile => `your ${tileWords(tile)}`;
const an = tile => `${tile[0] === '8' || tile === '1z' ? 'an' : 'a'} ${tileWords(tile)}`;
const ORDINALS = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'];
const nth = index => ORDINALS[index] ?? `${index + 1}th`;
const COUNT = ['no', 'one', 'two', 'three', 'four'];
const points = value => Math.round(value).toLocaleString('en');
const capital = text => `${text[0].toUpperCase()}${text.slice(1)}`;
const lower = text => `${text[0].toLowerCase()}${text.slice(1)}`;
const sentence = text => text.replace(/\.$/, '');
const list = (items, and = 'and') => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} ${and} ${items.at(-1)}`);
const stopped = () => new DOMException('Search stopped', 'AbortError');

/** Whether two moves are the same move: the same kind, and the same tile. */
export const sameMove = (a, b) => a.kind === b.kind && (a.tile ?? null) === (b.tile ?? null);

/** A decision's weights as the review keeps them (`weighted`, one entry a
 * listed choice) as moves: a riichi is one move whichever tile it names, as
 * the network weighs it, and a choice the network has no move for is left
 * out. */
export function movesOf(weighted) {
  const moves = [];
  for (const choice of weighted) {
    if (choice.weight == null || (choice.kind === 'riichi' && moves.some(move => move.kind === 'riichi'))) continue;
    moves.push({ kind: choice.kind, tile: choice.kind === 'riichi' ? null : choice.tile ?? null, weight: choice.weight });
  }
  return moves;
}

const marginOf = (mine, other) => Math.log(Math.max(mine, 1e-9)) - Math.log(Math.max(other, 1e-9));

/** One answer judged against the player's move: its weight and Trained AI's
 * choice's, the move it now rates highest, the margin of the player's move
 * over the best other, and whether the player's move now leads. */
export function judge(moves, played, preferred) {
  const mine = moves.find(move => sameMove(move, played));
  if (!mine) return { refused: 'The move played is not open on this table' };
  const theirs = moves.find(move => sameMove(move, preferred));
  let best = null, other = null;
  for (const move of moves) {
    if (!best || move.weight > best.weight) best = move;
    if (move !== mine && (!other || move.weight > other.weight)) other = move;
  }
  return { played: mine.weight, preferred: theirs ? theirs.weight : null, removesPreferred: !theirs,
    top: { kind: best.kind, tile: best.tile ?? null, weight: best.weight }, margin: marginOf(mine.weight, other?.weight ?? 0), flips: best === mine };
}

/** How exposed a tile is against a riichi, safest first, with the rough share
 * of such tiles that deal in mid-hand, which only orders them (the facts
 * study's, as review_facts names them). */
const TIERS = ['safe', 'honour, 3 seen', 'honour, 2 seen', 'suji 1/9', 'honour, 1 seen', 'suji 2/8', 'suji 3-7',
  'honour, none seen', 'no-suji 1/9', 'half-suji 4-6', 'no-suji 2/8', 'no-suji 3/7', 'no-suji 4-6'];
const RATE = { safe: 0, 'honour, 3 seen': 0.002, 'honour, 2 seen': 0.015, 'suji 1/9': 0.018, 'honour, 1 seen': 0.035,
  'suji 2/8': 0.027, 'suji 3-7': 0.045, 'honour, none seen': 0.075, 'no-suji 1/9': 0.055, 'half-suji 4-6': 0.075,
  'no-suji 2/8': 0.08, 'no-suji 3/7': 0.10, 'no-suji 4-6': 0.125 };
const TIER_WORDS = {
  'honour, 3 seen': 'an honour with every other copy in sight', 'honour, 2 seen': 'an honour with two other copies in sight',
  'honour, 1 seen': 'an honour with one other copy in sight', 'honour, none seen': 'an honour nobody has thrown',
  'suji 1/9': 'a terminal on suji', 'suji 2/8': 'on suji', 'suji 3-7': 'on suji', 'no-suji 1/9': 'a terminal with no suji',
  'half-suji 4-6': 'a middle tile on half suji', 'no-suji 2/8': 'a tile with no suji', 'no-suji 3/7': 'a tile with no suji',
  'no-suji 4-6': 'a middle tile with no suji',
};

/** The engine's facts about a table (review_facts) as the reasoning reads
 * them: each tile the seat may throw by its name, with its worst exposure
 * against every riichi, and what the hand holds. */
function factsOf(raw, position) {
  const threats = raw.threats.map(threat => threat.seat);
  const tiles = {};
  for (const d of raw.discards) {
    const tier = d.tiers.map(exposure => exposure.tier).sort((a, b) => TIERS.indexOf(b) - TIERS.indexOf(a))[0] ?? null;
    tiles[d.tile] = { tile: d.tile, shanten: d.shanten, acceptance: d.acceptance, tier, unseen: d.unseen, ownRiver: d.own_river,
      dora: d.dora, honour: d.tile[1] === 'z', single: d.held === 1, pairBreak: d.valued && d.held >= 2,
      // Waiting, the tiles that would improve the hand are those it wins on.
      waitsLeft: d.shanten === 0 ? d.acceptance : 0,
      forms: { regular: d.forms.regular, 'seven pairs': d.forms.seven_pairs, 'thirteen orphans': d.forms.thirteen_orphans } };
  }
  const held = [...new Set(position.players[position.seat].hand)];
  return { me: position.seat, tiles, threats, best: Math.min(...Object.values(tiles).map(tile => tile.shanten)),
    doraHeld: raw.dora_held, valuedPairs: raw.valued_pairs,
    safeHeld: threats.length ? held.filter(tile => raw.threats.every(threat => threat.safe.includes(tile))) : [] };
}

const meldTiles = meld => (meld.kind === 'chii' ? [0, 1, 2].map(step => `${Number(meld.tile[0]) + step}${meld.tile[1]}`)
  : Array(['kan', 'extended-kan', 'concealed-kan'].includes(meld.kind) ? 4 : 3).fill(meld.tile));

/** How many copies of each kind the deciding seat can see: its hand, every
 * river (a claimed discard counts with the set that took it), every set and
 * the indicators. */
export function visibleCounts(position) {
  const seen = Object.fromEntries(TILE_TYPES.map(tile => [tile, 0]));
  const count = tile => { seen[tile]++; };
  position.indicators.forEach(count);
  for (const player of position.players) {
    player.hand.forEach(count);
    player.discards.forEach(discard => { if (!discard.claimed) count(discard.tile); });
    player.melds.forEach(meld => meldTiles(meld).forEach(count));
  }
  return seen;
}

const doraOf = indicator => {
  const rank = Number(indicator[0]), suit = indicator[1];
  if (suit !== 'z') return `${rank === 9 ? 1 : rank + 1}${suit}`;
  return rank <= 4 ? `${rank === 4 ? 1 : rank + 1}z` : `${rank === 7 ? 5 : rank + 1}z`;
};
const indicatorFor = dora => TILE_TYPES.find(tile => doraOf(tile) === dora);
const total = table => table.players.reduce((sum, player) => sum + player.score, 0);

/** The player `delta` points up, taken evenly from the others, or down, given
 * evenly to them, the points on the table kept. A step past 8,000 asks the
 * player to imagine two things, so it costs two. */
export function scoreEdit(position, delta) {
  const me = position.seat;
  return { family: 'scores', key: 'scores', param: delta, cost: Math.abs(delta) > 8000 ? 2 : 1,
    words: `you had had ${points(Math.abs(delta))} points ${delta > 0 ? 'more, taken evenly from' : 'fewer, given evenly to'} the others`,
    apply: table => {
      const before = total(table);
      table.players[me].score += delta;
      table.players.forEach((player, seat) => { if (seat !== me) player.score = Math.round((player.score - delta / 3) / 100) * 100; });
      table.players[(me + 1) % 4].score += before - total(table);
    } };
}

/** The single changes asked about a table. Each is `{ family, key, words,
 * cost, apply(table) }`: `words` finish "if …", `cost` counts the facts a
 * player must change in their head, and `key` names what the change
 * touches, so that two changes touching one thing are never paired. Only
 * what the player could see is changed, and only in ways a game reaches. */
export function singleEdits(position, { played, preferred }) {
  const edits = [];
  const me = position.seat, mine = position.players[me], P = played.tile, A = preferred.tile;
  const seen = visibleCounts(position);
  const add = (family, key, words, apply, extra = {}) => edits.push({ family, key, words, cost: 1, apply, ...extra });

  // A tile of the hand swapped for one of which the player cannot see four.
  // The tile played keeps a copy, so the player's move stays open; a swap that
  // takes Trained AI's tile away is marked, since "it would not throw a tile
  // you do not hold" explains nothing.
  const kinds = [...new Set(mine.hand)];
  for (const out of kinds) {
    const copies = mine.hand.filter(tile => tile === out).length;
    if (out === P && copies === 1) continue;
    for (const into of TILE_TYPES) {
      if (into === out || seen[into] >= 4) continue;
      add('hand', `hand:${out}`, `your ${tileWords(out)} had been ${an(into)}`, table => {
        const hand = table.players[table.seat].hand;
        hand[hand.indexOf(out)] = into;
        if (table.drawn === out && copies === 1) table.drawn = into;
      }, { out, into, removesPreferred: out === A && copies === 1 });
    }
  }

  // An opponent's riichi taken back, or declared a discard earlier or later,
  // or declared by a closed opponent with its latest discard. Every discard
  // after a declaration is the tile just drawn, so a riichi declared with a
  // tile from the hand cannot have come a discard earlier: that tile would
  // then follow the declaration out of the hand, which no game plays.
  position.players.forEach((player, seat) => {
    if (seat === me) return;
    const at = player.discards.findIndex(discard => discard.riichi);
    if (player.riichi !== 'none') {
      add('riichi', `riichi:${seat}`, `${WINDS[seat]} had not declared riichi`, table => {
        const them = table.players[seat];
        them.discards.forEach(discard => { discard.riichi = false; });
        Object.assign(them, { riichi: 'none', ippatsu: false, score: them.score + 1000 });
        table.riichi_sticks = Math.max(0, table.riichi_sticks - 1);
      }, { undeclares: true, seat });
      for (const step of [-1, 1]) {
        const to = at + step;
        if (to < 0 || to >= player.discards.length || player.discards[to].claimed || (step < 0 && !player.discards[at].drawn)) continue;
        add('riichi', `riichi:${seat}`, `${WINDS[seat]} had declared riichi one discard ${step < 0 ? 'earlier' : 'later'}, on the ${tileWords(player.discards[to].tile)}`, table => {
          const them = table.players[seat];
          them.discards[at].riichi = false;
          them.discards[to].riichi = true;
          // Ippatsu lasts until the declarer's next discard.
          them.ippatsu = to === them.discards.length - 1;
          if (them.riichi === 'double' && to > 0) them.riichi = 'riichi';
        });
      }
    } else if (player.discards.length && player.melds.every(meld => meld.kind === 'concealed-kan') && player.score >= 1000) {
      const last = player.discards.at(-1);
      if (last.claimed) return;
      add('riichi', `riichi:${seat}`, `${WINDS[seat]} had declared riichi with their last discard, the ${tileWords(last.tile)}`, table => {
        const them = table.players[seat];
        them.discards.at(-1).riichi = true;
        Object.assign(them, { riichi: 'riichi', ippatsu: true, score: them.score - 1000 });
        table.riichi_sticks += 1;
      });
    }
  });

  // The river tiles that decide what is safe: each copy of either tile in a
  // river made a tile nobody can see, and either tile put into a riichi
  // player's river in place of their latest discard since the declaration.
  // Where none is left, a tile with room for every copy in that river is
  // taken instead, so that the copies changed together, as the Why line
  // changes every copy the player threw to take a furiten away, never put a
  // fifth in sight, which the rules refuse; with no such tile, there is no
  // such change.
  const neutral = (tile, copies) => TILE_TYPES.find(t => t !== P && t !== A && !seen[t] && t[1] !== 'z' && t[1] !== tile[1])
    ?? TILE_TYPES.find(t => t !== P && t !== A && seen[t] + copies <= 4);
  position.players.forEach((player, seat) => {
    player.discards.forEach((discard, index) => {
      if (discard.claimed || (discard.tile !== P && discard.tile !== A)) return;
      const into = neutral(discard.tile, player.discards.filter(other => !other.claimed && other.tile === discard.tile).length);
      if (!into) return;
      add('river', `river:${seat}:${index}`, `${seat === me ? 'your' : `${WINDS[seat]}'s`} ${nth(index)} discard had been ${an(into)}, not the ${tileWords(discard.tile)}`,
        table => { table.players[seat].discards[index].tile = into; }, { seat, index, from: discard.tile, into });
    });
    if (seat === me || player.riichi === 'none') return;
    const declared = player.discards.findIndex(discard => discard.riichi);
    for (const tile of [P, A]) {
      if (seen[tile] >= 4 || player.discards.some(discard => discard.tile === tile)) continue;
      const index = player.discards.findLastIndex((discard, at) => at > declared && !discard.claimed);
      if (index < 0) continue;
      add('river', `river:${seat}:${index}`, `${WINDS[seat]}'s ${nth(index)} discard had been the ${tileWords(tile)}, not the ${tileWords(player.discards[index].tile)}`,
        table => { table.players[seat].discards[index].tile = tile; }, { seat, from: player.discards[index].tile });
    }
  });

  // Each kind in question or held made the dora instead.
  for (const dora of new Set([P, A, ...kinds])) {
    const indicator = indicatorFor(dora);
    if (indicator === position.indicators[0] || seen[indicator] >= 4) continue;
    add('dora', 'dora', `the dora indicator had been the ${tileWords(indicator)}, not the ${tileWords(position.indicators[0])}, making the ${tileWords(dora)} dora`,
      table => { table.indicators[0] = indicator; }, { into: dora });
  }

  // The scores, the points on the table kept: the player ahead or behind by
  // a step, or everyone level, where they are not already.
  for (const delta of [-16000, -8000, 8000, 16000]) edits.push(scoreEdit(position, delta));
  if (new Set(position.players.map(player => player.score)).size > 1) {
    add('scores', 'scores', 'everyone had had the same score', table => {
      const before = total(table), level = Math.round(before / 400) * 100;
      table.players.forEach(player => { player.score = level; });
      table.players[(me + 1) % 4].score += before - 4 * level;
    });
  }

  // The hand of the game: the last hand, this hand's number in the south
  // round, and the first hand. The players are numbered by the hand number.
  const named = new Set([`${position.round} ${position.kyoku}`]);
  for (const [round, kyoku] of [[1, 4], [1, position.kyoku], [0, 1]]) {
    if (named.has(`${round} ${kyoku}`)) continue;
    named.add(`${round} ${kyoku}`);
    add('round', 'round', `this had been ${WINDS[round]} ${kyoku}${round === 1 && kyoku === 4 ? ', the last hand' : ''}`,
      table => { Object.assign(table, { round, kyoku }); });
  }
  return edits;
}

/** The table `edits` make of `position`, which is left as it was. */
export function applyEdits(position, ...edits) {
  const table = structuredClone(position);
  for (const edit of edits) edit.apply(table);
  return table;
}

/** What a change did, in the engine's facts about the two tiles and the hand:
 * among them the seats in riichi after it (`standing`), and those whose
 * riichi it took back or added, which a count alone would hide where a pair
 * of changes takes one back and adds another. */
function factDelta(F0, F1, P, A) {
  const p0 = F0.tiles[P], p1 = F1.tiles[P], a0 = F0.tiles[A], a1 = F1.tiles[A];
  const both = (x, y, read) => (x && y ? read(y) - read(x) : 0);
  return { shantenP: both(p0, p1, t => t.shanten), accP: both(p0, p1, t => t.acceptance), shantenA: both(a0, a1, t => t.shanten),
    dora: F1.doraHeld - F0.doraHeld, valuedPairs: F1.valuedPairs.length - F0.valuedPairs.length,
    standing: F1.threats, taken: F0.threats.filter(seat => !F1.threats.includes(seat)), added: F1.threats.filter(seat => !F0.threats.includes(seat)),
    tierA: a0 && a1 && a0.tier !== a1.tier ? [a0.tier, a1.tier] : null, tierP: p0 && p1 && p0.tier !== p1.tier ? [p0.tier, p1.tier] : null,
    unseenA: both(a0, a1, t => t.unseen), unseenP: both(p0, p1, t => t.unseen), safeHeld: F1.safeHeld.length - F0.safeHeld.length };
}

/** Whether a change put a riichi on the table or took one off it. */
const riichiMoved = d => d.taken.length > 0 || d.added.length > 0;

const TABLE = new Set(['scores', 'round']);

/** Every idea a change belongs to, by what it changed in the facts. */
function ideasOf(edit, d) {
  const ideas = new Set();
  if (TABLE.has(edit.family)) ideas.add('table');
  // A tile in question seen once more or less changes how safe it is, even
  // with nobody in riichi.
  if (['riichi', 'river'].includes(edit.family) || riichiMoved(d) || d.tierA || d.tierP || d.safeHeld || d.unseenA || d.unseenP) ideas.add('safety');
  if (d.shantenP < 0 || (d.shantenP === 0 && d.accP >= 4)) ideas.add('faster');
  if (d.dora > 0 || d.valuedPairs > 0 || edit.family === 'dora') ideas.add('worth more');
  if (!ideas.size) ideas.add(edit.family === 'hand' ? 'shape' : 'other');
  return [...ideas];
}

/** The one idea a change is said under. A hand swap is read by what it does
 * to the hand first. Moving the indicator also changes what is in sight,
 * which is the reason only when it makes a tile in question dead or live
 * again, or changes how safe it is against a riichi. */
function primary(change, unseen) {
  if (TABLE.has(change.family)) return 'table';
  if (change.family === 'hand') return ['faster', 'worth more', 'safety'].find(idea => change.ideas.includes(idea)) ?? 'shape';
  if (change.family !== 'dora') return 'safety';
  const d = change.delta, livens = (before, by) => by && before === 0;
  return d.tierA || d.tierP || livens(unseen.A, d.unseenA) || livens(unseen.P, d.unseenP) ? 'safety' : 'worth more';
}

/** How a tile in question stands after a change, its exposure `[was, is]`
 * against the riichi standing before and after: safe, no longer safe, or
 * safer or less safe by the order of TIERS. A tile exposed where nobody was
 * in riichi before is less safe. */
function exposureWords(name, [was, is]) {
  if (is === 'safe') return `${name} safe`;
  if (was === 'safe') return `${name} no longer safe`;
  return was != null && TIERS.indexOf(is) < TIERS.indexOf(was) ? `${name} safer` : `${name} less safe`;
}

/** What a change did, in a few words, read from the engine's facts after it
 * (`d`, factDelta's). Who is in riichi is said as the facts stand after the
 * change: nobody, only the riichi still standing where one was taken back,
 * or the riichi added counted among all those standing. */
function deltaWords(d, A, P, unseen) {
  const out = [];
  if (d.shantenP < 0) out.push('your hand a step closer to waiting');
  else if (d.shantenP > 0) out.push('your hand a step further from waiting');
  else if (d.accP >= 4) out.push(`${d.accP} more tiles that improve your hand`);
  if (d.dora > 0) out.push(`${d.dora} more dora in your hand`);
  if (d.valuedPairs > 0) out.push('a pair of a valued honour');
  if (d.taken.length && !d.standing.length) out.push('nobody in riichi');
  else {
    if (d.taken.length) out.push(`only ${list(d.standing.map(seat => WINDS[seat]))} in riichi`);
    else if (d.added.length === 1) out.push(d.standing.length === 1 ? 'a riichi' : `a ${nth(d.standing.length - 1)} riichi`);
    else if (d.added.length) out.push(`${COUNT[d.added.length]} ${d.added.length < d.standing.length ? 'more ' : ''}riichi`);
    if (d.tierA) out.push(exposureWords(the(A), d.tierA));
    if (d.tierP) out.push(exposureWords(your(P), d.tierP));
  }
  // Copies in sight matter where a tile was dead, every copy seen.
  if (d.unseenA && unseen.A === 0) out.push(`not every ${tileWords(A)} in sight`);
  if (d.unseenP && unseen.P === 0) out.push(`not every ${tileWords(P)} in sight`);
  if (d.safeHeld && !d.tierA && !d.tierP && !riichiMoved(d)) {
    out.push(d.safeHeld === -1 ? 'one safe tile fewer' : `${Math.abs(d.safeHeld)} ${d.safeHeld > 0 ? 'more' : 'fewer'} safe tile${Math.abs(d.safeHeld) === 1 ? '' : 's'} in your hand`);
  }
  return out;
}

/** What a change did to a special hand that Trained AI's tile keeps the
 * hand's distance by. */
function formWords(F0, F1, A) {
  const was = F0.tiles[A], is = F1?.tiles[A];
  if (!was || !is) return [];
  return ['thirteen orphans', 'seven pairs'].flatMap(form => {
    const before = was.forms[form], after = is.forms[form];
    if (before == null || before === after || before !== was.shanten || before === was.forms.regular) return [];
    return [`${form} a step ${after > before ? 'further' : 'closer'}`];
  });
}

/** What a change did, in the few words said in brackets after it: what it
 * did to a special hand that Trained AI's tile keeps, then to the facts. */
const changeSaid = (F0, change, A, P, unseen) => [...formWords(F0, change.facts, A), ...deltaWords(change.delta, A, P, unseen)];

/** The words said in brackets after a change of a question's table, as the
 * search says them for the change it shows, read from the engine's facts
 * (`facts`, review_facts) before and after: nothing is asked of Trained AI. */
export function changeWords({ position, played, preferred, facts }, ...edits) {
  const F0 = factsOf(facts(position), position), table = applyEdits(position, ...edits);
  const F1 = factsOf(facts(table), table), P = played.tile, A = preferred.tile;
  return changeSaid(F0, { facts: F1, delta: factDelta(F0, F1, P, A) }, A, P, { A: F0.tiles[A]?.unseen, P: F0.tiles[P]?.unseen });
}

/** The hand swaps worth asking, picked without the network: those that bring
 * the hand closer to waiting once the player's tile is thrown (the five that
 * leave the most tiles to improve it), a valued pair (two) and a dora (two).
 * A swap that touches either tile in question is left out. */
function pickSwaps(singles, base, F0, P, A, facts) {
  const after = F0.tiles[P]?.shanten ?? F0.best;
  const swaps = singles.filter(e => e.family === 'hand' && !e.removesPreferred && ![P, A].includes(e.out) && ![P, A].includes(e.into)).flatMap(edit => {
    let raw;
    try { raw = facts(applyEdits(base, edit)); } catch { return []; }
    const thrown = raw.discards.find(discard => discard.tile === P);
    return thrown ? [{ edit, closer: thrown.shanten < after, acceptance: thrown.acceptance, dora: raw.dora_held > F0.doraHeld,
      pair: raw.valued_pairs.some(tile => !F0.valuedPairs.includes(tile)) }] : [];
  });
  const most = (picked, count) => swaps.filter(picked).sort((x, y) => y.acceptance - x.acceptance).slice(0, count);
  return [...new Set([...most(s => s.closer, 5), ...most(s => s.pair, 2), ...most(s => s.dora, 2)].map(s => s.edit))];
}

/** Where the unedited table's answer stands apart from the review's own
 * weights, if it does: a move one has and the other has not, or a weight
 * further off than rounding. */
function unlikeReview(answer, weights) {
  if (!answer || answer.refused) return answer?.refused ?? 'no answer';
  const moves = answer.moves;
  if (moves.length !== weights.length) return 'other moves';
  const off = weights.find(move => !(Math.abs((moves.find(other => sameMove(other, move))?.weight ?? Infinity) - move.weight) <= SAME_WEIGHT));
  return off ? `${off.kind} ${off.tile ?? ''}`.trim() : null;
}

/** Asks Trained AI about tables in passes of `size`, minding `signal`, and
 * tells `onProgress` how many tables have been asked of how many are planned.
 * `onPass(answers)` hears the answers so far after each pass, and stops the
 * asking by answering false. */
function asker({ ask, signal, size, onProgress }) {
  let asked = 0, planned = 0;
  return {
    plan: count => { planned += count; onProgress({ asked, planned }); },
    get asked() { return asked; },
    async all(positions, onPass = () => {}) {
      const answers = [];
      for (let start = 0; start < positions.length; start += size) {
        if (signal?.aborted) throw stopped();
        const pass = positions.slice(start, start + size);
        const replies = await ask(pass, signal);
        if (signal?.aborted) throw stopped();
        if (!Array.isArray(replies) || replies.length !== pass.length) throw new Error('Trained AI returned the wrong number of answers');
        answers.push(...replies);
        asked += pass.length;
        onProgress({ asked, planned });
        if (onPass(answers) === false) return null;
      }
      return answers;
    },
  };
}

/** Changes judged from their answers, each with what it did to the facts. */
function judged(edits, answers, { base, played, preferred, F0, facts, withFacts = true }) {
  return edits.map((edit, index) => {
    const { apply: _apply, parts: _parts, ...said } = edit, answer = answers[index];
    const verdict = answer.refused ? answer : judge(answer.moves, played, preferred);
    if (verdict.refused) return { ...said, refused: verdict.refused };
    const change = { ...said, edit, ...verdict };
    if (withFacts) {
      const position = applyEdits(base, ...(edit.parts ?? [edit]));
      change.facts = factsOf(facts(position), position);
      change.delta = factDelta(F0, change.facts, played.tile, preferred.tile);
      change.ideas = ideasOf(edit, change.delta);
    }
    return change;
  });
}

/** Where the reasoning starts: the engine's facts about the unedited table
 * and its two tiles, and the review's own weights judged. Null where
 * version 1 has nothing to say: a move that is not a discard on either side,
 * or the same tile. */
function baselineOf({ position, played, preferred, weights, facts }) {
  if (played?.kind !== 'discard' || preferred?.kind !== 'discard' || played.tile === preferred.tile) return null;
  const F0 = factsOf(facts(position), position);
  const P = F0.tiles[played.tile], A = F0.tiles[preferred.tile];
  if (!P || !A) return null;
  const baseline = judge(weights, played, preferred);
  return baseline.refused ? null : { F0, P, A, baseline, unseen: { A: A.unseen, P: P.unseen } };
}

/** Whether the search is offered for a decision's question: a discard against
 * a discard, Trained AI's choice not so sure that nothing small moves it. */
export function searchOffered(question) {
  const asked = question.played?.kind === 'discard' && question.preferred?.kind === 'discard' && question.played.tile !== question.preferred.tile;
  return asked && judge(question.weights, question.played, question.preferred).margin >= MARGIN_FLOOR;
}

/** The single changes a search asks first: every one that is not a hand
 * swap, and the swaps picked without the network. */
function budgetOf(position, { played, preferred }, F0, facts) {
  const singles = singleEdits(position, { played, preferred });
  return [...singles.filter(edit => edit.family !== 'hand'), ...pickSwaps(singles, position, F0, played.tile, preferred.tile, facts)];
}

/** How many tables a search of the question asks before any score step or
 * pair: the unedited table and its budget of single changes, which is most
 * of what it asks. The page times it by the review's own time a position to
 * say about how long the search takes. None where nothing is searched. */
export function tablesFirstAsked({ position, played, preferred, weights, facts }) {
  const start = baselineOf({ position, played, preferred, weights, facts });
  return start && start.baseline.margin >= MARGIN_FLOOR ? budgetOf(position, { played, preferred }, start.F0, facts).length + 1 : 0;
}

/** What would change Trained AI's mind: the single changes, of the budget the
 * prototype measured, that make it throw the player's tile, grouped by what
 * they changed in the engine's facts; the smallest score step that does it;
 * and where no single change does, the best pair of the eight changes that
 * came nearest. The first table asked is the unedited one, and a search whose
 * unedited table Trained AI does not answer as the review recorded stops
 * there, since nothing it found could be compared. `onProgress({ asked,
 * planned })` hears every pass.
 *
 * Answers `{ status: 'searched', said, singles, pairs, thresholds, asked }`,
 * `said` being the findings in words (`saidLines`); `{ status: 'unlike the
 * review', said }` where it stopped at the unedited table; `{ status: 'too
 * sure' }` below the margin floor, where it asks nothing; and `{ status: 'not
 * a discard' }` for a decision version 1 does not reason about. */
export async function whatWouldChangeItsMind({ position, played, preferred, weights, facts, ask, signal, onProgress = () => {}, size = PASS }) {
  const start = baselineOf({ position, played, preferred, weights, facts });
  if (!start) return { status: 'not a discard' };
  const { F0, baseline, unseen } = start, P = played.tile, A = preferred.tile;
  if (baseline.margin < MARGIN_FLOOR) return { status: 'too sure', margin: baseline.margin };
  const budget = budgetOf(position, { played, preferred }, F0, facts);
  const net = asker({ ask, signal, size, onProgress });
  const context = { base: position, played, preferred, F0, facts };
  net.plan(budget.length + 1);
  let unlike = null;
  const answers = await net.all([position, ...budget.map(edit => applyEdits(position, edit))], replies => {
    unlike ??= unlikeReview(replies[0], weights);
    return !unlike;
  });
  if (unlike) return { status: 'unlike the review', unlike, asked: net.asked, said: { lead: UNLIKE, detail: '', items: [], still: '' } };
  const changes = judged(budget, answers.slice(1), context);

  // The smallest score step that makes Trained AI throw the player's tile,
  // halving between the steps asked, a thousand points apart.
  const thresholds = [];
  for (const sign of [1, -1]) {
    const steps = [8000, 16000].map(step => changes.find(change => change.family === 'scores' && change.param === step * sign));
    const first = steps.findIndex(change => change?.flips);
    if (first < 0) continue;
    let low = first === 0 ? 0 : 8000 * sign, high = (first === 0 ? 8000 : 16000) * sign, atHigh = steps[first], tried = 0;
    for (;;) {
      const middle = Math.round((low + high) / 2000) * 1000;
      if (middle === low || middle === high) break;
      const edit = scoreEdit(position, middle);
      net.plan(1);
      const [step] = judged([edit], await net.all([applyEdits(position, edit)]), { ...context, withFacts: false });
      tried++;
      if (step.flips) [high, atHigh] = [middle, step]; else low = middle;
    }
    thresholds.push({ ...atHigh, label: sign > 0 ? 'ahead' : 'behind', at: high, tried });
  }

  // Pairs of the eight changes nearest to flipping it, only when no single
  // change does, a score step past 8,000 points among them: two that touch
  // the same thing are never paired.
  const one = changes.filter(change => !change.refused && change.cost === 1 && !change.removesPreferred);
  let pairs = [];
  if (!one.some(change => change.flips) && !thresholds.length) {
    const chosen = [];
    for (const change of [...one].sort((x, y) => y.margin - x.margin)) {
      if (chosen.length < PAIRED && !chosen.some(other => other.key === change.key)) chosen.push(change);
    }
    const pairEdits = chosen.flatMap((a, i) => chosen.slice(i + 1).map(b => ({ family: `${a.family}+${b.family}`, key: `${a.key}|${b.key}`,
      cost: a.cost + b.cost, words: `${a.words}, and ${b.words}`, parts: [a.edit, b.edit], apply: table => { a.edit.apply(table); b.edit.apply(table); } })));
    net.plan(pairEdits.length);
    pairs = judged(pairEdits, await net.all(pairEdits.map(edit => applyEdits(position, ...edit.parts))), context);
  }
  const said = searchSaid({ changes, pairs, thresholds, A, P, F0, unseen });
  return { status: 'searched', asked: net.asked, singles: changes, pairs, thresholds, said };
}

/** What a search says when the unedited table is not the review's. */
const UNLIKE = 'This table does not replay as the review asked it, so no change was tried.';

const IDEAS = ['faster', 'worth more', 'safety', 'shape', 'table'];
const TITLES = { faster: 'A faster hand', 'worth more': 'A hand worth more', safety: 'What is safe', shape: 'The shape of the hand', table: 'The table' };

/** A move as the panel names it: a discard by its tile, and a quad of either
 * kind a kan of its tile, as the hand's list of decisions calls it. */
export function moveWords({ kind, tile }) {
  if (kind === 'discard') return the(tile);
  if (kind === 'riichi') return 'riichi';
  if (kind === 'tsumo') return 'the win';
  return /kan$/.test(kind) && tile ? `a kan of the ${tileWords(tile)}` : kind;
}

/** The weights after a change: the move it now rates highest first. */
function weightsAfter(change, A, P) {
  const { top } = change, at = tile => top.kind === 'discard' && top.tile === tile;
  if (at(P)) return `${your(P)} ${percent(change.played)}, ${the(A)} ${percent(change.preferred ?? 0)}`;
  if (at(A)) return `${the(A)} ${percent(change.preferred)}, ${your(P)} ${percent(change.played)}`;
  return `${moveWords(top)} ${percent(top.weight)}, ${your(P)} ${percent(change.played)}, ${the(A)} ${percent(change.preferred ?? 0)}`;
}

const byMargin = (x, y) => y.margin - x.margin;

/** The table's line, or null where nothing at the table flips the choice:
 * the smallest score step found either way, everyone level, and the round
 * that does it most. A step found past 8,000 points is said with the rest:
 * it is one change, if a large one, and the line of what moved nothing
 * leaves the score out once any step has moved it. */
function tableSaid(one, thresholds, A, P) {
  const parts = thresholds.map(step => `from ${points(Math.abs(step.at))} points ${step.at > 0 ? 'more' : 'fewer'}: ${weightsAfter(step, A, P)}`);
  const level = one.find(change => change.family === 'scores' && change.param == null && change.flips);
  if (level) parts.push(`if ${level.words}: ${weightsAfter(level, A, P)}`);
  const [round] = one.filter(change => change.family === 'round' && change.flips).sort(byMargin);
  if (round) parts.push(`if ${round.words}: ${weightsAfter(round, A, P)}`);
  return parts.length ? `${TITLES.table}: ${parts.join('; ')}.` : null;
}

/** The search's findings in version 1's words: a lead line, and either the
 * ideas that flip it, each with its strongest change and how many changes of
 * that idea were tried, and the table's line, or the pair that does, or the
 * nearest change; then what moved nothing. */
function searchSaid({ changes, pairs, thresholds, A, P, F0, unseen }) {
  const one = changes.filter(change => !change.refused && change.cost === 1 && !change.removesPreferred);
  const flips = one.filter(change => change.flips), table = tableSaid(one, thresholds, A, P);
  const said = { lead: '', detail: '', items: [], still: '' };
  if (flips.length || table) {
    said.lead = `One change and it throws ${your(P)}:`;
    for (const idea of IDEAS) {
      if (idea === 'table') {
        if (table) said.items.push(table);
        continue;
      }
      const members = flips.filter(change => primary(change, unseen) === idea).sort(byMargin);
      if (!members.length) continue;
      const [best] = members, also = changeSaid(F0, best, A, P, unseen);
      const tried = one.filter(change => primary(change, unseen) === idea).length;
      said.items.push(`${TITLES[idea]} (${members.length} of ${tried} tried): if ${best.words}${also.length ? ` (${list(also)})` : ''}: ${weightsAfter(best, A, P)}.`);
    }
  } else {
    const pair = pairs.filter(change => !change.refused && change.flips).sort(byMargin)[0];
    const near = [...one].sort(byMargin)[0];
    if (pair) {
      const also = deltaWords(pair.delta, A, P, unseen);
      said.lead = 'No single change does it. Two together:';
      said.detail = `if ${pair.words}${also.length ? ` (${list(also)})` : ''}: ${weightsAfter(pair, A, P)}.`;
    } else {
      said.lead = `Nothing asked makes it throw ${your(P)}.`;
      said.detail = near ? `Nearest: if ${near.words}: ${weightsAfter(near, A, P)}.` : '';
    }
  }
  const tried = changes.filter(change => !change.refused);
  const still = ['scores', 'round'].filter(family => tried.some(change => change.family === family) && !tried.some(change => change.family === family && change.flips));
  if (still.length) said.still = `Not moved by ${list(still.map(family => (family === 'scores' ? `the score (±${points(16000)})` : 'the round')), 'or')}.`;
  return said;
}

/** A search's findings, or a Why line, as lines of plain text: the lead the
 * panel sets in bold, its items each on a line of its own, and what moved
 * nothing. */
export function saidLines(said) {
  if ('label' in said) return [`${said.label}: ${said.text}`];
  return [[said.lead, said.detail].filter(Boolean).join(' '), ...said.items.map(item => `- ${item}`), ...(said.still ? [said.still] : [])];
}

const far = n => (n === 0 ? 'waiting' : n < 0 ? 'complete' : `${n} from waiting`);

/** Every engine fact on which Trained AI's tile `A` and the player's `P`
 * differ, with the side it favours (after the facts study's differences). */
function differences(A, P, F) {
  const out = [];
  const add = (family, favours, size, text, extra = {}) => out.push({ family, favours, size, text, ...extra });
  const a = `Trained AI's ${tileWords(A.tile)}`, p = your(P.tile);
  if (A.shanten !== P.shanten) {
    add('distance', A.shanten < P.shanten ? 'ai' : 'you', Math.abs(A.shanten - P.shanten),
      `${capital(p)} leaves the hand ${far(P.shanten)}; ${a} leaves it ${far(A.shanten)}.`);
  } else {
    const d = A.acceptance - P.acceptance, most = Math.max(A.acceptance, P.acceptance);
    if (Math.abs(d) >= Math.max(4, 0.2 * most)) {
      add('speed', d > 0 ? 'ai' : 'you', Math.abs(d) / Math.max(most, 1), A.shanten === 0
        ? `Both leave the hand waiting: ${A.waitsLeft} tiles left to win on after ${a}, ${P.waitsLeft} after ${p}.`
        : `Both leave the hand ${far(A.shanten)}; ${A.acceptance} unseen tiles would improve it after ${a}, ${P.acceptance} after ${p}.`);
    }
  }
  const exposed = tile => (tile.tier === 'safe' ? 'cannot deal in' : `is ${TIER_WORDS[tile.tier]}`);
  if (F.threats.length) {
    const ia = TIERS.indexOf(A.tier), ip = TIERS.indexOf(P.tier);
    if (ia !== ip && Math.abs(RATE[A.tier] - RATE[P.tier]) >= 0.02) {
      add('safety', ia < ip ? 'ai' : 'you', Math.abs(RATE[A.tier] - RATE[P.tier]),
        `Against ${list(F.threats.map(seat => WINDS[seat]))}'s riichi ${p} ${exposed(P)}; ${a} ${exposed(A)}.`);
    }
  } else if (A.unseen === 0 && P.unseen > 0) {
    // With nobody in riichi, a tile whose every copy is in sight cannot be
    // anyone's wait, bar thirteen orphans: the safest throw there is.
    add('safety', 'ai', 1, `Every copy of ${the(A.tile)} is in sight, so nobody can be waiting on it; ${COUNT[P.unseen]} ${tileWords(P.tile)} ${P.unseen === 1 ? 'is' : 'are'} still unseen.`, { dead: true });
  } else if (P.unseen === 0 && A.unseen > 0) {
    add('safety', 'you', 1, `Every copy of ${your(P.tile)} is in sight, so nobody can be waiting on it; ${COUNT[A.unseen]} ${tileWords(A.tile)} ${A.unseen === 1 ? 'is' : 'are'} still unseen.`, { dead: true });
  } else if (A.unseen === 0) {
    add('dead', 'neither', 0, 'Every copy of both tiles is in sight, so neither can deal in.');
  }
  for (const form of ['thirteen orphans', 'seven pairs']) {
    const fa = A.forms[form], fp = P.forms[form], least = Math.min(A.shanten, P.shanten);
    if (fa == null || fp == null || fa === fp || Math.min(fa, fp) > Math.max(3, least) || Math.min(fa, fp) > least + 1) continue;
    add('plan', fa < fp ? 'ai' : 'you', Math.abs(fa - fp), `${capital(fa < fp ? a : p)} keeps ${form} ${far(Math.min(fa, fp))}; ${fa < fp ? p : a} leaves it ${far(Math.max(fa, fp))}.`, { form });
  }
  if (A.ownRiver !== P.ownRiver) add('furiten', A.ownRiver ? 'ai' : 'you', 1, `You had already thrown ${an(A.ownRiver ? A.tile : P.tile)}, so a wait on it would be furiten.`);
  if (A.dora !== P.dora) add('value', A.dora < P.dora ? 'ai' : 'you', Math.abs(A.dora - P.dora), `${capital(A.dora > P.dora ? a : p)} is dora.`, { dora: true });
  if (A.pairBreak !== P.pairBreak) add('value', P.pairBreak ? 'ai' : 'you', 1, `${capital(A.pairBreak ? a : p)} breaks a pair of a valued honour.`);
  if (A.honour && P.honour && A.single && P.single && A.unseen !== P.unseen) {
    const [dead, live] = A.unseen < P.unseen ? [A, P] : [P, A];
    add('liveness', dead === A ? 'ai' : 'you', Math.abs(A.unseen - P.unseen),
      `Only ${dead.unseen} ${dead.unseen === 1 ? 'copy' : 'copies'} of ${the(dead.tile)} ${dead.unseen === 1 ? 'is' : 'are'} unseen; ${the(live.tile)} has ${live.unseen}.`);
  }
  return out;
}

/** The fact that leads among those favouring Trained AI's tile: against a
 * riichi with a slow hand, safety first; otherwise the distance, then the
 * rest. */
function leading(diffs, F) {
  const order = F.threats.length && F.best >= 2 ? ['safety', 'distance'] : ['distance', 'safety'];
  order.push('plan', 'speed', 'furiten', 'value', 'liveness');
  return diffs.filter(d => d.favours === 'ai').sort((x, y) => order.indexOf(x.family) - order.indexOf(y.family) || y.size - x.size)[0] ?? null;
}

/** Trained AI's weight on the tiles that share its tile's leading fact. */
function massOf(fact, F, weight, A, P) {
  const sum = shared => Object.values(F.tiles).filter(shared).reduce((total, tile) => total + (weight[tile.tile] ?? 0), 0);
  switch (fact?.family) {
    case 'distance': return { what: `tiles that keep the hand ${far(A.shanten)}`, mass: sum(t => t.shanten <= A.shanten) };
    case 'safety': return fact.dead ? { what: 'tiles with every copy in sight', mass: sum(t => t.unseen === 0) }
      : { what: 'tiles that cannot deal in', mass: sum(t => t.tier === 'safe') };
    case 'speed': return { what: `tiles that leave the hand ${far(A.shanten)} with more improving tiles than yours`,
      mass: sum(t => t.shanten <= A.shanten && t.acceptance > P.acceptance) };
    case 'plan': return { what: `tiles that keep ${fact.form} ${far(A.forms[fact.form])}`, mass: sum(t => t.forms[fact.form] != null && t.forms[fact.form] <= A.forms[fact.form]) };
    case 'furiten': return { what: 'tiles you had already thrown', mass: sum(t => t.ownRiver) };
    default: return null;
  }
}

/** Several changes made as one, the probe that takes a fact away where one
 * change alone would leave it standing, said in one clause: every riichi
 * taken back ("South and North had not declared riichi"), or every copy the
 * player threw of a tile made another, each the same tile ("your second and
 * ninth discards had each been a 1 characters, not the 8 bamboo"). */
function together(family, parts) {
  if (parts.length < 2) return parts;
  const words = family === 'riichi' ? `${list(parts.map(part => WINDS[part.seat]))} had not declared riichi`
    : `your ${list(parts.map(part => nth(part.index)))} discards had each been ${an(parts[0].into)}, not the ${tileWords(parts[0].from)}`;
  return [{ family, key: family, cost: parts.length, words, parts, apply: table => parts.forEach(part => part.apply(table)) }];
}

/** The change that takes the leading fact away, where it has a clean one: a
 * copy of the dead tile out of sight again, every riichi taken back, every
 * discard the player made of the furiten tile made another, another tile
 * made the dora. A change that leaves the fact standing would test nothing:
 * one copy of the furiten tile left in the river keeps a wait on it furiten,
 * so a copy another seat claimed, which cannot be changed, leaves no change
 * to try. Distance, speed and a plan have none either. */
function probeFor(fact, singles, me, river, A, P) {
  const first = picked => singles.filter(picked).slice(0, 1);
  if (fact?.family === 'safety' && fact.dead) return first(e => e.family === 'river' && e.from === A.tile);
  if (fact?.family === 'safety') return together('riichi', singles.filter(e => e.undeclares));
  if (fact?.family === 'furiten') {
    const tile = A.ownRiver ? A.tile : P.tile;
    if (river.some(discard => discard.tile === tile && discard.claimed)) return [];
    return together('river', singles.filter(e => e.family === 'river' && e.seat === me && e.from === tile));
  }
  if (fact?.family === 'value' && fact.dora) return first(e => e.family === 'dora' && e.into !== A.tile && e.into !== P.tile);
  return [];
}

/** Why Trained AI chose its tile, in the engine's facts: the fact that leads
 * among those favouring its tile, and two checks that it is the reason and
 * not just true. Trained AI must put 95% or more on tiles that share the
 * fact; and where the fact has a clean change, the table without it must
 * halve its lead over the player's tile, a change that also moved another
 * fact of the two tiles telling nothing. It is explained when both hold,
 * partly when a check fails, and not explained when no fact favours its
 * tile. Asks only where there is a change to try, in one pass with the
 * unedited table first, and stops there, unchecked, where Trained AI does
 * not answer that table as the review recorded.
 *
 * Answers `{ verdict, label, text, check }`, `check` being the workings for a
 * tooltip; or null for a decision version 1 does not reason about. */
export async function explainWhy({ position, played, preferred, weights, facts, ask, signal }) {
  const start = baselineOf({ position, played, preferred, weights, facts });
  if (!start) return null;
  const { F0, A, P, unseen } = start;
  const diffs = differences(A, P, F0), fact = leading(diffs, F0);
  const weight = Object.fromEntries(weights.filter(move => move.kind === 'discard').map(move => [move.tile, move.weight]));
  const mass = massOf(fact, F0, weight, A, P);
  const tries = probeFor(fact, singleEdits(position, { played, preferred }), F0.me, position.players[F0.me].discards, A, P);
  let probes = [];
  if (tries.length) {
    if (signal?.aborted) throw stopped();
    const answers = await ask([position, ...tries.map(edit => applyEdits(position, edit))], signal);
    if (signal?.aborted) throw stopped();
    const unlike = unlikeReview(answers?.[0], weights);
    if (unlike) return { verdict: 'unlike the review', unlike, label: 'Not checked', text: UNLIKE, check: '' };
    const lead = (a, p) => Math.log(Math.max(a, 1e-9)) - Math.log(Math.max(p, 1e-9));
    const before = lead(weight[A.tile] ?? 0, weight[P.tile] ?? 0);
    probes = judged(tries, answers.slice(1), { base: position, played, preferred, F0, facts }).filter(change => !change.refused).map(change => {
      const d = change.delta, after = change.removesPreferred ? null : lead(change.preferred, change.played);
      const other = { safety: d.shantenP || d.shantenA || d.dora, furiten: d.tierA || d.tierP || riichiMoved(d),
        value: d.unseenA || d.unseenP || d.tierA || d.tierP }[fact.family];
      return { ...change, halved: after != null && after <= before / 2, void: Boolean(other) };
    });
  }
  const probe = probes.find(change => !change.void) ?? null;
  const verdict = !fact ? 'not explained' : (mass?.mass ?? 0) >= MASS && (probe ? probe.halved : true) ? 'explained' : 'partly explained';
  return { verdict, ...whySaid({ verdict, fact, mass, probe, voided: probes.find(change => change.void), diffs, A: A.tile, P: P.tile, unseen }) };
}

/** The Why line in version 1's words, which leave out what the table above it
 * already shows, and the check's workings, for a tooltip on Partly and Not
 * explained. */
function whySaid({ verdict, fact, mass, probe, voided, diffs, A, P, unseen }) {
  const massSaid = mass ? `Trained AI puts ${percent(mass.mass)} on ${mass.what}.` : '';
  const facts = family => diffs.filter(d => d.favours === 'you' && family.includes(d.family)).map(d => lower(sentence(d.text)));
  if (verdict === 'explained') {
    return { label: 'Why', check: '', text: `${sentence(fact.text)}. ${massSaid}${probe ? ` If ${probe.words}: ${weightsAfter(probe, A, P)}.` : ''}` };
  }
  if (verdict === 'partly explained') {
    const yours = facts(['safety']);
    let text = `${sentence(fact.text)}.${massSaid ? ` ${massSaid}` : ''}`;
    if (probe && !probe.halved) text += ` If ${probe.words}, it still does not choose yours: ${weightsAfter(probe, A, P)}.`;
    if (yours.length) text += ` On your side: ${yours.join('; ')}.`;
    const check = [];
    if (!probe && voided) {
      const others = deltaWords(voided.delta, A, P, unseen).filter(said => !/closer|further|improve/.test(said));
      check.push(`The check that takes it away (${voided.words}) also ${others.length ? `leaves ${list(others)}` : 'changes another fact of the two tiles'}, so it cannot say which counts.`);
    }
    if (probe && !probe.halved) check.push('Taking it away did not halve Trained AI\'s lead.');
    if ((mass?.mass ?? 0) < MASS) check.push(`Explained needs ${Math.round(MASS * 100)}% or more of its weight on ${mass ? 'those tiles' : 'tiles that share the fact'}.`);
    return { label: 'Partly', text, check: check.join(' ') };
  }
  const yours = facts(['distance', 'speed', 'safety']), neutral = diffs.find(d => d.favours === 'neither');
  const text = yours.length ? `The counts favour ${your(P)}.` : neutral ? `${sentence(neutral.text)}.` : 'The counts do not separate the two tiles.';
  return { label: 'Not explained', text,
    check: `No fact the engine counts favours ${the(A)}${yours.length ? `; for yours, ${yours.join('; ')}` : ''}.` };
}
