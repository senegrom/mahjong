import { TILE_TYPES, WINDS, runTiles, tileWords } from './tiles.js';
import { sameMove } from './action-weights.js';

export const PHYSICAL_KEY = 'riichi.physical.v1';

export function parseTiles(text) {
  const compact = text.replace(/[\s,]+/g, '');
  if (!compact) return [];
  const groups = compact.match(/[1-9]+[mpsz]/g);
  if (!groups || groups.join('') !== compact) throw new Error('Use tile notation such as 123m456p789s11z');
  const tiles = groups.flatMap(group => [...group.slice(0, -1)].map(rank => `${rank}${group.at(-1)}`));
  if (tiles.some(tile => !TILE_TYPES.includes(tile))) throw new Error('Honours are 1z to 7z');
  return tiles;
}

export function emptyPosition() {
  return { seat: 0, turn: 0, phase: 'act', round: 0, kyoku: 1, counters: 0, riichi_sticks: 0,
    wall: 69, indicators: [], drawn: null, pending: null, pending_kind: 'discard', just_claimed: null,
    after_quad: false, first_turns: true,
    players: Array.from({ length: 4 }, () => ({ hand: [], melds: [], discards: [], score: 30000, riichi: 'none', ippatsu: false, furiten: false })) };
}

export function parsePhysical(text) {
  try {
    if (typeof text !== 'string' || text.length > 100_000) return null;
    const value = JSON.parse(text);
    const p = value?.position;
    const number = value => value == null || Number.isFinite(value);
    const flag = value => value == null || typeof value === 'boolean';
    const optionalTile = value => value == null || TILE_TYPES.includes(value);
    // This is a draft, possibly incomplete. Rust validates the full state
    // before it can ever reach the policy or produce a legal action.
    if (value?.version === 1 && p && JSON.stringify(p).length < 100_000
      && Array.isArray(p.players) && p.players.length === 4 && Array.isArray(p.indicators) && p.indicators.every(tile => TILE_TYPES.includes(tile))
      && ['act', 'call', 'draw', 'over'].includes(p.phase)
      && ['', 'discard', 'extended-kan', 'concealed-kan'].includes(p.pending_kind ?? '')
      && ['round', 'kyoku', 'counters', 'riichi_sticks', 'wall'].every(key => number(p[key]))
      && ['drawn', 'pending', 'just_claimed'].every(key => optionalTile(p[key]))
      && flag(p.first_turns) && flag(p.after_quad)
      && p.players.every(player => player && Array.isArray(player.hand) && Array.isArray(player.melds)
        && Array.isArray(player.discards) && player.hand.every(tile => TILE_TYPES.includes(tile))
        && number(player.score) && ['none', 'riichi', 'double'].includes(player.riichi) && flag(player.ippatsu) && flag(player.furiten)
        && optionalTile(player.drew_at_riichi)
        && player.melds.every(m => m && TILE_TYPES.includes(m.tile) && number(m.from) && optionalTile(m.claimed_tile)
          && (m.declared_turn == null || (Number.isInteger(m.declared_turn) && m.declared_turn >= 0 && m.declared_turn < MAX_ORDER))
          && ['chii', 'pon', 'kan', 'extended-kan', 'concealed-kan'].includes(m.kind))
        && player.discards.every(d => d && TILE_TYPES.includes(d.tile) && number(d.order) && flag(d.drawn) && flag(d.riichi) && flag(d.claimed)))
      && Number.isInteger(p.seat) && p.seat >= 0 && p.seat < 4 && Number.isInteger(p.turn) && p.turn >= 0 && p.turn < 4) return p;
  } catch { /* An unreadable draft must not stop the normal game. */ }
  return null;
}

/** The first number a draft lacks or holds out of its range, as a sentence, or
 * null when every number is one the engine takes. A number field the person
 * has emptied is null in the draft, and a number field takes whatever is
 * typed into it, a negative count or one past its `max` too. The engine's own
 * answers to those are "invalid type: unit value" and "invalid value: integer
 * `-1`, expected u32". The ranges are the engine's (`Position::build`); the
 * live wall's also depends on the kans, which the engine says itself. */
export function missingNumber(position) {
  const fields = [['wall', 'live wall count', 0, 70], ['kyoku', 'hand number', 1, 4], ['counters', 'honba count', 0, 100],
    ['riichi_sticks', 'riichi sticks', 0, 100]];
  for (const [key, name, least, most] of fields) {
    const value = position?.[key];
    if (!Number.isInteger(value)) return `Enter the ${name}`;
    if (value < least || value > most) return `The ${name} must be ${least} to ${most}`;
  }
  const within = (value, least, most) => Number.isInteger(value) && value >= least && value <= most;
  if (!within(position?.round, 0, 3)) return 'Choose the round wind';
  for (const [index, player] of (position?.players ?? []).entries()) {
    if (!Number.isInteger(player?.score)) return `Enter ${WINDS[index]}'s points`;
    if (!within(player.score, -100000, 200000)) return `${WINDS[index]}'s points must be -100,000 to 200,000`;
    if ((player.melds ?? []).some(m => !within(m?.from, 0, 3))) return `Choose who supplied each of ${WINDS[index]}'s called sets`;
    if ((player.discards ?? []).some(d => !within(d?.order, 0, MAX_ORDER - 1))) {
      return `Give each of ${WINDS[index]}'s discards an order number from 0 to ${MAX_ORDER - 1}`;
    }
  }
  return null;
}

/** Discard order numbers run below this; the engine refuses any other. */
export const MAX_ORDER = 400;

/** The order number the next discard at the table will carry. The one count
 * for the physical table's recorded moves, its editor's earlier discards and
 * a guided game's kans, which each used to count it. */
export function nextOrder(position) {
  return Math.max(-1, ...position.players.flatMap(player => player.discards.map(entry => entry.order))) + 1;
}

/** A quad's turn as recorded: the next discard's order number, or none where
 * that is past what a draft holds. */
export function quadTurn(position) {
  const turn = nextOrder(position);
  return turn < MAX_ORDER ? turn : undefined;
}

const SELF_QUADS = ['concealed-kan', 'extended-kan'];

/** The position as the engine is asked it. A draft keeps two facts of
 * Mortal's replay under names of its own, which the engine and every other
 * reader of the draft pass over: a seat's `drew_at_riichi`, the tile it drew
 * on the turn it declared riichi, and a quad's `declared_turn`, the order
 * number of the discard that turn ended with, or will. Here they become the
 * engine's `riichi_draw` and `order` where they still hold after any edit,
 * where the engine would otherwise refuse the table: the riichi draw for the
 * seat analysed alone while it is in riichi; a quad's turn while the set is
 * still a quad of the seat's own, that discard still its own or the turn
 * still being played, and for an added kan still after the discard its pon
 * took. Without them the engine guesses, as for a position typed in. The
 * engine's own two names are never a draft's: whatever a draft holds under
 * them is left out. Mortal's replay refuses a turn on one ground more, which
 * only the engine's pairing of sets and discards knows (analysisOf). */
export function analysisInput(position) {
  const p = structuredClone(position), made = nextOrder(p);
  const playingOn = (p.phase === 'act' && p.after_quad) || (p.phase === 'call' && SELF_QUADS.includes(p.pending_kind));
  p.players.forEach((player, index) => {
    const drawn = player.drew_at_riichi;
    delete player.drew_at_riichi;
    delete player.riichi_draw;
    if (drawn != null && index === p.seat && player.riichi !== 'none') player.riichi_draw = drawn;
    for (const meld of player.melds) {
      const turn = meld.declared_turn;
      delete meld.declared_turn;
      delete meld.order;
      if (turn == null || !SELF_QUADS.includes(meld.kind)) continue;
      const ended = player.discards.some(entry => entry.order === turn);
      const played = playingOn && index === p.turn && turn === made;
      // The discard an added kan's pon took, from the seat it came from: the
      // engine pairs a triplet with a claimed discard of the set's own tile.
      const supplier = p.players[(index + meld.from) % 4];
      const early = meld.kind === 'extended-kan' && Boolean(supplier?.discards.some(entry => entry.claimed
        && entry.tile === meld.tile && entry.order >= turn));
      if ((ended || played) && !early) meld.order = turn;
    }
  });
  return p;
}

/** The engine for a typed-in table, which the caller frees; `Analysis` is the
 * engine's PhysicalAnalysis. The table is told with the two facts
 * (analysisInput) only where the engine takes them, building the table and
 * replaying it for Mortal, and otherwise without them, as every table was
 * told before a draft kept them. An edit after the move can leave a fact
 * that fits the table no longer, in ways only the engine's own pairing of
 * sets and discards knows: a quad's turn on a discard that now follows a
 * call makes Mortal's replay refuse the whole table. So the engine itself is
 * asked, and a fact never costs a table the advice it had without it. */
export function analysisOf(Analysis, position) {
  const told = analysisInput(position);
  if (told.players.some(player => player.riichi_draw != null || player.melds.some(meld => meld.order != null))) {
    let engine = null;
    try {
      engine = new Analysis(told);
      engine.agent_observation_mortal();
      return engine;
    } catch {
      engine?.free();
    }
    for (const player of told.players) {
      delete player.riichi_draw;
      for (const meld of player.melds) delete meld.order;
    }
  }
  return new Analysis(told);
}

/** A typed-in table after a change to it, `after`, less a riichi draw the
 * change left without its declaration. The draw is of the turn a seat
 * declared on, and the table's editor can move the declaration to another
 * discard, or take it back and put it elsewhere, without touching the draw,
 * which would then be told of a turn it was not drawn on. A change that
 * records a draw itself (recordDiscard) keeps its own. */
export function withoutStaleDraw(before, after) {
  const declared = seat => (seat.riichi === 'none' ? null : seat.discards.find(entry => entry.riichi)?.order ?? null);
  after.players.forEach((player, index) => {
    const was = before.players[index];
    if (player.drew_at_riichi != null && player.drew_at_riichi === was?.drew_at_riichi && declared(player) !== declared(was)) {
      delete player.drew_at_riichi;
    }
  });
  return after;
}

function remove(hand, tile, required = true) {
  const index = hand.indexOf(tile);
  if (index < 0) { if (required) throw new Error(`This hand does not hold the ${tileWords(tile)}`); return; }
  hand.splice(index, 1);
}

export function recordDraw(position, seat, tile) {
  if (!TILE_TYPES.includes(tile)) throw new Error('Choose the tile drawn at the physical table');
  if (position.wall <= 0) throw new Error('The live wall is empty');
  const p = structuredClone(position);
  const player = p.players[seat];
  if (player.hand.length !== 13 - 3 * player.melds.length) throw new Error('Enter the concealed hand before recording its draw');
  player.hand.push(tile);
  player.furiten = player.riichi !== 'none' && player.furiten;
  p.after_quad = p.turn === seat && (p.after_quad
    || (p.phase === 'call' && ['extended-kan', 'concealed-kan'].includes(p.pending_kind)));
  // Ippatsu survives the kan's robbery window, then ends when the kan stands.
  if (p.after_quad) p.players.forEach(player => { player.ippatsu = false; });
  p.turn = seat; p.seat = seat; p.phase = 'act'; p.drawn = tile;
  p.pending = null; p.pending_kind = 'discard'; p.just_claimed = null;
  p.wall--;
  return p;
}

export function recordDiscard(position, seat, tile, riichi = false) {
  if (!TILE_TYPES.includes(tile)) throw new Error('Choose the tile discarded at the physical table');
  const p = structuredClone(position);
  const player = p.players[seat];
  const known = player.hand.length > 0;
  if (known) {
    if (player.hand.length !== 14 - 3 * player.melds.length) throw new Error('Record this player’s draw or call before discarding');
    remove(player.hand, tile);
  }
  if (riichi && player.riichi !== 'none') throw new Error('This player has already declared riichi');
  if (riichi && player.score < 1000) throw new Error('Riichi needs 1,000 points');
  const double = p.first_turns && player.discards.length === 0;
  const order = nextOrder(p);
  player.discards.push({ tile, order, drawn: p.turn === seat && p.drawn === tile, riichi, claimed: false });
  if (riichi) { player.riichi = double ? 'double' : 'riichi'; player.ippatsu = true; player.score -= 1000; p.riichi_sticks++; }
  else player.ippatsu = false;
  // The tile drawn on the turn of the declaration, where the hand is known:
  // Mortal reads the hand of that turn for as long as the riichi stands. A
  // declaration whose draw is not known keeps none, nor that of an earlier
  // one the table's editor took back.
  if (riichi) {
    if (known && p.turn === seat && p.drawn) player.drew_at_riichi = p.drawn;
    else delete player.drew_at_riichi;
  }
  if (p.players.every(player => player.discards.length > 0)) p.first_turns = false;
  // Act means this seat already drew (including a replacement) or just
  // called pon/chii. Neither discard consumes another live-wall tile.
  // Only an unknown hand not yet at its act decision includes an implicit
  // draw. A kan awaiting its replacement still consumes one wall tile.
  const implicitDraw = !known && !(p.turn === seat && p.phase === 'act');
  if (implicitDraw) {
    if (!Number.isInteger(p.wall) || p.wall <= 0) throw new Error('The live wall is empty or its count is missing');
    p.wall--;
  }
  p.turn = seat; p.phase = 'call'; p.pending = tile; p.pending_kind = 'discard';
  p.drawn = null; p.just_claimed = null; p.after_quad = false;
  return p;
}

/** Record only the analysed seat's real choice. Never fill in unseen tiles,
 * other players' responses, a replacement draw, or another dora indicator. */
export function recordChoice(position, choice, choices) {
  if (!choices.some(entry => sameMove(entry, choice))) {
    throw new Error('Analyse this position again before recording a choice');
  }
  if (choice.kind === 'discard' || choice.kind === 'riichi') return recordDiscard(position, position.seat, choice.tile, choice.kind === 'riichi');
  const p = structuredClone(position), player = p.players[p.seat];
  if (choice.kind === 'pass') {
    if (choices.some(entry => entry.causes_furiten || entry.kind === 'ron')) player.furiten = true;
    // Keep the discard available for the other seats' real responses.
    return p;
  }
  if (choice.kind === 'ron' || choice.kind === 'tsumo') { p.phase = 'over'; return p; }
  if (['pon', 'chii', 'kan'].includes(choice.kind)) {
    const offered = p.pending;
    const discard = p.players[p.turn].discards.reduce((last, entry) => !last || entry.order > last.order ? entry : last, null);
    if (!discard || discard.tile !== offered || discard.claimed) throw new Error('This discard is no longer available');
    const tiles = choice.kind === 'chii'
      ? runTiles(choice.tile)
      : Array(choice.kind === 'kan' ? 4 : 3).fill(offered);
    if (!tiles.includes(offered)) throw new Error('The called set must include the offered tile');
    tiles.splice(tiles.indexOf(offered), 1);
    for (const tile of tiles) remove(player.hand, tile);
    discard.claimed = true;
    player.melds.push({ kind: choice.kind, tile: choice.kind === 'chii' ? choice.tile : offered, from: (p.turn - p.seat + 4) % 4, claimed_tile: offered });
    player.furiten = false;
    p.just_claimed = choice.kind === 'kan' ? null : offered;
  } else if (choice.kind === 'concealed-kan') {
    for (let i = 0; i < 4; i++) remove(player.hand, choice.tile);
    // A quad's turn is that of the discard the turn will end with.
    player.melds.push({ kind: choice.kind, tile: choice.tile, from: 0, declared_turn: quadTurn(p) });
  } else if (choice.kind === 'extended-kan') {
    const meld = player.melds.find(m => m.kind === 'pon' && m.tile === choice.tile);
    if (!meld) throw new Error('The pon is missing');
    remove(player.hand, choice.tile); meld.kind = choice.kind; meld.declared_turn = quadTurn(p);
  } else throw new Error('Unknown choice');
  p.first_turns = false;
  if (['pon', 'chii', 'kan'].includes(choice.kind)) p.players.forEach(player => { player.ippatsu = false; });
  p.turn = p.seat; p.drawn = null; p.pending = null;
  p.after_quad = choice.kind.includes('kan');
  p.phase = p.after_quad ? 'draw' : 'act';
  if (choice.kind === 'concealed-kan' || choice.kind === 'extended-kan') {
    p.phase = 'call'; p.pending = choice.tile; p.pending_kind = choice.kind;
  }
  return p;
}
