import { TILE_TYPES } from './tiles.js';

export const PHYSICAL_KEY = 'riichi.physical.v1';
export const TILES = Object.freeze([...TILE_TYPES]);

export function parseTiles(text) {
  const compact = text.replace(/[\s,]+/g, '');
  if (!compact) return [];
  const groups = compact.match(/[1-9]+[mpsz]/g);
  if (!groups || groups.join('') !== compact) throw new Error('Use tile notation such as 123m456p789s11z');
  const tiles = groups.flatMap(group => [...group.slice(0, -1)].map(rank => `${rank}${group.at(-1)}`));
  if (tiles.some(tile => !TILES.includes(tile))) throw new Error('Honours are 1z to 7z');
  return tiles;
}

export function emptyPosition() {
  return { seat: 0, turn: 0, phase: 'act', round: 0, kyoku: 1, counters: 0, riichi_sticks: 0,
    wall: 69, indicators: [], drawn: null, pending: null, pending_kind: 'discard', just_claimed: null,
    after_quad: false, first_turns: true,
    players: Array.from({ length: 4 }, () => ({ hand: [], melds: [], discards: [], score: 30000, riichi: 'none', ippatsu: false, furiten: false })) };
}

export function readPhysical(storage) {
  try {
    const value = JSON.parse(storage?.getItem(PHYSICAL_KEY));
    const p = value?.position;
    // This is a draft, possibly incomplete. Rust validates the full state
    // before it can ever reach the policy or produce a legal action.
    if (value?.version === 1 && p && JSON.stringify(p).length < 100_000
      && Array.isArray(p.players) && p.players.length === 4 && Array.isArray(p.indicators)
      && p.players.every(player => player && Array.isArray(player.hand) && Array.isArray(player.melds)
        && Array.isArray(player.discards) && player.hand.every(tile => TILES.includes(tile))
        && player.melds.every(m => m && TILES.includes(m.tile)) && player.discards.every(d => d && TILES.includes(d.tile)))
      && Number.isInteger(p.seat) && p.seat >= 0 && p.seat < 4 && Number.isInteger(p.turn) && p.turn >= 0 && p.turn < 4) return p;
  } catch { /* An unreadable draft must not stop the normal game. */ }
  return emptyPosition();
}

function remove(hand, tile, required = true) {
  const index = hand.indexOf(tile);
  if (index < 0) { if (required) throw new Error(`The hand does not contain ${tile}`); return; }
  hand.splice(index, 1);
}

export function recordDraw(position, seat, tile) {
  if (!TILES.includes(tile)) throw new Error('Choose the tile drawn at the physical table');
  if (position.wall <= 0) throw new Error('The live wall is empty');
  const p = structuredClone(position);
  const player = p.players[seat];
  if (player.hand.length !== 13 - 3 * player.melds.length) throw new Error('Enter the concealed hand before recording its draw');
  player.hand.push(tile);
  player.furiten = player.riichi !== 'none' && player.furiten;
  p.after_quad = p.after_quad && p.turn === seat;
  // Ippatsu survives the kan's robbery window, then ends when the kan stands.
  if (p.after_quad) p.players.forEach(player => { player.ippatsu = false; });
  p.turn = seat; p.seat = seat; p.phase = 'act'; p.drawn = tile;
  p.pending = null; p.pending_kind = 'discard'; p.just_claimed = null;
  p.wall--;
  return p;
}

export function recordDiscard(position, seat, tile, riichi = false) {
  if (!TILES.includes(tile)) throw new Error('Choose the tile discarded at the physical table');
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
  const order = Math.max(-1, ...p.players.flatMap(player => player.discards.map(entry => entry.order))) + 1;
  player.discards.push({ tile, order, drawn: p.turn === seat && p.drawn === tile, riichi, claimed: false });
  if (riichi) { player.riichi = double ? 'double' : 'riichi'; player.ippatsu = true; player.score -= 1000; p.riichi_sticks++; }
  else player.ippatsu = false;
  if (p.players.every(player => player.discards.length > 0)) p.first_turns = false;
  // Unknown opponents' live draws are recorded together with their discards.
  if (!known) p.wall = Math.max(0, p.wall - 1);
  p.turn = seat; p.phase = 'call'; p.pending = tile; p.pending_kind = 'discard';
  p.drawn = null; p.just_claimed = null; p.after_quad = false;
  return p;
}

/** Record only the analysed seat's real choice. Never fill in unseen tiles,
 * other players' responses, a replacement draw, or another dora indicator. */
export function recordChoice(position, choice, choices) {
  if (!choices.some(entry => entry.kind === choice.kind && (entry.tile ?? null) === (choice.tile ?? null))) {
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
      ? [0, 1, 2].map(offset => `${Number(choice.tile[0]) + offset}${choice.tile[1]}`)
      : Array(choice.kind === 'kan' ? 4 : 3).fill(offered);
    tiles.splice(tiles.indexOf(offered), 1);
    for (const tile of tiles) remove(player.hand, tile);
    discard.claimed = true;
    player.melds.push({ kind: choice.kind, tile: choice.kind === 'chii' ? choice.tile : offered, from: (p.turn - p.seat + 4) % 4 });
    player.furiten = false;
    p.just_claimed = choice.kind === 'kan' ? null : offered;
  } else if (choice.kind === 'concealed-kan') {
    for (let i = 0; i < 4; i++) remove(player.hand, choice.tile);
    player.melds.push({ kind: choice.kind, tile: choice.tile, from: 0 });
  } else if (choice.kind === 'extended-kan') {
    const meld = player.melds.find(m => m.kind === 'pon' && m.tile === choice.tile);
    if (!meld) throw new Error('The pon is missing');
    remove(player.hand, choice.tile); meld.kind = choice.kind;
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
