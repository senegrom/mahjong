import { TILE_TYPES, sequenceWords, tileWords } from './tiles.js';

/** What the table says while a Trained opponent waits for the network. The
 * opponent's worker reports its own loading; before that worker starts, the
 * page is usually still saving the network for offline play (on a first
 * visit, the whole 116 MB), and that download's progress is the wait. On a
 * first visit that download cannot begin until the service worker has saved
 * the game and every tile graphic, about 100 MB more, which reports no
 * progress: the note says what is happening rather than nothing for minutes. */
export function waitingNote(note, offline) {
  if (note) return note;
  if (offline?.phase === 'ai') return `downloading the trained network ${offline.progress ?? 0}%`;
  return offline?.installing ? 'saving the game and tile graphics, then downloading the trained network' : '';
}

export function heldSafeCount(view) {
  const me = view?.seats?.[0];
  if (!me || view.phase === 'over') return 0;
  // Count physical, discardable concealed tiles, not all globally safe types.
  return [...me.hand, ...(me.drawn ? [me.drawn] : [])].filter((tile) => view.safe.includes(tile)).length;
}

/** A choice as its button says it. `robbing` is true in a window to rob a
 * quad, where the win is on the tile added to it, not on a discard. */
export function callLabel(choice, robbing = false) {
  const tile = tileWords(choice.tile);
  switch (choice.kind) {
    case 'chii': return `Chii — ${sequenceWords(choice.tile)}`;
    case 'pon': return 'Pon — triplet';
    case 'kan': return 'Kan — quad';
    case 'ron': return robbing ? 'Ron — rob the quad' : 'Ron — win on discard';
    case 'tsumo': return 'Tsumo — self-draw win';
    case 'riichi': return `Riichi — discard the ${tile}`;
    case 'concealed-kan': return `Kan — concealed quad of ${tile}`;
    case 'extended-kan': return `Kan — extend the ${tile} triplet`;
    default: return 'Pass';
  }
}

/** The kind of quad a call window offers to be robbed, `'extended-kan'` or
 * `'concealed-kan'`, or `null` at an ordinary discard. The engine puts a
 * declared quad's tile on the table as it puts a discard, so the window is
 * told apart by the quad itself: all four copies are in it, so no other
 * seat's tile can be one of them (EMA section 3.3.13). */
export function quadOnOffer(view) {
  if (view?.phase !== 'call' || !view.pending_discard) return null;
  const from = view.seats?.find(seat => seat.seat === view.pending_from);
  return from?.melds?.find(meld => ['extended-kan', 'concealed-kan'].includes(meld.kind)
    && meld.tiles.includes(view.pending_discard))?.kind ?? null;
}

/** What a win is worth, as a table announces it, written from the engine's
 * numbers rather than read out of its sentence: a win by discard as its
 * total, a dealer's self-draw as what each of the others pays ("4,000 all"),
 * and anyone else's as each non-dealer's share and then the dealer's
 * ("2,000 / 4,000"). A self-draw's first payment alone is only a part of the
 * hand. Counters are in every figure; riichi bets are not. */
export function winValue(win) {
  const { total, from_dealer: dealer, from_each_other: each } = win.payments;
  if (win.by !== 'self-draw') return total.toLocaleString();
  return dealer ? `${each.toLocaleString()} / ${dealer.toLocaleString()}` : `${each.toLocaleString()} all`;
}

/** A change of points as a score sheet writes it: "+8,000", "-2,000", and
 * "0" for none. */
export function signed(value) {
  if (value === 0) return '0';
  return value > 0 ? `+${value.toLocaleString()}` : value.toLocaleString();
}

/** A weight or a chance from 0 to 1 as a percentage to one decimal place.
 * Anything above nothing but under 0.1% is written "<0.1%", so that it does
 * not read as 0.0%. */
export function percent(weight) {
  return weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
}

export function callTiles(choice, pending) {
  if (choice.kind === 'chii' && /^[1-7][mps]$/.test(choice.tile ?? '')) {
    return [0, 1, 2].map((offset) => `${Number(choice.tile[0]) + offset}${choice.tile[1]}`);
  }
  if (choice.kind === 'pon' && pending) return [pending, pending, pending];
  if (choice.kind === 'kan' && pending) return Array(4).fill(pending);
  if (choice.kind === 'concealed-kan' || choice.kind === 'extended-kan') return Array(4).fill(choice.tile);
  return choice.tile ? [choice.tile] : choice.kind === 'ron' && pending ? [pending] : [];
}

export function acceptsHandKey(event, hand) {
  if (!hand || event.defaultPrevented || event.isComposing || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  const target = event.target;
  if (!target || !hand.contains(target)) return false;
  if (target.closest('input, select, textarea, a, [contenteditable]:not([contenteditable="false"])')) return false;
  const button = target.closest('button');
  return !button || button.classList.contains('tile');
}

/** Preserve meld shape, but move the tile actually claimed to its source-side
 * position. Concealed quads have no claimed tile; an added kan retains the pon's.
 */
export function meldTiles(meld) {
  const tiles = [...meld.tiles];
  const source = tiles.indexOf(meld.claimed_tile);
  if (meld.kind === 'concealed-kan' || source < 0) return tiles;
  const target = meld.from === 'left' ? 0 : meld.from === 'across' ? 1 : tiles.length - 1;
  tiles.splice(source, 1);
  tiles.splice(target, 0, meld.claimed_tile);
  return tiles;
}

// The game's view names an open kan 'claimed-kan'; the physical and guided
// tables name it 'kan'.
const MELD_WORDS = { chii: 'Chii', pon: 'Pon', kan: 'Open kan', 'claimed-kan': 'Open kan', 'extended-kan': 'Added kan', 'concealed-kan': 'Concealed kan' };

/** A called set in words, as a screen reader should say it: "Pon of 5 bamboo". */
export function meldWords(meld) {
  const lowest = [...(meld.tiles ?? [])].sort()[0];
  const tiles = meld.kind === 'chii' ? sequenceWords(lowest) : tileWords(lowest);
  return `${MELD_WORDS[meld.kind] ?? 'Called set'} of ${tiles}`;
}

export function placeLabel(row) {
  const place = ['1st', '2nd', '3rd', '4th'][row.place - 1] ?? String(row.place);
  return row.tied ? `Joint ${place}` : place;
}

/** Move among legal tile buttons, not array slots that may be disabled.
 * The returned marker always names the element that actually owns focus.
 */
export function moveHandFocus(hand, direction) {
  if (!hand || ![-1, 1].includes(direction)) return null;
  const buttons = [...hand.querySelectorAll('button[data-hand-index]')].filter(button => !button.disabled);
  if (!buttons.length) return null;
  const document = hand.ownerDocument;
  const active = document.activeElement;
  if (!hand.contains(active)) return null;
  const current = active?.closest('[data-hand-index]');
  const position = buttons.indexOf(current);
  const next = position < 0 ? buttons.length - 1 : Math.max(0, Math.min(buttons.length - 1, position + direction));
  const target = buttons[next];
  target.focus({ preventScroll: true });
  const focused = buttons.find(button => button === document.activeElement);
  return focused ? Number(focused.dataset.handIndex) : null;
}

/** Analyse each distinct legal discard once for a decision. The previews and
 * border hints share these exact, read-only engine results. In particular, do
 * not infer readiness from the 14-tile hand or require a riichi action: an open
 * hand can be tenpai too. Callers invalidate this map when choices change.
 */
export function analyzeDiscards(engine, choices = []) {
  const hints = new Map();
  if (!engine) return hints;
  for (const choice of choices) {
    if (choice.kind !== 'discard' || hints.has(choice.tile)) continue;
    hints.set(choice.tile, engine.discard_hint(choice.tile));
  }
  return hints;
}

/** Copies of every tile kind nobody can currently see from the human seat.
 * This is the same public information used for wait width: own concealed
 * tiles, unclaimed discards, all called sets and the exposed dora indicators.
 */
export function unseenTileCounts(view) {
  const left = new Map(TILE_TYPES.map(tile => [tile, 4]));
  const see = tile => {
    if (!left.has(tile)) return;
    left.set(tile, Math.max(0, left.get(tile) - 1));
  };
  for (const [index, seat] of (view?.seats ?? []).entries()) {
    if (index === 0) {
      for (const tile of seat.hand ?? []) see(tile);
      if (seat.drawn) see(seat.drawn);
    }
    for (const discard of seat.discards ?? []) if (!discard.claimed) see(discard.tile);
    for (const meld of seat.melds ?? []) for (const tile of meld.tiles ?? []) see(tile);
  }
  for (const tile of view?.dora_indicators ?? []) see(tile);
  return left;
}

// What each kind of set is called when it is made. A quad is a Kan however
// it came about: claimed from a discard, drawn concealed, or extended from
// a triplet already on the table.
const CALL_WORDS = {
  chii: 'Chii',
  pon: 'Pon',
  'claimed-kan': 'Kan',
  'extended-kan': 'Kan',
  'concealed-kan': 'Kan',
};

/**
 * The set a seat has just called, as the word said at the table, or `''`
 * when nothing was. `before` and `after` are that seat's meld kinds,
 * oldest first, either side of a change.
 *
 * A call is an event rather than a state: it is said when it happens and
 * then goes quiet, because the meld itself goes on showing what was
 * called. Sets that were already there when the seat came into view — a
 * restored match, the next deal, a different person in the chair — were
 * not called in front of anyone, so they say nothing. Extending a triplet
 * into a quad changes a set in place without adding one, and that is a
 * call too.
 */
export function calledSet(before, after) {
  if (!Array.isArray(before) || !Array.isArray(after)) return '';
  if (after.length < before.length) return '';
  const changed = after.findIndex((kind, at) => kind !== before[at]);
  if (changed === -1) return '';
  return CALL_WORDS[after[changed]] ?? '';
}

/**
 * What a seat's riichi or call says to a screen reader, or `''` when there is
 * nothing to say. Its listener hears the word without seeing which seat it
 * appeared at, so the seat is named: "South declared riichi", "West called
 * pon". `called` is the word calledSet gave, and riichi outlasts it, as the
 * seat's badge shows.
 */
export function seatAnnouncement(wind, riichi, called) {
  if (riichi) return `${wind} declared riichi`;
  return called ? `${wind} called ${called.toLowerCase()}` : '';
}
