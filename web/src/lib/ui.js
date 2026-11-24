import { TILE_TYPES, runTiles, sequenceWords, tileWords } from './tiles.js';
import { playedNetwork } from './trained-network.js';

/** What the page says while it starts, which waits for two things at once:
 * the engine and the selected face's tile graphics. `tiles` is how many of
 * the graphics are in, `{ done, total }`, or null before the first; `drawn`
 * says they all are, and `engine` that the engine is. On a slow link the
 * graphics are often in first, and the line went on reading "37/37" for
 * seconds while the engine was what was still loading: then it says so. */
export function startupWords(tiles, drawn, engine) {
  if (drawn && !engine) return 'Loading the game engine…';
  return tiles ? `Loading selected tile graphics… ${tiles.done}/${tiles.total}` : 'Loading the game and selected tile graphics…';
}

/** A network's download in whole megabytes: what is fetched and kept
 * (storedBytes), or the network's own size where its listing gives no other. */
const megabytes = network => Math.round((network.storedBytes ?? network.bytes) / 1e6);

/** What the table says while a Trained opponent waits for the network. The
 * opponent's worker reports its own loading; before that worker starts, the
 * page is saving the network for offline play (on a first visit, the whole
 * 116 MB), and that download is the wait. It is read from where the network
 * itself stands, `offline.network` (offline.js), never from the whole save's
 * phase: the opponent plays as soon as the network is in hand, while the
 * phase stays 'ai' until the runtime is saved too, and a first visit's
 * install, which saves the game beside the network, delays it no more.
 * While the network downloads, `offline.progress` is the percent of it
 * received. The note says how much the download is, the size of the
 * `network` the page plays, once, so that a player on a slow or metered link
 * knows what the wait is ("downloading the trained network (59 MB) 37%").
 *
 * `held` says the opponents' worker holds the network already (policy.js
 * networkInWorker): then nothing waits for the page's download, which a new
 * game may start again where the network is not stored, and only the
 * worker's own note is said. */
export function waitingNote(note, offline, { held = false, network = playedNetwork() } = {}) {
  if (note) return note;
  if (held) return '';
  if (offline?.network === 'downloading') return `downloading the trained network (${megabytes(network)} MB) ${offline.progress ?? 0}%`;
  if (offline?.network === 'waiting') return 'waiting for another window to download the trained network';
  return '';
}

/** Whether a note of the opponent's worker says it is fetching the network
 * itself, or waits its turn behind another window's download of it
 * ("downloading the network 12%", "waiting for another window to download
 * the network"), rather than loading it from this device or thinking. */
export function noteIsDownload(note) {
  return /^(downloading|waiting for another window)\b/.test(note ?? '');
}

/** Whether what a Trained opponent waits for is a download, the wait
 * waitingNote names, rather than the network starting from this device's
 * copy or thinking: the worker's own note where it has one (noteIsDownload),
 * and with none, where the network itself stands, `offline.network`
 * 'downloading' or 'waiting'. Never by the whole save's phase, which stays
 * 'ai' while the runtime is saved and the opponents already play, nor by a
 * first visit's install, which the network does not wait for. Only then does
 * play offer Club meanwhile: a start or a thought takes seconds, a download
 * minutes. Never where the opponents' worker holds the network (`held`), as
 * waitingNote has it. */
export function waitIsDownload(note, offline, { held = false } = {}) {
  if (note) return noteIsDownload(note);
  if (held) return false;
  return offline?.network === 'downloading' || offline?.network === 'waiting';
}

/** A waiting note parted from the figure it ends with, how far a download
 * has got: "downloading the trained network (59 MB) 37%" is its words and
 * "37%". Play's prompt says the words in a live region, which a screen
 * reader then hears once, and shows the figure beside them, outside it: said
 * with them, every percent of a download would be read out, a hundred lines
 * in a row on a first visit. */
export function partedNote(note) {
  const [, words, figure = ''] = /^(.*?)(?:\s+(\d+%))?$/s.exec(note ?? '');
  return { words, figure };
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

/** How far a hand is from waiting, by its shanten, in the words every review
 * gives it: "complete", "waiting", "2 from waiting". */
export function fromWaiting(shanten) {
  if (shanten < 0) return 'complete';
  return shanten === 0 ? 'waiting' : `${shanten} from waiting`;
}

export function callTiles(choice, pending) {
  if (choice.kind === 'chii' && /^[1-7][mps]$/.test(choice.tile ?? '')) return runTiles(choice.tile);
  if (choice.kind === 'pon' && pending) return [pending, pending, pending];
  if (choice.kind === 'kan' && pending) return Array(4).fill(pending);
  if (choice.kind === 'concealed-kan' || choice.kind === 'extended-kan') return Array(4).fill(choice.tile);
  return choice.tile ? [choice.tile] : choice.kind === 'ron' && pending ? [pending] : [];
}

/** Whether a button's click came from the keyboard, Enter or Space on the
 * focused button, rather than from a mouse, a finger or a pen. A click's
 * detail is its click count (UI Events): a press of a mouse button, a tap or
 * a pen's touch counts at least one, in every browser, while the click a
 * browser makes for a key counts none, as does a script's click(). A screen
 * reader's press counts none in Chromium and may count one elsewhere. Either
 * answer is safe for what it decides, whether what is on the screen moves to
 * show where focus went: focus moves either way, and a screen reader follows
 * it. */
export function pressedByKeyboard(event) {
  return event?.detail === 0;
}

/** Whether a click is the second or a later one of a multiple click, such as
 * a double click's second press: a click's detail counts them (UI Events).
 * Play draws each decision where the last one was, a tile or a button in the
 * same place, and against opponents that answer at once the next is there
 * before a double click's second press lands, which would answer it too: two
 * tiles thrown by one double click. So a control whose first press has
 * already acted lets the rest of a multiple click go, as the review's buttons
 * that take each other's place do. A click made for a key counts none, and is
 * never let go here (heldPress). */
export function repeatedClick(event) {
  return (event?.detail ?? 0) > 1;
}

/** Hands focus on to a heading that says where the player now is, and moves
 * the page to show it only when the caller asks: for a press by keyboard,
 * whose player follows focus and is shown a ring there, and not for a
 * mouse's or a finger's, which is shown none. It moves by as little as will
 * show the heading, unless the caller asks for the heading at the top of the
 * screen (`block` 'start'), below the bar its scroll margin keeps clear, for
 * what the heading opens to be on the screen with it: the final standings on
 * a phone, as the result's sheet steps aside for them. */
export function land(heading, scroll, block = 'nearest') {
  heading?.focus({ preventScroll: true });
  if (scroll) heading?.scrollIntoView({ block });
}

/** What play says wherever a press can do nothing because another window has
 * taken the match on (App's saveConflict): in the turn's prompt while a hand
 * is played, and beside the result's and the standings' buttons, which
 * cannot be pressed meanwhile. The page's own alert, with its Reload latest
 * match, is at the top of the page, above the screen from the result at
 * desktop width and under the result's sheet on a phone. */
export const RELOAD_TO_CONTINUE = 'Reload the latest match to continue here.';

/** Where a hand's result is a sheet over the table and the hand, as
 * ScoreScreen's styles have it: on a phone held upright, and on a small one
 * held sideways, up to 760 pixels wide. */
export const RESULT_SHEET = '(max-width: 760.98px)';

/** Where the game's review has room for its list of hands and decisions in a
 * column of its own beside the decision on show, rather than folded above it. */
export const REVIEW_WIDE = '(min-width: 1100px) and (min-height: 600px)';

/** What each word a review tags a move with says to a screen reader: the
 * word is for the eye, and too short to be heard alone. */
export const TAG_SAID = Object.freeze({ You: 'your move', Now: 'your pick', AI: "Trained AI's choice" });

/** Why Club reviews a hand in Trained AI's place: the network cannot be
 * reached now, or, once the match has no engine (another window took it
 * over), there is no hand left to put to Trained AI. */
export function reviewFallbackNote(engine) {
  return engine
    ? 'The trained network is not available right now, so Club reviews this hand. Trained AI takes over again once the network can be reached.'
    : 'This hand can no longer be put to Trained AI, so Club reviews it.';
}

export function acceptsHandKey(event, hand) {
  if (!hand || event.defaultPrevented || event.isComposing || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  const target = event.target;
  if (!target || !hand.contains(target)) return false;
  if (target.closest('input, select, textarea, a, [contenteditable]:not([contenteditable="false"])')) return false;
  const button = target.closest('button');
  return !button || button.classList.contains('tile');
}

/** Whether play's keys in the hand act on this key, each once however long
 * it is held: Enter and Space throw the marked tile, a digit the tile of its
 * place, r goes to the riichi choices and t wins by self-draw. A repeat of
 * one of these is cancelled, so that a held key throws no second tile. A
 * repeat of any other key is left to the browser, to scroll the page or to
 * move focus with. The arrows move the marker at every repeat, and Escape
 * clears it; both are handled before this is asked. */
export function handKeyActs(key) {
  return key === 'Enter' || key === ' ' || /^[0-9rt]$/i.test(key ?? '');
}

/** Whether a keydown is Enter held down on something Enter presses: a
 * button, a summary, a link or anything with role="button", anywhere in the
 * app and in every mode. Enter presses as it goes down and again at every
 * repeat, and what takes a pressed control's place, or wherever focus lands
 * as it goes, is pressed in turn: a tile or a call of the next decision drawn
 * where the last one was, or New game, whose confirmation then holds the page.
 * So a held key presses nothing twice; App cancels these repeats. Text fields
 * and selects are left alone, which Enter does not press. Space presses a
 * button as it is let go, once however long it is held, so only Enter needs
 * this. */
export function heldPress(event) {
  if (!event?.repeat || event.key !== 'Enter') return false;
  const target = event.target;
  if (!target?.closest || target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')) return false;
  return Boolean(target.closest('button, summary, a[href], [role="button"]'));
}

/** What a key does in the game's review, by name, or null: 'previous' and
 * 'next' for ← and →, 'previous-hand' and 'next-hand' with Shift, 'reveal'
 * for s and 'back' for Esc; and, while the player's `hand` is live for them
 * to pick a move from, 'pick-1' to 'pick-9' for its first nine tiles and
 * 'pick-drawn' for the tile just drawn (0), as in play. Keys are left to
 * whatever else has them as play leaves them: while a dialog is open over the
 * page (`dialogOpen`), with Ctrl, Alt or Meta held, while text is composed,
 * once something has handled them, and in a field. In the live hand ← and →
 * move along it, as in play, rather than step: 'previous-tile' and
 * 'next-tile'. */
export function reviewKey(event, { hand = null, dialogOpen = false } = {}) {
  if (dialogOpen || event.defaultPrevented || event.isComposing || event.ctrlKey || event.altKey || event.metaKey) return null;
  const target = event.target;
  if (target?.closest?.('input, select, textarea, [contenteditable]:not([contenteditable="false"])')) return null;
  // Chrome sends a keydown with no key when a field is filled in for the player.
  const key = event.key ?? '', inHand = Boolean(hand && target && hand.contains(target));
  if (key === 'ArrowLeft' || key === 'ArrowRight') {
    const step = key === 'ArrowLeft' ? 'previous' : 'next';
    if (event.shiftKey) return `${step}-hand`;
    return inHand ? `${step}-tile` : step;
  }
  if (key === 'Escape') return 'back';
  if (key.toLowerCase() === 's') return 'reveal';
  if (!hand || event.shiftKey) return null;
  if (/^[1-9]$/.test(key)) return `pick-${key}`;
  return key === '0' ? 'pick-drawn' : null;
}

/** Whether a keydown is Enter held down on a button of the review's step bar,
 * of Trained AI's reasoning or of the review's own controls, which the
 * browser would press again at every repeat. Enter presses a button as it
 * goes down, and all three hand focus on as a button goes: a reveal to Next,
 * so that held on Show answer or a tile of the live hand it would step past
 * the answer it had just shown; "What would change its mind?" to Cancel and
 * Cancel back to it, so that held on either it would start and stop the
 * search by turns; and Stop to Continue and Continue back to Stop, so that it
 * would stop and start the review by turns. So, as play's held keys throw no
 * tile, a held Enter presses them once. Space presses a button as it is let
 * go, once however long it is held, and the arrows step on as they repeat. */
export function heldEnter(event) {
  return Boolean(event.repeat && event.key === 'Enter' && event.target?.closest?.('[data-steps], [data-insight], [data-review-controls]'));
}

/** A called set as the table lays it out: its tiles, and `turned`, the slot
 * of the tile turned sideways on the side of the player it came from, or -1
 * for a concealed quad, which turns none. The tile actually claimed is moved
 * into that slot; an added kan keeps its pon's. A set without its claimed
 * tile, as the guided table passes its sets, keeps its order and is still
 * turned on the side it came from. */
export function meldLayout(meld) {
  const tiles = [...meld.tiles];
  if (meld.kind === 'concealed-kan') return { tiles, turned: -1 };
  const turned = meld.from === 'left' ? 0 : meld.from === 'across' ? 1 : tiles.length - 1;
  const source = tiles.indexOf(meld.claimed_tile);
  if (source >= 0) {
    tiles.splice(source, 1);
    tiles.splice(turned, 0, meld.claimed_tile);
  }
  return { tiles, turned };
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
