/** The whole-game review: every decision the player made in a finished match,
 * put to the trained network and ranked by how far the player's move departed
 * from its choice.
 *
 * Nothing about the review is saved with the match. It is rebuilt from the
 * saved match's own commands, replayed from the first deal on a separate
 * engine, so the saved match and its format are never touched. The engine
 * keeps a hand's decisions only until the next deal, so the replay stops at
 * the end of each hand to ask about that hand's decisions, which is what the
 * per-hand review asks (review-policy.js), here in batches.
 */
import { MatchSession } from './session.js';
import { reviewWithStrong } from './review-policy.js';
import { SEAT_NAMES, tileWords } from './tiles.js';
import { quadOnOffer } from './ui.js';

/** A decision is flagged when the player's move had under this weight... */
export const FLAG_PLAYED_BELOW = 0.05;
/** ...while the network's own choice had at least this. A network that is
 * not decisive is not asked to overrule anyone: near-ties never flag. */
export const FLAG_PREFERRED_FROM = 0.6;
/** How many of the flagged decisions the review leads with. */
export const LEADING = 5;
/** Positions asked in one forward pass. Natively eight cost about half as
 * much a position as one at a time (35 ms against 66); in the browser's
 * single-threaded runtime they cost about the same (157 ms against 163, in
 * Chromium on 14 October 2026), with identical answers, so a batch saves
 * messages there and costs nothing. */
export const BATCH = 8;
/** A device slower than this a position reviews a hand at a time unless the
 * player asks for the rest of the game: at half a second a position, a game
 * of 150 to 200 decisions takes 75 to 100 seconds. */
export const SLOW_MS = 500;

/** "East 1" or "South 3", and "East 1 · 1 honba" for a hand that repeats one:
 * a round and number come back only with another honba, so every hand of a
 * game has a name of its own. */
export const handName = view => `${SEAT_NAMES[view.round] ?? view.round} ${view.kyoku}${view.counters ? ` · ${view.counters} honba` : ''}`;

/** Every hand of a finished match, replayed from its first deal: the table at
 * each of the player's decisions, every player's tiles there, what the player
 * chose, and what happened between; how the hand ended and what it changed
 * for the player; and which saved command closed it. Throws when the replay
 * does not end exactly where the match did. */
export function scanMatch(Game, snapshot) {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  try {
    const hands = [];
    let lines = match.engine.advance(), hand = null;
    for (const [at, command] of snapshot.commands.entries()) {
      hand ??= { index: hands.length, name: handName(match.view), counters: match.view.counters, seat: match.view.seats[0].seat, frames: [] };
      if (command.type === 'choose') {
        hand.frames.push({ view: match.view, choices: match.choices, lines, command: at,
          played: { kind: command.kind, tile: command.tile ?? null } });
        lines = [];
      } else if (command.type === 'next') {
        // The tiles every player held at each of the decisions, in the order
        // of the table's seats, the tile just drawn held apart. This replay's
        // engine answers only once the hand is over, as it is here, since
        // while a hand is played they are what the table hides.
        hand.frames.forEach((frame, index) => { frame.hands = match.engine.review_hands(index); });
        const { outcome } = match.view;
        Object.assign(hand, { end: at, outcome: outcome?.line ?? '', change: outcome?.changes?.[0] ?? 0 });
        hands.push(hand);
        hand = null;
        lines = [];
      }
      match.apply(command);
      if (!match.over) lines = [...lines, ...match.engine.advance()];
    }
    if (!match.over || hand || match.stateKey() !== snapshot.state) throw new Error('This match could not be replayed for review');
    return hands;
  } finally { match.dispose(); }
}

/** A replay that moves forward through a match's commands, so a hand's
 * decisions can be read off the engine at the hand's end. */
class Replay {
  constructor(Game, snapshot) {
    this.commands = snapshot.commands;
    this.match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
    this.match.advance(false);
    this.at = 0;
  }
  /** Plays on to the saved command `end`, unapplied; false if it is behind. */
  seek(end) {
    if (end < this.at) return false;
    for (; this.at < end; this.at++) {
      this.match.apply(this.commands[this.at]);
      this.match.advance(false);
    }
    return true;
  }
  get engine() { return this.match.engine; }
  dispose() { this.match.dispose(); }
}

/** How far a reviewed move departed from the network, as the review shows it.
 * Riichi is two questions, as everywhere the network advises: whether to
 * declare, then, given the declaration, which tile. When the player and the
 * network both declared, they are judged on the tile; otherwise on the move.
 * Null where there was no choice, or nothing to compare. */
export function judgement(row) {
  if (row.agreed || row.forced) return null;
  const tile = row.played_kind === 'riichi' && row.advised_kind === 'riichi';
  const played = tile ? row.played_conditional_weight : row.played_weight;
  const preferred = tile ? row.preferred_conditional_weight : row.preferred_weight;
  if (!Number.isFinite(played) || !Number.isFinite(preferred)) return null;
  return { stage: tile ? 'tile' : 'move', played, preferred, gap: preferred - played,
    flagged: played < FLAG_PLAYED_BELOW && preferred >= FLAG_PREFERRED_FROM };
}

/** The flagged decisions of the reviewed hands, the widest gap first; equal
 * gaps keep the order they were played in. */
export function rankDecisions(decisions) {
  return decisions.map((decision, order) => ({ decision, order, gap: decision.judgement?.flagged ? decision.judgement.gap : null }))
    .filter(entry => entry.gap !== null)
    .sort((a, b) => b.gap - a.gap || a.order - b.order)
    .map(entry => entry.decision);
}

/** The lists of the player's decisions a review can show, by the names its
 * filter gives them: the flagged decisions, every decision where Trained AI
 * chose differently (the flagged ones among them), and every decision. */
export const FILTERS = Object.freeze(['flagged', 'differed', 'every']);
/** What each filter's button says, before how many decisions it holds. */
export const FILTER_LABELS = Object.freeze({ flagged: 'Flagged', differed: 'Trained AI chose differently', every: 'Every decision' });

/** Whether Trained AI chose differently: a real choice, made otherwise. A
 * forced move was no choice, so it never differs. */
export const differed = decision => !decision.agreed && !decision.forced;

/** The decisions a filter holds, in the order the whole-game review lists
 * them: the flagged ones, and those where Trained AI chose differently, the
 * widest gap first as `rankDecisions` ranks them, a different move whose
 * weights could not be compared after the rest; every decision in the order
 * it was made. Equal gaps keep the order they were played in. */
export function filterDecisions(decisions, filter) {
  if (filter === 'flagged') return rankDecisions(decisions);
  if (filter === 'every') return [...decisions];
  const gap = decision => decision.judgement?.gap ?? -Infinity;
  return decisions.map((decision, order) => ({ decision, order }))
    .filter(entry => differed(entry.decision))
    // Two moves that could not be compared give NaN, and keep their order.
    .sort((a, b) => (gap(b.decision) - gap(a.decision)) || a.order - b.order)
    .map(entry => entry.decision);
}

/** How many decisions each filter holds. */
export function filterCounts(decisions) {
  return { flagged: rankDecisions(decisions).length, differed: decisions.filter(differed).length, every: decisions.length };
}

/** What a list says as the player chooses its filter: how many decisions it
 * shows now, of which kind, `scope` such as " in this hand" saying where, and
 * in what order; and, where the list leads with a few and keeps the rest
 * behind a button, that it does. The flagged decisions among those where
 * Trained AI chose differently are counted too. */
export function filterSaid(filter, counts, { scope = '', ranked = filter !== 'every', leading = Infinity } = {}) {
  const count = counts[filter], s = count === 1 ? '' : 's';
  if (!count) {
    return { flagged: `No decision${scope} is flagged.`, differed: `Trained AI chose the same as you at every decision${scope}.`,
      every: `No decisions${scope}.` }[filter];
  }
  const shown = {
    flagged: `${count} flagged decision${s}${scope}`,
    differed: `${count} decision${s}${scope} where Trained AI chose differently${counts.flagged ? `, ${counts.flagged} of them flagged` : ''}`,
    every: count === 1 ? `the one decision${scope}` : `all ${count} decisions${scope}`,
  }[filter];
  const order = count === 1 ? '' : ranked ? ', the widest departure first' : ', in the order you made them';
  return `Showing ${shown}${order}.${count > leading ? ` The first ${leading} are listed.` : ''}`;
}

/** A place in the ranking as it is said: "1st", "2nd", "3rd", "11th", "22nd". */
export function ordinal(number) {
  const suffix = number % 100 >= 11 && number % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][number % 10] ?? 'th';
  return `${number}${suffix}`;
}

/** What the whole-game list says when the decision a player is on, or came
 * back to, has been put lower by the hands reviewed since: where it is now,
 * and whether every decision the filter holds had to be shown to keep it. A
 * ranked list moves a decision down when later ones departed further; the
 * list of every decision, when the decisions of an earlier hand arrive after
 * it. Not how many the list holds: that grows with every hand reviewed while
 * the place stays, so a count said once goes stale, and a status that kept it
 * current would be read out again at every hand. */
export function movedNote(rank, everyShown, filter = 'flagged') {
  const why = filter === 'every' ? 'Decisions reviewed since came before this one' : 'Decisions reviewed since departed further from Trained AI';
  const all = { flagged: 'every flagged decision', differed: 'every decision where Trained AI chose differently', every: 'every decision' }[filter];
  return `${why}: this one is now ${ordinal(rank + 1)}${everyShown ? `, and ${all} is shown` : ''}.`;
}

/** How many of the player's real choices were the network's own. A forced
 * move, such as the discard after riichi, was no choice, and is left out. */
export function agreement(decisions) {
  const counted = decisions.filter(decision => !decision.forced);
  const matched = counted.filter(decision => decision.agreed).length;
  return { matched, decisions: counted.length, share: counted.length ? matched / counted.length : null };
}

/** The review of one finished match. It owns its replay and its requests, and
 * reviews the hands queued, in order, a forward pass at a time; a slow device
 * holds after each hand unless the player has asked for the whole game.
 * `analyzeBatch(positions, signal)` asks the network; `onChange` hears every
 * change, and `onComplete` hears once that every hand is reviewed, whether or
 * not anyone is looking. */
export class MatchReview {
  constructor(Game, snapshot, { analyzeBatch, onChange = () => {}, onComplete = () => {}, now = () => performance.now() } = {}) {
    this.Game = Game;
    this.snapshot = snapshot;
    this.analyzeBatch = analyzeBatch;
    this.onChange = onChange;
    this.onComplete = onComplete;
    this.now = now;
    this.hands = scanMatch(Game, snapshot).map(hand => ({ ...hand, status: 'waiting', decisions: null }));
    this.queue = [];
    this.busy = false;
    this.failure = '';
    this.held = false;
    // Whether the player asked for the whole game on a slow device.
    this.patient = false;
    this.closed = false;
    // Whether anything was ever asked for, and whether the finished review
    // has been reported complete.
    this.requested = false;
    this.recorded = false;
    this.positionMs = null;
    this.asked = 0;
    this.replay = null;
    this.abort = new AbortController();
    this.current = null;
  }

  /** What one match is known by in the trend: its seed and its length. */
  get id() { return `${this.snapshot.seed}:${this.snapshot.commands.length}`; }
  get decisions() { return this.hands.flatMap(hand => hand.decisions ?? []); }
  get reviewedHands() { return this.hands.filter(hand => hand.status === 'done').length; }
  get complete() { return this.hands.every(hand => hand.status === 'done'); }
  /** Positions still to ask about: the decisions of the hands not yet done. */
  get remaining() { return this.hands.filter(hand => hand.status !== 'done').reduce((sum, hand) => sum + hand.frames.length, 0); }
  /** How long the rest would take at the speed seen so far, in seconds. */
  get estimate() { return this.positionMs == null ? null : Math.ceil(this.remaining * this.positionMs / 1000); }
  get slow() { return this.positionMs != null && this.positionMs > SLOW_MS; }
  /** What the trend keeps of a finished review. */
  get summary() {
    const { matched, decisions } = agreement(this.decisions);
    return { id: this.id, matched, decisions, flagged: rankDecisions(this.decisions).length };
  }

  notify() { if (!this.closed) this.onChange(this); }

  /** Reviews these hands next, in this order, then whatever was queued
   * already; every hand not yet reviewed by default. `patient` is the
   * player's word that a slow device should go on to the end. */
  review(indices = this.hands.map(hand => hand.index), { patient = false } = {}) {
    if (this.closed) return;
    this.requested = true;
    const wanted = indices.filter(index => this.hands[index] && this.hands[index].status !== 'done' && index !== this.current);
    this.queue = [...wanted, ...this.queue.filter(index => !wanted.includes(index))];
    this.patient ||= patient;
    this.held = false;
    this.failure = '';
    void this.pump();
    this.notify();
  }

  /** Goes on after a hold: the next hand, or with `patient` the rest. */
  resume({ patient = false } = {}) { this.review([], { patient }); }

  /** Stops now. The hand being reviewed goes back to waiting; the hands
   * already reviewed stay reviewed. */
  stop() {
    this.queue = [];
    this.held = false;
    if (this.busy) {
      this.abort.abort();
      this.abort = new AbortController();
    }
    this.notify();
  }

  async pump() {
    if (this.busy || this.closed) return;
    this.busy = true;
    this.notify();
    try {
      while (this.queue.length && !this.closed && !this.held) {
        const hand = this.hands[this.queue.shift()];
        this.current = hand.index;
        try {
          await this.reviewHand(hand);
        } catch (error) {
          if (this.closed) return;
          hand.status = 'waiting';
          // Stopped: whatever the player has queued since still goes on.
          if (error?.name === 'AbortError') continue;
          this.failure = error?.message ?? String(error);
          this.queue = [];
          break;
        } finally {
          this.current = null;
        }
        if (this.complete && !this.recorded) {
          this.recorded = true;
          this.onComplete(this);
        }
        // A slow device holds after each hand: the player chooses whether to
        // go on, one hand or the rest, knowing how long it would take.
        if (this.slow && !this.patient && this.queue.length) this.held = true;
        this.notify();
      }
    } finally {
      this.busy = false;
      this.notify();
    }
  }

  async reviewHand(hand) {
    hand.status = 'reviewing';
    hand.done = 0;
    this.notify();
    if (!this.replay?.seek(hand.end)) {
      this.replay?.dispose();
      this.replay = new Replay(this.Game, this.snapshot);
      this.replay.seek(hand.end);
    }
    const engine = this.replay.engine, signal = this.abort.signal;
    const notes = engine.review();
    if (notes.length !== hand.frames.length) throw new Error('This hand could not be replayed for review');
    const rows = await reviewWithStrong(engine, notes, null, {
      signal, skipForced: true,
      batch: { size: BATCH, analyze: (positions, signal) => this.timed(positions, signal) },
      onProgress: done => { hand.done = done; this.notify(); },
    });
    if (signal.aborted) throw new DOMException('Review changed', 'AbortError');
    hand.decisions = rows.map((row, index) => {
      const decision = { ...row, hand: hand.index, index, frame: hand.frames[index] };
      decision.judgement = judgement(decision);
      // The review's numbers for the moves a player compares: what they
      // played, what the network chose and, where it is flagged, every turn
      // move they could choose again. A response to a discard has none.
      if (!decision.agreed && !decision.forced && !decision.call_tile) {
        const measure = choice => engine.review_measure(index, choice.kind, choice.tile ?? undefined);
        const moves = decision.judgement?.flagged ? decision.weighted
          : decision.weighted.filter(choice => sameMove(choice, decision.played_kind, decision.played_tile)
            || sameMove(choice, decision.advised_kind, decision.advised_tile));
        decision.measures = moves.map(choice => ({ kind: choice.kind, tile: choice.tile ?? null, ...measure(choice) }));
      }
      return decision;
    });
    hand.status = 'done';
  }

  /** Asks one batch and keeps the time it took a position. The first batch
   * of a review can include loading the network, so it is not counted. */
  async timed(positions, signal) {
    const started = this.now();
    const answers = await this.analyzeBatch(positions, signal);
    const each = (this.now() - started) / positions.length;
    if (this.asked++ > 0) this.positionMs = this.positionMs == null ? each : (this.positionMs * 3 + each) / 4;
    return answers;
  }

  dispose() {
    if (this.closed) return;
    this.closed = true;
    this.queue = [];
    this.abort.abort();
    this.replay?.dispose();
    this.replay = null;
  }
}

/** Whether a choice is the move of this kind and tile. */
export function sameMove(choice, kind, tile) {
  return choice.kind === kind && (choice.tile ?? null) === (tile ?? null);
}

/** What each seat was doing at one of the player's decisions, in the order
 * of the view's seats, as a replay's table says it beside the gold edge of
 * the seat whose turn it was: the player to play, or to answer the tile
 * another seat had just let go, and that seat. */
export function seatNotes(view) {
  const notes = view.seats.map(() => '');
  if (view.phase === 'act') {
    const turn = view.seats.findIndex(seat => seat.turn);
    if (turn >= 0) notes[turn] = 'To play';
  } else if (view.phase === 'call') {
    notes[0] = 'To answer';
    const from = view.seats.findIndex(seat => seat.seat === view.pending_from);
    if (from > 0) notes[from] = quadOnOffer(view) ? 'Declared a kan' : 'Discarded';
  }
  return notes;
}

/** "South", "you and South", "South, West and North". */
const listed = names => names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`;

/** The table at a decision in a sentence or two, which a replay says as its
 * step changes, beside the step itself: whose move it was, how many tiles
 * were left to draw, and who had declared riichi. */
export function tableSaid(view) {
  const left = `${view.wall} tile${view.wall === 1 ? '' : 's'} left in the wall`;
  const move = view.phase === 'call' && view.pending_discard
    ? `${SEAT_NAMES[view.pending_from] ?? 'An opponent'} ${quadOnOffer(view) ? 'declared a kan of' : 'discarded'} the ${tileWords(view.pending_discard)}`
    : 'Your turn';
  const riichi = view.seats.filter(seat => seat.riichi).map(seat => (seat === view.seats[0] ? 'you' : SEAT_NAMES[seat.seat]));
  const declared = riichi.length ? ` ${listed(riichi).replace(/^./, letter => letter.toUpperCase())} ${riichi.length === 1 && riichi[0] !== 'you' ? 'has' : 'have'} declared riichi.` : '';
  return `${move}, ${left}.${declared}`;
}

/** Each move as a hand's list of decisions names it, in a word. */
const MOVE_NAMES = Object.freeze({ discard: 'Discard', riichi: 'Riichi', tsumo: 'Tsumo', ron: 'Ron', pon: 'Pon', chii: 'Chii',
  kan: 'Kan', 'concealed-kan': 'Kan', 'extended-kan': 'Kan', pass: 'Pass' });

/** A decision as a hand's list of decisions gives it: the move in a word,
 * the tile it concerns (the tile thrown or declared, the tile a self-drawn
 * win was made on, or the discard an answer was to), and whether Trained AI
 * agreed, a flagged decision saying that it was flagged. */
export function decisionInBrief(frame, decision) {
  const { kind, tile } = frame.played, view = frame.view;
  const verdict = !decision ? '' : decision.judgement?.flagged ? 'Flagged' : decision.forced ? 'Only move'
    : decision.agreed ? 'Trained AI agreed' : 'Trained AI differed';
  const concerned = view.phase === 'call' ? view.pending_discard : tile ?? (kind === 'tsumo' ? view.seats[0]?.drawn ?? null : null);
  return { move: MOVE_NAMES[kind] ?? kind, tile: concerned ?? null, verdict };
}
