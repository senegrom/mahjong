/** The game's review: every decision the player made in a match, put to the
 * trained network and ranked by how far the player's move departed from its
 * choice. From any hand's result it reviews the game so far, and once the
 * match is over the whole game; it is taken on to each later result as the
 * match goes on, so that a hand reviewed once is never asked about again.
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
import { callTiles, percent, quadOnOffer } from './ui.js';

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

/** Every hand of a match, replayed from its first deal: the table at each of
 * the player's decisions, every player's tiles there, what the player chose,
 * and what happened between; how the hand ended and what it changed for the
 * player; and which saved command closed it. A finished match is replayed to
 * its end. A match waiting at a hand's result is replayed to that hand, which
 * closes at the command that will deal the next, so that a hand closes at the
 * same command before the match goes on and after. Throws when the replay
 * does not end exactly where the match did, or ends in a hand being played or
 * one just dealt. */
export function scanMatch(Game, snapshot) {
  const match = new MatchSession(Game, snapshot.seed, snapshot.opponents);
  try {
    const hands = [];
    let lines = match.engine.advance(), hand = null;
    const begin = () => ({ index: hands.length, name: handName(match.view), counters: match.view.counters, seat: match.view.seats[0].seat, frames: [] });
    // The hand on the table is over; `end` is the saved command that deals
    // the next, or where that command will go.
    const close = end => {
      // The tiles every player held at each of the decisions, in the order
      // of the table's seats, the tile just drawn held apart. This replay's
      // engine answers only once the hand is over, as it is here, since
      // while a hand is played they are what the table hides.
      hand.frames.forEach((frame, index) => { frame.hands = match.engine.review_hands(index); });
      const { outcome } = match.view;
      Object.assign(hand, { end, outcome: outcome?.line ?? '', change: outcome?.changes?.[0] ?? 0 });
      hands.push(hand);
      hand = null;
      lines = [];
    };
    for (const [at, command] of snapshot.commands.entries()) {
      hand ??= begin();
      if (command.type === 'choose') {
        hand.frames.push({ view: match.view, choices: match.choices, lines, command: at,
          played: { kind: command.kind, tile: command.tile ?? null } });
        lines = [];
      } else if (command.type === 'next') close(at);
      match.apply(command);
      if (!match.over) lines = [...lines, ...match.engine.advance()];
    }
    // A match waiting at a hand's result: that hand is over, and closes where
    // the command that deals the next will go. A hand that asked the player
    // nothing, such as one won before the player's first turn, may have had
    // no command of its own to begin it.
    if (!match.over && match.engine.hand_is_over()) {
      hand ??= begin();
      close(snapshot.commands.length);
    }
    // A replay that does not end where the match was saved went another way,
    // and is refused. That is checked first, so that one that went another
    // way and also stopped in a hand gives that reason, the one that is true.
    // One that ends where the match was saved, but in a hand being played or
    // one just dealt, is refused too: the tiles then are what the table
    // hides. A finished match keeps its last hand on the table, over. The
    // page says either reason after "The game could not be replayed for
    // review:".
    if (match.stateKey() !== snapshot.state) throw new Error('The replay did not end where the match did');
    if (hand || !match.engine.hand_is_over()) throw new Error('A hand is still being played');
    return hands;
  } finally { match.dispose(); }
}

/** A replay that moves forward through a match's commands, so a hand's
 * decisions can be read off the engine at the hand's end. The commands grow
 * as the review is taken on to a later result (MatchReview.extend): the old
 * ones begin the new, so a replay plays on from wherever it is. */
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

/** A judged decision's two weights as the whole game's list gives them, under
 * the two moves: the player's move and Trained AI's choice, or where both
 * declared riichi the player's tile given riichi. */
export function weightsSaid({ stage, played, preferred }) {
  return `${stage === 'tile' ? 'Given riichi, your tile' : 'Your move'} ${percent(played)}, Trained AI's choice ${percent(preferred)}.`;
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
/** What each filter's button says, before how many decisions it holds: the
 * verdicts a replayed hand's list gives each decision (decisionInBrief). */
export const FILTER_LABELS = Object.freeze({ flagged: 'Flagged', differed: 'Trained AI differed', every: 'Every decision' });
/** A filter button's title, where it has one. The Flagged button's is the
 * rule a decision is flagged by, kept at hand but out of the way: a pointer
 * shows it as a tooltip, and a screen reader reads it as the button's
 * description, after its name. It is put in the words each decision's
 * weights are given in (weightsSaid), so it reads like the figures it applies
 * to. Like the app's other button titles, it is not a sentence. */
export const FILTER_TITLES = Object.freeze({
  flagged: `Your move under ${Math.round(FLAG_PLAYED_BELOW * 100)}%, Trained AI's choice ${Math.round(FLAG_PREFERRED_FROM * 100)}% or more`,
});

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

/** What a decision is known by across the game: its hand and its number in
 * the hand, from 0, as "3:12". */
export const decisionKey = (hand, index) => `${hand}:${index}`;

/** Whether a filter holds a decision. A decision not yet reviewed (null) is
 * known only to have been made, so only the list of every decision holds it. */
export function listedIn(decision, filter) {
  if (filter === 'every') return true;
  if (!decision) return false;
  return filter === 'flagged' ? Boolean(decision.judgement?.flagged) : differed(decision);
}

/** The places a filter holds, `{ hand, step }`, through the hands in the
 * order they were played and each hand's decisions in the order they were
 * made: what Previous and Next step through. */
export function playlist(hands, filter) {
  return hands.flatMap(hand => hand.frames.flatMap((_, step) =>
    (listedIn(hand.decisions?.[step] ?? null, filter) ? [{ hand: hand.index, step }] : [])));
}

/** The place in `list` before `place` (direction -1) or after it (1), in the
 * order played, or null at either end. The place need not be in the list
 * itself, as a decision the filter does not hold may be shown all the same. */
export function neighbour(list, place, direction) {
  const after = entry => entry.hand - place.hand || entry.step - place.step;
  if (direction > 0) return list.find(entry => after(entry) > 0) ?? null;
  for (let at = list.length - 1; at >= 0; at--) if (after(list[at]) < 0) return list[at];
  return null;
}

/** Where a hand is opened at: its first decision the filter holds, or its
 * first decision if the filter holds none there, as before the hand is
 * reviewed; null for a hand in which the player made no decision. */
export function handEntry(hands, filter, handIndex) {
  const hand = hands[handIndex];
  if (!hand?.frames.length) return null;
  const step = hand.frames.findIndex((_, index) => listedIn(hand.decisions?.[index] ?? null, filter));
  return { hand: hand.index, step: Math.max(0, step) };
}

/** Where a review opens: the newest hand in which the player made a decision,
 * which is the hand just played, or at the end the final hand, at its entry
 * (handEntry); null before any decision. */
export function landing(hands, filter) {
  for (let at = hands.length - 1; at >= 0; at--) if (hands[at].frames.length) return handEntry(hands, filter, at);
  return null;
}

/** What a list says as the player chooses its filter: how many decisions it
 * shows now, of which kind, `scope` such as " in this hand" saying where, and
 * in what order; and, where the list leads with a few and keeps the rest
 * behind a button, that it does. The flagged decisions among those where
 * Trained AI differed are counted too. It is said as a button is pressed,
 * so it names the list without saying that it is shown. */
export function filterSaid(filter, counts, { scope = '', ranked = filter !== 'every', leading = Infinity } = {}) {
  const count = counts[filter], s = count === 1 ? '' : 's';
  if (!count) {
    return { flagged: `None flagged${scope}.`, differed: `Trained AI agreed at every decision${scope}.`,
      every: `No decisions${scope}.` }[filter];
  }
  const shown = {
    flagged: `${count} flagged decision${s}${scope}`,
    differed: `${count} decision${s}${scope} where Trained AI differed${counts.flagged ? `, ${counts.flagged} of them flagged` : ''}`,
    every: count === 1 ? `The one decision${scope}` : `All ${count} decisions${scope}`,
  }[filter];
  const order = count === 1 ? '' : ranked ? ', widest departure first' : ', in the order played';
  return `${shown}${order}.${count > leading ? ` First ${leading} listed.` : ''}`;
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
 * current would be read out again at every hand. It leads with the place, the
 * one thing it must say; focus is on the decision, so it need not say which,
 * and a list that grew to keep it shows all it holds, whichever list it is. */
export function movedNote(rank, everyShown, filter = 'flagged') {
  const why = filter === 'every' ? 'decisions reviewed since came before it' : 'decisions reviewed since departed further';
  return `Now ${ordinal(rank + 1)}: ${why}${everyShown ? ', so the whole list is shown' : ''}.`;
}

/** How many of the player's real choices were the network's own. A forced
 * move, such as the discard after riichi, was no choice, and is left out. */
export function agreement(decisions) {
  const counted = decisions.filter(decision => !decision.forced);
  const matched = counted.filter(decision => decision.agreed).length;
  return { matched, decisions: counted.length, share: counted.length ? matched / counted.length : null };
}

/** The spells in which the hand's own notes ask Trained AI, a decision at a
 * time, of the worker the review of the game so far asks too (App). The
 * worker answers its requests in turn, so a note asked while the review goes
 * on would wait behind one of its batches, and the next note behind another:
 * the review waits at its gate for every spell to end instead. The notes
 * begin a spell as they begin to ask, and end it however they stop, whether
 * answered, failed, out of time, closed or asked again. `begin()` answers the
 * function that ends the spell it began, which ends it once however often it
 * is called: a spell counted out twice would open the gate on another still
 * under way. `gate()` is what a review's gate answers: null while no spell
 * is under way, or a promise that settles once every spell begun has ended. */
export class Asking {
  constructor() {
    this.spells = 0;
    this.ended = null;
    this.open = null;
  }

  begin() {
    if (!this.spells++) this.ended = new Promise(resolve => { this.open = resolve; });
    let over = false;
    return () => {
      if (over) return;
      over = true;
      if (--this.spells) return;
      // The gate answers null again before the review waiting at it goes on,
      // so a review that looks again finds it open, unless another spell has
      // begun meanwhile.
      const open = this.open;
      this.ended = this.open = null;
      open();
    };
  }

  gate() { return this.ended; }
}

/** The review of one match: of the game so far while it is played, and of the
 * whole game once it is over. It is taken on to each later result (`extend`),
 * keeping every hand it has reviewed. It owns its replay and its requests,
 * and reviews the hands queued, in order, a forward pass at a time; a slow
 * device holds after each hand unless the player has asked for every hand.
 * `analyzeBatch(positions, signal)` asks the network; `onChange` hears every
 * change, and `onComplete` hears once that every hand of the finished match
 * is reviewed, whether or not anyone is looking. `gate()` answers null while
 * the review may ask the network, or a promise that settles once it may; the
 * review waits there before each forward pass (passGate), so that the hand's
 * own notes, which ask the same worker a decision at a time, go first
 * (Asking). */
export class MatchReview {
  constructor(Game, snapshot, { analyzeBatch, gate = () => null, onChange = () => {}, onComplete = () => {}, now = () => performance.now() } = {}) {
    this.Game = Game;
    this.snapshot = snapshot;
    this.analyzeBatch = analyzeBatch;
    this.gate = gate;
    this.onChange = onChange;
    this.onComplete = onComplete;
    this.now = now;
    this.hands = scanMatch(Game, snapshot).map(hand => ({ ...hand, status: 'waiting', decisions: null }));
    this.queue = [];
    this.busy = false;
    this.failure = '';
    this.held = false;
    // Whether the player asked for every hand on a slow device.
    this.patient = false;
    this.closed = false;
    // Whether anything has been asked for since the review was last taken on
    // to new hands or to the match's end, and whether the finished match's
    // review has been reported complete.
    this.requested = false;
    this.recorded = false;
    this.positionMs = null;
    this.asked = 0;
    this.replay = null;
    this.abort = new AbortController();
    this.current = null;
  }

  /** What one match is known by in the trend: its seed and its length, which
   * is the finished match's once it is over, the only length the trend
   * keeps. */
  get id() { return `${this.snapshot.seed}:${this.snapshot.commands.length}`; }
  /** Whether the match is over. A finished match's last hand closed at its
   * saved 'next', the command that ended the game; a hand waiting at its
   * result closes where that command will go, past every command saved. */
  get over() {
    const last = this.hands.at(-1);
    return Boolean(last) && last.end < this.snapshot.commands.length;
  }
  /** The order the hands are asked about when none are named. A finished
   * match is asked from its first deal. A match under way is asked first
   * about the hand just played, whose result the player came from, then
   * about the rest from the first deal. */
  get order() {
    const indices = this.hands.map(hand => hand.index);
    return this.over ? indices : [...indices.slice(-1), ...indices.slice(0, -1)];
  }
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

  /** Tells onComplete, once, that every hand of the finished match is
   * reviewed. The game so far is not yet a game for the trend, which
   * compares whole games. */
  report() {
    if (this.closed || !this.over || !this.complete || this.recorded) return;
    this.recorded = true;
    this.onComplete(this);
  }

  /** Reviews these hands next, in this order, then whatever was queued
   * already; by default every hand not yet reviewed, in `order`. `patient`
   * is the player's word that a slow device should go on to the end. */
  review(indices = this.order, { patient = false } = {}) {
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

  /** Takes the review on to the same match saved later, at a later hand's
   * result or finished. Every hand already scanned is kept as it is, with
   * whatever was reviewed of it, since a hand closes at the same command
   * however far the match has gone; only the hands played since are added,
   * waiting. Throws, and changes nothing, for another match, or for one that
   * cannot be replayed. */
  extend(snapshot) {
    if (this.closed) return;
    const known = this.snapshot.commands, same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    if (snapshot.seed !== this.snapshot.seed || !same(snapshot.opponents, this.snapshot.opponents) || snapshot.commands.length < known.length
      || known.some((command, at) => !same(command, snapshot.commands[at]))) throw new Error('This is not the match under review');
    if (snapshot.commands.length === known.length) return;
    const scanned = scanMatch(this.Game, snapshot), wasOver = this.over;
    this.snapshot = snapshot;
    // The old commands begin the new, so a replay, even one a hand is being
    // reviewed on, plays on from where it is.
    if (this.replay) this.replay.commands = snapshot.commands;
    const added = scanned.slice(this.hands.length).map(hand => ({ ...hand, status: 'waiting', decisions: null }));
    this.hands.push(...added);
    // New hands, or the match's end, are not yet asked for: opening the
    // review asks for them, the hand just played first, and a slow device
    // holds again after it.
    if (added.length || this.over !== wasOver) {
      this.requested = false;
      this.patient = false;
    }
    // A game whose every hand was reviewed as it was played is whole the
    // moment it ends.
    this.report();
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
        this.report();
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
    // Said after the headline's "Review stopped:", which names the review.
    if (notes.length !== hand.frames.length) throw new Error('This hand could not be replayed');
    const rows = await reviewWithStrong(engine, notes, null, {
      signal, skipForced: true,
      batch: { size: BATCH, analyze: async (positions, signal) => {
        await this.passGate(signal);
        return this.timed(positions, signal);
      } },
      onProgress: done => { hand.done = done; this.notify(); },
    });
    if (signal.aborted) throw new DOMException('Review changed', 'AbortError');
    hand.decisions = rows.map((row, index) => {
      const decision = { ...row, hand: hand.index, index, frame: hand.frames[index] };
      decision.judgement = judgement(decision);
      // The review's numbers for the moves a player compares: what they
      // played, what the network chose and, where it is flagged, every turn
      // move they could have made instead. A response to a discard has none.
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

  /** Waits, before a forward pass, while the gate is shut: while the hand's
   * own notes ask Trained AI. Nothing is given up meanwhile. The review goes
   * on exactly where it was, at the next batch of the same hand, with the
   * same hands queued and the same word from the player: that they asked for
   * it, and whether a slow device should go on to the end or hold after the
   * hand for them. Each time the gate opens it is asked again, since the
   * notes may have begun to ask again by the time the review goes on. A
   * review stopped or closed while it waits stops waiting at once, as it
   * stops waiting for an answer, and a gate that fails stops the review as a
   * failed answer would, saying why, rather than leaving it waiting. */
  async passGate(signal) {
    for (let shut = this.gate(); shut; shut = this.gate()) {
      await new Promise((resolve, reject) => {
        const stop = () => reject(new DOMException('Review changed', 'AbortError'));
        if (signal?.aborted) return stop();
        signal?.addEventListener('abort', stop, { once: true });
        shut.then(resolve, reject).finally(() => signal?.removeEventListener('abort', stop));
      });
    }
  }

  /** Asks one batch and keeps the time it took a position. The first batch
   * of a review can include loading the network, so it is not counted. Time
   * spent at the gate is the notes' rather than the network's, so it is not
   * counted either: the clock starts once the gate is passed. */
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
 * were left to draw, and who had declared riichi. It is said at every step,
 * so it is put in the table's own words: "45 tiles left", as its centre gives
 * the wall, and "declared riichi", as a seat announces it. */
export function tableSaid(view) {
  const left = `${view.wall} tile${view.wall === 1 ? '' : 's'} left`;
  const move = view.phase === 'call' && view.pending_discard
    ? `${SEAT_NAMES[view.pending_from] ?? 'An opponent'} ${quadOnOffer(view) ? 'declared a kan of' : 'discarded'} the ${tileWords(view.pending_discard)}`
    : 'Your turn';
  const riichi = view.seats.filter(seat => seat.riichi).map(seat => (seat === view.seats[0] ? 'you' : SEAT_NAMES[seat.seat]));
  const declared = riichi.length ? ` ${listed(riichi).replace(/^./, letter => letter.toUpperCase())} declared riichi.` : '';
  return `${move}, ${left}.${declared}`;
}

/** Each move as the review's short lines name it, in a word. */
const MOVE_NAMES = Object.freeze({ discard: 'Discard', riichi: 'Riichi', tsumo: 'Tsumo', ron: 'Ron', pon: 'Pon', chii: 'Chii',
  kan: 'Kan', 'concealed-kan': 'Kan', 'extended-kan': 'Kan', pass: 'Pass' });

/** A move at a decision in the engine's words, "discards the red dragon", as
 * the decision's own choices name it. */
const moveWords = (frame, move) => frame.choices.find(choice => sameMove(choice, move.kind, move.tile))?.label
  ?? MOVE_NAMES[move.kind] ?? move.kind;

/** A decision as a line of the review gives it: the player's move, `you`,
 * and Trained AI's where it chose another, `ai`, each a word and its tiles
 * (moveToken); the tile on offer where the player answered another's
 * discard, `offered`; the verdict in words, and as flags whether the
 * decision is `flagged`, whether Trained AI `agreed`, whether the move was
 * `forced`, the only one, and whether it is `reviewed` yet. `said` is the
 * line as a screen reader hears it, the moves in the engine's words: "You:
 * discards the red dragon. Trained AI: discards the 8 bamboo. Flagged." */
export function decisionInBrief(frame, decision) {
  const { kind, tile } = frame.played, view = frame.view;
  const flagged = Boolean(decision?.judgement?.flagged), forced = Boolean(decision?.forced), agreed = Boolean(decision?.agreed);
  const verdict = !decision ? '' : flagged ? 'Flagged' : forced ? 'Only move' : agreed ? 'Trained AI agreed' : 'Trained AI differed';
  const ai = decision && !forced && !agreed ? moveToken(decision.advised_kind, decision.advised_tile ?? null) : null;
  const offered = view.phase === 'call' ? view.pending_discard ?? null : null;
  const answer = !decision ? 'Not reviewed yet.' : forced ? 'Only move.' : agreed ? 'Trained AI agreed.' : `Trained AI: ${decision.advised}.`;
  const said = `${offered ? `On the ${tileWords(offered)}. ` : ''}You: ${moveWords(frame, frame.played)}. ${answer}${flagged ? ' Flagged.' : ''}`;
  return { you: moveToken(kind, tile), ai, offered, verdict, flagged, agreed, forced, reviewed: Boolean(decision), said };
}

/** A move as the review's short lines give it: a word, and the tiles it
 * names. A discard is its tile alone; a riichi and a quad declared from the
 * hand, concealed or extended, are the word and their tile; a chii is the
 * word and its run; any other move, a pass, a pon, a quad called, a win, is
 * its word. */
export function moveToken(kind, tile) {
  if (kind === 'discard') return { word: '', tiles: tile ? [tile] : [] };
  const word = MOVE_NAMES[kind] ?? kind;
  if (kind === 'chii') return { word, tiles: tile ? callTiles({ kind, tile }) : [] };
  return { word, tiles: tile && ['riichi', 'concealed-kan', 'extended-kan'].includes(kind) ? [tile] : [] };
}

/** A move's weight as the review states it beside Trained AI's choice, and
 * whether it is `given` riichi. Where the player and Trained AI both declared
 * riichi, a decision is judged on the tile (`judgement`), so a riichi there
 * is weighed by its tile given the declaration; otherwise every move by its
 * own weight, a riichi by the declaration's. The weight is null where
 * Trained AI gave the move none. */
export function moveWeight(decision, move) {
  const given = move.kind === 'riichi' && decision.played_kind === 'riichi' && decision.advised_kind === 'riichi';
  const choice = decision.weighted?.find(entry => sameMove(entry, move.kind, move.tile));
  const weight = given ? choice?.conditionalWeight : choice?.weight;
  return { weight: Number.isFinite(weight) ? weight : null, given };
}

/** What a step of the review says as it is reached, in the one line a screen
 * reader hears: where it is, the table (tableSaid), the player's move and
 * Trained AI's, each with its weight, a move the player picked there, and
 * whether the decision is flagged. In place of Trained AI's half it says
 * that Trained AI agreed, that the move was the only one, that its move is
 * `hidden`, or that the hand is not reviewed yet. A hidden move hides the
 * player's weight too, which would give the answer away, but not the flag:
 * how far the player went wrong is not what was better. */
export function stepSaid({ hand, step, decision = null, hidden = false, pick = null }) {
  const frame = hand.frames[step], played = frame.played;
  const weighed = move => {
    if (!decision || decision.forced || hidden) return '';
    const { weight, given } = moveWeight(decision, move);
    return weight === null ? '' : `, ${percent(weight)}${given ? ' given riichi' : ''}`;
  };
  const you = `You: ${moveWords(frame, played)}${weighed(played)}.`;
  const advised = decision && { kind: decision.advised_kind, tile: decision.advised_tile ?? null };
  const ai = !decision ? 'Trained AI: not reviewed yet.' : decision.forced ? 'Only move.' : hidden ? "Trained AI's move hidden."
    : decision.agreed ? 'Trained AI agreed.' : `Trained AI: ${decision.advised}${weighed(advised)}.`;
  const now = pick && decision && !decision.forced && !hidden ? ` Now: ${moveWords(frame, pick)}${weighed(pick)}.` : '';
  return `${hand.name}, decision ${step + 1} of ${hand.frames.length}. ${tableSaid(frame.view)} ${you} ${ai}${now}`
    + `${decision?.judgement?.flagged ? ' Flagged.' : ''}`;
}
