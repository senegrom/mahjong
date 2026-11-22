/** A match owns its engine, AI work and replay log. No async continuation uses
 * a replaceable global Game. Saves replay legal commands on the real engine;
 * they never deserialize arbitrary internal Rust state or rerun neural choices.
 */
import { normalizeOpponents, opponentPreset, DEFAULT_OPPONENT, OPPONENT_TYPES, OPPONENT_POSITIONS } from './opponents.js';
import { normalizeTileFace } from './tile-faces.js';
import { NETWORKS } from './model-manifest.js';
import { opponentTable, trainedMove as decide } from './trained-move.js';

export const SAVE_KEY = 'riichi.match.v2';
export const SETTINGS_KEY = 'riichi.settings.v1';
const VERSION = 1;
// The settings record's own version: 2 since the site played generation 810,
// 3 since it played generation 1110, 4 since it played generation 1410, 5
// since it played generation 2610 and 6 since it played generation 3000, see
// readSettings. Saved matches keep VERSION, so a match saved before any of
// them still resumes.
const SETTINGS_VERSION = 6;
const MAX_COMMANDS = 20000;
const MAX_SAVE_BYTES = 2_000_000;
const PLAYER_ACTIONS = ['discard', 'riichi', 'tsumo', 'concealed-kan', 'extended-kan', 'ron', 'pon', 'kan', 'chii', 'pass'];

function require(condition, message) {
  if (!condition) throw new Error(message);
}

/** The review adviser a player starts with: the trained network. */
export const DEFAULT_REVIEW_ADVISER = 'strong';
/** The filters a game's review steps through its decisions by, as
 * game-review.js names them: the flagged decisions, those where Trained AI
 * differed, and every decision. Listed here again, since game-review.js
 * imports this module and cannot be imported by it. */
const REVIEW_FILTERS = ['flagged', 'differed', 'every'];

/** Trained is the default opponent and review adviser. A record from before
 * the site played generation 810 (version 1) keeps its hints, keys and tile
 * faces but not who plays and who reviews: when 810 arrived every player was
 * put in front of it once, at the owner's word, Club and Beginner included,
 * since the page writes the whole record on a first visit and a stored Club
 * cannot be told from one that was only an old default. From version 2 a
 * saved choice is kept as it stands, Club included. A setting the record does
 * not hold takes today's default.
 *
 * trainedNetwork is the version of the trained network the player picked. It
 * is written only once they pick one, so a player who never has plays the
 * default, the network published last, and moves on with it when a newer one
 * is published. Without a pick, and for a version this build does not list,
 * it is left out, and the page plays the default (trained-network.js). A
 * record from before the site played generation 1110 (version 2), 1410
 * (version 3), 2610 (version 4) or 3000 (version 5) keeps every other choice
 * but not its pick: when 1110 arrived every player was put in front of it
 * once, at the owner's word, whatever version they had picked before, and the
 * releases of 1410, 2610 and 3000 followed that example. A pick made since
 * writes a version 6 record, and from version 6 a pick is kept as it stands.
 *
 * reviewHideMove and reviewHideTiles say whether Trained AI's reviews keep
 * its move back, and the other players' tiles, until the player asks to see
 * them. Both start off, and a record from before they were offered holds
 * neither, so it reads them as off and the record needs no new version.
 * reviewFilter is which decisions the game's review steps through, the
 * flagged ones unless the player chose another filter, which is kept as they
 * left it; a record without one, or with one the review does not offer,
 * reads the flagged ones, so it needs no new version either. */
export function readSettings(storage, touch = false) {
  const defaults = { difficulty: DEFAULT_OPPONENT, hints: true, confirmDiscards: touch, shortcuts: true, tileFace: 'classic', reviewAdviser: DEFAULT_REVIEW_ADVISER,
    reviewHideMove: false, reviewHideTiles: false, reviewFilter: 'flagged' };
  try {
    const value = JSON.parse(storage?.getItem(SETTINGS_KEY));
    if (![1, 2, 3, 4, 5, SETTINGS_VERSION].includes(value?.version)) return defaults;
    defaults.tileFace = normalizeTileFace(value.tileFace);
    for (const key of ['hints', 'confirmDiscards', 'shortcuts', 'reviewHideMove', 'reviewHideTiles']) {
      if (typeof value[key] === 'boolean') defaults[key] = value[key];
    }
    if (REVIEW_FILTERS.includes(value.reviewFilter)) defaults.reviewFilter = value.reviewFilter;
    if (value.version === 1) return defaults;
    if (['club', 'strong'].includes(value.reviewAdviser)) defaults.reviewAdviser = value.reviewAdviser;
    if (value.version === SETTINGS_VERSION && NETWORKS.some(network => network.version === value.trainedNetwork)) {
      defaults.trainedNetwork = value.trainedNetwork;
    }
    if (OPPONENT_TYPES.includes(value.difficulty)) defaults.difficulty = value.difficulty;
    if (Array.isArray(value.opponents)) {
      // Only a complete valid table replaces the legacy single-tier setting.
      const opponents = normalizeOpponents(value.opponents);
      defaults.opponents = opponents;
      defaults.difficulty = opponentPreset(opponents);
    }
  } catch { /* Restricted storage should not prevent playing. */ }
  return defaults;
}

/** Preferences that are one choice stored as two, and are written together:
 * the table's name and its three seats. */
const WRITTEN_TOGETHER = [['difficulty', 'opponents']];

/** The record in storage as it stands, where this build's pages wrote it,
 * and otherwise null: none, one that cannot be read, or another version's. */
function storedSettings(storage) {
  try {
    const value = JSON.parse(storage?.getItem?.(SETTINGS_KEY));
    return value && typeof value === 'object' && !Array.isArray(value) && value.version === SETTINGS_VERSION ? value : null;
  } catch { return null; }
}

/** Writes the preferences only when they differ from this window's last
 * write, and then only those that differ, into the record as it stands in
 * storage now. A window that wrote its whole record at every change took
 * back whatever another window had changed meanwhile: a network picked in
 * one was lost to the next checkbox ticked in the other, and an equal copy
 * written each time the effect reruns would do the same at every move. A
 * preference set back to nothing, as a pick of the default network is, is
 * taken out of the record. A page's first write (no `last`), and a write
 * over a record this build's pages did not write, is the whole of this
 * window's preferences. Returns this window's preferences as it last wrote
 * them, for the next call; what storage holds may say more. */
export function writeSettings(storage, settings, last = null) {
  const mine = { version: SETTINGS_VERSION, ...settings };
  const text = JSON.stringify(mine);
  if (text === last) return last;
  let record = mine;
  const stored = last === null ? null : storedSettings(storage);
  if (stored) {
    const before = JSON.parse(last);
    const changed = new Set(Object.keys({ ...before, ...mine }).filter(key => JSON.stringify(before[key]) !== JSON.stringify(mine[key])));
    for (const keys of WRITTEN_TOGETHER) if (keys.some(key => changed.has(key))) keys.forEach(key => changed.add(key));
    record = { ...stored };
    for (const key of changed) {
      if (mine[key] === undefined) delete record[key];
      else record[key] = mine[key];
    }
  }
  try { storage?.setItem(SETTINGS_KEY, JSON.stringify(record)); } catch { return last; /* Gameplay still works. */ }
  return text;
}

export class MatchSession {
  constructor(Game, seed, difficulty, { ai, onChange = () => {}, onSave = () => {}, guard = (task) => task() } = {}) {
    require(Number.isSafeInteger(seed) && seed >= 0 && seed < 2 ** 31, 'Invalid match seed');
    const opponents = normalizeOpponents(difficulty);
    const preset = opponentPreset(opponents);
    this.engine = preset === 'custom' ? Game.with_opponents(seed, ...opponents) : new Game(seed, preset);
    this.seed = seed;
    this.initialDifficulty = preset;
    this.initialOpponents = Object.freeze([...opponents]);
    this.opponents = [...opponents];
    this.commands = [];
    this.events = [];
    this.ai = ai;
    this.guard = guard;
    this.onChange = onChange;
    this.onSave = onSave;
    this.closed = false;
    this.busy = false;
    this.thinking = false;
    this.failure = '';
    this.abort = new AbortController();
    // Club asked for while a Trained opponent is awaited, and whether the
    // match was given to Club that way (clubMeanwhile).
    this.standIn = false;
    this.stoodIn = false;
  }

  get view() { return this.engine.view(); }
  get choices() { return this.engine.choices(); }
  get over() { return this.engine.game_is_over(); }
  get difficulty() { return opponentPreset(this.opponents); }
  get needsRecovery() { return !this.closed && Boolean(this.failure) && this.engine.needs_opponent_move(); }
  get pendingOpponent() {
    if (this.closed || !this.engine.needs_opponent_move()) return null;
    const player = this.engine.opponent_player?.();
    if (!Number.isInteger(player)) return null;
    const seats = this.view.seats;
    const index = seats.findIndex(seat => seat.player === player);
    return index > 0 ? { player, position: OPPONENT_POSITIONS[index - 1] } : null;
  }
  get progressed() {
    const view = this.view;
    return !this.over && (this.commands.length > 0 || view.hands_played > 0 || view.phase === 'over'
      || view.seats.some((seat) => seat.discards.length || seat.melds.length || seat.riichi));
  }

  stateKey() { return JSON.stringify([this.view, this.choices, this.over]); }

  snapshot() {
    return { version: VERSION, format: 7, seed: this.seed, difficulty: this.initialDifficulty, opponents: [...this.initialOpponents],
      commands: this.commands.map((command) => ({ ...command })), state: this.stateKey() };
  }

  static restore(Game, text, options) {
    require(typeof text === 'string' && text.length <= MAX_SAVE_BYTES, 'Saved match is too large');
    const saved = JSON.parse(text);
    const format = saved?.format ?? 0;
    // A save without a format, or of format 2, 3 or 4, was written between 5
    // and 10 September 2026. Those are no longer replayed: the player is told
    // why, and the page keeps the save until New game replaces it.
    const retired = saved?.version === VERSION && [0, 2, 3, 4].includes(format);
    require(!retired, 'It was saved between 5 and 10 September 2026 by an earlier version of this site, and this version cannot resume a match that old');
    require(saved?.version === VERSION && [5, 6, 7].includes(format) && Array.isArray(saved.commands), 'Unsupported saved match');
    require(saved.commands.length <= MAX_COMMANDS && typeof saved.state === 'string', 'Invalid saved match');
    const opponents = normalizeOpponents(saved.opponents);
    require(saved.difficulty === opponentPreset(opponents), 'Saved opponents disagree with their preset');
    const session = new MatchSession(Game, saved.seed, opponents, options);
    try {
      session.advance(false);
      for (const command of saved.commands) {
        session.apply(command);
        session.advance(false);
      }
      // A save of format 5 or 6 is compared without what was added or
      // repaired since it was written, and must agree on everything else.
      // One that diverges fails closed, and the save is left untouched.
      const state = session.stateKey();
      const comparable = (value) => {
        if (format >= 7) return value;
        const data = JSON.parse(value), view = data[0];
        // Format 7 gives a win's value as numbers beside the sentence that is
        // still compared, and says a robbed quad was robbed where it used to
        // say the winner took a discard. Neither changes what happened.
        for (const win of view.outcome?.wins ?? []) delete win.payments;
        if (typeof view.outcome?.line === 'string') view.outcome.line = view.outcome.line.replace(/(\w+) robs (\w+)'s quad/g, "$1 wins on $2's discard");
        // Before format 7 a declared quad left the turn's drawn tile held
        // apart while the quad could be robbed, and after it was, even when
        // the quad had taken that tile. So at every call and every result,
        // quad or not, the concealed tiles are compared as one set: the tile
        // held apart joins them unless all four of its copies are in a quad.
        // The waits are left out, because the old ones were read from the
        // hand without the tile held apart; they follow from the tiles,
        // which are all still compared.
        if (view.phase === 'call' || view.phase === 'over') {
          for (const seat of view.seats ?? []) {
            const quadded = seat.melds?.some(meld => meld.tiles?.length === 4 && meld.tiles.every(tile => tile === seat.drawn));
            seat.hand = [...seat.hand ?? [], ...(seat.drawn && !quadded ? [seat.drawn] : [])].sort();
            delete seat.drawn;
          }
          delete view.waits;
          delete view.waits_left;
        }
        if (format >= 6) return JSON.stringify(data);
        // Format 6 adds scorer-owned tile attribution, not a rules change.
        // Old saved hands still verify all tiles, yaku/han and payments.
        for (const win of data[0].outcome?.wins ?? []) {
          delete win.scoring;
          for (const yaku of win.yaku ?? []) { delete yaku.id; delete yaku.tile; }
        }
        return JSON.stringify(data);
      };
      require(comparable(state) === comparable(saved.state), 'The saved match does not match this engine version');
      return session;
    } catch (error) {
      session.dispose();
      throw error;
    }
  }

  apply(command) {
    require(command && typeof command === 'object' && !Array.isArray(command), 'Invalid match command');
    require(this.commands.length < MAX_COMMANDS, 'This match has exceeded the save limit');
    let recorded;
    switch (command.type) {
      case 'choose': {
        require(PLAYER_ACTIONS.includes(command.kind), 'Invalid player action');
        const tile = command.tile ?? null;
        require(tile === null || (typeof tile === 'string' && /^[1-9][mps]$|^[1-7]z$/.test(tile)), 'Invalid tile');
        require(this.choices.some((choice) => choice.kind === command.kind && (choice.tile ?? null) === tile), 'That choice is no longer available');
        this.engine.choose(command.kind, tile ?? undefined);
        recorded = { type: 'choose', kind: command.kind, tile };
        break;
      }
      case 'opponent': {
        require(this.engine.needs_opponent_move(), 'No opponent decision is pending');
        // The trained opponents answer in Mortal's forty-six moves, which is
        // the space the network they run was trained in. Our own rules still
        // decide what may be done; the engine turns the answer back into a
        // move it can play.
        const mask = this.engine.opponent_mask_mortal();
        require(Number.isInteger(command.action) && command.action >= 0 && command.action < mask.length && mask[command.action], 'Invalid opponent action');
        // A declared reach plays nothing yet: the engine waits for the tile
        // it discards, the next command. run() applies the two back to back;
        // where a save stopped between them, run() asks the tile.
        this.engine.play_opponent_mortal(command.action);
        recorded = { type: 'opponent', action: command.action };
        break;
      }
      case 'next':
        require(!this.over && this.engine.hand_is_over(), 'The hand cannot be advanced');
        this.engine.next_hand();
        if (!this.engine.game_is_over()) this.events = [];
        recorded = { type: 'next' };
        break;
      case 'club':
        require(this.opponents.includes('neural'), 'Already using built-in opponents');
        this.engine.continue_with_club();
        this.opponents = this.opponents.map(type => type === 'neural' ? 'club' : type);
        recorded = { type: 'club' };
        break;
      case 'opponent-club': {
        const pending = this.pendingOpponent;
        require(pending && Number.isInteger(command.player) && command.player === pending.player, 'That trained opponent is not awaiting an answer');
        const index = this.view.seats.findIndex(seat => seat.player === command.player) - 1;
        require(index >= 0 && this.opponents[index] === 'neural', 'That opponent is not Trained');
        this.engine.continue_opponent_with_club(command.player);
        this.opponents = this.opponents.map((type, position) => position === index ? 'club' : type);
        recorded = { type: 'opponent-club', player: command.player };
        break;
      }
      default:
        throw new Error('Unrecognized saved match command');
    }
    // Store only successfully applied, sanitized commands.
    this.commands.push(recorded);
  }

  advance(save = true) {
    if (!this.over) this.events = [...this.events, ...this.engine.advance()].slice(-300);
    if (save) this.onSave(this.snapshot());
  }

  notify() { if (!this.closed) this.onChange(this); }

  async run(command = null) {
    if (this.closed || this.busy) return false;
    this.busy = true;
    this.failure = '';
    this.notify();
    try {
      return await this.guard(async () => {
        if (this.closed) return false;
        if (command) this.apply(command);
        this.advance();
        this.notify();
        let moves = 0;
        while (!this.over && this.engine.needs_opponent_move()) {
          require(moves++ < 400, 'The opponent sequence did not settle');
          this.thinking = true;
          this.notify();
          let actions = null;
          try { actions = await this.trainedMove(); }
          catch (error) {
            // Given to Club while it was asked (clubMeanwhile), the question
            // was dropped: it did not fail.
            if (this.closed || !this.standIn) throw error;
          }
          if (this.closed) return false;
          if (this.standIn) {
            // Club takes every Trained seat here, in the run that was
            // waiting, as recovery's own command does after a failure. An
            // answer that came just as it was asked for is let go unplayed.
            this.standIn = false;
            this.apply({ type: 'club' });
            this.stoodIn = true;
          } else {
            // A riichi and its tile are applied together, and saved once.
            for (const action of actions) this.apply({ type: 'opponent', action });
          }
          this.advance();
          this.notify();
        }
        return true;
      });
    } catch (error) {
      if (!this.closed) this.failure = error?.message ?? String(error);
      return false;
    } finally {
      this.standIn = false;
      // A reply/error from a disposed match must not touch its replacement.
      if (!this.closed) {
        this.busy = false;
        this.thinking = false;
        this.notify();
      }
    }
  }

  /** Gives the match's Trained seats to Club while one of them is still
   * asked, for a player who would rather play on than wait for the network
   * to download: the question is dropped, unanswered, and the run that asked
   * it goes on with Club, recording the same command as continueWithClub
   * does once an opponent has failed. Nothing else is stopped, so whatever
   * was fetching the network carries on. Answers whether it was taken: only
   * a match waiting on a Trained opponent can be. `stoodIn` says afterwards
   * that the match turned to Club this way, not after a failure, which the
   * page reads to keep the player's own table as their preference (App). */
  clubMeanwhile() {
    if (this.closed || !this.thinking || this.standIn) return false;
    this.standIn = true;
    // The question ends as its signal does; later ones have a signal of
    // their own, and the match's end stops whichever is current.
    const asked = this.abort;
    this.abort = new AbortController();
    asked.abort();
    return true;
  }

  /** The waiting Trained opponent's move, as Mortal's moves to apply in
   * order: Trained AI's move (trained-move.js), asked through `ai(planes,
   * mask, signal)`, which answers as analyzePolicy does. A riichi is two, its
   * declaration and its tile, the tile asked of a preview of the reach before
   * either is recorded, so that the last hand's rules can throw it undeclared
   * instead. What else is asked of the engine, the renamed game where Trained
   * AI hesitates, the preview and the rules' codes, is read only then, from
   * the engine as it still stands, since nothing moves it while the match is
   * busy, and not at all once the match is closed. A reach a save stopped
   * between its two moves is asked its tile of the engine's pending reach,
   * whose mask opens only tiles, so no rule is asked and the riichi stands. */
  async trainedMove() {
    return (await decide(this.ai, opponentTable(this.engine), this.abort.signal)).moves;
  }

  choose(choice) { return this.run({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null }); }
  nextHand() { return this.run({ type: 'next' }); }
  retry() { return this.run(); }
  continueWithClub() { return this.run({ type: 'club' }); }
  continueOpponentWithClub() {
    const pending = this.pendingOpponent;
    return pending ? this.run({ type: 'opponent-club', player: pending.player }) : Promise.resolve(false);
  }

  dispose() {
    if (this.closed) return;
    this.closed = true;
    this.abort.abort();
    this.engine.free();
  }
}
