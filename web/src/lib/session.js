/** A match owns its engine, AI work and replay log. No async continuation uses
 * a replaceable global Game. Saves replay legal commands on the real engine;
 * they never deserialize arbitrary internal Rust state or rerun neural choices.
 */
import { normalizeOpponents, opponentPreset, DEFAULT_OPPONENT, OPPONENT_TYPES, OPPONENT_POSITIONS } from './opponents.js';
import { normalizeTileFace } from './tile-faces.js';
import { NETWORKS } from './model-manifest.js';

export const SAVE_KEY = 'riichi.match.v2';
export const SETTINGS_KEY = 'riichi.settings.v1';
const VERSION = 1;
// The settings record's own version: 2 since the site played generation 810
// and 3 since it played generation 1110, see readSettings. Saved matches keep
// VERSION, so a match saved before either still resumes.
const SETTINGS_VERSION = 3;
const MAX_COMMANDS = 20000;
const MAX_SAVE_BYTES = 2_000_000;
const PLAYER_ACTIONS = ['discard', 'riichi', 'tsumo', 'concealed-kan', 'extended-kan', 'ron', 'pon', 'kan', 'chii', 'pass'];

function require(condition, message) {
  if (!condition) throw new Error(message);
}

/** The review adviser a player starts with: the trained network. */
export const DEFAULT_REVIEW_ADVISER = 'strong';

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
 * record from before the site played generation 1110 (version 2) keeps every
 * other choice but not its pick: when 1110 arrived every player was put in
 * front of it once, at the owner's word, whatever version they had picked
 * before. A player who picks 0.1.810 or 0.1.410 again writes a version 3
 * record, and from version 3 a pick is kept as it stands.
 *
 * reviewHideMove and reviewHideTiles say whether Trained AI's reviews keep
 * its move back, and the other players' tiles, until the player asks to see
 * them. Both start off, and a record from before they were offered holds
 * neither, so it reads them as off and the record needs no new version. */
export function readSettings(storage, touch = false) {
  const defaults = { difficulty: DEFAULT_OPPONENT, hints: true, confirmDiscards: touch, shortcuts: true, tileFace: 'classic', reviewAdviser: DEFAULT_REVIEW_ADVISER,
    reviewHideMove: false, reviewHideTiles: false };
  try {
    const value = JSON.parse(storage?.getItem(SETTINGS_KEY));
    if (![1, 2, SETTINGS_VERSION].includes(value?.version)) return defaults;
    defaults.tileFace = normalizeTileFace(value.tileFace);
    for (const key of ['hints', 'confirmDiscards', 'shortcuts', 'reviewHideMove', 'reviewHideTiles']) {
      if (typeof value[key] === 'boolean') defaults[key] = value[key];
    }
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

/** Writes the preferences only when they differ from the last write. Every
 * write replaces the whole record, so writing an equal copy each time the
 * effect reruns would overwrite a change another tab made meanwhile. Returns
 * what this tab last stored, for the next call. */
export function writeSettings(storage, settings, last = null) {
  const text = JSON.stringify({ version: SETTINGS_VERSION, ...settings });
  if (text === last) return last;
  try { storage?.setItem(SETTINGS_KEY, text); } catch { return last; /* Gameplay still works. */ }
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
        // A declared reach plays nothing yet: the engine asks again which
        // tile it discards, and the loop above comes round to answer.
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
          const action = await this.ai(this.engine.opponent_observation_mortal(), this.engine.opponent_mask_mortal(), this.abort.signal);
          if (this.closed) return false;
          this.apply({ type: 'opponent', action });
          this.advance();
          this.notify();
        }
        return true;
      });
    } catch (error) {
      if (!this.closed) this.failure = error?.message ?? String(error);
      return false;
    } finally {
      // A reply/error from a disposed match must not touch its replacement.
      if (!this.closed) {
        this.busy = false;
        this.thinking = false;
        this.notify();
      }
    }
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
