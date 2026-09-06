/** A match owns its engine, AI work and replay log. No async continuation uses
 * a replaceable global Game. Saves replay legal commands on the real engine;
 * they never deserialize arbitrary internal Rust state or rerun neural choices.
 */
import { normalizeOpponents, opponentPreset, OPPONENT_TYPES, OPPONENT_POSITIONS } from './opponents.js';

export const SAVE_KEY = 'riichi.match.v2';
export const SETTINGS_KEY = 'riichi.settings.v1';
const VERSION = 1;
const MAX_COMMANDS = 20000;
const MAX_SAVE_BYTES = 2_000_000;
const DIFFICULTIES = OPPONENT_TYPES;
const PLAYER_ACTIONS = ['discard', 'riichi', 'tsumo', 'concealed-kan', 'extended-kan', 'ron', 'pon', 'kan', 'chii', 'pass'];

function require(condition, message) {
  if (!condition) throw new Error(message);
}

export function readSettings(storage, touch = false) {
  const defaults = { difficulty: 'club', hints: true, confirmDiscards: touch, shortcuts: true };
  try {
    const value = JSON.parse(storage?.getItem(SETTINGS_KEY));
    if (value?.version !== VERSION) return defaults;
    for (const key of ['hints', 'confirmDiscards', 'shortcuts']) {
      if (typeof value[key] === 'boolean') defaults[key] = value[key];
    }
    if (DIFFICULTIES.includes(value.difficulty)) defaults.difficulty = value.difficulty;
    if (Array.isArray(value.opponents)) {
      // Only a complete valid table replaces the legacy single-tier setting.
      const opponents = normalizeOpponents(value.opponents);
      defaults.opponents = opponents;
      defaults.difficulty = opponentPreset(opponents);
    }
  } catch { /* Restricted storage should not prevent playing. */ }
  return defaults;
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
    return index > 0 ? { player, position: OPPONENT_POSITIONS[index - 1], seat: seats[index].seat } : null;
  }
  get progressed() {
    const view = this.view;
    return !this.over && (this.commands.length > 0 || view.hands_played > 0 || view.phase === 'over'
      || view.seats.some((seat) => seat.discards.length || seat.melds.length || seat.riichi));
  }

  stateKey() { return JSON.stringify([this.view, this.choices, this.over]); }

  snapshot() {
    return { version: VERSION, format: 4, seed: this.seed, difficulty: this.initialDifficulty, opponents: [...this.initialOpponents],
      commands: this.commands.map((command) => ({ ...command })), state: this.stateKey() };
  }

  static restore(Game, text, options) {
    require(typeof text === 'string' && text.length <= MAX_SAVE_BYTES, 'Saved match is too large');
    const saved = JSON.parse(text);
    require(saved?.version === VERSION && (saved.format === undefined || saved.format === 2 || saved.format === 3 || saved.format === 4) && Array.isArray(saved.commands), 'Unsupported saved match');
    require(saved.commands.length <= MAX_COMMANDS && typeof saved.state === 'string', 'Invalid saved match');
    require(saved.format === 4 || !saved.commands.some(command => command?.type === 'opponent-club'), 'Unsupported legacy controller change');
    const opponents = saved.format === 4 ? normalizeOpponents(saved.opponents) : normalizeOpponents(saved.difficulty);
    require(saved.format !== 4 || saved.difficulty === opponentPreset(opponents), 'Saved opponents disagree with their preset');
    const session = new MatchSession(Game, saved.seed, opponents, options);
    try {
      session.advance(false);
      for (const command of saved.commands) {
        const legacyCall = saved.format === undefined && session.difficulty === 'neural'
          && command.type === 'choose' && session.view.phase === 'call';
        session.apply(command);
        session.advance(false);
        // The old binding implicitly passed all unasked opponents after a
        // human call. Preserve those historical decisions as explicit legal
        // passes, never by asking a new network to rewrite the past.
        while (legacyCall && !session.over && session.view.phase === 'call' && session.engine.needs_opponent_move()) {
          require(session.engine.opponent_mask()[70], 'Cannot migrate a historical claim');
          session.apply({ type: 'opponent', action: 70 });
          session.advance(false);
        }
      }
      // Prior saves did not include the claimed tile in meld presentation.
      // Compare every former field, then migrate only this additive metadata.
      // Divergent rules/commands still fail closed and leave the save untouched.
      const state = session.stateKey();
      const comparable = (value) => {
        if (saved.format === 4) return value;
        const data = JSON.parse(value);
        // Only new identity/controller metadata is ignored for old saves. The
        // original uniform controllers are replayed, never inferred from UI.
        for (const seat of data[0].seats ?? []) { delete seat.player; delete seat.controller; }
        if (saved.format === 3) return JSON.stringify(data);
        // These derived hints were incorrect on 14-tile hands. They are not
        // authoritative game state; every tile, score, phase and legal choice
        // must still agree exactly with the replayed record.
        delete data[0].waits;
        delete data[0].waits_left;
        for (const win of data[0].outcome?.wins ?? []) {
          delete win.dora_indicators;
          delete win.ura_indicators;
          delete win.ura_dora;
        }
        return JSON.stringify(data, (key, field) => saved.format === undefined && key === 'claimed_tile' ? undefined : field);
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
        const mask = this.engine.opponent_mask();
        require(Number.isInteger(command.action) && command.action >= 0 && command.action < mask.length && mask[command.action], 'Invalid opponent action');
        this.engine.play_opponent(command.action);
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
          const action = await this.ai(this.engine.opponent_observation(), this.engine.opponent_mask(), this.abort.signal);
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
