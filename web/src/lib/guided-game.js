import { TILES, emptyPosition, parsePhysical, missingNumber, recordDraw, recordDiscard, recordChoice } from './physical-position.js';
import { tileWords } from './tiles.js';

export const GUIDED_KEY = 'riichi.guided.v1';
const WINDS = ['East', 'South', 'West', 'North'];
const STAGES = ['setup', 'hand', 'dora', 'turn', 'decision', 'responses', 'kan-response', 'indicator', 'over'];
const AGENTS = ['beginner', 'club', 'quick', 'strong'];
const sameChoice = (a, b) => a.kind === b.kind && (a.tile ?? null) === (b.tile ?? null);

export function emptyGuided() {
  const position = emptyPosition();
  position.wall = 70; position.phase = 'draw';
  return { state: { position, stage: 'setup', nextSeat: 0, needsDraw: true, agent: 'club', result: '' }, past: [], log: [] };
}

export function setTiles(meld) {
  return meld.kind === 'chii'
    ? [0, 1, 2].map(n => `${Number(meld.tile[0]) + n}${meld.tile[1]}`)
    : Array(meld.kind.includes('kan') ? 4 : 3).fill(meld.tile);
}

/** Count only tiles actually seen, with claimed discards counted in their set. */
export function visibleCounts(position) {
  const counts = new Map(TILES.map(tile => [tile, 0]));
  const add = tile => {
    if (!counts.has(tile)) throw new Error('Choose a valid tile');
    const count = counts.get(tile) + 1;
    if (count > 4) throw new Error(`All four copies of ${tileWords(tile)} are already accounted for. Undo the mistaken entry.`);
    counts.set(tile, count);
  };
  position.indicators.forEach(add);
  position.players.forEach(player => {
    player.hand.forEach(add);
    player.melds.forEach(meld => setTiles(meld).forEach(add));
    player.discards.filter(d => !d.claimed).forEach(d => add(d.tile));
  });
  return counts;
}

export function doraTiles(position) {
  return position.indicators.map(tile => {
    const rank = Number(tile[0]), suit = tile[1];
    return `${suit !== 'z' ? rank % 9 + 1 : rank <= 4 ? rank % 4 + 1 : (rank - 4) % 3 + 5}${suit}`;
  });
}

function numbers(p) {
  const missing = missingNumber(p);
  if (missing) throw new Error(missing);
  if (![p.seat, p.round].every(n => Number.isInteger(n) && n >= 0 && n < 4)
    || p.kyoku < 1 || p.kyoku > 4 || p.counters < 0 || p.counters > 100
    || p.riichi_sticks < 0 || p.riichi_sticks > 100
    || p.players.some(player => player.score < -100000 || player.score > 200000)) {
    throw new Error('Check the seat, round, points, honba and riichi sticks');
  }
}

function requireStage(state, ...stages) {
  if (!stages.includes(state.stage)) throw new Error('That event is no longer the next step. Use the current prompt.');
}

function requireTile(tile) {
  if (!TILES.includes(tile)) throw new Error('Choose a tile');
}

function startTurn(state, seat, needsDraw = true, replacement = false) {
  const p = state.position;
  state.nextSeat = seat; state.needsDraw = needsDraw; state.stage = 'turn';
  p.turn = seat; p.phase = 'draw'; p.pending = null; p.pending_kind = 'discard';
  p.drawn = null; p.just_claimed = null; p.after_quad = replacement;
  if (needsDraw && p.wall === 0) {
    state.stage = 'over'; p.phase = 'over'; state.result = 'Exhaustive draw';
  }
}

function completeKan(state) {
  state.stage = 'indicator';
  state.nextSeat = state.position.turn;
  state.needsDraw = true;
  state.position.players.forEach(player => { player.ippatsu = false; });
}

function checkKan(p) {
  if (p.wall <= 0) throw new Error('A kan needs a replacement draw; the live wall is empty');
  if (p.players.flatMap(player => player.melds).filter(m => m.kind.includes('kan')).length >= 4) {
    throw new Error('Four kans have already been declared');
  }
}

/** One table event at a time. validateDecision is the real rules engine in the UI.
 * Unknown opponents' hands remain empty throughout; no wall is simulated. */
export function guidedEvent(game, event, validateDecision = () => {}) {
  const state = structuredClone(game.state);
  let p = state.position, label;
  const actor = () => WINDS[state.nextSeat];
  switch (event.type) {
    case 'setup':
      requireStage(state, 'setup'); numbers(p);
      state.stage = 'hand';
      label = `${WINDS[p.round]} ${p.kyoku} · You are ${WINDS[p.seat]} · ${p.players[p.seat].score.toLocaleString()} points`;
      break;
    case 'hand':
      requireStage(state, 'hand');
      if (p.players[p.seat].hand.length !== 13) throw new Error('Choose exactly 13 starting tiles');
      state.stage = 'dora'; label = 'Starting hand recorded · 13 tiles';
      break;
    case 'indicator':
      requireStage(state, 'dora', 'indicator'); requireTile(event.tile);
      p.indicators.push(event.tile);
      label = `Dora indicator: ${tileWords(event.tile)}`;
      if (state.stage === 'dora') startTurn(state, 0);
      else startTurn(state, state.nextSeat, true, true);
      break;
    case 'draw':
      requireStage(state, 'turn');
      if (state.nextSeat !== p.seat || !state.needsDraw) throw new Error('This prompt is not for your draw');
      label = `You draw ${tileWords(event.tile)}${p.after_quad ? ' · replacement' : ''}`;
      p = state.position = recordDraw(p, p.seat, event.tile);
      state.stage = 'decision'; state.needsDraw = false;
      break;
    case 'discard': {
      requireStage(state, 'turn'); requireTile(event.tile);
      if (state.nextSeat === p.seat) throw new Error('Choose one of your legal moves');
      if (state.needsDraw && p.wall <= 0) throw new Error('The live wall is empty');
      const player = p.players[state.nextSeat];
      if (event.riichi && (!state.needsDraw || p.wall < 5 || player.melds.some(m => m.kind !== 'concealed-kan'))) {
        throw new Error('Riichi requires a closed hand, a draw, and at least four live tiles after it');
      }
      const wall = p.wall;
      p = state.position = recordDiscard(p, state.nextSeat, event.tile, Boolean(event.riichi));
      // The manual editor combines an unknown draw with every discard. A
      // guided pon/chii already establishes that this turn has no draw.
      if (!state.needsDraw) p.wall = wall;
      p.players[state.nextSeat].discards.at(-1).drawn = state.needsDraw && (Boolean(event.drawn) || player.riichi !== 'none');
      state.stage = 'decision';
      label = `${actor()}: ${event.riichi ? 'riichi · ' : ''}discard ${tileWords(event.tile)}`;
      break;
    }
    case 'choice': {
      requireStage(state, 'decision');
      const { choice, choices } = event;
      if (!choice || !Array.isArray(choices) || !choices.some(c => sameChoice(c, choice))) throw new Error('Analyse this decision again');
      p = state.position = recordChoice(p, choice, choices);
      label = `You: ${choice.label ?? choice.kind}`;
      if (choice.kind === 'pass') state.stage = p.pending_kind === 'discard' ? 'responses' : 'kan-response';
      else if (choice.kind === 'ron' || choice.kind === 'tsumo') { state.stage = 'over'; state.result = `You: ${choice.kind}`; }
      else if (choice.kind === 'discard' || choice.kind === 'riichi') state.stage = 'responses';
      else if (choice.kind === 'kan') completeKan(state);
      else if (choice.kind.includes('kan')) state.stage = 'kan-response';
      // Chii and pon lead straight to the next discard decision, without a draw.
      else { state.stage = 'decision'; state.nextSeat = p.seat; state.needsDraw = false; }
      break;
    }
    case 'continue':
      requireStage(state, 'responses', 'kan-response');
      if (state.stage === 'kan-response') { completeKan(state); label = 'Kan stands · no ron'; }
      else { startTurn(state, (p.turn + 1) % 4); label = state.stage === 'over' ? 'Exhaustive draw' : 'No other calls'; }
      break;
    case 'call': {
      requireStage(state, 'responses');
      const seat = event.seat, kind = event.kind;
      if (!Number.isInteger(seat) || seat < 0 || seat > 3 || seat === p.turn || seat === p.seat) throw new Error('Choose the opponent who called');
      if (!['chii', 'pon', 'kan'].includes(kind)) throw new Error('Choose chii, pon or open kan');
      const player = p.players[seat];
      if (player.riichi !== 'none' || player.melds.length >= 4 || p.wall <= 0) throw new Error('This opponent cannot call this discard');
      const offered = p.pending;
      if (kind === 'kan') checkKan(p);
      const meld = { kind, tile: kind === 'chii' ? event.tile : offered, from: (p.turn - seat + 4) % 4 };
      if (kind === 'chii' && (!/^[1-7][mps]$/.test(event.tile ?? '') || meld.from !== 3 || !setTiles(meld).includes(offered))) {
        throw new Error('Chii must include the discard in a sequence from the left');
      }
      const discard = p.players[p.turn].discards.at(-1);
      if (!discard || discard.claimed || discard.tile !== offered) throw new Error('That discard is no longer available');
      discard.claimed = true; player.melds.push(meld); player.furiten = false;
      p.first_turns = false; p.players.forEach(player => { player.ippatsu = false; });
      label = `${WINDS[seat]}: ${kind} · ${setTiles(meld).map(tileWords).join(', ')}`;
      startTurn(state, seat, kind === 'kan', kind === 'kan');
      if (kind === 'kan') completeKan(state);
      break;
    }
    case 'kan': {
      requireStage(state, 'turn'); requireTile(event.tile);
      if (state.nextSeat === p.seat || !state.needsDraw) throw new Error('A kan can only be declared after a draw');
      checkKan(p);
      const player = p.players[state.nextSeat];
      if (!['concealed-kan', 'extended-kan'].includes(event.kind)) throw new Error('Choose the kind of kan');
      if (event.kind === 'extended-kan') {
        const meld = player.melds.find(m => m.kind === 'pon' && m.tile === event.tile);
        if (!meld) throw new Error('This opponent has no matching pon to extend');
        meld.kind = 'extended-kan';
      } else {
        if (player.melds.length >= 4) throw new Error('This opponent already has four sets');
        player.melds.push({ kind: 'concealed-kan', tile: event.tile, from: 0 });
      }
      // Count the opponent's unobserved draw before the kan, then the
      // replacement separately when their next event is entered.
      p.wall--;
      if (p.wall === 0) throw new Error('There is no replacement draw left after this draw');
      p.first_turns = false; p.turn = state.nextSeat;
      p.phase = 'call'; p.drawn = null; p.just_claimed = null;
      p.pending = event.tile; p.pending_kind = event.kind; p.after_quad = true;
      state.stage = 'decision'; state.needsDraw = false;
      label = `${actor()}: ${event.kind} · ${tileWords(event.tile)}`;
      break;
    }
    case 'finish':
      requireStage(state, 'turn', 'decision', 'responses', 'kan-response', 'indicator');
      if (typeof event.result !== 'string' || !event.result.trim() || event.result.length > 120) throw new Error('Choose the hand result');
      state.stage = 'over'; p.phase = 'over'; state.result = event.result; label = event.result;
      break;
    case 'next-hand': {
      requireStage(state, 'over'); numbers(p);
      const old = p;
      p = state.position = emptyPosition(); p.wall = 70; p.phase = 'draw';
      const shift = event.repeat ? 0 : 1;
      p.seat = (old.seat - shift + 4) % 4;
      p.round = old.round + (shift && old.kyoku === 4 ? 1 : 0);
      if (p.round > 3) throw new Error('North 4 has finished. Start a new game.');
      p.kyoku = event.repeat ? old.kyoku : old.kyoku % 4 + 1;
      p.counters = event.repeat || state.result === 'Exhaustive draw' ? old.counters + 1 : 0;
      p.riichi_sticks = old.riichi_sticks;
      p.players.forEach((player, i) => { player.score = old.players[(i + shift) % 4].score; });
      state.stage = 'setup'; state.nextSeat = 0; state.needsDraw = true; state.result = '';
      label = event.repeat ? 'Next hand · dealer repeats' : 'Next hand · dealer moves';
      break;
    }
    default: throw new Error('Unknown guided event');
  }
  visibleCounts(p);
  if (state.stage === 'decision') validateDecision(p);
  return { state, past: [...game.past.slice(-29), { state: structuredClone(game.state), logLength: game.log.length }], log: [...game.log, label] };
}

/** Save setup/score inputs without turning every keystroke into a game move. */
export function editGuided(game, edit) {
  const next = structuredClone(game);
  edit(next.state);
  visibleCounts(next.state.position);
  return next;
}

export function undoGuided(game) {
  const previous = game.past.at(-1);
  return previous ? { state: structuredClone(previous.state), past: game.past.slice(0, -1), log: game.log.slice(0, previous.logLength) } : game;
}

export function parseGuided(text) {
  try {
    if (typeof text !== 'string' || text.length > 2_000_000) return null;
    const value = JSON.parse(text), game = value?.game;
    const validState = state => state && STAGES.includes(state.stage) && AGENTS.includes(state.agent)
      && Number.isInteger(state.nextSeat) && state.nextSeat >= 0 && state.nextSeat < 4
      && typeof state.needsDraw === 'boolean' && typeof state.result === 'string' && state.result.length <= 120
      && parsePhysical(JSON.stringify({ version: 1, position: state.position }))
      && state.position.indicators.length <= 5
      && state.position.players.every(p => p.hand.length <= 14 && p.melds.length <= 4 && p.discards.length <= 100)
      && visibleCounts(state.position);
    if (value?.version !== 1 || !game || !validState(game.state) || !Array.isArray(game.past) || game.past.length > 30
      || !Array.isArray(game.log) || game.log.length > 10000 || game.log.some(line => typeof line !== 'string' || line.length > 500)
      || game.past.some(entry => !validState(entry.state) || !Number.isInteger(entry.logLength) || entry.logLength < 0 || entry.logLength > game.log.length)) return null;
    return game;
  } catch { return null; }
}

export const GUIDED_FORMAT = {
  key: GUIDED_KEY, lock: 'riichi.guided-game.writer', empty: emptyGuided, parse: parseGuided,
  encode: game => JSON.stringify({ version: 1, game }),
};
