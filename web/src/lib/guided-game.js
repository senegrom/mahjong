import { TILES, emptyPosition, parsePhysical, missingNumber, recordDraw, recordDiscard, recordChoice } from './physical-position.js';
import { tileWords } from './tiles.js';
import { rememberEnding, applySettlement, validEndingState, verifySettlement } from './guided-settlement.js';

export const GUIDED_KEY = 'riichi.guided.v1';
const WINDS = ['East', 'South', 'West', 'North'];
const STAGES = ['setup', 'hand', 'dora', 'turn', 'decision', 'responses', 'claim-response', 'kan-response', 'indicator', 'over'];
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
  if (needsDraw && p.wall === 0) rememberEnding(state, 'draw');
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

/** Commit a set only after the table has resolved higher-priority calls. */
function applyClaim(state, claim) {
  const p = state.position, { seat, kind, tile } = claim;
  if (p.phase !== 'call' || p.pending_kind !== 'discard') throw new Error('A set needs a pending discard');
  if (!Number.isInteger(seat) || seat < 0 || seat > 3 || seat === p.turn) throw new Error('Choose the player who called');
  if (!['chii', 'pon', 'kan'].includes(kind)) throw new Error('Choose chii, pon or open kan');
  const player = p.players[seat], offered = p.pending;
  if (player.riichi !== 'none' || player.melds.length >= 4 || p.wall <= 0) throw new Error('This player cannot call this discard');
  if (kind === 'kan') checkKan(p);
  const meld = { kind, tile: kind === 'chii' ? tile : offered, from: (p.turn - seat + 4) % 4 };
  if (kind === 'chii' && (!/^[1-7][mps]$/.test(tile ?? '') || meld.from !== 3 || !setTiles(meld).includes(offered))) {
    throw new Error('Chii must include the discard in a sequence from the left');
  }
  if (seat === p.seat) {
    state.position = recordChoice(p, claim, [claim]);
    state.stage = 'decision'; state.nextSeat = seat; state.needsDraw = false;
  } else {
    const discard = p.players[p.turn].discards.at(-1);
    if (!discard || discard.claimed || discard.tile !== offered) throw new Error('That discard is no longer available');
    discard.claimed = true; player.melds.push(meld); player.furiten = false;
    p.first_turns = false; p.players.forEach(player => { player.ippatsu = false; });
    startTurn(state, seat, kind === 'kan', kind === 'kan');
    if (kind !== 'kan') p.just_claimed = offered;
  }
  if (kind === 'kan') completeKan(state);
}

function queueClaim(state, claim) {
  // Validate the eventual move now, but keep its tiles at their original
  // positions so ron can still cancel the set and be scored correctly.
  const preview = structuredClone(state);
  applyClaim(preview, claim); visibleCounts(preview.position);
  state.claim = claim; state.stage = 'claim-response';
}

function validClaim(state) {
  if (state.stage !== 'claim-response') return state.claim == null;
  if (!state.claim) return false;
  const preview = structuredClone(state);
  applyClaim(preview, state.claim); visibleCounts(preview.position);
  return true;
}

function forbiddenAfterCall(p, player) {
  if (!p.just_claimed) return [];
  const forbidden = [p.just_claimed], meld = player.melds.at(-1);
  if (meld?.kind === 'chii') {
    const low = Number(meld.tile[0]), called = Number(p.just_claimed[0]);
    if (called === low && low <= 6) forbidden.push(`${low + 3}${meld.tile[1]}`);
    if (called === low + 2 && low >= 2) forbidden.push(`${low - 1}${meld.tile[1]}`);
  }
  return forbidden;
}

/** One table event at a time. validateDecision is the real rules engine in the UI.
 * Unknown opponents' hands remain empty throughout; no wall is simulated. */
export function guidedEvent(game, event, validateDecision = () => {}, scoreSettlement) {
  const state = structuredClone(game.state);
  let p = state.position, label;
  const actor = () => WINDS[state.nextSeat];
  switch (event.type) {
    case 'setup':
      requireStage(state, 'setup'); numbers(p);
      state.opening = p.players.map(player => player.score);
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
      if (!state.needsDraw && forbiddenAfterCall(p, player).includes(event.tile)) {
        throw new Error('Swap-calling is not allowed: choose a different discard after the call');
      }
      // EMA 2025 permits one live tile after the draw. This prompt still
      // includes the opponent's hidden draw, which recordDiscard counts below.
      if (event.riichi && (!state.needsDraw || p.wall < 2 || player.melds.some(m => m.kind !== 'concealed-kan'))) {
        throw new Error('Riichi requires a closed hand, a draw, and at least one live tile after it');
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
      if (['pon', 'chii', 'kan'].includes(choice.kind)) {
        queueClaim(state, { seat: p.seat, kind: choice.kind, tile: choice.tile ?? null });
        if (choices.some(c => c.kind === 'ron' || c.causes_furiten)) p.players[p.seat].furiten = true;
        label = `You call ${choice.label ?? choice.kind} · awaiting other responses`;
        break;
      }
      if (choice.kind === 'ron' || choice.kind === 'tsumo') rememberEnding(state, choice.kind, [p.seat]);
      p = state.position = recordChoice(p, choice, choices);
      label = `You: ${choice.label ?? choice.kind}`;
      if (choice.kind === 'pass') state.stage = p.pending_kind === 'discard' ? 'responses' : 'kan-response';
      else if (choice.kind === 'ron' || choice.kind === 'tsumo') { state.stage = 'over'; state.result = `You: ${choice.kind}`; }
      else if (choice.kind === 'discard' || choice.kind === 'riichi') state.stage = 'responses';
      else if (choice.kind.includes('kan')) state.stage = 'kan-response';
      break;
    }
    case 'continue':
      requireStage(state, 'responses', 'claim-response', 'kan-response');
      if (state.stage === 'claim-response') {
        const claim = state.claim;
        applyClaim(state, claim); p = state.position; delete state.claim;
        label = `${WINDS[claim.seat]}: ${claim.kind} stands · no higher-priority calls`;
      } else if (state.stage === 'kan-response') { completeKan(state); label = 'Kan stands · no ron'; }
      else { startTurn(state, (p.turn + 1) % 4); label = state.stage === 'over' ? 'Exhaustive draw' : 'No other calls'; }
      break;
    case 'call': {
      requireStage(state, 'responses', 'claim-response');
      const seat = event.seat, kind = event.kind;
      if (!Number.isInteger(seat) || seat < 0 || seat > 3 || seat === p.turn || seat === p.seat) throw new Error('Choose the opponent who called');
      if (state.claim && (state.claim.kind !== 'chii' || !['pon', 'kan'].includes(kind) || seat === state.claim.seat)) {
        throw new Error('Only a simultaneous pon or kan takes precedence over chii; otherwise record ron or confirm the call');
      }
      queueClaim(state, { seat, kind, tile: event.tile ?? null });
      label = `${WINDS[seat]} calls ${kind} · awaiting other responses`;
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
    case 'finish': {
      requireStage(state, 'turn', 'decision', 'responses', 'claim-response', 'kan-response', 'indicator');
      if (typeof event.result !== 'string' || !event.result.trim() || event.result.length > 120) throw new Error('Choose the hand result');
      const kind = event.kind ?? (event.result === 'Exhaustive draw' ? 'draw' : 'manual');
      const winners = ['ron', 'tsumo'].includes(kind) ? [event.winner] : [];
      if (kind === 'ron' && (p.phase !== 'call' || event.winner === p.turn)) throw new Error('Ron needs another player’s pending discard or kan');
      if (kind === 'tsumo' && !((state.stage === 'turn' && state.needsDraw && event.winner === state.nextSeat)
        || (p.phase === 'act' && p.drawn && event.winner === p.turn))) throw new Error('Tsumo needs the current player’s actual draw');
      if (kind === 'draw' && (p.wall !== 0 || p.phase !== 'call' || p.pending_kind !== 'discard')) throw new Error('An exhaustive draw needs the final discard and an empty live wall');
      rememberEnding(state, kind, winners);
      delete state.claim;
      state.stage = 'over'; p.phase = 'over'; state.result = event.result; label = event.result;
      break;
    }
    case 'settle':
      requireStage(state, 'over');
      label = applySettlement(state, event.input ?? {}, scoreSettlement);
      break;
    case 'next-hand': {
      requireStage(state, 'over'); numbers(p);
      if (state.ending && state.ending.kind !== 'manual') {
        if (!state.settlement) throw new Error('Calculate and apply the hand settlement first');
        if (event.repeat !== state.settlement.repeat) throw new Error('Dealer continuation is determined by the settled result');
      }
      const nextCounters = state.settlement?.next_counters;
      const old = p;
      p = state.position = emptyPosition(); p.wall = 70; p.phase = 'draw';
      const shift = event.repeat ? 0 : 1;
      p.seat = (old.seat - shift + 4) % 4;
      p.round = old.round + (shift && old.kyoku === 4 ? 1 : 0);
      if (p.round > 3) throw new Error('North 4 has finished. Start a new game.');
      p.kyoku = event.repeat ? old.kyoku : old.kyoku % 4 + 1;
      p.counters = nextCounters ?? (event.repeat || state.result === 'Exhaustive draw' ? old.counters + 1 : 0);
      p.riichi_sticks = old.riichi_sticks;
      p.players.forEach((player, i) => { player.score = old.players[(i + shift) % 4].score; });
      state.stage = 'setup'; state.nextSeat = 0; state.needsDraw = true; state.result = '';
      delete state.ending; delete state.settlement; delete state.opening;
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
  if (game.state.ending && game.state.ending.kind !== 'manual'
    && JSON.stringify({ ...next.state, agent: game.state.agent }) !== JSON.stringify(game.state)) {
    throw new Error('Undo the hand result before changing the scored table');
  }
  visibleCounts(next.state.position);
  return next;
}

export function undoGuided(game) {
  const previous = game.past.at(-1);
  return previous ? { state: structuredClone(previous.state), past: game.past.slice(0, -1), log: game.log.slice(0, previous.logLength) } : game;
}

export function parseGuided(text, scoreSettlement) {
  try {
    if (typeof text !== 'string' || text.length > 2_000_000) return null;
    const value = JSON.parse(text), game = value?.game;
    const validState = state => state && STAGES.includes(state.stage) && AGENTS.includes(state.agent)
      && Number.isInteger(state.nextSeat) && state.nextSeat >= 0 && state.nextSeat < 4
      && typeof state.needsDraw === 'boolean' && typeof state.result === 'string' && state.result.length <= 120
      && parsePhysical(JSON.stringify({ version: 1, position: state.position }))
      && state.position.indicators.length <= 5
      && state.position.players.every(p => p.hand.length <= 14 && p.melds.length <= 4 && p.discards.length <= 100)
      && visibleCounts(state.position) && validClaim(state) && validEndingState(state)
      && (!scoreSettlement || verifySettlement(state, scoreSettlement));
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
