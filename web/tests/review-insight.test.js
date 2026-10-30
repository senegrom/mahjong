import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { BATCH } from '../src/lib/game-review.js';
import { MARGIN_FLOOR, PASS, applyEdits, changesMeet, explainWhy, firstHandGives, judge, moveWords, notChecked, scoreEdit, searchOffered,
  singleEdits, tablesAtMost, tablesFirstAsked, visibleCounts, whatWouldChangeItsMind } from '../src/lib/review-insight.js';
import { trainedAsk } from '../src/lib/review-insight-policy.js';
import { saidLines, saidOfChange, searchWhereOnly, standInAsk, standInQuestion } from './fixtures/insight-panel.js';
import { planesDigest, recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { mirroringNetwork } from './fixtures/renamed-planes.js';
import { clubAdvisedGame, eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const questions = recordedQuestions(Game, PhysicalAnalysis);
const question = id => questions.find(entry => entry.id === id);
const facts = review_facts;
const searched = questions.filter(entry => searchOffered(entry));

/** What version 1's panel says for each of the prototype's questions, from
 * its recorded answers (insight/recommendation/terse.txt): the Why line, then
 * the search's findings. Some lines say otherwise than terse.txt. East 3 #5
 * tries one change fewer of what is safe: North declared riichi with a tile
 * from its hand, so it cannot have declared a discard earlier, the tile
 * then following the declaration out of its hand. East 2 · 2 honba #14's
 * probe changes both of the player's 8 bamboo, since one left in the river
 * keeps a wait on it furiten, and says so in one clause; Trained AI was
 * asked that table since. South 4 #18's Why line no longer says that nobody
 * can be waiting on the player's 8 circles, every copy in sight: a 6-7 or a
 * 7-9 of circles waits on it holding none. And no table is made that no game
 * reaches: East 1 #20, the game's first deal, where nobody has scored, asks
 * no score step, and no hand is made East 1, which leaves South 4 #18 no
 * other hand to try. */
const PANELS = {
  'e3-5': [
    'Partly: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.',
    'One change and it throws your west wind:',
    '- A faster hand (7 of 7 tried): if your white dragon had been a 2 circles (your hand a step closer to waiting and one safe tile fewer): your west wind 99.6%, the 7 characters 0.2%.',
    '- A hand worth more (4 of 14 tried): if the dora indicator had been the 8 bamboo, not the 9 bamboo, making the 9 bamboo dora (2 more dora in your hand): your west wind 94.3%, the 7 characters 2.0%.',
    '- What is safe (2 of 8 tried): if South had declared riichi with their last discard, the west wind (a second riichi and the 7 characters no longer safe): your west wind 70.8%, the 7 characters <0.1%.',
    'Not moved by the score (±16,000) or the round.',
  ],
  's3-4': [
    'Why: Against North\'s riichi your 7 characters is a tile with no suji; Trained AI\'s white dragon cannot deal in. Trained AI puts 98.3% on tiles that cannot deal in. If North had not declared riichi: the 1 circles 39.7%, your 7 characters 5.8%, the white dragon 1.8%.',
    'No single change does it. Two together: if North\'s third discard had been the 7 characters, not the 2 bamboo, and North\'s first discard had been a 2 characters, not the white dragon (the white dragon no longer safe and your 7 characters safe): your 7 characters 94.4%, the white dragon <0.1%.',
    'Not moved by the score (±16,000) or the round.',
  ],
  'e1-20': [
    'Not explained: Every copy of both tiles is in sight, so neither can deal in.',
    'One change and it throws your west wind:',
    '- What is safe (9 of 15 tried): if South\'s first discard had been a 1 characters, not the north wind (not every north wind in sight): your west wind 96.0%, the north wind 2.4%.',
    'Not moved by the round.',
  ],
  'e2b-14': [
    'Why: You had already thrown an 8 bamboo, so a wait on it would be furiten. Trained AI puts 99.2% on tiles you had already thrown. If your second and ninth discards had each been a 1 characters, not the 8 bamboo: the 2 bamboo 66.4%, your 6 bamboo <0.1%, the 8 bamboo <0.1%.',
  ],
  's1-9': [
    'Why: Your 6 circles leaves the hand 5 from waiting; Trained AI\'s 3 bamboo leaves it 4 from waiting. Trained AI puts 98.3% on tiles that keep the hand 4 from waiting.',
    'One change and it throws your 6 circles:',
    '- A hand worth more (2 of 14 tried): if the dora indicator had been the 3 bamboo, not the 9 bamboo, making the 4 bamboo dora (1 more dora in your hand): your 6 circles 60.7%, the 3 bamboo 20.3%.',
    'Not moved by the score (±16,000) or the round.',
  ],
  's4-18': [
    'Partly: Your 8 circles leaves the hand 3 from waiting; Trained AI\'s 9 characters leaves it 2 from waiting. Trained AI puts 84.9% on tiles that keep the hand 2 from waiting.',
    'One change and it throws your 8 circles:',
    '- What is safe (2 of 7 tried): if South had declared riichi with their last discard, the 8 bamboo (a riichi, the 9 characters less safe and your 8 circles safe): your 8 circles 98.3%, the 9 characters 1.5%.',
    'Not moved by the score (±16,000).',
  ],
  'why7-1m': [
    'Partly: Trained AI\'s 7 characters keeps thirteen orphans 4 from waiting; your 1 characters leaves it 5 from waiting. Trained AI puts 66.0% on tiles that keep thirteen orphans 4 from waiting.',
    'One change and it throws your 1 characters:',
    '- A faster hand (3 of 7 tried): if your east wind had been a 3 bamboo (thirteen orphans a step further and your hand a step closer to waiting): your 1 characters 94.8%, the 7 characters <0.1%.',
    '- A hand worth more (12 of 15 tried): if the dora indicator had been the green dragon, not the white dragon, making the red dragon dora (2 more dora in your hand): your 1 characters 99.7%, the 7 characters <0.1%.',
    '- The table: from 7,000 points more: your 1 characters 56.0%, the 7 characters 30.2%; if this had been South 4, the last hand: your 1 characters 78.4%, the 7 characters 1.0%.',
  ],
  'why7-2s': [
    'Not explained: The counts favour your 2 bamboo.',
  ],
};

/** A question's Why line and search, Trained AI answering as the recorded
 * run did: the search with an `ask` of its own, so that what it asked can be
 * counted apart from the Why line's probe. */
async function panel(entry, { weights = entry.weights } = {}) {
  const why = await explainWhy({ ...entry, weights, facts, ask: recordedAsk(entry).ask });
  const asking = recordedAsk(entry);
  const search = await whatWouldChangeItsMind({ ...entry, weights, facts, ask: asking.ask });
  return { why, search, ...asking };
}

test('the Why line and the search say for the prototype\'s eight questions what version 1 says', async () => {
  for (const entry of questions) {
    const { why, search } = await panel(entry);
    assert.deepEqual([...saidLines(why), ...(search.said ? saidLines(search.said) : [])], PANELS[entry.id], entry.title);
  }
});

test('a Why line that is only partly or not explained carries the check\'s workings for its tooltip', async () => {
  const checks = {};
  for (const entry of questions) checks[entry.id] = (await panel(entry)).why.check;
  assert.deepEqual(checks, {
    'e3-5': 'The check that takes it away (your fourth discard had been a 2 circles, not the 7 characters) also leaves the 7 characters no longer safe, so it cannot say which counts. Explained needs 95% or more of its weight on those tiles.',
    's3-4': '',
    'e1-20': 'No fact the engine counts favours the north wind.',
    'e2b-14': '',
    's1-9': '',
    's4-18': 'Explained needs 95% or more of its weight on those tiles.',
    'why7-1m': 'Explained needs 95% or more of its weight on those tiles.',
    'why7-2s': 'No fact the engine counts favours the 7 characters; for yours, both leave the hand 4 from waiting; 18 unseen tiles would improve it after Trained AI\'s 7 characters, 67 after your 2 bamboo.',
  });
});

test('a pass is no longer than the review\'s own batch', () => {
  assert.equal(PASS, BATCH);
});

test('every table the recorded run answered is written by the page as the run asked it, but the one pair named', () => {
  for (const entry of questions) assert.deepEqual(recordedAsk(entry).otherwise, [], entry.title);
});

test('the search asks only tables the recorded run asked, the unedited one first, in passes of at most the review\'s batch', async () => {
  const passes = {}, planned = {};
  for (const entry of searched) {
    const asked = [], { ask, unanswered } = recordedAsk(entry);
    await whatWouldChangeItsMind({ ...entry, facts, ask: positions => { asked.push(positions); return ask(positions); },
      onProgress: step => { planned[entry.id] ??= step.planned; } });
    passes[entry.id] = asked.map(pass => pass.length);
    assert.deepEqual(asked[0][0], entry.position, `${entry.title}: the unedited table first`);
    assert.ok(asked.every(pass => pass.length <= PASS), entry.title);
    if (entry.id !== 's3-4') { assert.deepEqual(unanswered, [], entry.title); continue; }
    // The prototype paired the eight changes nearest to flipping South 3 #4,
    // one of them a wall that version 1 leaves alone, so the eighth it pairs
    // instead, the west wind swapped for a 4 characters, was never paired.
    const held = table => table.players[table.seat].hand;
    const count = (table, tile) => held(table).filter(kind => kind === tile).length;
    assert.equal(unanswered.length, 7);
    for (const table of unanswered) {
      assert.equal(count(table, '3z'), count(entry.position, '3z') - 1);
      assert.ok(count(table, '4m') > count(entry.position, '4m'));
    }
  }
  // The unedited table and every change version 1 makes of the prototype's
  // budget, in passes of eight: no wall, no dealer, no call taken back, no
  // score levelled that was level already or whose points do not share out
  // in hundreds (East 3 #5 and South 3 #4, a riichi stick on the table), no
  // riichi moved a discard earlier where the declaration's tile came from the
  // hand (the same two), no score changed at the game's first deal (East 1
  // #20) and no hand made East 1; then the score steps one at a time, and the
  // pairs.
  assert.deepEqual(passes, { 'e3-5': [8, 8, 8, 8, 4], 's3-4': [8, 8, 8, 7, 8, 8, 8, 4], 'e1-20': [8, 8, 7], 's1-9': [8, 8, 8, 5],
    's4-18': [8, 8, 8, 4], 'why7-1m': [8, 8, 8, 7, 1, 1, 1] });
  // What the button's range counts, given no engine to ask, against what the
  // search plans: first the unedited table and the single changes, the hand
  // swaps counted as many as it picks at most, nine, where it picked as few
  // as five; and room from the start for the 28 pairs it may try after them,
  // which the range counts at most.
  assert.deepEqual(planned, { 'e3-5': 36 + 28, 's3-4': 31 + 28, 'e1-20': 23 + 28, 's1-9': 29 + 28, 's4-18': 28 + 28, 'why7-1m': 31 + 28 });
  const first = { 'e3-5': 36, 's3-4': 33, 'e1-20': 27, 's1-9': 31, 's4-18': 30, 'why7-1m': 31, 'e2b-14': 0, 'why7-2s': 0 };
  for (const entry of questions) {
    assert.equal(tablesFirstAsked(entry), first[entry.id], entry.title);
    assert.equal(tablesAtMost(entry), first[entry.id] && first[entry.id] + 28, entry.title);
  }
});

test("a search's progress never runs back: its plan holds from the start room for what it may ask after the single changes, gives back what it did not ask, and never asks more than the button's range", async () => {
  const steps = {};
  for (const entry of searched) {
    const heard = [];
    const search = await whatWouldChangeItsMind({ ...entry, facts, ask: recordedAsk(entry).ask, onProgress: step => heard.push(step) });
    const shares = heard.map(({ asked, planned }) => asked / planned);
    steps[entry.id] = heard.map(({ asked, planned }) => `${asked}/${planned}`).join(' ');
    assert.ok(shares.every((share, at) => !at || share >= shares[at - 1]), `${entry.title}: ${steps[entry.id]}`);
    assert.deepEqual(heard.at(-1), { asked: search.asked, planned: search.asked }, `${entry.title}: every table planned asked at the end`);
    assert.ok(search.asked <= tablesAtMost(entry), `${entry.title}: ${search.asked} of at most ${tablesAtMost(entry)}`);
  }
  // East 3 #5, where single changes flip the choice, fills its bar early;
  // South 3 #4, whose pairs ask 59 tables, nearly twice what the button's
  // old estimate counted, fills it as it goes; and why-7's 1 characters
  // takes three score steps from the room.
  assert.equal(steps['e3-5'], '0/64 8/64 16/64 24/64 32/64 36/64 36/36');
  assert.equal(steps['s3-4'], '0/59 8/59 16/59 24/59 31/59 39/59 47/59 55/59 59/59');
  assert.equal(steps['why7-1m'], '0/59 8/59 16/59 24/59 31/59 32/59 33/59 34/59 34/34');
});

test('each change is grouped by what it did to the engine\'s facts, as the prototype grouped it', async () => {
  let compared = 0;
  for (const entry of searched) {
    const recorded = new Map(entry.record.answers.map(answer => [answer[0], answer[8]]));
    const { search } = await panel(entry);
    for (const change of [...search.singles, ...search.pairs].filter(asked => !asked.refused)) {
      assert.deepEqual(change.ideas, recorded.get(change.words), `${entry.title}: ${change.words}`);
      compared++;
    }
  }
  assert.equal(compared, 193);
});

test('nothing is searched where Trained AI chose by more than e^7 to one', async () => {
  for (const entry of questions) {
    const { margin } = judge(entry.weights, entry.played, entry.preferred);
    assert.equal(searchOffered(entry), margin >= MARGIN_FLOOR, entry.title);
    if (searchOffered(entry)) continue;
    const { search, asked } = await panel(entry);
    assert.deepEqual([search.status, asked], ['too sure', []], entry.title);
  }
  assert.deepEqual(questions.filter(entry => !searchOffered(entry)).map(entry => entry.id), ['e2b-14', 'why7-2s']);
});

test('a Why line takes a furiten away by changing every copy the player threw', async () => {
  // The player had thrown the 8 bamboo as their second and their ninth
  // discards: either changed alone, the other still makes a wait on it
  // furiten, so the probe changes both, in one table asked beside the
  // unedited one. Trained AI's lead of 9.2 falls to 0.4, which halves it.
  const entry = question('e2b-14'), { ask, asked, unanswered } = recordedAsk(entry), tables = [];
  const thrown = table => table.players[table.seat].discards.flatMap((discard, index) => (discard.tile === '8s' ? [index] : []));
  assert.deepEqual(thrown(entry.position), [1, 8]);
  const why = await explainWhy({ ...entry, facts, ask: positions => { tables.push(...positions); return ask(positions); } });
  assert.deepEqual([why.verdict, asked, unanswered], ['explained', [2], []]);
  assert.deepEqual([tables[0], thrown(tables[1])], [entry.position, []]);
});

test('pairs are tried only where no single change flips the choice', async () => {
  for (const entry of searched) {
    const { search } = await panel(entry);
    const flipped = search.singles.some(change => change.flips && change.cost === 1 && !change.removesPreferred);
    assert.equal(search.pairs.length, flipped ? 0 : 28, entry.title);
    assert.equal(flipped, entry.id !== 's3-4', entry.title);
  }
});

test('the score at which Trained AI changes its mind is found by halving between the steps asked', async () => {
  for (const entry of searched) {
    const { search } = await panel(entry);
    assert.deepEqual(search.thresholds.map(({ label, at, tried }) => ({ label, at, asked: tried })),
      entry.record.thresholds.map(({ label, at, asked }) => ({ label, at, asked })), entry.title);
  }
  // Why-7's 1 characters: 8,000 points more flips it, so 4,000, 6,000 and
  // 7,000 are asked, and 7,000 is the first that flips it.
  assert.equal((await panel(question('why7-1m'))).search.thresholds[0].at, 7000);
});

test('a search stops at the unedited table where Trained AI does not answer it as the review recorded', async () => {
  // A review recorded with another table's weights: those Trained AI gave
  // East 3 #5 with the dora indicator moved to the 8 bamboo.
  const entry = question('e3-5');
  const moved = entry.record.answers.find(answer => answer[0].startsWith('the dora indicator had been the 8 bamboo'));
  const weights = [{ kind: 'discard', tile: '3z', weight: moved[1] }, { kind: 'discard', tile: '7m', weight: moved[2] }];
  const { why, search, asked } = await panel(entry, { weights });
  assert.equal(search.status, 'unlike the review');
  assert.deepEqual(asked, [PASS], 'nothing is asked after the first pass');
  assert.deepEqual(saidLines(search.said), ['Asked again, Trained AI answered this table differently, so no change was tried.']);
  // The Why line's probe asked the same table: it says its fact, on the
  // review's weights, not checked, and why in its tooltip.
  assert.equal(why.verdict, 'unlike the review');
  assert.deepEqual(saidLines(why), ['Not checked: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 2.0% on tiles you had already thrown.']);
  assert.equal(why.check, 'Asked again, Trained AI answered this table differently, so the check was not run.');
});

test('a Why line whose check cannot ask Trained AI says its fact all the same, not checked, with why in its tooltip', async () => {
  const entry = question('e3-5');
  const failing = await explainWhy({ ...entry, facts, ask: async () => { throw new Error('The trained opponent did not answer in time'); } });
  assert.deepEqual(saidLines(failing), ['Not checked: You had already thrown a 7 characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles you had already thrown.']);
  assert.deepEqual([failing.verdict, failing.failure, failing.check],
    ['not checked', 'The trained opponent did not answer in time', 'The check was not run: The trained opponent did not answer in time.']);
  // A line stopped while it asks is stopped, not said.
  const stop = new AbortController();
  await assert.rejects(explainWhy({ ...entry, facts, signal: stop.signal,
    ask: async () => { stop.abort(); throw new DOMException('Asking stopped', 'AbortError'); } }), { name: 'AbortError' });
});

test("a Why line that asked nothing is said as not checked, its fact kept, where a search finds that Trained AI answers the decision's table otherwise", async () => {
  // South 1 #9's line is explained without asking: its fact is the distance,
  // which has no change to try.
  const asked = [], why = await explainWhy({ ...question('s1-9'), facts, ask: async positions => { asked.push(...positions); return []; } });
  assert.deepEqual([why.verdict, asked], ['explained', []]);
  const unchecked = notChecked(why);
  assert.deepEqual(saidLines(unchecked), ["Not checked: Your 6 circles leaves the hand 5 from waiting; Trained AI's 3 bamboo leaves it 4 from waiting. Trained AI puts 98.3% on tiles that keep the hand 4 from waiting."]);
  assert.deepEqual([unchecked.verdict, unchecked.check], ['unlike the review', 'Asked again, Trained AI answered this table differently, so the check was not run.']);
  // A line with a probe keeps its fact and its weight, not what the probe
  // found: South 3 #4's.
  assert.deepEqual(saidLines(notChecked((await panel(question('s3-4'))).why)), ["Not checked: Against North's riichi your 7 characters is a tile with no suji; Trained AI's white dragon cannot deal in. Trained AI puts 98.3% on tiles that cannot deal in."]);
});

test('a search stopped by its signal asks nothing more, and says it was stopped', async () => {
  const entry = question('e3-5'), { ask } = recordedAsk(entry), stop = new AbortController();
  let passes = 0;
  await assert.rejects(whatWouldChangeItsMind({ ...entry, facts, signal: stop.signal,
    ask: async positions => { passes++; const answers = await ask(positions); stop.abort(); return answers; } }), { name: 'AbortError' });
  assert.equal(passes, 1);
});

test('every change makes a table the rules accept, changed as its words say and in nothing else', () => {
  const where = (before, after, path = '') => {
    if (typeof before !== 'object' || before === null) return JSON.stringify(before) === JSON.stringify(after) ? [] : [path];
    return [...new Set([...Object.keys(before), ...Object.keys(after)])].flatMap(key => where(before[key], after[key], `${path}.${key}`));
  };
  const allowed = {
    hand: /^\.(players\.\d\.hand\.\d+|drawn)$/,
    riichi: /^\.(players\.\d\.(discards\.\d+\.riichi|riichi|ippatsu|score)|riichi_sticks)$/,
    river: /^\.players\.\d\.discards\.\d+\.tile$/,
    dora: /^\.indicators\.0$/,
    scores: /^\.players\.\d\.score$/,
    round: /^\.(round|kyoku)$/,
  };
  let tables = 0;
  for (const entry of questions) {
    const before = structuredClone(entry.position), sum = table => table.players.reduce((total, player) => total + player.score, 0);
    for (const edit of singleEdits(entry.position, entry)) {
      const table = applyEdits(entry.position, edit), changed = where(entry.position, table);
      assert.ok(changed.length && changed.every(path => allowed[edit.family].test(path)), `${entry.title}: ${edit.words} changed ${changed}`);
      if (edit.family === 'river') assert.equal(changed.length, 1, edit.words);
      if (edit.family === 'scores') assert.equal(sum(table), sum(entry.position), edit.words);
      // A step up is taken from the others, and a step down given to them.
      if (edit.family === 'scores' && edit.param != null) {
        assert.equal(table.players[table.seat].score - entry.position.players[table.seat].score, edit.param, edit.words);
        assert.match(edit.words, edit.param > 0 ? /points more, taken evenly from the others$/ : /points fewer, given evenly to the others$/);
      }
      assert.doesNotThrow(() => facts(table), `${entry.title}: ${edit.words}`);
      tables++;
    }
    assert.deepEqual(entry.position, before, `${entry.title}: the unedited table is left as it was`);
  }
  assert.ok(tables > 2000, `only ${tables} tables`);
  // North's riichi taken back in South 3 #4 hands back the thousand it bet.
  const entry = question('s3-4'), north = entry.position.players[3];
  const undeclared = applyEdits(entry.position, singleEdits(entry.position, entry).find(edit => edit.words === 'North had not declared riichi'));
  assert.deepEqual([undeclared.players[3].riichi, undeclared.players[3].score, undeclared.riichi_sticks],
    ['none', north.score + 1000, entry.position.riichi_sticks - 1]);
});

test('a search stops where the unedited table\'s weights stand further from the review\'s than rounding, and goes on within it', async () => {
  // The review's own weights for East 3 #5 with the west wind's moved: by
  // 2e-4 Trained AI's answer to the unedited table is not the review's, and
  // by 5e-5 it is, a forward pass beside other tables rounding no further.
  const entry = question('e3-5');
  const moved = by => entry.weights.map(move => (move.tile === '3z' ? { ...move, weight: move.weight + by } : move));
  const far = await panel(entry, { weights: moved(2e-4) });
  assert.deepEqual([far.search.status, far.search.unlike, far.asked], ['unlike the review', 'discard 3z', [PASS]]);
  const near = await panel(entry, { weights: moved(5e-5) });
  assert.deepEqual([near.search.status, near.unanswered], ['searched', []]);
});

/** The tables of a finished game's decisions where the player throws a tile
 * and has another to throw, by the hand's name and the decision's number
 * ("South 3 #25"), each with the tiles the player could throw. */
function discardTables(snapshot) {
  const tables = new Map();
  eachDecision(Game, snapshot, ({ hand, index, engine, position }) => {
    const discards = engine.review_choices(index).filter(choice => choice.kind === 'discard').map(choice => choice.tile);
    if (position.phase === 'act' && discards.length > 1) tables.set(`${hand.name} #${index + 1}`, { position, discards });
  });
  return tables;
}
const reviewCheck = discardTables(reviewCheckGame(Game)), clubFive = discardTables(clubAdvisedGame(Game, 5));
const clubSixteen = discardTables(clubAdvisedGame(Game, 16));
const asked = (P, A) => ({ played: { kind: 'discard', tile: P }, preferred: { kind: 'discard', tile: A } });
const honour = tile => tile[1] === 'z';

test('no change, alone or paired, has a seat in riichi throw a tile from its hand after its declaration', () => {
  // Every discard after a declaration is the tile just drawn. Only a riichi
  // change moves a declaration, the test above holding every other change to
  // tiles and scores, and a riichi moved a discard earlier keeps the rule
  // only where the tile it was declared with had been drawn. No opponent in
  // these games declared with the tile it drew, so none is moved earlier.
  const fromHand = table => table.players.some(player => player.riichi !== 'none'
    && player.discards.slice(player.discards.findIndex(discard => discard.riichi) + 1).some(discard => !discard.drawn));
  let tables = 0;
  for (const [game, decisions] of [['The review check\'s game', reviewCheck], ['Club seed 5', clubFive]]) {
    for (const [where, { position, discards }] of decisions) {
      assert.ok(!fromHand(position), `${game}, ${where}`);
      const riichi = singleEdits(position, asked(discards[0], discards[1])).filter(edit => edit.family === 'riichi');
      assert.ok(!riichi.some(edit => /one discard earlier/.test(edit.words)), `${game}, ${where}`);
      riichi.forEach((edit, at) => {
        for (const other of [null, ...riichi.slice(at + 1).filter(next => !changesMeet(next, edit))]) {
          assert.ok(!fromHand(applyEdits(position, edit, ...(other ? [other] : []))), `${game}, ${where}: ${edit.words}${other ? `, and ${other.words}` : ''}`);
          tables++;
        }
      });
    }
  }
  assert.equal(tables, 1072);
  // East 3 #5's North declared with the 7 circles from its hand. Had it
  // declared with the 7 circles just drawn, its riichi could have come a
  // discard earlier, the 7 circles then thrown as drawn after it.
  const entry = question('e3-5'), drawn = structuredClone(entry.position), north = drawn.players[3];
  assert.deepEqual(north.discards.map(discard => [discard.tile, discard.drawn, discard.riichi]).slice(-3), [['2m', true, false], ['7p', false, true], ['6p', true, false]]);
  assert.ok(!singleEdits(entry.position, entry).some(edit => /one discard earlier/.test(edit.words)));
  north.discards.at(-2).drawn = true;
  const earlier = singleEdits(drawn, entry).find(edit => /one discard earlier/.test(edit.words));
  assert.equal(earlier.words, 'North had declared riichi one discard earlier, on the 2 characters');
  const table = applyEdits(drawn, earlier);
  assert.ok(!fromHand(table));
  assert.doesNotThrow(() => facts(table));
});

test('a change touches every discard it changes, so that no pair of changes names one discard twice', () => {
  // East 3 #5's budget holds South declaring riichi with its last discard,
  // the west wind, and that discard made a 1 characters: paired, they would
  // say the riichi was declared with the west wind on a table where it was
  // declared with the 1 characters.
  const entry = question('e3-5'), singles = singleEdits(entry.position, entry), named = words => singles.find(edit => edit.words === words);
  assert.ok(changesMeet(named('South had declared riichi with their last discard, the west wind'), named('South\'s fourth discard had been a 1 characters, not the west wind')));
  // Each riichi and river change of every discard decision of three whole
  // games, the first tile the player could throw against each other: the
  // discards it changes, a declaration moved or a tile replaced, are among
  // what it touches, and two that may be paired change none in common. No
  // other change touches a river (the test of every change above).
  const changed = (position, edit) => {
    const table = applyEdits(position, edit);
    return position.players.flatMap((player, seat) => player.discards.flatMap((discard, index) =>
      (JSON.stringify(discard) === JSON.stringify(table.players[seat].discards[index]) ? [] : [`river:${seat}:${index}`])));
  };
  let edits = 0, pairs = 0;
  for (const [game, decisions] of [['The review check\'s game', reviewCheck], ['Club seed 5', clubFive], ['Club seed 16', clubSixteen]]) {
    for (const [where, { position, discards }] of decisions) {
      for (const A of discards.slice(1)) {
        const rivers = singleEdits(position, asked(discards[0], A)).filter(edit => ['riichi', 'river'].includes(edit.family))
          .map(edit => ({ edit, changes: changed(position, edit) }));
        for (const { edit, changes } of rivers) {
          assert.ok(changes.every(thing => edit.touches.includes(thing)), `${game}, ${where}: ${edit.words} changes ${changes}`);
          edits++;
        }
        rivers.forEach((one, at) => {
          for (const other of rivers.slice(at + 1).filter(next => !changesMeet(next.edit, one.edit))) {
            assert.ok(!one.changes.some(thing => other.changes.includes(thing)), `${game}, ${where}: ${one.edit.words}, and ${other.edit.words}`);
            pairs++;
          }
        });
      }
    }
  }
  assert.deepEqual([edits, pairs], [11838, 20465]);
});

const threeGames = [['The review check\'s game', reviewCheck], ['Club seed 5', clubFive], ['Club seed 16', clubSixteen]];

test('no change, alone or paired, moves points at the game\'s first deal, and none makes a later hand the first', () => {
  // At East 1 with no counters nobody has scored: each player holds the
  // starting 30,000, less a riichi bet this hand. Score steps were asked
  // there of points nobody had won, Club seed 16's East 1 #7 "from 6,000
  // points fewer" of 32,000, 32,000, 32,000 and 24,000, and "this had been
  // East 1" of later hands' scores. Only riichi and score changes move
  // points (the test of every change above), so a pair that moves any holds
  // one of them.
  const unscored = table => table.players.every(player => player.score === 30000 - (player.riichi === 'none' ? 0 : 1000));
  let decisions = 0, tables = 0;
  for (const [game, decisionsOf] of threeGames) {
    for (const [where, { position, discards }] of decisionsOf) {
      const edits = singleEdits(position, asked(discards[0], discards[1]));
      assert.ok(!edits.some(edit => /this had been East 1/.test(edit.words)), `${game}, ${where}`);
      if (position.round !== 0 || position.kyoku !== 1 || position.counters) continue;
      assert.ok(unscored(position), `${game}, ${where}`);
      const moving = edits.filter(edit => ['riichi', 'scores'].includes(edit.family));
      moving.forEach((edit, at) => {
        for (const other of [null, ...moving.slice(at + 1).filter(next => !changesMeet(next, edit))]) {
          assert.ok(unscored(applyEdits(position, edit, ...(other ? [other] : []))), `${game}, ${where}: ${edit.words}${other ? `, and ${other.words}` : ''}`);
          tables++;
        }
      });
      decisions++;
    }
  }
  assert.deepEqual([decisions, tables], [37, 152]);
});

test('at East 1 with counters, the scores are changed only as far as the dealer\'s wins and the draws can have moved them', async () => {
  // The review check's East 1 #1 typed in again two counters on: the dealer
  // won 12,000 from the player, West, then waited at a draw with South, the
  // player and North paying 1,500 each. The dealer has paid nothing, and
  // South and North have been paid 1,500 at most. A step up for the player
  // takes points from them all; a step down would give South more than the
  // draws can have paid it.
  const position = structuredClone(reviewCheck.get('East 1 #1').position);
  Object.assign(position, { counters: 2 });
  [43500, 31500, 16500, 28500].forEach((score, seat) => { position.players[seat].score = score; });
  assert.equal(position.seat, 2);
  assert.doesNotThrow(() => facts(position));
  const [P, A] = reviewCheck.get('East 1 #1').discards;
  assert.deepEqual(singleEdits(position, asked(P, A)).filter(edit => edit.family === 'scores').map(edit => edit.words), [
    'you had had 8,000 points more, taken evenly from the others',
    'you had had 16,000 points more, taken evenly from the others',
    'everyone had had the same score',
  ]);
  const down = applyEdits(position, scoreEdit(position, -8000));
  assert.deepEqual([firstHandGives(position), firstHandGives(down), down.players[1].score], [true, false, 34200]);
  // Where no score change moves Trained AI, stood in for, the line says so
  // without the reach of steps it never asked.
  const asking = standInQuestion(position, P, A);
  const { said } = await whatWouldChangeItsMind({ ...asking, facts, ask: standInAsk(asking, facts) });
  assert.equal(said.still, 'Not moved by the score or the round.');
});

test('everyone is made level only where the points on the table share out in hundreds, and then has one score', () => {
  // With a riichi stick on the table 119,000 points do not: Club seed 16's
  // South 3 #21 was asked "if everyone had had the same score" of 29,800,
  // 29,800, 29,600 and 29,800, a table with a rank of its own.
  let level = 0, uneven = 0;
  for (const [game, decisionsOf] of threeGames) {
    for (const [where, { position, discards }] of decisionsOf) {
      const edit = singleEdits(position, asked(discards[0], discards[1])).find(each => each.words === 'everyone had had the same score');
      const scores = position.players.map(player => player.score), points = scores.reduce((sum, score) => sum + score, 0);
      if (new Set(scores).size === 1 || points % 400) {
        assert.equal(edit, undefined, `${game}, ${where}`);
        uneven += new Set(scores).size > 1;
        continue;
      }
      const table = applyEdits(position, edit);
      assert.deepEqual(table.players.map(player => player.score), Array(4).fill(points / 4), `${game}, ${where}`);
      level++;
    }
  }
  assert.deepEqual([level, uneven], [198, 62]);
});

/** What the panel says a change of a question's table did, or a pair of
 * changes, named by their words, in its brackets, where Trained AI, stood in
 * for, throws the player's tile on that table and on no other
 * (searchWhereOnly). */
async function saidOf(question, ...words) {
  const singles = singleEdits(question.position, question);
  const { said } = await searchWhereOnly(question, facts, ...words.map(text => singles.find(edit => edit.words === text)));
  return saidOfChange([said.detail, ...said.items].find(line => words.every(text => line.includes(text))));
}
/** The question at a decision of `tables` (`where`), the player's tile `P`
 * against Trained AI's `A`, its weights stood in for. */
const questionAt = (tables, where, [P, A]) => standInQuestion(tables.get(where).position, P, A);

test('a change is said as the engine\'s facts stand after it: what is safer, who is still in riichi and how many riichi stand', async () => {
  // A 4 bamboo in East's river puts the player's 1 bamboo on suji, while the
  // 2 characters it replaces were safe against East's riichi.
  assert.equal(await saidOf(questionAt(reviewCheck, 'East 2 · 1 honba #17', ['1s', '2m']), 'East\'s 14th discard had been a 4 bamboo, not the 2 characters'),
    'the 2 characters no longer safe and your 1 bamboo safer');
  // South and North in riichi: either taken back leaves the other standing,
  // and a riichi from West would be the third.
  assert.equal(await saidOf(questionAt(clubFive, 'South 3 #25', ['1m', '6m']), 'South had not declared riichi'), 'only North in riichi and the 6 characters safer');
  assert.equal(await saidOf(questionAt(clubFive, 'South 3 #25', ['1m', '6m']), 'West had declared riichi with their last discard, the 5 bamboo'),
    'a third riichi and your 1 characters no longer safe');
  // Nobody in riichi at the first discard: two declared at once are two, said
  // of the pair that alone makes Trained AI throw the player's tile.
  assert.equal(await saidOf(questionAt(reviewCheck, 'East 1 #1', ['8s', '3m']), 'East had declared riichi with their last discard, the 9 circles',
    'South had declared riichi with their last discard, the north wind'), 'two riichi, the 3 characters less safe and your 8 bamboo less safe');
  // North's riichi, the only one, taken back, on the recorded question's own
  // weights.
  assert.equal(await saidOf(question('s3-4'), 'North had not declared riichi'), 'nobody in riichi');
});

test('only an honour with every copy in sight is said to be one nobody can wait on, since a sequence waits on a suit tile it holds none of', async () => {
  // Every question of three whole games where either tile has every copy in
  // sight, Trained AI's weights stood in for, which set only the figures
  // said. A suit tile once was said to be dead in 1,351 of them: "Every copy
  // of the 1 bamboo is in sight, so nobody can be waiting on it", though a
  // 2-3 waits on it holding none.
  let lines = 0, dead = 0;
  for (const [game, decisions] of [['The review check\'s game', reviewCheck], ['Club seed 5', clubFive], ['Club seed 16', clubSixteen]]) {
    for (const [where, { position, discards }] of decisions) {
      const unseen = Object.fromEntries(facts(position).discards.map(discard => [discard.tile, discard.unseen]));
      for (const [P, A] of discards.flatMap(mine => discards.filter(other => other !== mine).map(other => [mine, other]))) {
        if (unseen[P] && unseen[A]) continue;
        const asking = standInQuestion(position, P, A);
        const why = await explainWhy({ ...asking, facts, ask: async tables => tables.map(() => ({ moves: asking.weights })) });
        const said = `${saidLines(why)[0]} ${why.check}`, at = `${game}, ${where}: ${P} against ${A}: ${said}`;
        assert.doesNotMatch(said, /very copy of (the |your )?\d/, at);
        if (!honour(P) || !honour(A)) assert.doesNotMatch(said, /both tiles/, at);
        dead += /very copy of (the |your )?[a-z]/.test(said);
        lines++;
      }
    }
  }
  assert.deepEqual([lines, dead], [3626, 917]);
});

test('a dora indicator moved onto the last unseen copy of an honour in question makes it dead, said under what is safe', async () => {
  // The review check's East 1 #6, nobody in riichi: the player threw the
  // west wind, one copy of it unseen, and Trained AI the 7 bamboo. With the
  // west wind for the indicator every west wind is in sight, so the change
  // is said under what is safe, the dora it brings beside it; it was said
  // under a hand worth more, the west wind's death left unsaid.
  const asking = questionAt(reviewCheck, 'East 1 #6', ['3z', '7s']);
  const words = 'the dora indicator had been the west wind, not the north wind, making the north wind dora';
  const { said } = await searchWhereOnly(asking, facts, singleEdits(asking.position, asking).find(edit => edit.words === words));
  const line = said.items.find(item => item.includes(words));
  assert.match(line, /^What is safe \(1 of \d+ tried\): /);
  assert.equal(saidOfChange(line), '1 more dora in your hand and every west wind in sight');
});

test('a dora is probed by moving the first indicator only where that leaves the two tiles worth the same as dora', async () => {
  // The review check's East 2 · 1 honba #5: the indicators are the 5 circles
  // and, after a quad, the 2 bamboo. The 6 circles is dora by the first, and
  // moving it takes that away; the 3 bamboo is dora by the second, which
  // moving the first leaves standing, so its Why line asks nothing. It once
  // said "If the dora indicator had been the 5 characters, not the 5 circles,
  // making the 6 characters dora, it still does not choose yours" of a table
  // where the 3 bamboo was dora still. Trained AI's weights stood in for set
  // only the figures said.
  const { position } = reviewCheck.get('East 2 · 1 honba #5');
  assert.deepEqual(position.indicators, ['5p', '2s']);
  const why = async (P, A) => {
    const asking = standInQuestion(position, P, A), tables = [];
    const said = await explainWhy({ ...asking, facts, ask: async positions => { tables.push(...positions); return positions.map(() => ({ moves: asking.weights })); } });
    return { said: saidLines(said)[0], tables };
  };
  const first = await why('6p', '7p');
  assert.match(first.said, /^Partly: Your 6 circles is dora\. If the dora indicator had been the 1 characters, not the 5 circles, making the 2 characters dora, /);
  assert.deepEqual([first.tables.length, facts(first.tables[1]).discards.find(discard => discard.tile === '6p').dora], [2, 0]);
  const second = await why('3s', '2m');
  assert.deepEqual([second.said, second.tables], ['Partly: Your 3 bamboo is dora.', []]);
});

test('a pair of hand swaps that moves the hand two steps says so', async () => {
  // The review check's East 1 #14, the player's red dragon against the 3
  // characters: the north wind made an 8 circles, or the 5 bamboo a 7
  // bamboo, each brings the hand a step closer once the red dragon is
  // thrown, and the two together leave it waiting, which was said as "a step
  // closer to waiting".
  const asking = questionAt(reviewCheck, 'East 1 #14', ['7z', '3m']), swaps = ['your north wind had been an 8 circles', 'your 5 bamboo had been a 7 bamboo'];
  const singles = singleEdits(asking.position, asking), distance = (...words) => facts(applyEdits(asking.position,
    ...words.map(text => singles.find(edit => edit.words === text)))).discards.find(discard => discard.tile === '7z').shanten;
  assert.deepEqual([distance(), distance(swaps[0]), distance(swaps[1]), distance(...swaps)], [2, 1, 1, 0]);
  assert.equal(await saidOf(asking, ...swaps), 'your hand waiting');
});

/** South 1 #9's search, Trained AI throwing the 6 circles on a table where
 * `flips(gain)` holds of the points the player gained, the scores not
 * levelled, as the recorded run answered the dora change that makes it throw
 * them; and answering every other table as it answered the unedited one, so
 * that the score alone moves it. */
async function scoreSearch(flips) {
  const entry = question('s1-9'), { answers } = recordedAsk(entry), seat = entry.position.seat;
  const dora = singleEdits(entry.position, entry).find(edit => edit.words.startsWith('the dora indicator had been the 3 bamboo,'));
  const thrown = answers.get(JSON.stringify(applyEdits(entry.position, dora)));
  const gain = table => table.players[seat].score - entry.position.players[seat].score;
  const level = table => new Set(table.players.map(player => player.score)).size === 1;
  return whatWouldChangeItsMind({ ...entry, facts, ask: async positions => positions.map(table => (flips(gain(table)) && !level(table) ? thrown
    : { moves: entry.weights })) });
}

test('a score step past 8,000 points that alone makes it throw the player\'s tile is the table\'s finding, and no pair is tried', async () => {
  const found = await scoreSearch(gain => gain >= 10000);
  assert.deepEqual(found.thresholds.map(({ label, at }) => [label, at]), [['ahead', 10000]]);
  assert.deepEqual(saidLines(found.said), [
    'One change and it throws your 6 circles:',
    '- The table: from 10,000 points more: your 6 circles 60.7%, the 3 bamboo 20.3%.',
    'Not moved by the round.',
  ]);
  assert.deepEqual(found.pairs, []);
});

test('the table\'s line gives the score at which Trained AI changes its mind each way it was found', async () => {
  // 8,000 points fewer flips it, which once said only the step found ahead.
  const found = await scoreSearch(gain => gain >= 10000 || gain <= -5000);
  assert.deepEqual(found.thresholds.map(({ label, at }) => [label, at]), [['ahead', 10000], ['behind', -5000]]);
  assert.deepEqual(saidLines(found.said), [
    'One change and it throws your 6 circles:',
    '- The table: from 10,000 points more: your 6 circles 60.7%, the 3 bamboo 20.3%; from 5,000 points fewer: your 6 circles 60.7%, the 3 bamboo 20.3%.',
    'Not moved by the round.',
  ]);
});

test('a quad is named a kan of its tile, as the hand\'s list of decisions names it', () => {
  // The quads open at two decisions of Club-advised games: green dragons
  // added to a pon, and four green dragons held.
  const kans = [];
  for (const [seed, name, number] of [[5, 'South 3', 20], [16, 'South 1', 5]]) {
    eachDecision(Game, clubAdvisedGame(Game, seed), ({ hand, index, engine }) => {
      if (hand.name === name && index === number - 1) kans.push(...engine.review_choices(index).filter(choice => /kan$/.test(choice.kind)));
    });
  }
  assert.deepEqual(kans.map(move => [move.kind, moveWords(move)]), [['extended-kan', 'a kan of the green dragon'], ['concealed-kan', 'a kan of the green dragon']]);
  assert.deepEqual([{ kind: 'discard', tile: '2p' }, { kind: 'riichi', tile: null }, { kind: 'tsumo', tile: null }].map(moveWords), ['the 2 circles', 'riichi', 'the win']);
});

test('every copy of a tile the player threw is changed at once to take a furiten away, on a table the rules accept', () => {
  // Each copy becomes the same tile. Where every tile of the other suits is
  // in sight it is one with room for them all: in Club seed 16's South 3 #20
  // the player had thrown two 1 circles, and the 1 characters, three of them
  // in sight, once took both, a fifth copy the rules refuse; the 2
  // characters, two in sight, take them now.
  const crowded = [];
  let tables = 0;
  for (const [game, decisions] of [['The review check\'s game', reviewCheck], ['Club seed 5', clubFive], ['Club seed 16', clubSixteen]]) {
    for (const [where, { position, discards }] of decisions) {
      const river = position.players[position.seat].discards, seen = visibleCounts(position);
      for (const tile of discards) {
        const copies = river.filter(discard => discard.tile === tile);
        if (copies.length < 2 || copies.some(discard => discard.claimed)) continue;
        const edits = singleEdits(position, asked(discards.find(other => other !== tile), tile))
          .filter(edit => edit.family === 'river' && edit.seat === position.seat && edit.from === tile);
        const [into, ...others] = new Set(edits.map(edit => edit.into)), table = applyEdits(position, ...edits);
        assert.deepEqual([edits.length, others], [copies.length, []], `${game}, ${where}: the ${tile}`);
        assert.ok(!table.players[table.seat].discards.some(discard => discard.tile === tile), `${game}, ${where}: the ${tile}`);
        assert.doesNotThrow(() => { facts(table); new PhysicalAnalysis(table).free(); }, `${game}, ${where}: the ${tile} made ${into}`);
        if (seen[into]) crowded.push(`${game}, ${where}: ${copies.length} copies of ${tile} made ${into}, ${seen[into]} in sight`);
        tables++;
      }
    }
  }
  assert.deepEqual([tables, crowded], [14, ['Club seed 16, South 3 #20: 2 copies of 1p made 2m, 2 in sight']]);
});

test('a Why line takes every riichi back at once, said in one clause', async () => {
  // Club seed 16's South 3 #26, East and West in riichi: the player threw the
  // 5 circles, and Trained AI the 7 characters, which cannot deal in; with
  // neither riichi it would throw the 5 circles. Trained AI answers as it
  // answered both tables in the page's runtime, read as the page reads it.
  const recorded = JSON.parse(readFileSync(new URL('./fixtures/review-insight-two-riichi.json', import.meta.url), 'utf8'));
  const answers = new Map([recorded.unedited, recorded.probe].map(({ digest, weights }) => [digest, weights]));
  // Asked as the page asks, a table Trained AI hesitates over again of the
  // renamed game, which answers here as the recorded answer renamed.
  const network = mirroringNetwork((planes, mask) => ({ weights: answers.get(planesDigest(planes, mask)) }));
  const ask = trainedAsk({ PhysicalAnalysis, analyzeBatch: network.analyzeBatch });
  const { position } = clubSixteen.get(`${recorded.hand} #${recorded.decision}`), [{ moves: weights }] = await ask([position]);
  assert.deepEqual(network.requests, [{ size: 1, renamed: false }, { size: 1, renamed: true }], 'at 79.3% the unedited table is asked again');
  const why = async played => {
    const tables = [];
    const said = await explainWhy({ position, ...asked(played, '7m'), weights, facts, ask: positions => { tables.push(...positions); return ask(positions); } });
    return { said: saidLines(said), tables };
  };
  const thrown = await why('5p');
  assert.deepEqual(thrown.said, ['Partly: Against East and West\'s riichi your 5 circles is a middle tile on half suji; Trained AI\'s 7 characters cannot deal in. Trained AI puts 79.3% on tiles that cannot deal in.']);
  const [unedited, probe] = thrown.tables;
  assert.deepEqual([unedited, probe.players.map(player => player.riichi), probe.riichi_sticks], [position, ['none', 'none', 'none', 'none'], position.riichi_sticks - 2]);
  // Had the player thrown the 6 characters, taking both riichi back would not
  // have halved Trained AI's lead over it, and the line says so.
  assert.deepEqual((await why('6m')).said, ['Partly: Against East and West\'s riichi your 6 characters is a middle tile with no suji; Trained AI\'s 7 characters cannot deal in. Trained AI puts 79.3% on tiles that cannot deal in. If East and West had not declared riichi, it still does not choose yours: the 5 circles 92.7%, your 6 characters <0.1%, the 7 characters 0.5%.']);
});

test('a Why line weighs a safety fact on the tiles as safe as Trained AI\'s or safer, where its own tile can deal in', async () => {
  // The review check's East 3 #4, North in riichi: the player threw the 7
  // characters, a tile with no suji, and Trained AI the south wind, an honour
  // nobody has thrown, which can deal in. Its weight was once counted on the
  // tiles that cannot deal in alone, none of which it chose: "Trained AI puts
  // 0.0% on tiles that cannot deal in." Trained AI answers as it answered the
  // table, and the table without North's riichi, in the page's runtime.
  const recorded = JSON.parse(readFileSync(new URL('./fixtures/review-insight-honour-against-riichi.json', import.meta.url), 'utf8'));
  const answers = new Map([recorded.unedited, recorded.probe].map(({ digest, weights }) => [digest, weights]));
  // Asked as the page asks, a table Trained AI hesitates over again of the
  // renamed game, which answers here as the recorded answer renamed.
  const network = mirroringNetwork((planes, mask) => ({ weights: answers.get(planesDigest(planes, mask)) }));
  const ask = trainedAsk({ PhysicalAnalysis, analyzeBatch: network.analyzeBatch });
  const { position } = reviewCheck.get(`${recorded.hand} #${recorded.decision}`), [{ moves: weights }] = await ask([position]);
  const tables = [];
  const why = await explainWhy({ position, ...asked('7m', '2z'), weights, facts, ask: positions => { tables.push(...positions); return ask(positions); } });
  assert.deepEqual(saidLines(why), ['Why: Against North\'s riichi your 7 characters is a tile with no suji; Trained AI\'s south wind is an honour nobody has thrown. Trained AI puts 99.8% on tiles as safe as the south wind or safer. If North had not declared riichi: the 9 characters 56.8%, your 7 characters <0.1%, the south wind 1.3%.']);
  assert.deepEqual(tables.map(table => table.players.map(player => player.riichi)), [position.players.map(player => player.riichi), ['none', 'none', 'none', 'none']]);
  // Sure of the table, Trained AI asked it once; at 56.8% without the riichi,
  // it asked that table again, renamed, in a pass of its own.
  assert.deepEqual(network.requests, [{ size: 1, renamed: false }, { size: 2, renamed: false }, { size: 1, renamed: true }]);
});
