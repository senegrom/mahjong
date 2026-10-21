import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game, PhysicalAnalysis, review_facts } from '../src/wasm/riichi.js';
import { BATCH } from '../src/lib/game-review.js';
import { MARGIN_FLOOR, PASS, applyEdits, changeWords, explainWhy, judge, moveWords, saidLines, searchOffered, singleEdits,
  tablesFirstAsked, visibleCounts, whatWouldChangeItsMind } from '../src/lib/review-insight.js';
import { recordedAsk, recordedQuestions } from './fixtures/recorded-insight.js';
import { clubAdvisedGame, eachDecision, reviewCheckGame } from './fixtures/review-check-game.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const questions = recordedQuestions(Game, PhysicalAnalysis);
const question = id => questions.find(entry => entry.id === id);
const facts = review_facts;
const searched = questions.filter(entry => searchOffered(entry));

/** What version 1's panel says for each of the prototype's questions, from
 * its recorded answers (insight/recommendation/terse.txt): the Why line, then
 * the search's findings. Two lines say otherwise than terse.txt. East 3 #5
 * tries one change fewer of what is safe: North declared riichi with a tile
 * from its hand, so it cannot have declared a discard earlier, the tile
 * then following the declaration out of its hand. East 2 · 2 honba #14's
 * probe changes both of the player's 8 bamboo, since one left in the river
 * keeps a wait on it furiten; Trained AI was asked that table since. */
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
    'Not moved by the score (±16,000) or the round.',
  ],
  'e2b-14': [
    'Why: You had already thrown an 8 bamboo, so a wait on it would be furiten. Trained AI puts 99.2% on tiles you had already thrown. If your second discard had been a 1 characters, not the 8 bamboo, and your ninth discard had been a 1 characters, not the 8 bamboo: the 2 bamboo 66.4%, your 6 bamboo <0.1%, the 8 bamboo <0.1%.',
  ],
  's1-9': [
    'Why: Your 6 circles leaves the hand 5 from waiting; Trained AI\'s 3 bamboo leaves it 4 from waiting. Trained AI puts 98.3% on tiles that keep the hand 4 from waiting.',
    'One change and it throws your 6 circles:',
    '- A hand worth more (2 of 14 tried): if the dora indicator had been the 3 bamboo, not the 9 bamboo, making the 4 bamboo dora (1 more dora in your hand): your 6 circles 60.7%, the 3 bamboo 20.3%.',
    'Not moved by the score (±16,000) or the round.',
  ],
  's4-18': [
    'Partly: Your 8 circles leaves the hand 3 from waiting; Trained AI\'s 9 characters leaves it 2 from waiting. Trained AI puts 84.9% on tiles that keep the hand 2 from waiting. On your side: every copy of your 8 circles is in sight, so nobody can be waiting on it; one 9 characters is still unseen.',
    'One change and it throws your 8 circles:',
    '- What is safe (2 of 7 tried): if South had declared riichi with their last discard, the 8 bamboo (a riichi, the 9 characters less safe and your 8 circles safe): your 8 circles 98.3%, the 9 characters 1.5%.',
    'Not moved by the score (±16,000) or the round.',
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
  const passes = {};
  for (const entry of searched) {
    const asked = [], { ask, unanswered } = recordedAsk(entry);
    await whatWouldChangeItsMind({ ...entry, facts, ask: positions => { asked.push(positions); return ask(positions); } });
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
  // score levelled that was level already, and no riichi moved a discard
  // earlier where the declaration's tile came from the hand (East 3 #5 and
  // South 3 #4); then the score steps one at a time, and the pairs.
  assert.deepEqual(passes, { 'e3-5': [8, 8, 8, 8, 6], 's3-4': [8, 8, 8, 8, 1, 8, 8, 8, 4], 'e1-20': [8, 8, 8, 3], 's1-9': [8, 8, 8, 6],
    's4-18': [8, 8, 8, 5], 'why7-1m': [8, 8, 8, 8, 1, 1, 1] });
  // What the button's estimate counts before the search begins: the unedited
  // table and the single changes, the passes before any score step or pair.
  const first = { 'e3-5': 38, 's3-4': 33, 'e1-20': 27, 's1-9': 30, 's4-18': 29, 'why7-1m': 32, 'e2b-14': 0, 'why7-2s': 0 };
  for (const entry of questions) assert.equal(tablesFirstAsked({ ...entry, facts }), first[entry.id], entry.title);
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
  assert.equal(compared, 204);
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
  assert.deepEqual(saidLines(search.said), ['This table does not replay as the review asked it, so no change was tried.']);
  assert.equal(why.verdict, 'unlike the review');
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
        for (const other of [null, ...riichi.slice(at + 1).filter(next => next.key !== edit.key)]) {
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

test('a change is said as the engine\'s facts stand after it: what is safer, who is still in riichi and how many riichi stand', () => {
  const said = (tables, where, [P, A], ...words) => {
    const question = { position: tables.get(where).position, ...asked(P, A), facts }, singles = singleEdits(question.position, question);
    return changeWords(question, ...words.map(text => singles.find(edit => edit.words === text)));
  };
  // A 4 bamboo in East's river puts the player's 1 bamboo on suji, while the
  // 2 characters it replaces were safe against East's riichi.
  assert.deepEqual(said(reviewCheck, 'East 2 · 1 honba #17', ['1s', '2m'], 'East\'s 14th discard had been a 4 bamboo, not the 2 characters'),
    ['the 2 characters no longer safe', 'your 1 bamboo safer']);
  // South and North in riichi: either taken back leaves the other standing,
  // and a riichi from West would be the third.
  assert.deepEqual(said(clubFive, 'South 3 #25', ['1m', '6m'], 'South had not declared riichi'), ['only North in riichi', 'the 6 characters safer']);
  assert.deepEqual(said(clubFive, 'South 3 #25', ['1m', '6m'], 'West had declared riichi with their last discard, the 5 bamboo'),
    ['a third riichi', 'your 1 characters no longer safe']);
  // Nobody in riichi at the first discard: two declared at once are two.
  assert.deepEqual(said(reviewCheck, 'East 1 #1', ['8s', '3m'], 'East had declared riichi with their last discard, the 9 circles',
    'South had declared riichi with their last discard, the north wind'), ['two riichi', 'the 3 characters less safe', 'your 8 bamboo less safe']);
  // North's riichi, the only one, taken back.
  const entry = question('s3-4');
  assert.deepEqual(changeWords({ ...entry, facts }, singleEdits(entry.position, entry).find(edit => edit.words === 'North had not declared riichi')),
    ['nobody in riichi']);
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
