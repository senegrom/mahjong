/** Split the browser suite into shards, and run it over stand-in scripts,
 * without starting Chromium. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const suite = fileURLToPath(new URL('../scripts/browser-suite.mjs', import.meta.url));
const source = readFileSync(suite, 'utf8');
// The suite's table as its source has it: each script with its seconds.
const table = [...source.match(/const scripts = \[([\s\S]*?)\];/)[1].matchAll(/\['([^']+)', ([^\]]*)\]/g)]
  .map(([, script, seconds]) => [script, Number(seconds)]);

/** Runs the suite (or a copy of it) with BROWSER_SHARD as given, or unset. */
function run(args, shard, script = suite) {
  const env = { ...process.env };
  delete env.BROWSER_SHARD;
  if (shard !== undefined) env.BROWSER_SHARD = shard;
  return spawnSync(process.execPath, [script, ...args], { env, encoding: 'utf8' });
}
function list(shard, names = []) {
  const listed = run(['--list', ...names], shard);
  assert.equal(listed.status, 0, listed.stderr);
  return listed.stdout.split('\n').filter(Boolean);
}
/** A copy of the suite, its source as given, in a folder of its own. */
async function copyOfSuite(t, text = source) {
  const folder = await mkdtemp(join(tmpdir(), 'mahjong-suite-'));
  t.after(() => rm(folder, { recursive: true, force: true }));
  await mkdir(join(folder, 'scripts'));
  const copy = join(folder, 'scripts', 'browser-suite.mjs');
  await writeFile(copy, text);
  return { folder, copy };
}
/** A copy of the suite with stand-ins under its scripts' names: each exits
 * with the code given for its try, and writes its try's number into a picture
 * in test-results. */
async function standIns(t, exits) {
  const { folder, copy } = await copyOfSuite(t);
  for (const [name, codes] of Object.entries(exits)) {
    await writeFile(join(folder, 'scripts', name), `import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
const tries = existsSync('${name}.tries') ? Number(readFileSync('${name}.tries', 'utf8')) + 1 : 1;
writeFileSync('${name}.tries', String(tries));
mkdirSync('test-results', { recursive: true });
writeFileSync('test-results/${name}.png', 'try ' + tries);
console.log('ran ${name}, try ' + tries);
process.exitCode = ${JSON.stringify(codes)}[tries - 1];
`);
  }
  return { folder, copy, evidence: join(folder, 'test-results') };
}

test('every script in the table lands in exactly one shard, and without a shard all run in order', () => {
  const all = list();
  assert.equal(all.length, table.length, 'a line for each entry in the table');
  assert.deepEqual(all, table.map(([script]) => script));
  assert.equal(new Set(all).size, all.length);
  for (const name of all) assert.ok(existsSync(new URL(`../scripts/${name}`, import.meta.url)), name);
  assert.deepEqual(list('1/1'), all);
  for (const count of [2, 3, 4]) {
    const shards = Array.from({ length: count }, (_, index) => list(`${index + 1}/${count}`));
    assert.ok(shards.every(names => names.length), `each of ${count} shards runs something`);
    assert.deepEqual(shards.flat().toSorted(), all.toSorted(), `${count} shards`);
    for (const names of shards) assert.deepEqual(names, all.filter(name => names.includes(name)), 'in the suite\'s order');
  }
});

// Longest first, no shard of three takes longer than the longest script, or
// than an even third of the whole plus two thirds of the fourth longest: the
// last script put on the heaviest shard went to the lightest shard of the
// moment, and after the first three no script is longer than the fourth. The
// bound holds however the table is timed again. Shortest first, today's table
// would put 923 s in one shard, over the 735 s it allows.
test('no shard of three takes longer than splitting longest first allows', () => {
  const seconds = new Map(table), sorted = [...seconds.values()].toSorted((a, b) => b - a);
  const total = sorted.reduce((sum, value) => sum + value, 0);
  const bound = Math.max(sorted[0], total / 3 + (2 / 3) * (sorted[3] ?? 0));
  const loads = ['1/3', '2/3', '3/3'].map(shard => list(shard).reduce((sum, name) => sum + seconds.get(name), 0));
  assert.ok(Math.max(...loads) <= bound, `${loads.join(', ')} s in the shards, at most ${Math.round(bound)} s allowed`);
});

test('every entry in the table has a finite number of seconds, 0 or more', async t => {
  const { copy } = await copyOfSuite(t);
  for (const entry of ["['component-check.mjs']", "['component-check.mjs', -1]", "['component-check.mjs', NaN]",
    "['component-check.mjs', Infinity]", "['component-check.mjs', '43']", "['component-check.mjs', null]"]) {
    const text = source.replace("['component-check.mjs', 43]", entry);
    assert.notEqual(text, source);
    await writeFile(copy, text);
    const listed = run(['--list'], undefined, copy);
    assert.notEqual(listed.status, 0, entry);
    assert.match(listed.stderr, /Seconds must be a finite number, 0 or more, for component-check\.mjs/, entry);
  }
  await writeFile(copy, source.replace("['component-check.mjs', 43]", "['component-check.mjs', 0]"));
  assert.equal(run(['--list'], undefined, copy).status, 0);
});

test('a shard is i/n with 1 <= i <= n, and a named script must be in the suite, and runs whatever the shard', () => {
  for (const shard of ['0/3', '4/3', '3', '1/0', '1.5/3', 'a/b']) {
    const listed = run(['--list'], shard);
    assert.notEqual(listed.status, 0, shard);
    assert.match(listed.stderr, /BROWSER_SHARD must be i\/n/, shard);
  }
  const unknown = run(['--list', 'no-such-check.mjs']);
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /Not in the browser suite: no-such-check\.mjs/);
  assert.deepEqual(list(undefined, ['offline-check.mjs', 'component-check.mjs']), ['component-check.mjs', 'offline-check.mjs']);
  // Named, a script runs in a shard that would leave it out.
  const elsewhere = ['1/3', '2/3', '3/3'].find(shard => !list(shard).includes('game-review-check.mjs'));
  assert.deepEqual(list(elsewhere, ['game-review-check.mjs']), ['game-review-check.mjs']);
});

test('a failed script runs once more at the end, and fails the suite only if it fails again', async t => {
  // Every try of the first fails, and the first try of the last.
  const exits = { 'component-check.mjs': [3, 3], 'agent-defaults-check.mjs': [0], 'cleanup-check.mjs': [1, 0] };
  const { copy, evidence } = await standIns(t, exits);
  // An earlier run's evidence: a picture the first stand-in writes over, and
  // a first try of its own.
  await mkdir(join(evidence, 'first-try'), { recursive: true });
  await writeFile(join(evidence, 'first-try', 'earlier.png'), 'an earlier run');
  await writeFile(join(evidence, 'component-check.mjs.png'), 'an earlier run');
  await utimes(join(evidence, 'component-check.mjs.png'), new Date(2026, 0, 1), new Date(2026, 0, 1));
  // 1/3 holds none of them, and named they run all the same.
  const suiteRun = run(Object.keys(exits), '1/3', copy);
  assert.equal(suiteRun.status, 1);
  assert.deepEqual(suiteRun.stdout.match(/^ran .*/gm), ['ran component-check.mjs, try 1', 'ran agent-defaults-check.mjs, try 1',
    'ran cleanup-check.mjs, try 1', 'ran component-check.mjs, try 2', 'ran cleanup-check.mjs, try 2']);
  assert.match(suiteRun.stdout, /\n2\/3 browser scripts passed\nFLAKY cleanup-check\.mjs: passed on its second try\n/);
  assert.deepEqual(suiteRun.stderr.match(/^FAIL .*/gm), ['FAIL component-check.mjs']);
  // The second tries' pictures are where every picture goes, and the first
  // tries' are kept beside them, in place of the earlier run's.
  assert.deepEqual((await readdir(join(evidence, 'first-try'))).toSorted(), ['cleanup-check.mjs.png', 'component-check.mjs.png']);
  for (const name of ['component-check.mjs', 'cleanup-check.mjs']) {
    assert.equal(await readFile(join(evidence, 'first-try', `${name}.png`), 'utf8'), 'try 1', name);
    assert.equal(await readFile(join(evidence, `${name}.png`), 'utf8'), 'try 2', name);
  }
  assert.equal(await readFile(join(evidence, 'agent-defaults-check.mjs.png'), 'utf8'), 'try 1');
});

test('a script that passes on its second try passes the suite, and is named as flaky', async t => {
  const { copy } = await standIns(t, { 'agent-defaults-check.mjs': [0], 'cleanup-check.mjs': [1, 0] });
  const suiteRun = run(['agent-defaults-check.mjs', 'cleanup-check.mjs'], undefined, copy);
  assert.equal(suiteRun.status, 0, suiteRun.stderr);
  assert.match(suiteRun.stdout, /\n2\/2 browser scripts passed\nFLAKY cleanup-check\.mjs: passed on its second try\n/);
  assert.doesNotMatch(suiteRun.stderr, /^FAIL /m);
});
