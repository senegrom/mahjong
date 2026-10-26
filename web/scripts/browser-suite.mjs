/** One browser-suite inventory for local verification and CI.
 * Each process retains its own browser/context lifecycle and failure evidence.
 * Every script runs, whichever failed before it. At the end each failed script
 * runs once more, and only one that fails again fails the suite; one that
 * passes on its second try is named as FLAKY. What a failed script wrote to
 * test-results on its first try is kept in test-results/first-try.
 * BROWSER_SHARD=i/n runs only the i-th of n shards (CI runs three side by
 * side), script names narrow that to those scripts, and --list prints what
 * would run without running it:
 *   BROWSER_SHARD=2/3 node scripts/browser-suite.mjs [--list] [script.mjs ...] */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

// In run order, each with the seconds it took in CI on 23 October 2026 (main,
// run 39946819194), which balance the shards. The timestamps of the == lines
// in a CI log give new ones.
const scripts = [
  ['component-check.mjs', 43], ['agent-defaults-check.mjs', 14], ['cleanup-check.mjs', 17],
  ['webapp-reliability-check.mjs', 10], ['trained-default-check.mjs', 11], ['network-picker-check.mjs', 63],
  ['guided-game-check.mjs', 26], ['adviser-riichi-check.mjs', 7], ['adviser-review-check.mjs', 16],
  ['agent-watch-check.mjs', 85], ['ui-regression.mjs', 84], ['tile-effects-check.mjs', 8],
  ['full-review-check.mjs', 308], ['game-review-check.mjs', 735], ['discard-readiness-check.mjs', 32],
  ['mixed-opponents-check.mjs', 27], ['ui-polish-check.mjs', 9], ['trained-model-check.mjs', 12],
  ['toolchain-check.mjs', 26], ['offline-check.mjs', 59],
];

const shard = process.env.BROWSER_SHARD || '1/1';
const [index, count] = shard.split('/').map(Number);
if (!/^\d+\/\d+$/.test(shard) || index < 1 || index > count) {
  throw new Error(`BROWSER_SHARD must be i/n with 1 <= i <= n, not "${shard}"`);
}
const named = process.argv.slice(2).filter(argument => argument !== '--list');
const unknown = named.filter(name => !scripts.some(([script]) => script === name));
if (unknown.length) throw new Error(`Not in the browser suite: ${unknown.join(', ')}`);

// Longest first, each to the shard with the fewest seconds so far (the first
// of equals): every script lands in exactly one shard, the same one each time.
const load = Array(count).fill(0), shardOf = new Map();
for (const [script, seconds] of scripts.toSorted((a, b) => b[1] - a[1])) {
  const least = load.indexOf(Math.min(...load));
  load[least] += seconds;
  shardOf.set(script, least + 1);
}
const chosen = scripts.map(([script]) => script)
  .filter(script => shardOf.get(script) === index && (!named.length || named.includes(script)));

if (process.argv.includes('--list')) {
  for (const script of chosen) console.log(script);
} else {
  const web = fileURLToPath(new URL('../', import.meta.url)), evidence = join(web, 'test-results');
  /** Runs one script, and says whether it passed. */
  const passes = script => {
    const run = spawnSync(process.execPath, [fileURLToPath(new URL(script, import.meta.url))], {
      cwd: web, stdio: 'inherit',
    });
    if (run.error) console.error(run.error);
    return !run.error && !run.signal && run.status === 0;
  };
  /** When each file in test-results was last written, by its name. */
  const written = () => new Map(existsSync(evidence) ? readdirSync(evidence, { withFileTypes: true })
    .filter(entry => entry.isFile()).map(({ name }) => [name, statSync(join(evidence, name)).mtimeMs]) : []);
  // Each script that failed, with the files its first try wrote.
  const failed = new Map();
  let before = written();
  for (const script of chosen) {
    console.log(`\n== Browser regression: ${script}`);
    const passed = passes(script), after = written();
    if (!passed) failed.set(script, [...after.keys()].filter(name => after.get(name) !== before.get(name)));
    before = after;
  }
  // The first tries' pictures and reports show the failures, and the second
  // tries would write over them, so they move to first-try, in place of an
  // earlier run's.
  if (failed.size) {
    const firstTry = join(evidence, 'first-try');
    rmSync(firstTry, { recursive: true, force: true });
    mkdirSync(firstTry, { recursive: true });
    for (const name of new Set([...failed.values()].flat())) renameSync(join(evidence, name), join(firstTry, name));
  }
  const flaky = [], broken = [];
  for (const script of failed.keys()) {
    console.log(`\n== Browser regression: ${script}, second try`);
    (passes(script) ? flaky : broken).push(script);
  }
  console.log(`\n${chosen.length - broken.length}/${chosen.length} browser scripts passed`);
  for (const script of flaky) console.log(`FLAKY ${script}: passed on its second try`);
  for (const script of broken) console.error(`FAIL ${script}`);
  if (broken.length) process.exitCode = 1;
}
