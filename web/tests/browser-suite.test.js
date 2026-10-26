/** Split the browser suite into shards, and run it over stand-in scripts,
 * without starting Chromium. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const suite = fileURLToPath(new URL('../scripts/browser-suite.mjs', import.meta.url));

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

test('every script lands in exactly one shard, and without a shard all run in order', () => {
  const all = list();
  assert.ok(all.length >= 20);
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

test('a shard is i/n with 1 <= i <= n, and named scripts must be in the suite', () => {
  for (const shard of ['0/3', '4/3', '3', '1/0', '1.5/3', 'a/b']) {
    const listed = run(['--list'], shard);
    assert.notEqual(listed.status, 0, shard);
    assert.match(listed.stderr, /BROWSER_SHARD must be i\/n/, shard);
  }
  const unknown = run(['--list', 'no-such-check.mjs']);
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /Not in the browser suite: no-such-check\.mjs/);
  assert.deepEqual(list(undefined, ['offline-check.mjs', 'component-check.mjs']), ['component-check.mjs', 'offline-check.mjs']);
});

test('a failing script does not stop the ones after it, and the failures are named', async t => {
  const folder = await mkdtemp(join(tmpdir(), 'mahjong-suite-'));
  t.after(() => rm(folder, { recursive: true, force: true }));
  await mkdir(join(folder, 'scripts'));
  const copy = join(folder, 'scripts', 'browser-suite.mjs');
  await copyFile(suite, copy);
  // Stand-ins under the suite's own names, run by a copy of it: the first and
  // the last fail.
  const exits = { 'component-check.mjs': 3, 'agent-defaults-check.mjs': 0, 'cleanup-check.mjs': 1 };
  for (const [name, code] of Object.entries(exits)) {
    await writeFile(join(folder, 'scripts', name), `console.log('ran ${name}'); process.exitCode = ${code};\n`);
  }
  const suiteRun = run(Object.keys(exits), undefined, copy);
  assert.equal(suiteRun.status, 1);
  for (const name of Object.keys(exits)) assert.match(suiteRun.stdout, new RegExp(`ran ${name}`));
  assert.match(suiteRun.stdout, /1\/3 browser scripts passed/);
  assert.match(suiteRun.stderr, /FAIL component-check\.mjs\nFAIL cleanup-check\.mjs/);
});
