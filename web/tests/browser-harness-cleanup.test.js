/** Run the shared browser-check harness without starting Chromium or a server. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setImmediate } from 'node:timers/promises';
import { browserChecks, chromePath } from '../scripts/browser-harness.mjs';

const quiet = t => {
  t.mock.method(console, 'log', () => {});
  t.mock.method(console, 'error', () => {});
};

test('consecutive cases release their own contexts before the next case', async t => {
  quiet(t);
  const { check, contexts, results } = browserChecks();
  let live = 0, maximum = 0;
  for (let index = 0; index < 50; index++) {
    await check(`case ${index}`, async () => {
      for (let context = 0; context < 2; context++) {
        live++;
        contexts.push({ async close() { await setImmediate(); live--; } });
      }
      maximum = Math.max(maximum, live);
      assert.equal(live, 2, 'shared tabs must remain alive until their test finishes');
    });
    assert.equal(live, 0, 'the next case must not inherit active browser contexts');
    assert.equal(contexts.length, 0);
  }
  assert.equal(maximum, 2);
  assert.equal(results.length, 50);
  assert.ok(results.every(result => result.passed));
});

test('assertion failures still close every context', async t => {
  quiet(t);
  const { check, contexts, results } = browserChecks();
  let closed = 0;
  assert.equal(await check('failed assertion', async () => {
    contexts.push({ async close() { closed++; } }, { async close() { closed++; } });
    throw new Error('original assertion');
  }), false);
  assert.equal(closed, 2);
  assert.equal(contexts.length, 0);
  assert.equal(results[0].passed, false);
  assert.match(results[0].error, /original assertion/);
});

test('cleanup errors fail the case and do not prevent other contexts closing', async t => {
  quiet(t);
  const { check, contexts, results } = browserChecks();
  let otherClosed = false;
  await check('cleanup failure', async () => {
    contexts.push(
      { close() { throw new Error('failed closing context'); } },
      { async close() { await setImmediate(); otherClosed = true; } },
    );
  });
  assert.equal(otherClosed, true);
  assert.equal(contexts.length, 0);
  assert.equal(results.length, 1);
  assert.equal(results[0].passed, false);
  assert.match(results[0].error, /failed closing context/);
});

test('assertion and cleanup errors are both reported', async t => {
  quiet(t);
  const { check, contexts, results } = browserChecks();
  await check('combined failures', async () => {
    contexts.push({ async close() { throw new Error('cleanup diagnostic'); } });
    throw new Error('assertion diagnostic');
  });
  assert.equal(contexts.length, 0);
  assert.equal(results.length, 1);
  assert.equal(results[0].passed, false);
  assert.match(results[0].error, /assertion diagnostic/);
  assert.match(results[0].error, /cleanup diagnostic/);
});

test('a context opened for a case is closed with it', async t => {
  quiet(t);
  const { check, openContext, contexts } = browserChecks();
  let closed = false;
  const browser = { async createBrowserContext() { return { async close() { closed = true; } }; } };
  await check('opened context', async () => {
    await openContext(browser);
    assert.equal(contexts.length, 1);
  });
  assert.equal(closed, true);
});

test('the report saves every result and a failure fails the process', async t => {
  quiet(t);
  const folder = await mkdtemp(join(tmpdir(), 'mahjong-harness-'));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const { check, report } = browserChecks();
  await check('passes', async () => {});
  const previous = process.exitCode;
  try {
    await report('sample checks', join(folder, 'nested', 'report.json'));
    assert.equal(process.exitCode, previous, 'all passed: the exit status is left alone');
    await check('fails', async () => { throw new Error('broken'); });
    await report('sample checks', join(folder, 'nested', 'report.json'));
    assert.equal(process.exitCode, 1);
  } finally { process.exitCode = previous; }
  const saved = JSON.parse(await readFile(join(folder, 'nested', 'report.json'), 'utf8'));
  assert.deepEqual(saved.map(({ name, passed }) => ({ name, passed })),
    [{ name: 'passes', passed: true }, { name: 'fails', passed: false }]);
});

test('CHROME_BIN chooses the browser', t => {
  const previous = process.env.CHROME_BIN;
  t.after(() => { if (previous === undefined) delete process.env.CHROME_BIN; else process.env.CHROME_BIN = previous; });
  process.env.CHROME_BIN = 'custom-chrome';
  assert.equal(chromePath(), 'custom-chrome');
});

test('every browser check finds, launches and reports through the shared harness', async () => {
  const folder = new URL('../scripts/', import.meta.url);
  const scripts = (await readdir(folder)).filter(name => /-check\.mjs$|^ui-regression\.mjs$/.test(name));
  assert.ok(scripts.length >= 17);
  for (const name of scripts) {
    const source = await readFile(new URL(name, folder), 'utf8');
    assert.match(source, /from '\.\/browser-harness\.mjs'/, name);
    assert.doesNotMatch(source, /puppeteer\.launch|CHROME_BIN|process\.exitCode/, name);
  }
});
