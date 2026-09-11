/** Run the actual harness wrapper without starting Chromium or an HTTP server. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import ts from 'typescript';

function harness(name) {
  const source = readFileSync(new URL(`../scripts/${name}`, import.meta.url), 'utf8');
  const tree = ts.createSourceFile(name, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const wrapper = tree.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'check');
  assert.ok(wrapper, `${name} must have a test wrapper`);
  const contexts = [], results = [];
  const check = vm.runInNewContext(`(${wrapper.getText(tree)})`, {
    contexts, results, console: { log() {}, error() {} },
  });
  return { contexts, results, check };
}

for (const name of ['ui-regression.mjs', 'full-review-check.mjs']) {
  test(`${name}: consecutive cases release their own contexts before the next case`, async () => {
    const { contexts, results, check } = harness(name);
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

  test(`${name}: assertion failures still close every context`, async () => {
    const { contexts, results, check } = harness(name);
    let closed = 0;
    await check('failed assertion', async () => {
      contexts.push({ async close() { closed++; } }, { async close() { closed++; } });
      throw new Error('original assertion');
    });
    assert.equal(closed, 2);
    assert.equal(contexts.length, 0);
    assert.equal(results[0].passed, false);
    assert.match(results[0].error, /original assertion/);
  });

  test(`${name}: cleanup errors fail the case and do not prevent other contexts closing`, async () => {
    const { contexts, results, check } = harness(name);
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

  test(`${name}: assertion and cleanup errors are both reported`, async () => {
    const { contexts, results, check } = harness(name);
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
}
