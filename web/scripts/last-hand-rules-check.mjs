/** The last hand's rules as the mounted Physical adviser plays them on the
 * real engine; only the network's answers are controlled. The table is the
 * probe of riichi_core's all_last tests, typed in: North to move in South 4,
 * the 9 circles its one riichi tile, which, thrown undeclared, keeps North
 * first through every win while South leads it by under 1,000. Trained AI is
 * sure of the riichi, and of its tile: at a gap of 0 the rule throws the
 * tile undeclared, and the adviser says so; at 1,500 the riichi stands. At
 * a phone's width, where the tag is likeliest to overflow. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer, normalizePath } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import init, { PhysicalAnalysis } from '../src/wasm/riichi.js';
import { MORTAL_REACH } from '../src/lib/action-weights.js';
import { PHYSICAL_KEY } from '../src/lib/physical-position.js';
import { DAMA, DAMA_RULE, RULE_WORDS, ruleSaid } from '../src/lib/trained-move.js';
import { probe } from '../tests/fixtures/last-hand-tables.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const root = fileURLToPath(new URL('../', import.meta.url));
const cases = browserChecks();
/** The 9 circles' discard among Mortal's moves, and its tile kind. */
const NINE_CIRCLES = 17;

/** The probe at `gap`, as Physical play saves it; the plan that answers the
 * adviser's two questions, each sure, so that neither is asked again of the
 * renamed game (suits.js); and the engine's two moves on the 9 circles. */
function table(gap) {
  const position = probe(gap), engine = new PhysicalAnalysis(position);
  try {
    const allowed = mask => Array.from(mask, Boolean);
    const first = allowed(engine.agent_mask_mortal()), after = allowed(engine.agent_mask_after_reach());
    assert.ok(first[MORTAL_REACH] && first[NINE_CIRCLES], 'riichi and the plain discard of the 9 circles are open');
    assert.deepEqual(after.flatMap((open, move) => (open ? [move] : [])), [NINE_CIRCLES], 'the 9 circles are the one riichi tile');
    assert.equal(engine.agent_all_last_dama()[NINE_CIRCLES], gap < 1000 ? DAMA.KEEPS_FIRST : DAMA.SHORT);
    const weights = entries => Array.from({ length: 46 }, (_, move) => entries[move] ?? 0);
    const plan = [{ action: MORTAL_REACH, weights: weights({ [MORTAL_REACH]: .95, [NINE_CIRCLES]: .05 }), mask: first },
      { action: NINE_CIRCLES, weights: weights({ [NINE_CIRCLES]: 1 }), mask: after }];
    const choices = engine.agent_choices();
    const plain = choices.find(choice => choice.kind === 'discard' && choice.tile === '9p');
    const riichi = choices.find(choice => choice.kind === 'riichi' && choice.tile === '9p');
    assert.ok(plain && riichi);
    return { saved: JSON.stringify({ version: 1, position }), plan, plain, riichi };
  } finally { engine.free(); }
}

let server, browser;
try {
  await mkdir(resolve(root, 'test-results'), { recursive: true });
  // Test-only Vite replacement of the network's client, as the adviser's
  // riichi check makes it, with ids in forward slashes on Windows too.
  const policy = normalizePath(resolve(root, 'src/lib/policy.js'));
  server = await createServer({ root, configFile: false, logLevel: 'warn',
    plugins: [{ name: 'controlled-riichi-inference', enforce: 'pre', load(id) {
      if (id === policy) return readFileSync(resolve(root, 'tests/fixtures/riichi-policy.js'), 'utf8');
    } }, svelte()], server: { host: '127.0.0.1', port: 0 } });
  await server.listen(); browser = await launchChrome();
  for (const gap of [0, 1500]) {
    const ruled = gap < 1000;
    await cases.check(`South 4 at a gap of ${gap} at 390px: ${ruled ? 'the riichi thrown undeclared, by rule' : 'the riichi declared'}`, async () => {
      const { saved, plan, plain, riichi } = table(gap), problems = [];
      const page = await (await cases.openContext(browser)).newPage();
      page.on('pageerror', error => problems.push(error.message));
      page.on('console', message => { if (['warn', 'error'].includes(message.type())) problems.push(message.text()); });
      await page.setViewport({ width: 390, height: 1000 });
      // The table is in the page's storage before it loads, as a player's
      // own would be.
      await page.evaluateOnNewDocument(({ plan, key, saved }) => {
        window.riichiPlan = plan; window.riichiCalls = 0;
        if (localStorage.getItem(key) === null) localStorage.setItem(key, saved);
      }, { plan, key: PHYSICAL_KEY, saved });
      await page.goto(`${server.resolvedUrls.local[0]}tests/fixtures/agent-defaults.html`, { waitUntil: 'networkidle0' });
      await page.waitForSelector('.physical-editor:not(:disabled)');
      await page.click('[data-available]');
      await page.waitForSelector('[aria-label="Physical play agent"] option[value="full"]');
      await page.select('[aria-label="Physical play agent"]', 'full');
      await page.click('.physical-toolbar .primary');
      await page.waitForSelector('.recommendation, .physical-play .failure');
      const advice = await page.evaluate(() => {
        const failure = document.querySelector('.physical-play .failure')?.textContent ?? null;
        const line = document.querySelector('.recommendation > span:not(.worth)');
        if (!line) return { failure };
        const shown = line.cloneNode(true), tag = line.querySelector('.rule');
        for (const hidden of shown.querySelectorAll('.visually-hidden')) hidden.remove();
        return { failure, shown: shown.textContent, title: tag?.title ?? null, heard: line.querySelector('.visually-hidden')?.textContent ?? null,
          tagInView: !tag || tag.getBoundingClientRect().right <= innerWidth,
          best: [...document.querySelectorAll('.weight-row.best')].map(row => row.dataset.choice),
          primary: [...document.querySelectorAll('.record-choices button.primary')].map(button => button.textContent.trim()),
          declared: document.querySelector('.reach-heading meter')?.value ?? null,
          calls: window.riichiCalls, overflow: document.documentElement.scrollWidth > innerWidth + 1 };
      });
      await page.screenshot({ path: resolve(root, `test-results/last-hand-rules-${gap}.png`), fullPage: true });
      assert.equal(advice.failure, null);
      assert.equal(advice.calls, plan.length, 'the riichi and its tile, each asked once');
      const move = ruled ? plain : riichi;
      assert.deepEqual({ shown: advice.shown, title: advice.title, heard: advice.heard }, ruled
        ? { shown: `Next choice: ${plain.label} · ${RULE_WORDS.tag}`, title: RULE_WORDS[DAMA_RULE], heard: ruleSaid({ rules: [DAMA_RULE] }) }
        : { shown: `Next choice: ${riichi.label}`, title: null, heard: null });
      // The move played is ringed and recorded first; the weights stay the
      // network's, the riichi it was sure of on top.
      assert.deepEqual(advice.best, [move.label]);
      assert.deepEqual(advice.primary, [move.label]);
      assert.equal(advice.declared, plan[0].weights[MORTAL_REACH]);
      assert.ok(advice.tagInView, 'the tag within the screen');
      assert.equal(advice.overflow, false, 'no horizontal overflow');
      assert.deepEqual(problems, []);
    });
  }
} finally {
  await browser?.close(); await server?.close();
  await cases.report('last-hand rules checks', resolve(root, 'test-results/last-hand-rules.json'));
}
