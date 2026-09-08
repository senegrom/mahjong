/** Production UI + real-WASM discard previews, readiness rings and draw spacing.
 * No engine stubs or test hooks are included in the application build.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url));
const dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const opening = [['discard','6p'], ['discard','9s'], ['discard','8s'], ['discard','7p'],
  ['chii','4m'], ['discard','4z'], ['chii','2m'], ['discard','6z']];
function fixture(seed, commands = []) {
  const m = new MatchSession(Game, seed, 'club');
  try {
    m.advance(false);
    for (const [kind, tile] of commands) { m.apply({ type:'choose', kind, tile }); m.advance(false); }
    const expected = {};
    for (const c of m.choices.filter(c => c.kind === 'discard')) expected[c.tile] = m.engine.discard_hint(c.tile);
    return { snapshot:m.snapshot(), expected };
  } finally { m.dispose(); }
}
const initial = fixture(1), closed = fixture(81, [['discard','9s'], ['discard','5z']]);
const openHand = fixture(1, opening), restricted = fixture(1, opening.slice(0, 7));
const calling = fixture(1, opening.slice(0, 4)), riichi = fixture(31, [['riichi','8m']]);
const types = { '.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css',
  '.svg':'image/svg+xml', '.png':'image/png', '.webp':'image/webp', '.wasm':'application/wasm', '.json':'application/json' };
const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!path.startsWith('/mahjong/')) { res.writeHead(404).end(); return; }
    const file = resolve(dist, path.slice('/mahjong/'.length) || 'index.html');
    if (!file.startsWith(dist + sep)) { res.writeHead(403).end(); return; }
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type':types[extname(file)] ?? 'application/octet-stream', 'Cache-Control':'no-store' });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(404).end(); }
});
const contexts = [], results = [];
let browser;
async function check(name, task) {
  try { await task(); results.push({ name, passed:true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed:false, error:error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { while (contexts.length) await contexts.pop().close(); }
}
async function open(f, width = 1100, height = 900) {
  const context = await browser.createBrowserContext(); contexts.push(context);
  const p = await context.newPage(); p.problems = [];
  p.on('pageerror', error => p.problems.push(error.message));
  await p.setViewport({ width, height, hasTouch:width < 600, isMobile:width < 600, deviceScaleFactor:1 });
  await p.emulateMediaFeatures([{ name:'prefers-reduced-motion', value:'reduce' }]);
  await p.evaluateOnNewDocument((key, settings, snapshot) => {
    localStorage.setItem(key, JSON.stringify(snapshot));
    localStorage.setItem(settings, JSON.stringify({ version:1, difficulty:'club', hints:true, confirmDiscards:true, shortcuts:true }));
  }, SAVE_KEY, SETTINGS_KEY, f.snapshot);
  await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/`, { waitUntil:'networkidle0' });
  await p.waitForSelector('.hand');
  if (Object.keys(f.expected).length) await p.waitForSelector('.hand button:not(:disabled)');
  return p;
}
const saved = p => p.evaluate(key => localStorage.getItem(key), SAVE_KEY);
const waits = p => p.$$eval('.hint .wait .tile', els => els.map(el => el.getAttribute('aria-label').split(',')[0]));
const shot = (p, name) => p.screenshot({ path:resolve(output, name + '.png'), fullPage:true });
async function assertRings(p, expected) {
  const actual = await p.$$eval('.hand button', els => els.map(el => ({ tile:el.dataset.tile,
    readiness:el.dataset.readiness ?? null, disabled:el.disabled, ring:getComputedStyle(el).getPropertyValue('--ring').trim() })));
  for (const tile of actual) {
    const n = expected[tile.tile]?.shanten;
    const readiness = tile.disabled ? null : n === 0 ? 'ready' : n === 1 ? 'one-away' : null;
    assert.equal(tile.readiness, readiness, `Wrong readiness for ${tile.tile}`);
    if (readiness) assert.notEqual(tile.ring, 'none');
  }
  assert.deepEqual(p.problems, []);
}
try {
  await mkdir(output, { recursive:true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const chrome = process.env.CHROME_BIN || ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);
  assert.ok(chrome, 'Set CHROME_BIN to Chrome/Chromium');
  browser = await puppeteer.launch({ executablePath:chrome, headless:true, args:['--no-sandbox','--disable-dev-shm-usage'] });
  await check('silver and gold rings match every legal hypothetical discard', async () => {
    const p = await open(closed); await assertRings(p, closed.expected);
    assert.equal(await p.$$eval('[data-readiness=ready]', els => els.length), 2);
    assert.equal(await p.$$eval('[data-readiness=one-away]', els => els.length), 12);
    const colour = await p.$eval('.hand [data-tile="9p"]', el => getComputedStyle(el, '::before').backgroundColor);
    assert.equal(colour, 'rgb(197, 203, 211)');
    await shot(p, 'readiness-desktop');
  });
  await check('changing selection recomputes exact waits and clears stale waits without a move', async () => {
    const p = await open(closed), before = await saved(p);
    for (const [tile, expected] of [['2z',['green dragon']], ['6z',['south wind']], ['7m',[]], ['2z',['green dragon']]]) {
      await p.click(`.hand button[data-tile="${tile}"]`);
      assert.deepEqual(await waits(p), expected);
      assert.match(await p.$eval('.hint', el => el.textContent), /After discarding/i);
      if (tile === '7m') assert.match(await p.$eval('.hint', el => el.textContent), /1 tile from a wait/);
      assert.equal(await saved(p), before);
    }
    await p.keyboard.press('Escape');
    assert.doesNotMatch(await p.$eval('.hint', el => el.textContent), /After discarding/i);
    await assertRings(p, closed.expected);
  });
  await check('keyboard selection and native Tab preview the same focused discard', async () => {
    const p = await open(closed), before = await saved(p); await p.focus('.hand');
    await p.keyboard.press('ArrowLeft'); assert.deepEqual(await waits(p), []); // Drawn 9 circles.
    await p.keyboard.press('ArrowLeft'); assert.deepEqual(await waits(p), ['south wind']);
    await p.keyboard.down('Shift'); await p.keyboard.press('Tab'); await p.keyboard.up('Shift'); assert.deepEqual(await waits(p), ['green dragon']);
    assert.equal(await p.evaluate(() => document.activeElement.dataset.tile), '2z');
    assert.equal(await saved(p), before); await assertRings(p, closed.expected);
  });
  await check('an open hand gets gold readiness without being offered riichi', async () => {
    const p = await open(openHand); await assertRings(p, openHand.expected);
    assert.equal(await p.$('[data-choice=riichi]'), null);
    assert.equal(await p.$eval('.hand [data-tile="7z"]', el => el.dataset.readiness), 'ready');
    await p.click('.hand [data-tile="7z"]'); assert.deepEqual(await waits(p), ['2 characters']);
    assert.equal(await p.$('.my-melds [data-readiness]'), null);
  });
  await check('forbidden post-call tiles and call windows never receive readiness rings', async () => {
    const p = await open(restricted); await assertRings(p, restricted.expected);
    const barred = await p.$eval('.hand [data-tile="3m"]', el => ({ disabled:el.disabled, readiness:el.dataset.readiness }));
    assert.ok(barred.disabled); assert.equal(barred.readiness, undefined);
    const call = await open(calling); assert.equal(await call.$('[data-readiness]'), null); assert.deepEqual(call.problems, []);
    const locked = await open(riichi); await assertRings(locked, riichi.expected);
    assert.equal(await locked.$$eval('[data-readiness=ready]', els => els.length), 1);
  });
  await check('drawn tiles have no special border and hints can be switched off', async () => {
    const p = await open(initial);
    const drawn = await p.$eval('.hand button[data-drawn=true]', el => ({ ringed:el.classList.contains('ringed'), label:el.getAttribute('aria-label') }));
    assert.equal(drawn.ringed, false); assert.match(drawn.label, /just drawn/);
    const c = await open(closed); await c.click('.options summary'); await c.click('.option-fields input');
    assert.equal(await c.$('[data-readiness]'), null); assert.equal(await c.$('.hint'), null);
    assert.equal(await c.$eval('.hand .hand-tile[data-hand-drawn=true]', el => getComputedStyle(el).marginInlineStart), '18px');
    await c.click('.option-fields input'); await assertRings(c, closed.expected);
  });
  await check('new decisions refresh all readiness hints rather than reusing the old hand', async () => {
    const p = await open(closed); await p.click('.hand [data-tile="2z"]'); await p.click('.confirm-discard .primary');
    await p.waitForFunction(key => JSON.parse(localStorage.getItem(key)).commands.length === 3, {}, SAVE_KEY);
    await p.waitForSelector('.hand button:not(:disabled)');
    const next = fixture(81, [['discard','9s'], ['discard','5z'], ['discard','2z']]);
    await assertRings(p, next.expected);
    assert.doesNotMatch(await p.$eval('.hint', el => el.textContent), /After discarding/);
  });
  for (const [width, height] of [[320,568],[375,667],[390,844],[568,320],[844,390],[1440,1000]]) {
    await check(`separate drawn tile retains equal size and fits ${width}x${height}`, async () => {
      for (const f of [closed, openHand]) {
        const p = await open(f, width, height);
        const boxes = await p.$$eval('.hand .hand-tile', els => els.map(el => {
          const r = el.getBoundingClientRect();
          return { width:r.width, right:r.right, left:r.left, drawn:el.dataset.handDrawn, margin:parseFloat(getComputedStyle(el).marginInlineStart) };
        }));
        const expectedGap = width <= 760 || (width >= 640 && height <= 500) ? 14 : 18;
        const drawn = boxes.find(b => b.drawn); assert.ok(drawn); assert.equal(drawn.margin, expectedGap);
        for (const box of boxes) { assert.ok(Math.abs(box.width - drawn.width) < .2); assert.ok(box.right <= width); }
        assert.ok(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        await assertRings(p, f.expected);
        if (f === closed && width === 390) { await p.click('.hand [data-tile="2z"]'); await shot(p, 'readiness-phone-preview'); }
      }
    });
  }
} finally {
  await writeFile(resolve(output, 'discard-readiness-report.json'), JSON.stringify(results, null, 2));
  for (const context of contexts) await context.close();
  await browser?.close(); if (server.listening) await new Promise(done => server.close(done));
}
console.log(`${results.filter(r => r.passed).length}/${results.length} discard-readiness browser checks passed`);
if (results.some(r => !r.passed)) process.exitCode = 1;
