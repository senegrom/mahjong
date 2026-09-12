/** Real production service worker, real model, real CacheStorage. Close/reopen
 * the browser with HTTP cache cleared and the server refusing every asset. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, extname, sep, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import puppeteer from 'puppeteer-core';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MODEL_FILES } from '../src/lib/model-package.js';
import { TILE_IMAGE_URLS } from '../src/lib/tile-faces.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const manifest = JSON.parse(await readFile(resolve(dist, 'offline-manifest.json'), 'utf8'));
const source = await readFile(new URL('../src/offline/service-worker.js', import.meta.url), 'utf8');
const modelPath = MODEL_FILES.full, runtimePath = manifest.entries.find(e => e.url.startsWith('ort/') && e.url.endsWith('.wasm')).url;
const count = new Map(), refused = [], overrides = new Map();
let unavailable = false, failPath = null, holdPath = null, holdResolve = null, holdSeenResolve = null;
const mime = { '.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css',
  '.wasm':'application/wasm', '.svg':'image/svg+xml', '.png':'image/png', '.webp':'image/webp', '.json':'application/json' };
const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  if (!path.startsWith('/mahjong/')) { res.writeHead(404).end(); return; }
  const name = path.slice('/mahjong/'.length) || 'index.html';
  count.set(name, (count.get(name) ?? 0) + 1);
  if (unavailable || name === failPath) { refused.push(name); res.writeHead(503).end(); return; }
  if (name === holdPath) {
    holdSeenResolve?.();
    await new Promise(done => { holdResolve = done; });
  }
  try {
    const file = resolve(dist, name);
    if (!file.startsWith(dist + sep)) { res.writeHead(403).end(); return; }
    const body = overrides.get(name) ?? await readFile(file);
    res.writeHead(200, { 'Content-Type': mime[extname(name)] ?? 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(404).end(); }
});
const chrome = process.env.CHROME_BIN || ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);
assert.ok(chrome);
const fixture = (() => {
  const session = new MatchSession(Game, 81, ['neural', 'club', 'neural']);
  try { session.advance(false); return session.snapshot(); } finally { session.dispose(); }
})();
const results = [], browsers = new Set(), dirs = [];
async function launch(profile) {
  const browser = await puppeteer.launch({ executablePath: chrome, userDataDir: profile, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  browsers.add(browser); return browser;
}
async function close(browser) { await browser.close(); browsers.delete(browser); }
async function page(browser, { seed = true, offline = false, strength = 'neural' } = {}) {
  let initial = fixture;
  if (strength !== 'neural') {
    const session = new MatchSession(Game, 81, strength);
    try { session.advance(false); initial = session.snapshot(); } finally { session.dispose(); }
  }
  const p = await browser.newPage(); p.errors = [];
  p.on('pageerror', error => p.errors.push(error.message));
  await p.setViewport({ width: 390, height: 844, hasTouch: true, isMobile: true });
  await p.setCacheEnabled(false);
  if (offline) await p.setOfflineMode(true);
  if (seed) await p.evaluateOnNewDocument((key, settings, snapshot) => {
    if (!localStorage.getItem(key)) localStorage.setItem(key, JSON.stringify(snapshot));
    localStorage.setItem(settings, JSON.stringify({version:1,difficulty:snapshot.difficulty,opponents:snapshot.opponents,hints:true,confirmDiscards:true,shortcuts:true}));
    // Instrument Image only in this test. The application has no test hooks.
    const ImageClass = window.Image;
    window.preloadedImages = [];
    window.Image = class extends ImageClass { constructor(...args) { super(...args); window.preloadedImages.push(this); } };
  }, SAVE_KEY, SETTINGS_KEY, initial);
  await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/?opponents=${strength}&source=home-screen`, { waitUntil:'domcontentloaded' });
  return p;
}
const hand = p => p.waitForSelector('.hand', { timeout: 120000 });
const ready = p => p.waitForFunction(() => /game \+ AI ready/.test(document.querySelector('[data-offline-status]')?.textContent), { timeout: 120000 });
const saved = p => p.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
async function play(p, turns = 5) {
  for (let n = 0; n < turns; n++) {
    await p.waitForFunction(() => document.querySelector('.failure') || document.querySelector('.screen')
      || document.querySelector('.hand button:not(:disabled)') || document.querySelector('.call-options button:not(:disabled)'), { timeout: 45000 });
    assert.equal(await p.$('.failure'), null);
    if (await p.$('.screen')) break;
    const before = await saved(p);
    const call = await p.$('.call-options [data-choice=ron],.call-options [data-choice=tsumo],.call-options [data-choice=pass]');
    if (call) await call.click();
    else { await p.click('.hand button:not(:disabled)'); await p.click('.confirm-discard .primary'); }
    await p.waitForFunction((key,n) => JSON.parse(localStorage.getItem(key)).commands.length > n, {}, SAVE_KEY, before.commands.length);
  }
  // Runtime progress can say 'network ready' before opponents finish moving.
  // Wait for a real player decision, not a transient status message.
  await p.waitForFunction(() => document.querySelector('.failure, .screen, .standings, .hand button:not(:disabled), .call-options button:not(:disabled)'), { timeout: 45000 });
  assert.equal(await p.$('.failure'), null);
  assert.deepEqual(p.errors, []);
}
async function check(name, fn) {
  unavailable = false; failPath = null; holdPath = null; overrides.clear(); count.clear(); refused.length = 0;
  try { await fn(); results.push({ name, passed:true }); console.log(`PASS ${name}`); }
  catch (error) { results.push({ name, passed:false, error:error.stack }); console.error(`FAIL ${name}\n${error.stack}`); }
  finally { holdResolve?.(); holdResolve = null; holdSeenResolve = null; for (const b of [...browsers]) await close(b); }
}
async function profile() { const dir = await mkdtemp(join(tmpdir(), 'mahjong-offline-browser-')); dirs.push(dir); return dir; }
try {
  await mkdir(output, { recursive:true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  for (const strength of ['beginner', 'club']) await check(`${strength}: complete game and graphics save automatically and restart offline with NO AI download or button`, async () => {
    const dir = await profile(); let b = await launch(dir); const p = await page(b, { strength });
    await hand(p);
    await p.waitForSelector('[data-core-ready=true]');
    const images = await p.evaluate(() => window.preloadedImages.map(image => image.complete && image.naturalWidth > 0));
    assert.equal(images.length, TILE_IMAGE_URLS.length + 1); assert.ok(images.every(Boolean));
    assert.ok(await p.evaluate(async entries => {
      const cache = await caches.open('mahjong-offline-v1:/mahjong/');
      return (await Promise.all(entries.filter(e => e.group === 'core').map(e =>
        cache.match(new URL(`__offline_content__/${e.hash}`, location.href).href)))).every(Boolean);
    }, manifest.entries), 'Every core resource must be saved without interacting');
    for (const entry of manifest.entries.filter(e => e.group === 'ai')) assert.equal(count.get(entry.url) ?? 0, 0, entry.url);
    const beforeFaceChange = await saved(p);
    await p.click('.settings-trigger'); await p.click('.options summary');
    for (const face of ['matisse', 'classic', 'matisse']) {
      await p.select('select[aria-label="Tile face"]', face);
      await p.waitForFunction((key, face) => JSON.parse(localStorage.getItem(key)).tileFace === face
        && [...document.querySelectorAll('.hand img')].every(img => img.src.includes('/matisse/') === (face === 'matisse')), {}, SETTINGS_KEY, face);
      assert.deepEqual(await saved(p), beforeFaceChange, 'Changing artwork must preserve the current match');
    }
    const cdp = await p.createCDPSession(); await cdp.send('Network.clearBrowserCache');
    await close(b); unavailable = true; count.clear(); refused.length = 0;
    b = await launch(dir); const cold = await page(b, { seed: false, offline: true, strength });
    await hand(cold); await cold.waitForSelector('[data-core-ready=true]');
    assert.equal(await cold.$eval('select[aria-label="Tile face"]', select => select.value), 'matisse');
    assert.ok(await cold.$$eval('.hand img', images => images.length > 0 && images.every(image => image.src.includes('/matisse/') && image.complete && image.naturalWidth > 0)));
    await play(cold, 4);
    const before = await saved(cold);
    assert.ok(!before.commands.some(command => command.type === 'opponent'), 'Built-in game must not request the network');
    cold.on('dialog', dialog => void dialog.accept());
    await cold.click('.settings-trigger'); await cold.click('.mobile-new-game'); await hand(cold); await play(cold, 3);
    assert.notEqual((await saved(cold)).seed, before.seed);
    const art = manifest.entries.filter(e => e.url.endsWith('.svg') || e.url.includes('white-dragon')).map(e => e.url);
    assert.ok(await cold.evaluate(async urls => (await Promise.all(urls.map(async url => {
      const image = new Image(); image.src = url; await image.decode(); return image.naturalWidth > 0;
    }))).every(Boolean), art));
    assert.deepEqual(refused.filter(name => name !== 'sw.js'), []);
    if (strength === 'club') await cold.screenshot({ path: resolve(output, 'automatic-game-tiles-offline.png'), fullPage: true });
    assert.deepEqual(cold.errors, []);
  });
  await check('the optional button adds ONLY trained AI bytes and does not change the match', async () => {
    const b = await launch(await profile()), p = await page(b, { strength: 'club' });
    await hand(p); await p.waitForSelector('[data-core-ready=true]');
    const before = await saved(p), previous = new Map(count);
    assert.equal(count.get(modelPath) ?? 0, 0);
    await p.click('.settings-trigger'); await p.click('.offline-settings summary');
    assert.match(await p.$eval('[data-core-status]', el => el.textContent), /automatic/);
    assert.match(await p.$eval('[data-ai-status]', el => el.textContent), /optional/);
    assert.match(await p.$eval('[data-download-ai]', el => el.textContent), /Download trained AI/);
    await p.click('[data-download-ai]'); await ready(p);
    assert.deepEqual(await saved(p), before);
    for (const entry of manifest.entries.filter(e => e.group === 'core')) {
      assert.equal(count.get(entry.url) ?? 0, previous.get(entry.url) ?? 0, `AI button redownloaded core asset: ${entry.url}`);
    }
    assert.equal(count.get(modelPath), 1);
    await p.screenshot({ path: resolve(output, 'automatic-game-optional-ai.png'), fullPage: true });
    assert.deepEqual(p.errors, []);
  });
  await check('missing tile and icon cache entries repair automatically on reconnect without downloading AI', async () => {
    const b = await launch(await profile()), p = await page(b, { strength: 'club' });
    await hand(p); await p.waitForSelector('[data-core-ready=true]');
    const missing = manifest.entries.filter(e => e.url === 'tiles/Chun.svg' || e.url === 'favicon.ico');
    assert.equal(missing.length, 2);
    const previous = new Map(count);
    await p.evaluate(async entries => {
      const cache = await caches.open('mahjong-offline-v1:/mahjong/');
      for (const entry of entries) await cache.delete(new URL(`__offline_content__/${entry.hash}`, location.href).href);
      window.dispatchEvent(new Event('online'));
    }, missing);
    await p.waitForFunction(async entries => {
      const cache = await caches.open('mahjong-offline-v1:/mahjong/');
      return (await Promise.all(entries.map(entry => cache.match(new URL(`__offline_content__/${entry.hash}`, location.href).href)))).every(Boolean);
    }, {}, missing);
    for (const entry of missing) assert.equal(count.get(entry.url), (previous.get(entry.url) ?? 0) + 1);
    for (const entry of manifest.entries.filter(e => e.group === 'ai')) assert.equal(count.get(entry.url) ?? 0, 0);
    assert.deepEqual(p.errors, []);
  });
  await check('first play waits for every tile face, including unused tiles and dragon artwork', async () => {
    holdPath = 'tiles/Chun.svg';
    const seen = new Promise(done => { holdSeenResolve = done; });
    const b = await launch(await profile()), p = await page(b);
    await seen;
    assert.equal(await p.$('.hand'), null, 'No partially downloaded hand');
    holdPath = null; holdResolve();
    await hand(p); await ready(p);
    const images = await p.evaluate(() => window.preloadedImages.map(image => ({ url:image.src, complete:image.complete && image.naturalWidth > 0 })));
    assert.equal(images.length, TILE_IMAGE_URLS.length + 1);
    assert.ok(images.every(image => image.complete));
    const faces = manifest.entries.filter(e => e.url.startsWith('tiles/') && e.url.endsWith('.svg'));
    assert.deepEqual(faces.map(face => face.url).sort(), [...TILE_IMAGE_URLS].sort());
    for (const face of faces) assert.ok(count.has(face.url), face.url);
    assert.equal(count.get(modelPath), 1);
    await p.screenshot({path:resolve(output,'offline-first-download.png'),fullPage:true});
    assert.deepEqual(p.errors, []);
  });
  await check('cold browser restart with HTTP cache cleared plays real trained opponents on a plane', async () => {
    const dir = await profile(); let b = await launch(dir); const p = await page(b);
    await hand(p); await ready(p); await play(p, 3);
    const snapshot = await saved(p);
    const cdp = await p.createCDPSession(); await cdp.send('Network.clearBrowserCache');
    await close(b); unavailable = true; refused.length = 0; count.clear();
    b = await launch(dir); const cold = await page(b, { seed:false, offline:true });
    await hand(cold); await ready(cold);
    assert.deepEqual((await saved(cold)).commands, snapshot.commands);
    await play(cold, 7);
    assert.ok((await saved(cold)).commands.filter(c=>c.type==='opponent').length > snapshot.commands.filter(c=>c.type==='opponent').length);
    // Decode all art again after the process restart, not just currently held tiles.
    const art = manifest.entries.filter(e=>e.url.endsWith('.svg') || e.url.includes('white-dragon')).map(e=>e.url);
    assert.ok(await cold.evaluate(async urls => (await Promise.all(urls.map(async url=>{
      const image=new Image(); image.src=url; await image.decode(); return image.complete&&image.naturalWidth>0;
    }))).every(Boolean), art));
    // Native SW update probes may occur; the game itself must not hit network.
    assert.deepEqual(refused.filter(name=>name!=='sw.js'), []);
    const beforeRestart = await saved(cold); cold.on('dialog', d=>void d.accept());
    await cold.click('.settings-trigger'); await cold.click('.mobile-new-game'); await hand(cold); await play(cold, 3);
    assert.notEqual((await saved(cold)).seed, beforeRestart.seed);
    assert.deepEqual(refused.filter(name=>name!=='sw.js'), []);
    await cold.screenshot({path:resolve(output,'offline-plane-real-ai.png'),fullPage:true});
  });
  await check('interrupted runtime download stays incomplete and resumes without re-downloading good weights', async () => {
    failPath = runtimePath;
    // The hashed duplicate has the same body: fail both network locations.
    for (const entry of manifest.entries.filter(e=>e.group==='ai'&&e.url.endsWith('.wasm'))) overrides.set(entry.url, Buffer.from('incomplete'));
    const b=await launch(await profile()), p=await page(b); await hand(p);
    await p.waitForFunction(()=>document.body.textContent.includes('AI download incomplete'),{timeout:120000});
    assert.doesNotMatch(await p.$eval('[data-offline-status]',el=>el.textContent),/AI ready/);
    await p.waitForFunction(async()=>{
      const reg=await navigator.serviceWorker.getRegistration();
      return Boolean(reg?.active);
    });
    // Allow outstanding successful sibling requests to be written before retry.
    await new Promise(done=>setTimeout(done,200));
    assert.equal(count.get(modelPath),1);
    failPath=null; overrides.clear();
    await p.click('.settings-trigger'); await p.click('.offline-settings summary'); await p.click('.offline-settings button'); await ready(p); await p.click('.mobile-preferences-head button');
    assert.equal(count.get(modelPath),1); assert.ok(count.get(runtimePath)>=1);
    await play(p,4); assert.deepEqual(p.errors,[]);
  });
  await check('cached Trained remains selectable offline without an online availability check', async () => {
    const b=await launch(await profile()),p=await page(b);await hand(p);await ready(p);
    unavailable=true; refused.length=0;await p.setOfflineMode(true);await p.reload({waitUntil:'domcontentloaded'});
    await hand(p);await ready(p);await p.select('select[aria-label="opponent strength"]','custom');
    await p.waitForSelector('.custom-dialog[open]');
    for(const position of ['Left','Opposite','Right'])assert.equal(await p.$eval(`select[aria-label="${position} opponent"] option[value=neural]`,el=>el.disabled),false);
    assert.deepEqual(refused.filter(name=>name!=='sw.js'),[]);
    assert.deepEqual(p.errors,[]);
  });
  await check('UI-only upgrades reuse the entire AI download and do not interrupt a running match',async()=>{
    const b=await launch(await profile()),p=await page(b);await hand(p);await ready(p);await play(p,2);
    const previous=await saved(p), modelLoads=count.get(modelPath), runtimeLoads=count.get(runtimePath);
    const config=structuredClone(manifest),html=Buffer.concat([await readFile(resolve(dist,'index.html')),Buffer.from('\n<!-- offline update test -->')]);
    const entry=config.entries.find(e=>e.url==='index.html');entry.hash=createHash('sha256').update(html).digest('hex');entry.bytes=html.length;config.version+='-test';
    overrides.set('index.html',html);overrides.set('sw.js',Buffer.from(source.replace('/* OFFLINE_CONFIG */ null',JSON.stringify(config))));
    await p.evaluate(async()=>{const r=await navigator.serviceWorker.getRegistration();await r.update();});
    await p.waitForFunction(async()=>Boolean((await navigator.serviceWorker.getRegistration())?.waiting),{timeout:120000});
    assert.equal(count.get(modelPath),modelLoads);assert.equal(count.get(runtimePath),runtimeLoads);
    assert.deepEqual((await saved(p)).commands,previous.commands);await play(p,3);
    assert.deepEqual(p.errors,[]);
  });
} finally {
  await writeFile(resolve(output,'offline-report.json'),JSON.stringify(results,null,2));
  for(const b of [...browsers])await close(b);
  if(server.listening)await new Promise(done=>server.close(done));
  for(const dir of dirs)await rm(dir,{recursive:true,force:true});
}
console.log(`${results.filter(r=>r.passed).length}/${results.length} offline browser checks passed`);
if(results.some(r=>!r.passed))process.exitCode=1;
