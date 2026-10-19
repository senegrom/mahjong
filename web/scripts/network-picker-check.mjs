/** Picking the trained network, in the production build with its real
 * service worker. A pick reloads the page, which then asks for that
 * network's address alone, names it wherever Trained plays or advises, and
 * is reported on by the worker; picking the default again brings the default
 * back and leaves no pick, and a pick made in Agent watch reloads into Agent
 * watch. Settings saved before the default was published play it once,
 * whatever they picked, keep every other choice and resume a saved match;
 * a pick made after that stays. On a phone a seat's badge says Trained
 * without the version, and the opponents menu shows the version whole, not
 * cut off before its chevron. The networks' host is unreachable for the whole browser and the
 * page's own requests for a network are answered here, so nothing downloads
 * 116 MB: what is checked is which address each page asks for. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { observeNetworkRequests } from './network-requests.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });
const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const addressOf = network => `${network.origin}/${network.object}`;
// Every listed network but the default, newest first: each stays for a
// player to pick.
const others = NETWORKS.filter(network => network !== MANIFEST);
assert.ok(others.length, 'a network played before the default must stay listed');
const built = JSON.parse(await readFile(resolve(dist, 'offline-manifest.json'), 'utf8'));
const server = createServer(createFixtureHandler({ root: dist }));
const cases = browserChecks();
let browser;

// Every other choice a settings record can hold, none of them a default, with
// two Trained seats so that the page asks for the network it plays.
const chosen = { difficulty: 'custom', opponents: ['neural', 'club', 'neural'], hints: false, confirmDiscards: true,
  shortcuts: false, tileFace: 'dali', reviewAdviser: 'club' };
// A match at that table saved part way through its first hand, as a page
// before 1110 left it. Its Trained moves are replayed from the save, never
// asked of a network: each Trained seat passes on every call and otherwise
// throws the first tile it may. The Club seat can end a hand early, so the
// first deal still under way after thirty moves is taken.
const savedMatch = (() => {
  for (let seed = 7; seed < 40; seed++) {
    const match = new MatchSession(Game, seed, chosen.opponents);
    try {
      match.advance(false);
      for (let step = 0; step < 30 && match.view.phase !== 'over'; step++) {
        if (match.engine.needs_opponent_move()) {
          const mask = match.engine.opponent_mask_mortal();
          const action = mask[45] ? 45 : mask.findIndex((allowed, index) => allowed && index < 37);
          match.apply({ type: 'opponent', action: action >= 0 ? action : mask.findIndex(Boolean) });
        } else {
          const choice = match.choices.find(entry => entry.kind === 'pass') ?? match.choices.find(entry => entry.kind === 'discard');
          if (!choice) break;
          match.apply({ type: 'choose', kind: choice.kind, tile: choice.tile ?? null });
        }
        match.advance(false);
      }
      const trained = match.commands.filter(command => command.type === 'opponent').length;
      if (match.view.phase !== 'over' && !match.over && trained >= 10) return match.snapshot();
    } finally { match.dispose(); }
  }
  throw new Error('No deal was still under way after thirty moves');
})();

/** A page whose requests for any listed network are recorded in `asked`:
 * the page's own, which the availability probe (HEAD, answered as present)
 * and the offline save (GET, which fails as it does offline) make, each with
 * the document that made it, and its workers' and service worker's, which
 * reach no server. `saved` is stored before the first load only, as an
 * earlier visit left it. */
async function open(asked, mode, saved = {}) {
  const context = await cases.openContext(browser);
  await observeNetworkRequests(context, (request, from) => {
    if (NETWORKS.some(network => addressOf(network) === request.url)) asked.push({ method: request.method, url: request.url, from });
  });
  const page = await context.newPage();
  page.problems = [];
  page.on('pageerror', error => page.problems.push(error.message));
  await page.setViewport({ width: 1100, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.exposeFunction('reportNetworkRequest', (method, url, documentId) => asked.push({ method, url, documentId, from: 'page' }));
  await page.evaluateOnNewDocument((urls, saved) => {
    for (const [key, value] of Object.entries(saved)) if (localStorage.getItem(key) === null) localStorage.setItem(key, value);
    const documentId = `${performance.timeOrigin}-${Math.random()}`;
    Object.defineProperty(window, 'networkDocument', { value: documentId });
    const realFetch = window.fetch;
    window.fetch = (input, options) => {
      const target = input instanceof Request ? input.url : String(input);
      if (!urls.includes(target)) return realFetch(input, options);
      const method = String(options?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase();
      void window.reportNetworkRequest(method, target, documentId);
      return method === 'HEAD' ? Promise.resolve(new Response(null, { status: 200 })) : Promise.reject(new TypeError('Failed to fetch'));
    };
  }, NETWORKS.map(addressOf), saved);
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/${mode ? `?mode=${mode}` : ''}`, { waitUntil: 'domcontentloaded' });
  return page;
}
const settings = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SETTINGS_KEY);
const stored = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SAVE_KEY);
const addresses = requests => [...new Set(requests.map(request => request.url))];
/** Waits until the page now open, or its opponents' worker, has asked for
 * the bytes of `network`, which it does to save it as a match starts, and
 * returns every network address they asked for. A report the page left
 * behind as it reloaded is not the new page's. */
async function askedFor(page, asked, network) {
  const documentId = await page.evaluate(() => window.networkDocument);
  const mine = () => asked.filter(request => request.from !== 'page' || request.documentId === documentId);
  for (const deadline = Date.now() + 180000; !mine().some(request => request.method === 'GET' && request.url === addressOf(network));) {
    assert.ok(Date.now() < deadline, `the page never asked for the bytes of ${network.version}`);
    await new Promise(done => setTimeout(done, 100));
  }
  // The download is the last thing a page asks for; give a stray request time to show.
  await new Promise(done => setTimeout(done, 1500));
  return addresses(mine());
}
/** What the page names: the settings summary, the picker, the opponents
 * menu and each Trained seat's badge. */
const named = page => page.evaluate(() => ({
  summary: document.querySelector('[data-trained-network]').textContent.trim(),
  picker: document.querySelector('select[aria-label="Trained network"]').value,
  menu: document.querySelector('select[aria-label="opponent strength"] option[value="neural"]')?.textContent.trim() ?? null,
  badges: [...document.querySelectorAll('.opponent-type[data-controller="neural"]')]
    .map(badge => `${badge.textContent.trim()} | ${badge.getAttribute('aria-label')}`),
}));
/** Picks a version in the game settings and answers the confirmation; an
 * accepted pick reloads the page. Returns what the confirmation said. */
async function pick(page, version, accept = true) {
  await page.evaluate(() => {
    const section = document.querySelector('.network-choice');
    if (!section.open) section.querySelector('summary').click();
  });
  const said = new Promise(done => page.once('dialog', async dialog => {
    const message = dialog.message();
    await (accept ? dialog.accept() : dialog.dismiss());
    done(message);
  }));
  const reloaded = accept ? page.waitForNavigation({ waitUntil: 'domcontentloaded' }) : null;
  // The change is sent after this evaluation returns: an accepted pick
  // navigates away from the document it ran in.
  await page.evaluate(version => {
    const picker = document.querySelector('select[aria-label="Trained network"]');
    picker.value = version;
    setTimeout(() => picker.dispatchEvent(new Event('change', { bubbles: true })));
  }, version);
  const message = await said;
  await reloaded;
  return message;
}
/** The production worker's own status reply for this page's player's pick. */
const workerStatus = (page, version) => page.evaluate(version => new Promise((done, fail) => {
  const channel = new MessageChannel();
  channel.port1.onmessage = ({ data }) => data.error ? fail(new Error(data.error)) : done(data.value);
  navigator.serviceWorker.controller.postMessage({ type: 'MAHJONG_STATUS', version }, [channel.port2]);
}), version);
const playing = (network, seats = 3) => ({ summary: `Trained network ${network.version}`, picker: network.version, menu: `Trained ${network.version}`,
  badges: Array(seats).fill(`Trained ${network.version} | Trained ${network.version} opponent`) });

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome({ args: [`--host-resolver-rules=MAP ${new URL(MANIFEST.origin).hostname} ~NOTFOUND`] });

  await cases.check('the build lists every network for its service worker', async () => {
    assert.deepEqual(built.networks.map(network => network.version), NETWORKS.map(network => network.version));
    assert.deepEqual(built.networks, NETWORKS.map(network => ({ ...network })));
  });

  await cases.check(`picking ${others.map(network => network.version).join(' or ')} plays it alone, names it everywhere, and picking the default again restores ${MANIFEST.version}`, async () => {
    const asked = [];
    const page = await open(asked);
    await page.waitForSelector('.board');
    assert.deepEqual(await askedFor(page, asked, MANIFEST), [addressOf(MANIFEST)], 'a first visit asks for the default alone');
    assert.deepEqual(await named(page), playing(MANIFEST));
    assert.equal(Object.hasOwn(await settings(page), 'trainedNetwork'), false, 'no pick is written before one is made');
    // The picker offers every listed network, the default first and marked.
    assert.deepEqual(await page.$$eval('select[aria-label="Trained network"] option', options => options.map(option => [option.value, option.textContent.trim()])),
      NETWORKS.map(network => [network.version, network === MANIFEST ? `${network.version} (default)` : network.version]));

    // A pick that is not confirmed changes nothing.
    assert.equal(await pick(page, others[0].version, false), `Switch to Trained ${others[0].version}? The page reloads to load it.`);
    assert.equal(await page.$eval('select[aria-label="Trained network"]', el => el.value), MANIFEST.version);
    assert.equal(Object.hasOwn(await settings(page), 'trainedNetwork'), false);

    for (const older of others) {
      asked.length = 0;
      await pick(page, older.version);
      await page.waitForSelector('.board');
      assert.equal((await settings(page)).trainedNetwork, older.version);
      assert.deepEqual(await askedFor(page, asked, older), [addressOf(older)], `the reloaded page asks for ${older.version} alone`);
      assert.deepEqual(await named(page), playing(older));
      assert.ok((await page.$eval('[data-ai-status]', el => el.textContent)).includes(`Trained AI ${older.version} — optional`));
      // The service worker reports on the network the page plays.
      const reply = await workerStatus(page, older.version);
      assert.deepEqual(reply.network, { url: addressOf(older), version: older.version, sha256: older.sha256, bytes: older.bytes });
      assert.equal(reply.aiReady, false);
      assert.equal((await workerStatus(page)).network.version, MANIFEST.version, 'and on the default for a page without a pick');
      await page.screenshot({ path: resolve(output, `network-picker-${older.version}.png`), fullPage: true });

      // A fresh page reads the pick back from the settings.
      asked.length = 0;
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.waitForSelector('.board');
      assert.deepEqual(await askedFor(page, asked, older), [addressOf(older)]);
      assert.deepEqual(await named(page), playing(older));
    }

    asked.length = 0;
    await pick(page, MANIFEST.version);
    await page.waitForSelector('.board');
    // Picking the default is no pick, so a newer default would replace it.
    assert.equal(Object.hasOwn(await settings(page), 'trainedNetwork'), false, 'picking the default again leaves no pick');
    assert.deepEqual(await askedFor(page, asked, MANIFEST), [addressOf(MANIFEST)], 'back on the default, no other network is asked for');
    assert.deepEqual(await named(page), playing(MANIFEST));
    assert.deepEqual(page.problems, []);
  });

  // What builds before generation 1110 wrote: settings record version 2,
  // which held a pick only once a player made one. Every such pick is
  // dropped once, so that the player meets the default.
  const before = [...others.map(network => [`a pick of ${network.version}`, network.version]), ['no pick', undefined], ['a pick no build lists', '0.1.36']];
  for (const [index, [which, trainedNetwork]] of before.entries()) {
    // The first also makes a pick afterwards, and the second resumes a match.
    const later = index === 0 ? others[0] : null, resumes = index === 1;
    const also = [later && `, then keeps ${later.version} once it is picked again`, resumes && ', and resumes the match it saved'].filter(Boolean).join('');
    await cases.check(`a version 2 settings record with ${which} plays the default, ${MANIFEST.version}, keeps every other choice${also}`, async () => {
      const asked = [];
      const saved = { [SETTINGS_KEY]: JSON.stringify({ version: 2, ...chosen, trainedNetwork }) };
      if (resumes) saved[SAVE_KEY] = JSON.stringify(savedMatch);
      const page = await open(asked, undefined, saved);
      await page.waitForSelector('.board');
      assert.deepEqual(await askedFor(page, asked, MANIFEST), [addressOf(MANIFEST)], `the page asks for ${MANIFEST.version} alone`);
      // The page writes the record back as it loads: version 3, every other
      // choice as it was, and no pick.
      await page.waitForFunction(key => JSON.parse(localStorage.getItem(key))?.version === 3, {}, SETTINGS_KEY);
      assert.deepEqual(await settings(page), { version: 3, ...chosen });
      assert.equal(await page.$eval('select[aria-label="opponent strength"]', el => el.value), 'custom');
      if (resumes) {
        // The match goes on where it was saved, its Trained seats now played by the default.
        const { commands, seed, opponents } = await stored(page);
        assert.deepEqual({ seed, opponents, commands: commands.slice(0, savedMatch.commands.length) },
          { seed: savedMatch.seed, opponents: savedMatch.opponents, commands: savedMatch.commands });
        assert.doesNotMatch(await page.$eval('main', el => el.textContent), /could not be restored/);
        await page.screenshot({ path: resolve(output, 'network-picker-reset-resumed.png'), fullPage: true });
      }
      assert.deepEqual(await named(page), playing(MANIFEST, 2));
      if (later) {
        asked.length = 0;
        await pick(page, later.version);
        await page.waitForSelector('.board');
        assert.deepEqual(await askedFor(page, asked, later), [addressOf(later)]);
        for (let load = 0; load < 2; load++) {
          assert.deepEqual(await settings(page), { version: 3, ...chosen, trainedNetwork: later.version });
          assert.deepEqual(await named(page), playing(later, 2));
          asked.length = 0;
          await page.reload({ waitUntil: 'domcontentloaded' });
          await page.waitForSelector('.board');
          assert.deepEqual(await askedFor(page, asked, later), [addressOf(later)], `${later.version} is still played after a reload`);
        }
      }
      assert.deepEqual(page.problems, []);
    });
  }

  await cases.check('on a phone each Trained seat badge says Trained in one line, a screen reader still hears the version, and the opponents menu shows it whole', async () => {
    const page = await open([]);
    await page.waitForSelector('.board');
    const seen = {}, menus = {};
    for (const width of [390, 360, 320]) {
      await page.setViewport({ width, height: 800, isMobile: true, hasTouch: true });
      await page.waitForFunction(() => getComputedStyle(document.querySelector('.score-short')).display !== 'none');
      // The badge's text as laid out, one box a line. How many lines the whole
      // header takes depends on the platform's fonts, so it is not asserted:
      // with the version the badge took a header's last line to itself.
      seen[width] = await page.$$eval('.opponent-type[data-controller="neural"]', found => found.map(badge => {
        const range = document.createRange();
        range.selectNodeContents(badge);
        const lines = new Set([...range.getClientRects()].filter(box => box.width > 0).map(box => Math.round(box.top))).size;
        return { shown: badge.innerText.trim(), label: badge.getAttribute('aria-label'), lines };
      }));
      // The menu's text, laid out in its own font, against the room its
      // padding leaves: a select cuts its text off where the right padding,
      // which holds the chevron, begins.
      menus[width] = await page.$eval('select[aria-label="opponent strength"]', select => {
        const style = getComputedStyle(select), shown = select.selectedOptions[0].textContent;
        const probe = Object.assign(document.createElement('span'), { textContent: shown });
        Object.assign(probe.style, { font: style.font, letterSpacing: style.letterSpacing, whiteSpace: 'pre', position: 'absolute' });
        document.body.append(probe);
        const text = probe.getBoundingClientRect().width;
        probe.remove();
        const room = select.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        return { shown, whole: text <= room, text: +text.toFixed(1), room };
      });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `no sideways scroll at ${width}`);
      await page.screenshot({ path: resolve(output, `network-picker-phone-${width}.png`) });
    }
    const badge = Array(3).fill({ shown: 'Trained', label: `Trained ${MANIFEST.version} opponent`, lines: 1 });
    assert.deepEqual(seen, { 390: badge, 360: badge, 320: badge });
    for (const [width, menu] of Object.entries(menus)) {
      assert.equal(menu.shown, `Trained ${MANIFEST.version}`);
      assert.ok(menu.whole, `at ${width} px the menu cuts off its ${menu.text} px of text at ${menu.room} px`);
    }
    assert.deepEqual(page.problems, []);
  });

  await cases.check('a pick made in Agent watch reloads into Agent watch, and names the network in the lineup', async () => {
    const asked = [];
    const page = await open(asked, 'watch');
    const followed = () => page.$eval('select[aria-label="Followed agent"] option[value="full"]', option => option.textContent.trim());
    await page.waitForSelector('.agent-watch');
    await page.waitForFunction(() => document.querySelector('select[aria-label="Followed agent"]')?.value === 'full');
    assert.equal(await followed(), `Trained ${MANIFEST.version}`);
    // Only the watched game is lost to the reload, and the confirmation says so.
    const [older] = others;
    assert.match(await pick(page, older.version), /The page reloads to load it, and the watched game ends\.$/);
    await page.waitForSelector('.agent-watch');
    assert.equal(new URL(page.url()).searchParams.get('mode'), 'watch');
    await page.waitForFunction(() => document.querySelector('select[aria-label="Followed agent"]')?.value === 'full');
    assert.equal(await followed(), `Trained ${older.version}`);
    assert.equal(await page.$eval('[data-trained-network]', el => el.textContent.trim()), `Trained network ${older.version}`);
    await pick(page, MANIFEST.version);
    await page.waitForSelector('.agent-watch');
    await page.waitForFunction(() => document.querySelector('select[aria-label="Followed agent"]')?.value === 'full');
    assert.equal(await followed(), `Trained ${MANIFEST.version}`);
    assert.deepEqual(page.problems, []);
  });
} finally {
  await browser?.close();
  if (server.listening) await new Promise(done => server.close(done));
  await cases.report('network-picker checks', resolve(output, 'network-picker.json'));
}
