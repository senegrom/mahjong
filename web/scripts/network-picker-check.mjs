/** Picking the trained network, in the production build with its real
 * service worker. A pick reloads the page, which then asks for that
 * network's address alone, names it wherever Trained plays or advises, and
 * is reported on by the worker; picking the default again brings the default
 * back, and a pick made in Agent watch reloads into Agent watch. The
 * networks' host is unreachable for the whole browser and the page's own
 * requests for a network are answered here, so nothing downloads 116 MB:
 * what is checked is which address each page asks for. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserChecks, launchChrome } from './browser-harness.mjs';
import { createFixtureHandler } from './static-fixture-server.mjs';
import { observeNetworkRequests } from './network-requests.mjs';
import { SETTINGS_KEY } from '../src/lib/session.js';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';

const web = fileURLToPath(new URL('../', import.meta.url)), dist = resolve(web, 'dist'), output = resolve(web, 'test-results');
const addressOf = network => `${network.origin}/${network.object}`;
const older = NETWORKS.find(network => network.version === '0.1.410');
assert.ok(older, 'generation 410 must be listed');
assert.notEqual(older, MANIFEST);
const built = JSON.parse(await readFile(resolve(dist, 'offline-manifest.json'), 'utf8'));
const server = createServer(createFixtureHandler({ root: dist }));
const cases = browserChecks();
let browser;

/** A page whose requests for any listed network are recorded in `asked`:
 * the page's own, which the availability probe (HEAD, answered as present)
 * and the offline save (GET, which fails as it does offline) make, each with
 * the document that made it, and its workers' and service worker's, which
 * reach no server. */
async function open(asked, mode) {
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
  await page.evaluateOnNewDocument(urls => {
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
  }, NETWORKS.map(addressOf));
  await page.goto(`http://127.0.0.1:${server.address().port}/mahjong/${mode ? `?mode=${mode}` : ''}`, { waitUntil: 'domcontentloaded' });
  return page;
}
const settings = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), SETTINGS_KEY);
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
const playing = network => ({ summary: `Trained network ${network.version}`, picker: network.version, menu: `Trained ${network.version}`,
  badges: Array(3).fill(`Trained ${network.version} | Trained ${network.version} opponent`) });

try {
  await mkdir(output, { recursive: true });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await launchChrome({ args: [`--host-resolver-rules=MAP ${new URL(MANIFEST.origin).hostname} ~NOTFOUND`] });

  await cases.check('the build lists every network for its service worker', async () => {
    assert.deepEqual(built.networks.map(network => network.version), NETWORKS.map(network => network.version));
    assert.deepEqual(built.networks, NETWORKS.map(network => ({ ...network })));
  });

  await cases.check('picking 0.1.410 plays it alone, names it everywhere, and picking the default again restores 0.1.810', async () => {
    const asked = [];
    const page = await open(asked);
    await page.waitForSelector('.board');
    assert.deepEqual(await askedFor(page, asked, MANIFEST), [addressOf(MANIFEST)], 'a first visit asks for the default alone');
    assert.deepEqual(await named(page), playing(MANIFEST));
    assert.equal(Object.hasOwn(await settings(page), 'trainedNetwork'), false, 'no pick is written before one is made');

    // A pick that is not confirmed changes nothing.
    assert.equal(await pick(page, older.version, false), `Switch to Trained ${older.version}? The page reloads to load it.`);
    assert.equal(await page.$eval('select[aria-label="Trained network"]', el => el.value), MANIFEST.version);
    assert.equal(Object.hasOwn(await settings(page), 'trainedNetwork'), false);

    asked.length = 0;
    await pick(page, older.version);
    await page.waitForSelector('.board');
    assert.equal((await settings(page)).trainedNetwork, older.version);
    assert.deepEqual(await askedFor(page, asked, older), [addressOf(older)], 'the reloaded page asks for 410 alone, never 810');
    assert.deepEqual(await named(page), playing(older));
    assert.ok((await page.$eval('[data-ai-status]', el => el.textContent)).includes(`Trained AI ${older.version} — optional`));
    // The service worker reports on the network the page plays.
    const reply = await workerStatus(page, older.version);
    assert.deepEqual(reply.network, { url: addressOf(older), version: older.version, sha256: older.sha256, bytes: older.bytes });
    assert.equal(reply.aiReady, false);
    assert.equal((await workerStatus(page)).network.version, MANIFEST.version, 'and on the default for a page without a pick');
    await page.screenshot({ path: resolve(output, 'network-picker-410.png'), fullPage: true });

    // A fresh page reads the pick back from the settings.
    asked.length = 0;
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.board');
    assert.deepEqual(await askedFor(page, asked, older), [addressOf(older)]);

    asked.length = 0;
    await pick(page, MANIFEST.version);
    await page.waitForSelector('.board');
    assert.equal((await settings(page)).trainedNetwork, MANIFEST.version);
    assert.deepEqual(await askedFor(page, asked, MANIFEST), [addressOf(MANIFEST)], 'back on the default, 410 is not asked for');
    assert.deepEqual(await named(page), playing(MANIFEST));
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
