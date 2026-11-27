/** What the browser checks share: finding and launching Chrome, and running
 * named cases into a report whose failures set the exit status. */
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

// CHROME_BIN wins; otherwise the usual installs on Linux, Windows and macOS.
const CHROME = [
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
];

export function chromePath() {
  const found = process.env.CHROME_BIN || CHROME.find(existsSync);
  assert.ok(found, 'Set CHROME_BIN to a Chrome or Chromium executable');
  return found;
}

// The page runs a review's batches on a desktop's graphics card where Chrome
// offers it one (gpu-batches.js), and CI's Chrome offers none. On a desktop
// with a card the checks' Chrome did, and the checks that walk a review's
// batches met the card's as well, in other sizes, and failed there alone.
// Without its WebGPU service Chrome offers no adapter and still draws on the
// card, so a check sees the page it sees in CI (webgpu-check.mjs holds
// Chrome to that).
const NO_CARD = '--disable-features=WebGPUService';

/** Headless Chrome with the flags a CI container needs; options add to them.
 * Chrome has 60 s to start, not puppeteer's 30, and one that fails to start is
 * tried once more: in CI a Chrome once did not answer within 30 s, and the
 * check failed before any case ran. It offers the page no graphics card to
 * compute on unless `card` says to, which only the card's own check does. */
export async function launchChrome({ args = [], card = false, ...options } = {}) {
  const { default: puppeteer } = await import('puppeteer-core');
  const executablePath = chromePath();
  const launch = () => puppeteer.launch({ executablePath, headless: true, timeout: 60000, ...options,
    args: ['--no-sandbox', '--disable-dev-shm-usage', ...(card ? [] : [NO_CARD]), ...args] });
  try {
    return await launch();
  } catch (error) {
    console.error(`Chrome did not start, trying once more: ${error.message}`);
    return launch();
  }
}

/** Opens a <details> by pressing its summary, then waits for its toggle
 * event and two frames, so that a press inside it lands where it is aimed.
 * The settings lay an opened panel out where it stands first, and move it
 * below the other summaries (`order: 1`) only after its toggle event: a
 * press aimed before the move lands on what took its place, at 1100x900 the
 * panel's own summary, which closes it again. */
export async function openPanel(page, summary) {
  await page.evaluate(selector => {
    window.__panelSettled = new Promise((done, fail) => {
      setTimeout(() => fail(new Error(`${selector} did not toggle its panel`)), 5000);
      document.querySelector(selector).parentElement.addEventListener('toggle',
        () => requestAnimationFrame(() => requestAnimationFrame(done)), { once: true });
    });
  }, summary);
  await page.click(summary);
  await page.evaluate(() => window.__panelSettled);
}

/** How well the words of the elements matching `selector` can be read: their
 * colour as drawn, with their own opacity and every ancestor's, against each
 * pixel behind each piece of text's box, in one screenshot of the screen with
 * no words drawn (for light words, the lightest pixel). The first is brought
 * onto the middle of the screen, and every one on the screen with it is read.
 * Answers the worst one's ratio, colour, the pixel behind it that gives that
 * ratio, and its words. */
export async function wordsContrast(page, selector) {
  // Each piece of text in them, its own box: a tile drawn among the words
  // is no part of what is behind them.
  const words = await page.$$eval(selector, elements => {
    elements[0]?.scrollIntoView({ block: 'center', inline: 'nearest' });
    const found = [];
    for (const element of elements) {
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const parent = node.parentElement;
        if (!node.textContent.trim() || parent.closest('.visually-hidden')) continue;
        let opacity = 1;
        for (let up = parent; up; up = up.parentElement) opacity *= Number(getComputedStyle(up).opacity);
        const range = document.createRange();
        range.selectNodeContents(node);
        for (const box of range.getClientRects()) {
          // Only what a box that scrolls or clips lets be seen of it.
          let [left, top, right, bottom] = [box.left, box.top, box.right, box.bottom];
          for (let up = parent.parentElement; up; up = up.parentElement) {
            const style = getComputedStyle(up);
            if (style.overflowX === 'visible' && style.overflowY === 'visible') continue;
            const clip = up.getBoundingClientRect();
            [left, top, right, bottom] = [Math.max(left, clip.left), Math.max(top, clip.top), Math.min(right, clip.right), Math.min(bottom, clip.bottom)];
          }
          found.push({ colour: getComputedStyle(parent).color, opacity, text: node.textContent.trim().slice(0, 40),
            box: [left, top, right, bottom].map(edge => Math.round(edge * devicePixelRatio)) });
        }
      }
    }
    return found.filter(({ box }) => box[2] - box[0] >= 2 && box[3] - box[1] >= 2 && box[1] >= 0 && box[3] <= innerHeight * devicePixelRatio);
  });
  assert.ok(words.length, `no words match ${selector} on the screen`);
  // The words go at once, with no transition: under reduced motion every
  // change of style eases over a hundredth of a millisecond (app.css), a
  // frame in which the word is still drawn, and a picture taken in that
  // frame held the word, which then read as its own ground, 1 to 1 (once, in
  // CI). The picture waits until no word has a colour left, and for the
  // frames that paint that.
  const hidden = await page.addStyleTag({ content: '*, *::before, *::after { transition: none !important; color: transparent !important; -webkit-text-fill-color: transparent !important; text-shadow: none !important; text-decoration-color: transparent !important; }' });
  let shot;
  try {
    await page.waitForFunction(selector => [...document.querySelectorAll(selector)].every(element =>
      [element, ...element.querySelectorAll('*')].every(node => /, 0\)$|^transparent$/.test(getComputedStyle(node).color))), { timeout: 10000 }, selector);
    await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(done)))));
    shot = await page.screenshot({ encoding: 'base64', type: 'png' });
  } finally { await hidden.evaluate(node => node.remove()); }
  // The screenshot is read back in the page, which decodes a PNG already.
  return page.evaluate(async (shot, words) => {
    const image = new Image();
    image.src = `data:image/png;base64,${shot}`;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const light = rgb => rgb.map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4)
      .reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
    let worst = null;
    for (const { colour, opacity, text, box } of words) {
      const [left, top] = [Math.max(0, box[0]), Math.max(0, box[1])];
      const [right, bottom] = [Math.min(image.width, box[2]), Math.min(image.height, box[3])];
      const pixels = context.getImageData(left, top, right - left, bottom - top).data;
      const parts = colour.match(/[\d.]+/g).map(Number), alpha = (parts[3] ?? 1) * opacity;
      for (let at = 0; at < pixels.length; at += 4) {
        const behind = [pixels[at], pixels[at + 1], pixels[at + 2]];
        const drawn = parts.slice(0, 3).map((v, i) => v * alpha + behind[i] * (1 - alpha));
        const x = light(drawn), y = light(behind), ratio = (Math.max(x, y) + .05) / (Math.min(x, y) + .05);
        if (!worst || ratio < worst.ratio) worst = { ratio, ground: `rgb(${behind.join(', ')})`, colour, text };
      }
    }
    return worst;
  }, shot, words);
}

/** Named cases, run one after another. A case can open several browser
 * contexts: pushed to `contexts`, they are all closed before the next case,
 * whether it passed or not, and a context that fails to close fails it too. */
export function browserChecks() {
  const results = [], contexts = [];
  async function check(name, body) {
    const failures = [];
    try { await body(); } catch (error) { failures.push(error); }
    // No context may carry its engine, workers and service workers on into
    // the next case.
    const closed = await Promise.allSettled(contexts.splice(0).map(async context => context.close()));
    for (const result of closed) if (result.status === 'rejected') failures.push(result.reason);
    if (failures.length) {
      const error = failures.map(failure => failure?.stack ?? String(failure)).join('\n');
      results.push({ name, passed: false, error });
      console.error(`FAIL ${name}\n${error}`);
    } else {
      results.push({ name, passed: true });
      console.log(`PASS ${name}`);
    }
    return !failures.length;
  }
  /** A fresh context for the running case, closed when it ends. */
  async function openContext(browser) {
    const context = await browser.createBrowserContext();
    contexts.push(context);
    return context;
  }
  /** Saves the results as JSON (when given a file), prints the tally and
   * fails the process if any case failed. */
  async function report(label, file) {
    if (file) {
      await mkdir(dirname(file), { recursive: true });
      await writeFile(file, JSON.stringify(results, null, 2));
    }
    console.log(`${results.filter(result => result.passed).length}/${results.length} ${label} passed`);
    if (results.some(result => !result.passed)) process.exitCode = 1;
  }
  return { check, openContext, report, results, contexts };
}
