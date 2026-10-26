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

/** Headless Chrome with the flags a CI container needs; options add to them.
 * Chrome has 60 s to start, not puppeteer's 30, and one that fails to start is
 * tried once more: in CI a Chrome once did not answer within 30 s, and the
 * check failed before any case ran. */
export async function launchChrome({ args = [], ...options } = {}) {
  const { default: puppeteer } = await import('puppeteer-core');
  const executablePath = chromePath();
  const launch = () => puppeteer.launch({ executablePath, headless: true, timeout: 60000, ...options,
    args: ['--no-sandbox', '--disable-dev-shm-usage', ...args] });
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
