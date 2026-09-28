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

/** Headless Chrome with the flags a CI container needs; options add to them. */
export async function launchChrome({ args = [], ...options } = {}) {
  const { default: puppeteer } = await import('puppeteer-core');
  return puppeteer.launch({ executablePath: chromePath(), headless: true, ...options,
    args: ['--no-sandbox', '--disable-dev-shm-usage', ...args] });
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
