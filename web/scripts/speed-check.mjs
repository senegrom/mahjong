// How long the trained opponent takes to answer, measured in the page.
//
// The plan asks for under 200 ms a decision on a laptop, which is the
// difference between a table that keeps up with a person and one they wait
// for. The network runs in a worker, so this times the round trip a move
// actually takes rather than the tensor arithmetic on its own.
import puppeteer from 'puppeteer-core';

const URL = process.argv[2] ?? 'http://127.0.0.1:8732/?opponents=neural';
const WANTED = Number(process.argv[3] ?? 60);

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--disable-gpu', '--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 850 });
  await page.goto(URL, { waitUntil: 'networkidle2' });

  // Time each call into the worker, from the page's own side. The worker
  // also posts progress while it loads, so the reply is matched by the id
  // the request carried: timing to the first message back instead measures
  // how long it takes to say "loading the network", which is no time at all.
  await page.evaluate(() => {
    window.__times = [];
    const original = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message, ...rest) {
      const id = message?.id;
      if (id === undefined) return original.call(this, message, ...rest);
      const sent = performance.now();
      const listen = (event) => {
        const data = event.data ?? {};
        if (data.id !== id) return;
        if (data.action === undefined && data.error === undefined) return;
        window.__times.push(performance.now() - sent);
        this.removeEventListener('message', listen);
      };
      this.addEventListener('message', listen);
      return original.call(this, message, ...rest);
    };
  });

  const began = Date.now();
  while (Date.now() - began < 240_000) {
    const times = await page.evaluate(() => window.__times.length);
    if (times >= WANTED) break;
    const state = await page.evaluate(() => ({
      mine: [...document.querySelectorAll('.hand button.tile')].some((b) => !b.disabled),
      over: !!document.querySelector('[aria-label="how the hand ended"]'),
    }));
    if (state.over) {
      await page.evaluate(() => {
        const next = [...document.querySelectorAll('button')].find(
          (b) => b.textContent.includes('Next hand') || b.textContent.includes('Play again'),
        );
        next?.click();
      });
    } else if (state.mine) {
      await page.evaluate(() => {
        const tile = [...document.querySelectorAll('.hand button.tile')].find(
          (b) => !b.disabled,
        );
        tile?.click();
      });
    } else {
      await page.evaluate(() => {
        const pass = [...document.querySelectorAll('.controls button')].find(
          (b) => b.textContent.trim() === 'Pass',
        );
        pass?.click();
      });
    }
    await new Promise((r) => setTimeout(r, 120));
  }

  const times = await page.evaluate(() => window.__times);
  if (times.length < 10) throw new Error(`only ${times.length} decisions were timed`);
  const sorted = [...times].sort((a, b) => a - b);
  const at = (share) => sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;

  console.log(`decisions timed: ${times.length}`);
  console.log(`mean ${mean.toFixed(0)} ms`);
  console.log(`median ${at(0.5).toFixed(0)} ms, 90th ${at(0.9).toFixed(0)} ms, slowest ${sorted[sorted.length - 1].toFixed(0)} ms`);
  console.log(`first answer ${times[0].toFixed(0)} ms`);
  if (at(0.9) > 200) throw new Error(`nine in ten answers must be under 200 ms, not ${at(0.9).toFixed(0)}`);
  console.log('speed check passed');
} finally {
  await browser.close();
}
