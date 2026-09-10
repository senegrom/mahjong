import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';

// Render the production settings and its actual offline readiness text.
// The rest of App needs an engine/browser and is outside this UI boundary.
const source = await readFile(new URL('../src/App.svelte', import.meta.url), 'utf8');
const start = source.indexOf('<details class="offline-settings">');
const end = source.indexOf('</details>', start) + '</details>'.length;
const code = compile(`<script>let { offline, downloadAi = () => {} } = $props();</script>${source.slice(start, end)}`,
  { generate: 'server' }).js.code.replace(/from (['"])([^'"]+)\1/g,
  (_, _quote, name) => `from ${JSON.stringify(import.meta.resolve(name))}`);
const { default: Status } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const cached = { coreReady: true, aiReady: true, hasModel: true, phase: 'ready', progress: 0 };
const show = (change = {}) => render(Status, { props: { offline: { ...cached, ...change } } }).body;

test('an interrupted trained download shows a retry while the game itself stays ready', () => {
  const html = show({ aiReady: false, phase: 'incomplete', warning: 'Trained download interrupted' });
  assert.match(html, /Offline: game ready/);
  assert.doesNotMatch(html, /Offline: game \+ AI ready/);
  assert.match(html, /Trained download interrupted/);
  assert.match(html, /<button[^>]*data-download-ai[^>]*>Retry trained AI download/);
  assert.doesNotMatch(html, /<button[^>]*data-download-ai[^>]*disabled/);
});

test('trained readiness updates during the download and after it completes', () => {
  assert.match(show({ aiReady: false, phase: 'ai', progress: 40 }), /Saving AI… 40%/);
  assert.match(show({ aiReady: false }), /Download trained AI for offline play/);
  const html = show();
  assert.match(html, /Offline: game \+ AI ready/);
  assert.doesNotMatch(html, /data-download-ai/);
});

test('a build without the trained network never offers its download', () => {
  const html = show({ aiReady: false, hasModel: false });
  assert.match(html, /Offline: game ready/);
  assert.doesNotMatch(html, /data-download-ai/);
});
