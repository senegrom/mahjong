import test from 'node:test';
import assert from 'node:assert/strict';
import { render } from 'svelte/server';
import { componentLoader } from './fixtures/load-component.js';

// Render the production settings and its actual offline readiness text.
// App owns the engine/browser lifecycle outside this presentation boundary.
const Status = await componentLoader()('app/OfflineStatus.svelte');
const cached = { coreReady: true, aiReady: true, hasModel: true, phase: 'ready', progress: 0 };
const show = (change = {}) => render(Status, { props: { offline: { ...cached, ...change }, downloadAi: () => {} } }).body;

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

test('until the worker reports its build, no trained download is offered', () => {
  const html = show({ aiReady: false, hasModel: false });
  assert.match(html, /Offline: game ready/);
  assert.doesNotMatch(html, /data-download-ai/);
});
