import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { compile, parse } from 'svelte/compiler';
import { render } from 'svelte/server';

// Render the production settings and its actual derived readiness values.
// The rest of App needs an engine/browser and is outside this UI boundary.
const source = await readFile(new URL('../src/App.svelte', import.meta.url), 'utf8');
const ast = parse(source, { modern: true });
const declarations = ast.instance.content.body.filter(node => node.type === 'VariableDeclaration'
  && node.declarations.some(d => ['selectedAiReady', 'selectedAiAvailable'].includes(d.id?.name)))
  .map(node => source.slice(node.start, node.end)).join('\n');
const start = source.indexOf('<details class="offline-settings">');
const end = source.indexOf('</details>', start) + '</details>'.length;
const code = compile(`<script>let { offline, trainedModel, downloadAi = () => {} } = $props();\n${declarations}</script>${source.slice(start, end)}`,
  { generate: 'server' }).js.code.replace(/from (['"])([^'"]+)\1/g,
  (_, _quote, name) => `from ${JSON.stringify(import.meta.resolve(name))}`);
const { default: Status } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const quickCached = { coreReady: true, aiReady: true, strongReady: false, hasModel: true,
  hasStrongModel: true, phase: 'ready', progress: 0 };
const show = (trainedModel, change = {}) => render(Status, { props: { trainedModel, offline: { ...quickCached, ...change } } }).body;

test('Strong download failure shows a retry even while Quick remains ready', () => {
  const html = show('strong', { phase: 'incomplete', warning: 'Strong download interrupted' });
  assert.match(html, /Offline: game ready/);
  assert.doesNotMatch(html, /Offline: game \+ AI ready/);
  assert.match(html, /Strong download interrupted/);
  assert.match(html, /<button[^>]*data-download-ai[^>]*>Retry trained AI download/);
  assert.doesNotMatch(html, /<button[^>]*data-download-ai[^>]*disabled/);
});

test('selected model readiness updates during download and after completion', () => {
  assert.match(show('strong', { phase: 'ai', progress: 40 }), /Saving AI… 40%/);
  assert.match(show('strong'), /Download trained AI for offline play/);
  for (const html of [show('quick'), show('strong', { strongReady: true })]) {
    assert.match(html, /Offline: game \+ AI ready/);
    assert.doesNotMatch(html, /data-download-ai/);
  }
});
