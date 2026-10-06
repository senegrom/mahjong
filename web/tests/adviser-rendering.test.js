import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { compile } from 'svelte/compiler';
import { render } from 'svelte/server';

// Compile the production presentation components. Only Tile's artwork and the
// agent-name import (which otherwise initializes the browser worker) are stood
// in for; grouping, numeric formatting, branches, labels and buttons are real.
const uri = code => `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;
const loaded = new Map();
async function component(name) {
  if (!loaded.has(name)) loaded.set(name, (async () => {
    const file = new URL(`../src/lib/${name}`, import.meta.url);
    const source = await readFile(file, 'utf8');
    let server;
    for (const generate of ['client', 'server']) {
      const result = compile(source, { filename: fileURLToPath(file), generate });
      assert.deepEqual(result.warnings.map(({ code, message }) => ({ code, message })), []);
      if (generate === 'server') server = result.js.code;
    }
    const tile = compile('<script>let { tile } = $props();</script><span data-tile={tile}></span>', { generate: 'server' }).js.code;
    const absolute = code => code.replace(/from (['"])(svelte[^'"]*)\1/g, (_, _quote, specifier) => `from ${JSON.stringify(import.meta.resolve(specifier))}`);
    for (const specifier of new Set([...server.matchAll(/from (['"])([^'"]+)\1/g)].map(match => match[2]))) {
      const target = specifier === './Tile.svelte' ? uri(absolute(tile))
        : specifier === './agents.js' ? uri("export const AGENTS = { full: 'Trained', club: 'Club', beginner: 'Beginner' };")
        : specifier.startsWith('.') ? new URL(specifier, file).href : import.meta.resolve(specifier);
      server = server.replaceAll(`'${specifier}'`, JSON.stringify(target)).replaceAll(`"${specifier}"`, JSON.stringify(target));
    }
    return (await import(uri(server))).default;
  })());
  return loaded.get(name);
}
const html = async (name, props) => render(await component(name), { props }).body;
const discard = { kind: 'discard', tile: '3m', label: 'Discard 3m', weight: .2 };
const reachA = { kind: 'riichi', tile: '1m', label: 'Riichi 1m', weight: .8, conditionalWeight: .25 };
const reachB = { kind: 'riichi', tile: '2m', label: 'Riichi 2m', weight: .8, conditionalWeight: .75 };
const analysis = extra => ({ agent: 'full', kind: 'policy', choice: reachB, choices: [reachA, reachB, discard], ...extra });

test('riichi is rendered once, with separate conditional meters and the actual selected tile', async () => {
  const body = await html('AgentWeights.svelte', { analysis: analysis(), onchoose() {} });
  assert.equal((body.match(/Riichi declaration weight/g) ?? []).length, 1);
  assert.equal((body.match(/80\.0%/g) ?? []).length, 1);
  assert.match(body, /75\.0%/); assert.match(body, /25\.0%/); assert.match(body, /20\.0%/);
  assert.match(body, /Riichi 2m, given riichi/);
  assert.match(body, /class="weight-row[^"]*\bbest\b[^"]*" data-choice="Riichi 2m"/);
  assert.equal((body.match(/<button /g) ?? []).length, 3, 'only actual moves are clickable, not the declaration heading');
  assert.match(body, /aria-label="Play Riichi 1m"/); assert.match(body, /aria-label="Play Riichi 2m"/);
  assert.doesNotMatch(body, /plays its highest weight|60\.0%/);
});

test('display order and aggregate weights do not change the blue selected-move marker', async () => {
  const top = { ...discard, tile: '5m', label: 'Discard 5m', weight: .6 };
  const chosen = { ...discard, weight: .4 };
  const body = await html('AgentWeights.svelte', { analysis: analysis({ choice: chosen, choices: [top, chosen] }) });
  assert.match(body, /60\.0%/); assert.match(body, /40\.0%/);
  assert.match(body, /class="weight-row[^"]*\bbest\b[^"]*" data-choice="Discard 3m"/);
  assert.equal((body.match(/class="weight-row[^"]*\bbest\b/g) ?? []).length, 1);
  assert.match(body, /Agent choice/);
});

test('an unevaluated conditional stage is explicit, and its candidate moves remain available', async () => {
  const a = { ...reachA, weight: .2, conditionalWeight: null };
  const b = { ...reachB, weight: .2, conditionalWeight: null };
  const chosen = { ...discard, weight: .8 };
  const body = await html('AgentWeights.svelte', { analysis: analysis({ choice: chosen, choices: [chosen, a, b] }), onchoose() {}, disabled: true });
  assert.equal((body.match(/Not evaluated/g) ?? []).length, 2);
  assert.equal((body.match(/<meter /g) ?? []).length, 2, 'plain move and declaration only');
  assert.equal((body.match(/<button [^>]*disabled/g) ?? []).length, 3);
  assert.doesNotMatch(body, /NaN%|undefined%/);
});

test('zero conditional probability is not confused with missing data or an unscored kan', async () => {
  const extra = { kind: 'concealed-kan', tile: '7z', label: 'Extra kan', weight: null };
  const body = await html('AgentWeights.svelte', { analysis: analysis({ choices: [{ ...reachA, conditionalWeight: 0 }, { ...reachB, conditionalWeight: 1 }, extra] }) });
  assert.match(body, />0\.0%</); assert.match(body, /Unscored/);
  assert.doesNotMatch(body, /Not evaluated/);
});

test('training reward has explicit mixed units and is hidden when unavailable', async () => {
  const body = await html('AgentWeights.svelte', { analysis: analysis({ value: -1.25 }) });
  assert.match(body, /Estimated training reward −1\.25/);
  assert.match(body, /point change ÷ 4,000/); assert.match(body, /final placement bonus/);
  assert.match(body, /not a predicted finishing place, place gain or win probability/);
  assert.doesNotMatch(body, /Hand worth|in places at the table/);
  const absent = await html('AgentWeights.svelte', { analysis: analysis({ value: null }) });
  assert.doesNotMatch(absent, /Estimated training reward|point change ÷/);
});

test('built-in selection does not acquire percentages or a riichi probability group', async () => {
  const body = await html('AgentWeights.svelte', { analysis: { agent: 'club', kind: 'selection', choice: reachA,
    choices: [{ ...reachA, weight: 1 }, { ...discard, weight: 0 }] } });
  assert.match(body, /Selected/);
  assert.doesNotMatch(body, /<meter |\d\.\d%|Riichi declaration weight|Estimated training reward/);
});

test('historical review distinguishes shared declaration weights from different conditional discards', async () => {
  const body = await html('ReviewPreference.svelte', { note: { advised_kind: 'riichi', played_kind: 'riichi', agreed: false,
    preferred_weight: .8, played_weight: .8, preferred_conditional_weight: .75, played_conditional_weight: .25 } });
  assert.equal((body.match(/to declare riichi/g) ?? []).length, 2);
  assert.match(body, /Discard given riichi: 75\.0%/);
  assert.match(body, /Discard given riichi: 25\.0%/);
  assert.doesNotMatch(body, /60\.0%/);
});

test('historical unevaluated riichi and unscored moves never acquire invented percentages', async () => {
  const body = await html('ReviewPreference.svelte', { note: { advised_kind: 'discard', played_kind: 'riichi', agreed: false,
    preferred_weight: .8, played_weight: .2, preferred_conditional_weight: null, played_conditional_weight: null } });
  assert.match(body, /Discard given riichi: Not evaluated/);
  const unscored = await html('ReviewPreference.svelte', { note: { advised_kind: 'discard', played_kind: 'concealed-kan', agreed: false,
    preferred_weight: .8, played_weight: null } });
  assert.match(unscored, /Your move:/); assert.match(unscored, /Unscored/);
  assert.doesNotMatch(unscored, /NaN%|undefined%|given riichi/);
});
