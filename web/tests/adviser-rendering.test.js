import test from 'node:test';
import assert from 'node:assert/strict';
import { render } from 'svelte/server';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';
import { playNetwork } from '../src/lib/trained-network.js';
import { DAMA_RULE, DECLINE_RULE } from '../src/lib/trained-move.js';
import { componentLoader, moduleUrl, svelteModule } from './fixtures/load-component.js';
import { text } from './fixtures/markup-text.js';

// Render the production presentation components. Only Tile, for its artwork,
// and the agent-name import (which otherwise initializes the browser worker)
// are stood in for; grouping, numeric formatting, branches, labels and
// buttons are real. The stand-in names a Trained agent as agents.js does,
// with the network the page plays.
const network = new URL('../src/lib/trained-network.js', import.meta.url).href;
const component = componentLoader({
  'Tile.svelte': await svelteModule('<script>let { tile } = $props();</script><span data-tile={tile}></span>'),
  'agents.js': moduleUrl(`import { trainedName } from ${JSON.stringify(network)};
export const AGENTS = { full: 'Trained', club: 'Club', beginner: 'Beginner' };
export const agentName = agent => agent === 'full' ? trainedName() : AGENTS[agent];`),
});
const html = async (name, props) => render(await component(name), { props }).body;
const discard = { kind: 'discard', tile: '3m', label: 'Discard 3m', weight: .2 };
const reachA = { kind: 'riichi', tile: '1m', label: 'Riichi 1m', weight: .8, conditionalWeight: .25 };
const reachB = { kind: 'riichi', tile: '2m', label: 'Riichi 2m', weight: .8, conditionalWeight: .75 };
const analysis = extra => ({ agent: 'full', kind: 'policy', choice: reachB, choices: [reachA, reachB, discard], ...extra });
/** What the eye reads of some markup, and what a screen reader hears of it,
 * each as textContent gives it: without what is visually hidden, or without
 * what is hidden from a screen reader, neither holding more than text and a
 * tag of text. */
const seen = markup => text(markup.replace(/<span class="visually-hidden[^"]*">[^<]*<\/span>/g, ''), { joined: true });
const heard = markup => text(markup.replace(/<span aria-hidden="true">(?:[^<]|<(span|strong)[^>]*>[^<]*<\/\1>)*<\/span>/g, ''), { joined: true });
/** The last hand's lines, as the rules' titles and notes give them. */
const DAMA_LINE = 'Last hand: every win keeps it first, so no riichi', DECLINE_LINE = 'Last hand: this win would leave it last';

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

test('the value is named and explained by the reward the network the page plays learned from, and hidden when unavailable', async t => {
  t.after(() => playNetwork(undefined));
  const shown = async version => {
    playNetwork(version);
    return html('AgentWeights.svelte', { analysis: analysis({ value: -1.25 }) });
  };
  // 810 and 410 learned from the hybrid reward, in mixed units.
  for (const version of ['0.1.810', '0.1.410']) {
    const body = await shown(version);
    assert.match(body, /Estimated training reward −1\.25/);
    assert.match(body, new RegExp(`Trained ${version.replaceAll('.', '\\.')} learned from a mixed reward`));
    assert.match(body, /point change ÷ 4,000/); assert.match(body, /final placement bonus/);
    assert.match(body, /not a predicted finishing place, place gain or win probability/);
    assert.doesNotMatch(body, /Hand worth|in places at the table|placement alone/);
  }
  // 1110 was last trained on placement alone: no points term, no mixed units.
  const body = await shown('0.1.1110');
  assert.match(body, /title="Estimated placement bonus, not a win probability">Estimated placement reward −1\.25</);
  assert.match(body, /Trained 0\.1\.1110 was last trained on the placement bonus \(\+1\.5 for first, \+0\.5 for second, −0\.5 for third and −1\.5 for fourth\) the scoreboard promises as each hand ends, points counting only through the places they win\. An estimate, not a win probability\./);
  assert.doesNotMatch(body, /training reward|÷ 4,000|mixed|finishing place/);
  // 2610 and 1410 learned the same at alpha 0.05: the hand's points are paid
  // beside the bonus, and the tooltip and the words under the value say how
  // much.
  for (const version of ['0.1.2610', '0.1.1410']) {
    const weighed = await shown(version);
    assert.match(weighed, /title="Estimated placement bonus plus 0\.05 × point change ÷ 4,000, not a win probability">Estimated placement reward −1\.25</);
    assert.match(weighed, new RegExp(`Trained ${version.replaceAll('.', '\\.')} was last trained on the placement bonus`));
    assert.match(weighed, /was last trained on the placement bonus \(\+1\.5 for first, \+0\.5 for second, −0\.5 for third and −1\.5 for fourth\) the scoreboard promises as each hand ends, plus 0\.05 × the hand(’|'|&#39;)s point change ÷ 4,000\. An estimate, not a win probability\./);
    assert.doesNotMatch(weighed, /training reward|mixed|finishing place|only through the places/);
  }
  // With no value nothing is said of one, whichever network plays.
  for (const network of NETWORKS) {
    playNetwork(network.version);
    const absent = await html('AgentWeights.svelte', { analysis: analysis({ value: null }) });
    assert.doesNotMatch(absent, /Estimated|reward-help|learned from/);
  }
});

test('built-in selection does not acquire percentages or a riichi probability group', async () => {
  const body = await html('AgentWeights.svelte', { analysis: { agent: 'club', kind: 'selection', choice: reachA,
    choices: [{ ...reachA, weight: 1 }, { ...discard, weight: 0 }] } });
  assert.match(body, /Selected/);
  assert.match(body, /<strong>Club<\/strong>/);
  assert.doesNotMatch(body, /<meter |\d\.\d%|Riichi declaration weight|Estimated (training|placement) reward/);
});

test("a move one of the last hand's rules made says so after the next choice, the rule's line its title and, to a screen reader, said in full", async () => {
  // The probe: the network put 97% on a riichi on the 9 circles and 1% on
  // their plain discard, which the rule throws, the tile keeping it first.
  const plain = { kind: 'discard', tile: '9p', label: 'Discard 9p', weight: .01 };
  const reach = { kind: 'riichi', tile: '9p', label: 'Riichi 9p', weight: .97, conditionalWeight: 1 };
  const other = { kind: 'discard', tile: '3s', label: 'Discard 3s', weight: .02 };
  const advised = ruled => analysis({ choice: plain, choices: [reach, other, plain], ruled });
  const line = markup => /<div class="recommendation[^"]*">([^]*?)<\/div>/.exec(markup)[1];
  const body = await html('AgentWeights.svelte', { analysis: advised({ rules: [DAMA_RULE] }), onchoose() {} });
  assert.match(seen(line(body)), /Next choice: Discard 9p · by rule$/);
  assert.match(line(body), new RegExp(`<span class="rule[^"]*" title="${DAMA_LINE}">by rule</span>`));
  assert.match(heard(line(body)), /Next choice: Discard 9p, by rule \(last hand: every win keeps it first, so no riichi\)$/);
  // The rows keep the network's weights: the riichi it wanted on top, and
  // the move played ringed with its own small weight.
  assert.match(body, /class="weight-row[^"]*\bbest\b[^"]*" data-choice="Discard 9p"/);
  assert.equal((body.match(/class="weight-row[^"]*\bbest\b/g) ?? []).length, 1);
  assert.match(body, /97\.0%/); assert.match(body, /1\.0%/); assert.match(body, /100\.0%/);
  // Both rules: decline's line, then dama's.
  const both = line(await html('AgentWeights.svelte', { analysis: advised({ rules: [DECLINE_RULE, DAMA_RULE] }) }));
  assert.match(both, new RegExp(`title="${DECLINE_LINE} · ${DAMA_LINE}">by rule<`));
  assert.match(heard(both), /, by rule \(last hand: this win would leave it last · Last hand: every win keeps it first, so no riichi\)$/);
  // No rule, and Club, which no rule plays: no tag.
  assert.doesNotMatch(await html('AgentWeights.svelte', { analysis: advised(null) }), /by rule|visually-hidden/);
  assert.doesNotMatch(await html('AgentWeights.svelte', { analysis: { agent: 'club', kind: 'selection', choice: plain,
    choices: [{ ...plain, weight: 1 }, { ...reach, weight: 0 }] } }), /by rule/);
});

test('trained advice names the network that gave it by version', async () => {
  const body = await html('AgentWeights.svelte', { analysis: analysis() });
  assert.match(body, new RegExp(`<strong>Trained ${MANIFEST.version.replaceAll('.', '\\.')}</strong>`));
});

test('historical review distinguishes shared declaration weights from different conditional discards', async () => {
  const body = await html('ReviewPreference.svelte', { note: { advised_kind: 'riichi', played_kind: 'riichi', agreed: false,
    preferred_weight: .8, played_weight: .8, preferred_conditional_weight: .75, played_conditional_weight: .25 } });
  assert.equal((body.match(/to declare riichi/g) ?? []).length, 2);
  assert.match(body, /Discard given riichi: 75\.0%/);
  assert.match(body, /Discard given riichi: 25\.0%/);
  assert.doesNotMatch(body, /60\.0%/);
});

test("the preference line's title says what its percentages are, and at a riichi that its two weights are separate decisions", async () => {
  const title = markup => /<p class="policy-preference[^"]*" title="([^"]*)"/.exec(markup)?.[1];
  const weights = 'How strongly Trained AI prefers each move, not a chance of winning';
  const discards = { advised_kind: 'discard', played_kind: 'discard', agreed: false, preferred_weight: .8, played_weight: .1 };
  const plain = await html('ReviewPreference.svelte', { note: discards });
  assert.equal(title(plain), weights);
  // Asked for, not read: nothing of it in the line's own words.
  assert.doesNotMatch(text(plain), /chance of winning|separate decisions/);
  // A riichi on the line, Trained AI's, the player's or one they both
  // declared, adds that its declaration and its discard are not one move.
  const reach = { preferred_conditional_weight: .75, played_conditional_weight: .25 };
  for (const kinds of [{ advised_kind: 'riichi' }, { played_kind: 'riichi' }, { advised_kind: 'riichi', played_kind: 'riichi', agreed: true }]) {
    assert.equal(title(await html('ReviewPreference.svelte', { note: { ...discards, ...reach, ...kinds } })),
      `${weights}. Declaring riichi and the discard given riichi are weighed as separate decisions, not as one move.`, JSON.stringify(kinds));
  }
});

test("the preference line says \"by rule\" in place of the weight of a move one of the last hand's rules made, its line the title and said in full", async () => {
  // The probe, the player declaring riichi on the tile the rule threw
  // undeclared: the player's move keeps its weights.
  const note = { advised_kind: 'discard', advised_tile: '9p', played_kind: 'riichi', played_tile: '9p', agreed: false,
    preferred_weight: .01, played_weight: .97, preferred_conditional_weight: null, played_conditional_weight: 1, ruled: { rules: [DAMA_RULE] } };
  const body = await html('ReviewPreference.svelte', { note });
  assert.equal(seen(body), 'Trained preference: by rule · Your move: 97.0% to declare riichi · Discard given riichi: 100.0%');
  assert.equal(heard(body), 'Trained preference, by rule (last hand: every win keeps it first, so no riichi) · Your move: 97.0% to declare riichi '
    + '· Discard given riichi: 100.0%');
  assert.match(body, new RegExp(`<strong class="rule[^"]*" title="${DAMA_LINE}">by rule</strong>`));
  assert.doesNotMatch(body, /1\.0%/, "the rule's move is not weighed");
  // Agreed, the line says the rule alone; both rules give both lines.
  const agreed = await html('ReviewPreference.svelte', { note: { ...note, played_kind: 'discard', agreed: true, ruled: { rules: [DECLINE_RULE, DAMA_RULE] } } });
  assert.equal(seen(agreed), 'Trained preference: by rule');
  assert.match(agreed, new RegExp(`title="${DECLINE_LINE} · ${DAMA_LINE}">by rule<`));
  // No rule, the weight as ever.
  assert.equal(seen(await html('ReviewPreference.svelte', { note: { ...note, ruled: null } })),
    'Trained preference: 1.0% · Your move: 97.0% to declare riichi · Discard given riichi: 100.0%');
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
