<script>
  import Tile from './Tile.svelte';
  import { AGENTS } from './agents.js';
  import { likeliest } from './beliefs.js';
  import { groupPolicyChoices } from './action-weights.js';
  let { analysis = null, onchoose = null, disabled = false, dora = [] } = $props();
  const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
  // A signed estimate of the model's mixed training target, not a place gain.
  const worth = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
  // A share of a hand, read as copies: a guess of .15 against thirteen tiles
  // is about two of them.
  const copies = entry => entry.expected == null
    ? percent(entry.chance)
    : `${entry.expected.toFixed(1)}×`;
  let reads = $derived(analysis?.beliefs?.map(belief => ({ ...belief, top: likeliest(belief, 7) })) ?? []);
  let rows = $derived(analysis?.kind === 'policy'
    ? groupPolicyChoices(analysis.choices) : (analysis?.choices ?? []).map(choice => ({ choice })));
</script>

{#if analysis}
  <section class="weights" aria-label="Agent decision weights">
    <div class="recommendation"><strong>{AGENTS[analysis.agent]}</strong><span>Next choice: {analysis.choice.label}</span>
      {#if analysis.value != null}<span class="worth" title="Estimated mixed training reward, not an expected finishing place or win probability">Estimated training reward {worth(analysis.value)}</span>{/if}
    </div>
    {#if analysis.value != null}<p class="reward-help">The current model's reward combines the hand's point change ÷ 4,000 with its final placement bonus. This is not a predicted finishing place, place gain or win probability.</p>{/if}
    <p>{analysis.kind === 'policy' ? 'Policy weights are preferences, not win probabilities. The agent selects in its original action space, then chooses a discard if it declares riichi. Grouped display weights do not determine its selected move.' : 'Built-in agents do not have policy percentages. Selected shows the exact move this agent will play, including any sampled tie-break.'}</p>
    {#if onchoose}<p>The blue border marks the agent’s choice. Select a move below to play it after confirmation.</p>{/if}
    {#snippet rowContent(choice, weight, conditional = false)}
      <span class="choice-label">{#if choice.tile}<Tile tile={choice.tile} size="tiny" dora={dora.includes(choice.tile)} />{/if}{choice.label}</span>
      {#if analysis.kind === 'policy' && weight == null}<strong title={conditional ? 'The agent did not evaluate the conditional discard stage' : 'The trained agent scores only the first kan of each kind'}>{conditional ? 'Not evaluated' : 'Unscored'}</strong>
      {:else if analysis.kind === 'policy'}<meter min="0" max="1" value={weight} aria-label={conditional ? `${choice.label}, given riichi` : choice.label}></meter><strong>{percent(weight)}</strong>
      {:else}<strong>{weight ? 'Selected' : '—'}</strong>{/if}
    {/snippet}
    {#snippet choiceRow(choice, conditional = false)}
      {@const selected = choice.kind === analysis.choice.kind && (choice.tile ?? null) === (analysis.choice.tile ?? null)}
      {@const weight = conditional ? choice.conditionalWeight : choice.weight}
      <div class="weight-row" class:best={selected} data-choice={choice.label}>
        {#if onchoose}
          <button class="choice-action" {disabled} onclick={() => onchoose(choice)} aria-label={`Play ${choice.label}`}>{@render rowContent(choice, weight, conditional)}</button>
        {:else}{@render rowContent(choice, weight, conditional)}{/if}
        {#if selected}<span class="selected-label">Agent choice</span>{/if}
      </div>
    {/snippet}
    <div class="weight-list">
      {#each rows as row, index (index)}
        {#if row.choice}{@render choiceRow(row.choice)}
        {:else}
          <div class="reach-group" role="group" aria-label="Riichi declaration and conditional discards">
            <div class="reach-heading"><strong>Declare riichi</strong><meter min="0" max="1" value={row.weight} aria-label="Riichi declaration weight"></meter><strong>{percent(row.weight)}</strong></div>
            <p>Discard weights given riichi form a separate distribution, not additional declaration weights.</p>
            {#if row.choices.every(choice => choice.conditionalWeight == null)}<p>Conditional discard weights are shown only when the agent evaluates this stage.</p>{/if}
            <div class="reach-discards">
              {#each row.choices as choice, index (index)}{@render choiceRow(choice, true)}{/each}
            </div>
          </div>
        {/if}
      {/each}
    </div>
    {#if reads.length}
      <section class="beliefs" aria-label="What the network reads the other hands as">
        <h3>Reading the other hands</h3>
        <p>Trained against the hands themselves, from the discards and calls it can see. A number is how many copies of that tile it expects the seat to be holding.</p>
        {#each reads as belief (belief.seat)}
          <div class="belief">
            <span class="who">{belief.relative}{#if belief.held != null}<em>{belief.held} tiles</em>{/if}</span>
            <span class="guesses">
              {#each belief.top as entry (entry.tile)}
                <span class="guess"><Tile tile={entry.tile} size="tiny" dora={dora.includes(entry.tile)} /><strong>{copies(entry)}</strong></span>
              {/each}
            </span>
          </div>
        {/each}
      </section>
    {/if}
  </section>
{/if}

<style>
  .weights { padding: 14px; border: 1px solid #d8a12a66; border-radius: 12px; background: #0003; min-width: 0; }
  .recommendation { display: flex; flex-wrap: wrap; gap: 6px 16px; }
  .recommendation > span { color: var(--gold); }
  p { font-size: .78rem; line-height: 1.45; opacity: .8; margin: 8px 0 12px; }
  .weight-list { max-height: 360px; overflow-y: auto; }
  .weight-row { position: relative; flex-wrap: wrap; display: flex; align-items: center; gap: 8px; padding: 5px 6px; border: 2px solid transparent; border-top-color: #ffffff15; border-radius: 8px; font-size: .82rem; margin-bottom: 4px; }
  .choice-action { display: flex; align-items: center; gap: 8px; width: 100%; min-height: 44px; border: 0; padding: 4px 0; background: transparent; color: inherit; font: inherit; text-align: left; cursor: pointer; }
  .choice-action:focus-visible { outline: 2px solid #4ea3ff; outline-offset: 2px; border-radius: 4px; }
  .choice-action:disabled { opacity: .5; cursor: default; }
  .choice-label { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; }
  .weight-row strong { min-width: 56px; text-align: right; font-variant-numeric: tabular-nums; }
  meter { flex-shrink: 0; width: 90px; accent-color: var(--gold); }
  .best { color: var(--gold); border-color: #4ea3ff; }
  .selected-label { flex-basis: 100%; font-size: .7rem; }
  .reach-group { margin: 4px 0 10px; border: 1px solid #ffffff28; border-radius: 8px; padding: 8px; }
  .reach-heading { display: flex; align-items: center; gap: 8px; font-size: .82rem; font-variant-numeric: tabular-nums; }
  .reach-heading > strong:first-child { flex: 1; min-width: 0; }
  .reach-heading > strong:last-child { min-width: 56px; text-align: right; }
  .reach-group p { margin: 6px 0; }
  .reach-discards { border-left: 2px solid #ffffff28; padding-left: 4px; }
  .worth { color: var(--gold); font-variant-numeric: tabular-nums; }
  .beliefs { margin-top: 14px; padding-top: 12px; border-top: 1px solid #ffffff22; }
  .beliefs h3 { margin: 0; font-size: .84rem; font-weight: 600; }
  .belief { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; padding: 5px 0; border-top: 1px solid #ffffff10; }
  .who { display: flex; align-items: baseline; gap: 6px; min-width: 84px; font-size: .8rem; }
  .who em { font-style: normal; font-size: .72rem; opacity: .6; font-variant-numeric: tabular-nums; }
  .guesses { display: flex; flex-wrap: wrap; gap: 4px 8px; flex: 1; min-width: 0; }
  .guess { display: flex; align-items: center; gap: 3px; font-size: .76rem; }
  .guess strong { min-width: 0; font-weight: 500; opacity: .85; font-variant-numeric: tabular-nums; }
  @media (max-width: 400px) { meter { width: 48px; } }
</style>
