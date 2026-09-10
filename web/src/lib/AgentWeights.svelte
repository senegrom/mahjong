<script>
  import Tile from './Tile.svelte';
  import { AGENTS } from './agents.js';
  import { likeliest } from './beliefs.js';
  let { analysis = null, onchoose = null, disabled = false, dora = [] } = $props();
  const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
  // The critic answers in placements, where a full point is one place at the
  // table. Shown signed, as a lead or a deficit, not dressed up as a chance.
  const worth = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
  // A share of a hand, read as copies: a guess of .15 against thirteen tiles
  // is about two of them.
  const copies = entry => entry.expected == null
    ? percent(entry.chance)
    : `${entry.expected.toFixed(1)}×`;
  let reads = $derived(analysis?.beliefs?.map(belief => ({ ...belief, top: likeliest(belief, 7) })) ?? []);
</script>

{#if analysis}
  <section class="weights" aria-label="Agent decision weights">
    <div class="recommendation"><strong>{AGENTS[analysis.agent]}</strong><span>Next choice: {analysis.choice.label}</span>
      {#if analysis.value != null}<span class="worth" title="What the network's critic makes of this hand for this seat, in places at the table">Hand worth {worth(analysis.value)}</span>{/if}
    </div>
    <p>{analysis.kind === 'policy' ? 'Policy weights across legal moves. These are preferences, not win probabilities; the agent plays its highest weight.' : 'Built-in agents do not have policy percentages. Selected shows the exact move this agent will play, including any sampled tie-break.'}</p>
    {#if onchoose}<p>The blue border marks the agent’s choice. Select a move below to play it after confirmation.</p>{/if}
    {#snippet rowContent(choice)}
      <span class="choice-label">{#if choice.tile}<Tile tile={choice.tile} size="tiny" dora={dora.includes(choice.tile)} />{/if}{choice.label}</span>
      {#if analysis.kind === 'policy' && choice.weight == null}<strong title="The trained agent scores only the first kan of each kind">Unscored</strong>
      {:else if analysis.kind === 'policy'}<meter min="0" max="1" value={choice.weight} aria-label={choice.label}></meter><strong>{percent(choice.weight)}</strong>
      {:else}<strong>{choice.weight ? 'Selected' : '—'}</strong>{/if}
    {/snippet}
    <div class="weight-list">
      {#each analysis.choices as choice, index (index)}
        <div class="weight-row" class:best={choice.kind === analysis.choice.kind && choice.tile === analysis.choice.tile}>
          {#if onchoose}
            <button class="choice-action" {disabled} onclick={() => onchoose(choice)} aria-label={`Play ${choice.label}`}>{@render rowContent(choice)}</button>
          {:else}{@render rowContent(choice)}{/if}
        </div>
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
  .weight-row { display: flex; align-items: center; gap: 8px; padding: 5px 6px; border: 2px solid transparent; border-top-color: #ffffff15; border-radius: 8px; font-size: .82rem; margin-bottom: 4px; }
  .choice-action { display: flex; align-items: center; gap: 8px; width: 100%; min-height: 44px; border: 0; padding: 4px 0; background: transparent; color: inherit; font: inherit; text-align: left; cursor: pointer; }
  .choice-action:focus-visible { outline: 2px solid #4ea3ff; outline-offset: 2px; border-radius: 4px; }
  .choice-action:disabled { opacity: .5; cursor: default; }
  .choice-label { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; }
  .weight-row strong { min-width: 56px; text-align: right; font-variant-numeric: tabular-nums; }
  meter { width: 90px; accent-color: var(--gold); }
  .best { color: var(--gold); border-color: #4ea3ff; }
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
