<script>
  import Tile from './Tile.svelte';
  import { AGENTS } from './agents.js';
  let { analysis = null, onchoose = null, disabled = false, dora = [] } = $props();
  const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
</script>

{#if analysis}
  <section class="weights" aria-label="Agent decision weights">
    <div class="recommendation"><strong>{AGENTS[analysis.agent]}</strong><span>Next choice: {analysis.choice.label}</span></div>
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
  @media (max-width: 400px) { meter { width: 48px; } }
</style>
