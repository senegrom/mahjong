<script>
  // Presentation only. The selected move and both distributions come from
  // reviewWithStrong's existing one- or two-stage policy evaluation.
  let { note } = $props();
  const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
</script>

{#snippet preference(weight, kind, conditional)}
  <strong>{weight == null ? 'Unscored' : percent(weight)}</strong>
  {#if kind === 'riichi'}
    <span> to declare riichi · Discard given riichi: {conditional == null ? 'Not evaluated' : percent(conditional)}</span>
  {/if}
{/snippet}

<p class="policy-preference">Trained preference: {@render preference(note.preferred_weight, note.advised_kind, note.preferred_conditional_weight)}
  {#if !note.agreed}<span class="played"> · Your move: {@render preference(note.played_weight, note.played_kind, note.played_conditional_weight)}</span>{/if}
</p>

<style>
  .policy-preference { margin: 0; font-size: .85rem; font-variant-numeric: tabular-nums; }
  strong { color: var(--gold); }
  .played { opacity: .75; }
</style>
