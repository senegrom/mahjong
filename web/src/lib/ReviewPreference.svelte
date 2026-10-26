<script>
  import { percent } from './ui.js';
  // Presentation only. The selected move and both distributions come from
  // reviewWithStrong's existing one- or two-stage policy evaluation. What the
  // percentages are is the line's title, there when it is asked for rather
  // than read above every hand's notes; a riichi on the line adds that its
  // declaration and its discard are weighed apart. A title shows only under a
  // mouse, so the tile guide (AppSettings) says both as well.
  let { note } = $props();
  const WEIGHTS = 'How strongly Trained AI prefers each move, not a chance of winning';
  let riichi = $derived(note.advised_kind === 'riichi' || (!note.agreed && note.played_kind === 'riichi'));
  let title = $derived(riichi ? `${WEIGHTS}. Declaring riichi and the discard given riichi are weighed as separate decisions, not as one move.` : WEIGHTS);
</script>

{#snippet preference(weight, kind, conditional)}
  <strong>{weight == null ? 'Unscored' : percent(weight)}</strong>
  {#if kind === 'riichi'}
    <span> to declare riichi · Discard given riichi: {conditional == null ? 'Not evaluated' : percent(conditional)}</span>
  {/if}
{/snippet}

<p class="policy-preference" {title}>Trained preference: {@render preference(note.preferred_weight, note.advised_kind, note.preferred_conditional_weight)}
  {#if !note.agreed}<span class="played"> · Your move: {@render preference(note.played_weight, note.played_kind, note.played_conditional_weight)}</span>{/if}
</p>

<style>
  .policy-preference { margin: 0; font-size: .85rem; font-variant-numeric: tabular-nums; }
  strong { color: var(--gold); }
  .played { opacity: .75; }
</style>
