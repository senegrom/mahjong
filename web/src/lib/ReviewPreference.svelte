<script>
  import { percent } from './ui.js';
  import { RULE_WORDS, ruleLine, ruleSaid } from './trained-move.js';
  // Presentation only. The selected move and both distributions come from
  // reviewWithStrong's existing one- or two-stage policy evaluation. What the
  // percentages are is the line's title, there when it is asked for rather
  // than read above every hand's notes; a riichi on the line adds that its
  // declaration and its discard are weighed apart. A title shows only under a
  // mouse, so the tile guide (AppSettings) says both as well. Where one of
  // the last hand's rules made Trained AI's move (`ruled`), its weight did
  // not choose it: the line says "by rule" in its place, the rule's line the
  // tag's title, and, for a screen reader, in full.
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

<p class="policy-preference" {title}>Trained preference{#if note.ruled}<span aria-hidden="true">: <strong class="rule" title={ruleLine(note.ruled)}>{RULE_WORDS.tag}</strong></span><span
  class="visually-hidden">{ruleSaid(note.ruled)}</span>{:else}: {@render preference(note.preferred_weight, note.advised_kind, note.preferred_conditional_weight)}{/if}
  {#if !note.agreed}<span class="played"> · Your move: {@render preference(note.played_weight, note.played_kind, note.played_conditional_weight)}</span>{/if}
</p>

<style>
  .policy-preference { margin: 0; font-size: .85rem; font-variant-numeric: tabular-nums; }
  strong { color: var(--gold); }
  /* "by rule", its dotted line saying that its title says more. */
  .rule { text-decoration: underline dotted; text-underline-offset: 3px; cursor: help; }
  .played { opacity: .75; }
</style>
