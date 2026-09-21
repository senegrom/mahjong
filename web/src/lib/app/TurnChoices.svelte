<script>
  import Tile from '../Tile.svelte';
  import { callLabel, callTiles } from '../ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from '../tiles.js';

  // All actions return to App's guarded match session. The controls do not
  // choose a fallback move or mutate a saved game on their own.
  let {
    view, shownDora, busy, thinking, failure, saveConflict, loadNote,
    pendingOpponent, myTurn, confirmDiscards, shortcuts, touch,
    selectedTile, callChoices, choose, discard, oncancel,
  } = $props();
</script>

<p id="hand-help" class="prompt" role="status">
  {#if saveConflict}Reload the latest match to continue here.
  {:else if failure}Choose a recovery option above to continue.
  {:else if busy}{thinking ? (loadNote || (pendingOpponent ? `${pendingOpponent.position} Trained opponent is thinking…` : 'Trained opponents are thinking…')) : 'Playing the turn…'}
  {:else if myTurn}
    {confirmDiscards ? 'Your turn. Select a tile, then confirm the discard.' : 'Your turn. Choose a tile to discard.'}
    {#if !touch && shortcuts}<span class="key-help">Focus your hand to use arrows and Enter.</span>{/if}
  {:else if !callChoices.length}Waiting for the others…
  {:else}Choose a call, or pass.{/if}
</p>
{#if view.phase === 'call' && view.pending_discard && callChoices.length}
  <div class="call-stage" aria-live="polite">
    <span class="call-kicker">Discard</span>
    <Tile tile={view.pending_discard} dora={shownDora.includes(view.pending_discard)} />
    <span class="call-message"><strong>{NAMES[view.pending_from] ?? 'An opponent'}</strong><span>discarded the {tileWords(view.pending_discard)}</span></span>
  </div>
{/if}
{#if selectedTile && myTurn && !busy}
  <div class="confirm-discard" aria-live="polite">
    <span>Selected: {tileWords(selectedTile)}</span>
    <button class="primary" onclick={() => discard(selectedTile)}>Discard {tileWords(selectedTile)}</button>
    <button onclick={oncancel}>Cancel</button>
  </div>
{/if}
{#if callChoices.length}
  <div class="call-options">
    {#each callChoices as choice, index (index)}
      <button class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} class:win-call={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}
        aria-label={callLabel(choice)} disabled={busy || Boolean(failure)} onclick={() => choose(choice)}>
        <span class="call-label">{callLabel(choice)}</span>
        {#if callTiles(choice, view.pending_discard).length}
          <span class="call-preview" aria-hidden="true">
            {#each callTiles(choice, view.pending_discard) as tile, slot (slot)}<Tile {tile} size="tiny" />{/each}
          </span>
        {/if}
      </button>
    {/each}
  </div>
{/if}

<style>
  button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 12px; font: inherit; }
  button { cursor: pointer; touch-action: manipulation; }
  button:hover:not(:disabled) { background: #0007; }
  button:disabled { opacity: .55; cursor: default; }
  .prompt { margin: 0; font-size: .9rem; }
  .key-help { display: block; font-size: .78rem; }
  .call-options { display: flex; flex-wrap: wrap; gap: 8px; }
  .call-options button { display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; text-align: left; }
  .call-label { min-width: 0; overflow-wrap: anywhere; }
  .call-preview { display: inline-flex; align-items: center; gap: 4px; }
  .confirm-discard { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .confirm-discard > span { font-size: .85rem; }
  button.primary { background: var(--button-accent); color: var(--button-text); border-color: var(--button-accent); font-weight: 600; }
  button.primary:hover { background: var(--button-accent-hover); }
  @media (max-width: 760px) {
    .prompt { font-size: .82rem; }
    .call-options { gap: 6px; }
    .call-options button { font-size: .8rem; padding: 6px 10px; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .prompt { font-size: .8rem; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }

  .call-stage {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    align-items: center;
    gap: 9px 12px;
    padding: 10px 12px;
    border: 1px solid rgba(216, 161, 42, .45);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(216,161,42,.11), rgba(0,0,0,.23));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    animation: offer-in .24s ease-out both;
  }
  .call-kicker {
    grid-column: 1 / -1;
    margin-bottom: -5px;
    color: var(--gold);
    font-size: .64rem;
    font-weight: 750;
    letter-spacing: .14em;
    text-transform: uppercase;
  }
  .call-message { display: grid; min-width: 0; line-height: 1.25; }
  .call-message span { opacity: .78; font-size: .82rem; }
  .call-options { animation: choices-in .22s .06s ease-out both; }
  .call-options button.win-call {
    border-color: #efc45e;
    background: linear-gradient(180deg, #b93b24, #8e2414);
    box-shadow: 0 5px 16px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.16);
  }
  @keyframes offer-in { from { opacity: 0; transform: translateY(5px) scale(.985); } to { opacity: 1; transform: none; } }
  @keyframes choices-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .call-stage { grid-template-columns: auto auto minmax(0,1fr); padding: 9px 10px; }
  }
</style>
