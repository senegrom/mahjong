<script lang="ts">
  import type { TurnChoicesProps } from './types';
  import Tile from '../Tile.svelte';
  import { callLabel, callTiles, quadOnOffer } from '../ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from '../tiles.js';

  // All actions return to App's guarded match session. The controls do not
  // choose a fallback move or mutate a saved game on their own.
  let {
    view, shownDora, busy, thinking, failure, saveConflict, loadNote,
    pendingOpponent, myTurn, confirmDiscards, shortcuts, touch,
    selectedTile, callChoices, choose, discard, oncancel,
  }: TurnChoicesProps = $props();
  // The discard a call is offered on, and the tile chosen while its discard
  // waits to be confirmed. Each has a panel that comes and goes with it.
  let offered = $derived(view.phase === 'call' && callChoices.length ? view.pending_discard : null);
  let chosen = $derived(myTurn && !busy ? selectedTile : null);
  let discarder = $derived((view.pending_from ? NAMES[view.pending_from] : undefined) ?? 'An opponent');
  // A quad that may be robbed is offered as its tile, the way a discard is,
  // but nobody discarded it: the win takes the tile added to the quad.
  let robbing = $derived(Boolean(quadOnOffer(view)));
  // A screen reader listens to a live region only once it exists, and JAWS
  // and NVDA often miss one inserted along with its words, as the panels
  // are. So their news is said by one region that is always here.
  let announcement = $derived(offered ? `${discarder} ${robbing ? 'declared a kan of' : 'discarded'} the ${tileWords(offered)}`
    : chosen ? `Selected: ${tileWords(chosen)}` : '');
</script>

<p class="visually-hidden" role="status">{announcement}</p>
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
{#if offered}
  <div class="call-stage">
    <span class="call-kicker">{robbing ? 'Kan' : 'Discard'}</span>
    <Tile tile={offered} dora={shownDora.includes(offered)} />
    <span class="call-message"><strong>{discarder}</strong><span>{robbing ? 'declared a kan of' : 'discarded'} the {tileWords(offered)}</span></span>
  </div>
{/if}
{#if chosen}
  <div class="confirm-discard">
    <span>Selected: {tileWords(chosen)}</span>
    <button class="app-control primary" onclick={() => { if (selectedTile) discard(selectedTile); }}>Discard {tileWords(chosen)}</button>
    <button class="app-control" onclick={oncancel}>Cancel</button>
  </div>
{/if}
{#if callChoices.length}
  <div class="call-options">
    {#each callChoices as choice, index (index)}
      <button class="app-control" class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} class:win-call={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}
        aria-label={callLabel(choice, robbing)} disabled={busy || Boolean(failure || saveConflict)} onclick={() => choose(choice)}>
        <span class="call-label">{callLabel(choice, robbing)}</span>
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
  .prompt { margin: 0; font-size: .9rem; }
  .key-help { display: block; font-size: .78rem; }
  .call-options { display: flex; flex-wrap: wrap; gap: 8px; }
  .call-options button { display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; text-align: left; }
  .call-label { min-width: 0; overflow-wrap: anywhere; }
  .call-preview { display: inline-flex; align-items: center; gap: 4px; }
  .confirm-discard { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .confirm-discard > span { font-size: .85rem; }
  @media (max-width: 760px) {
    .prompt { font-size: .82rem; }
    .call-options { gap: 6px; }
    .call-options button { font-size: .8rem; padding: 6px 10px; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .prompt { font-size: .8rem; }
  }

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
