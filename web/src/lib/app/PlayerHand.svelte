<script>
  import Tile from '../Tile.svelte';
  import HandTile from '../HandTile.svelte';
  import Discards from '../Discards.svelte';
  import Melds from '../Melds.svelte';
  import { heldSafeCount, analyzeDiscards, unseenTileCounts } from '../ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from '../tiles.js';

  // Selection and keyboard focus stay owned by App across turn changes.
  // Only read-only presentation and hypothetical discard analysis live here.
  let {
    view, engine, closed, hints, busy, blocked, discardChoices, picked, selected,
    canDiscard, selectTile, syncHandFocus, handElement = $bindable(null),
  } = $props();
  let me = $derived(view?.seats[0]);
  let handTiles = $derived(me ? [...me.hand, ...(me.drawn ? [me.drawn] : [])] : []);
  let myTurn = $derived(view?.phase === 'act' && me?.turn);
  let shownDora = $derived(hints ? (view?.dora_types ?? []) : []);
  let safeCount = $derived(heldSafeCount(view));
  let previewTile = $derived(handTiles[selected ?? picked] ?? null);
  // Recompute once when the engine offers a new decision, not on every hover,
  // selection or animation frame. The selected tile and all readiness rings
  // use the same hypothetical-discard results, including open hands.
  let discardHints = $derived(hints && myTurn && !busy && !blocked && !closed
    ? analyzeDiscards(engine, discardChoices) : new Map());
  let discardHint = $derived(previewTile ? discardHints.get(previewTile) ?? null : null);
  let displayWaits = $derived(discardHint?.waits ?? view?.waits ?? []);
  let displayLeft = $derived(discardHint?.waits_left ?? view?.waits_left ?? []);
  let remainingByTile = $derived(hints && view ? unseenTileCounts(view) : new Map());
</script>

<section class="mine" class:turn={myTurn} aria-label="your seat">
  <header>
    <strong>You are {NAMES[me.seat]}</strong>
    <span class="score">{me.score.toLocaleString()}</span>
    {#if me.seat === 'east'}<span class="my-dealer">Dealer</span>{/if}
    {#if me.riichi}<span class="riichi">Riichi</span>{/if}
    {#if hints}
      <span class="hint">
        {#if !discardHint && view.shanten < 0}Complete tile shape
        {:else if displayWaits.length}
          {discardHint ? `After discarding ${tileWords(previewTile)}, waiting on` : me.riichi || !me.drawn ? 'Waiting on' : 'Wait before this draw:'}
          {#each displayWaits as wait, index (index)}
            <span class="wait"><Tile tile={wait} size="tiny" dora={shownDora.includes(wait)} /><span class="remaining" class:none={displayLeft[index] === 0} aria-label="{displayLeft[index]} unseen">{displayLeft[index]}</span></span>
          {/each}
        {:else if (discardHint?.shanten ?? view.shanten) === 0}
          Select a discard to see its waits
        {:else}
          {#if discardHint}After discarding {tileWords(previewTile)}: {/if}
          {discardHint?.shanten ?? view.shanten} tile{(discardHint?.shanten ?? view.shanten) === 1 ? '' : 's'} from a wait
        {/if}
      </span>
    {/if}
  </header>
  {#if hints && (safeCount || view.dora.length || view.furiten)}
    <div class="hand-facts">
      {#if safeCount}<span class="safe-note">{safeCount} held tile{safeCount === 1 ? '' : 's'} safe against declared riichi</span>{/if}
      {#if view.dora.length}<span class="dora-note">{view.dora.length} dora han in hand</span>{/if}
      {#if view.furiten}<span class="furiten">Furiten — self-draw wins only</span>{/if}
    </div>
  {/if}
  <div class="hand" class:has-draw={Boolean(me.drawn)} role="group" aria-label="your tiles" aria-describedby={view.phase === 'over' ? undefined : 'hand-help'} tabindex="-1" bind:this={handElement} onfocusin={syncHandFocus}>
    {#each handTiles as tile, index (index)}
      <HandTile {tile} handIndex={index} onclick={() => selectTile(tile, index)}
        disabled={!canDiscard(tile)} muted={view.phase === 'over'} selected={myTurn && (picked === index || selected === index)}
        drawn={Boolean(me.drawn) && index === me.hand.length}
        discardShanten={discardHints.get(tile)?.shanten ?? null}
        remaining={remainingByTile.get(tile) ?? null} showRemaining={hints}
        safe={hints && view.phase !== 'over' && view.safe.includes(tile)} dora={shownDora.includes(tile)} />
    {/each}
  </div>
  {#if me.melds.length}<div class="my-melds"><Melds melds={me.melds} size="small" dora={shownDora} /></div>{/if}
  <div class="own-discards">
    <span class="caption">Your discards ({me.discards.length})</span>
    <Discards discards={me.discards} compact={false} dora={shownDora} />
  </div>
</section>

<style>
  .mine { display: grid; gap: 10px; padding: 10px 12px; border-radius: 12px; background: #0004; border: 1px solid #d8a12a60; min-width: 0; }
  .mine header { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; font-size: .9rem; }
  .score { font-variant-numeric: tabular-nums; }
  .riichi, .furiten { color: var(--warning-text); font-weight: 600; }
  .hint { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-left: auto; font-size: .85rem; }
  .wait { position: relative; display: inline-flex; margin-right: 4px; }
  .remaining { position: absolute; right: -3px; bottom: -2px; min-width: 12px; padding: 0 2px; border-radius: 6px; background: #231d12; color: #fff; font-size: .65rem; line-height: 1.2; text-align: center; }
  .remaining.none { background: #932812; }
  .hand-facts { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: .8rem; }
  .safe-note { color: #9cddb5; }
  .dora-note { color: var(--gold); }
  .hand { --draw-gap: 12px; display: flex; gap: 5px; align-items: end; flex-wrap: wrap; padding: 6px 4px; min-width: 0; }
  .hand :global(button.tile[data-drawn=true]) { margin-inline-start: var(--draw-gap); }
  /* The hand takes focus on every turn, so its ring is a hint, not a
     frame: softer than a control's, and set out from the tiles. */
  .hand:focus-visible { border-radius: 8px; outline: 2px solid rgba(216, 161, 42, 0.45); outline-offset: 6px; }
  .my-melds { padding: 4px; }
  /* Always in view: the row is your furiten record and what the table
     sees of you, so it is not folded away on any screen. */
  .own-discards { display: grid; gap: 4px; }
  .own-discards .caption { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6; }
  @media (max-width: 760px) {
    .mine { padding: 8px; gap: 4px; }
    .mine header { font-size: .8rem; }
    .hint { margin-left: 0; font-size: .75rem; }
    .hand { display: grid; grid-template-columns: repeat(7,minmax(0,1fr)); gap: 6px; padding: 6px 3px; }
    .hand :global(button.tile) { width: 100%; min-height: 44px; }
    /* Reserve the extra gap within the grid, keeping all tile faces equal
       sized and the drawn tile inside the viewport even on a 320px phone. */
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .mine { padding: 6px 8px; gap: 4px; }
    .mine header { font-size: .8rem; }
    .hand { display: grid; grid-template-columns: repeat(7,minmax(0,44px)); gap: 5px; padding: 6px 3px; }
    .own-discards .caption { font-size: .62rem; }
    .hand :global(button.tile) { width: 100%; min-height: 44px; }
    /* Reserve the extra gap within the grid, keeping all tile faces equal
       sized and the drawn tile inside the viewport even on a 320px phone. */
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
    .hint { margin-left: 0; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }

  .my-dealer {
    padding: 1px 7px;
    border: 1px solid rgba(216, 161, 42, .55);
    border-radius: 999px;
    color: var(--gold);
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  .mine.turn {
    border-color: rgba(216, 161, 42, .9);
    box-shadow: 0 0 0 1px rgba(216, 161, 42, .16), 0 0 24px rgba(216, 161, 42, .10);
  }

  /* Counts sit below both hand tiles and wait tiles; never cover artwork. */
  .wait {
    position: static;
    display: inline-grid;
    justify-items: center;
    align-items: end;
    gap: 3px;
    margin-right: 5px;
  }
  .remaining {
    position: static;
    min-width: 18px;
    padding: 2px 3px 0;
    border-top: 1px solid rgba(247, 242, 228, .26);
    border-radius: 0;
    background: none;
    color: rgba(247, 242, 228, .75);
    font-size: .62rem;
    font-weight: 650;
    line-height: 1;
  }
  .remaining.none {
    border-color: rgba(255, 184, 164, .55);
    background: none;
    color: var(--warning-text);
  }

  .hand { --draw-gap: 18px; align-items: flex-end; }
  .hand :global(button.tile[data-drawn=true]) { margin-inline-start: 0; }
  .hand :global(.hand-tile[data-hand-drawn=true]) { margin-inline-start: var(--draw-gap); }

  @media (min-width: 761px) and (min-height: 501px) {
    .mine {
      --tile-width: 52px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(4, 30, 20, .43);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
    }
  }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .mine { border-radius: 15px; background: rgba(3,29,19,.40); }
    .hand { --draw-gap: 14px; }
    .hand :global(.hand-tile) { width: 100%; }
    .hand :global(.hand-tile[data-hand-drawn=true]) { margin-inline-start: var(--draw-gap); }
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
  }
</style>
