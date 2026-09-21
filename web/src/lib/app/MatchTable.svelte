<script>
  import Tile from '../Tile.svelte';
  import Seat from '../Seat.svelte';
  import Discards from '../Discards.svelte';
  import Melds from '../Melds.svelte';
  import { SEAT_NAMES as NAMES } from '../tiles.js';

  let { view, hints, thinking, pendingOpponent } = $props();
  let tableDialog = $state(null);
  let shownDora = $derived(hints ? (view?.dora_types ?? []) : []);
  let right = $derived(view?.seats[1]);
  let across = $derived(view?.seats[2]);
  let left = $derived(view?.seats[3]);
  let uraIndicators = $derived(view?.outcome?.wins?.find(win => win.ura_indicators?.length)?.ura_indicators ?? []);
  function inspectTable() { tableDialog?.showModal(); }
</script>

<div class="board">
  <div class="place across"><Seat seat={across} side="across" dealer={across.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === across.player} /></div>
  <div class="place left"><Seat seat={left} side="left" dealer={left.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === left.player} /></div>
  <div class="centre" aria-label="the table">
    <div class="round-details">
      <div class="round"><strong>{NAMES[view.round]} {view.kyoku}</strong><span>Round</span></div>
      <div class="wall"><strong>{view.wall}</strong><span>tiles left</span></div>
    </div>
    <div class="table-indicators">
      <div class="indicator-line"><span>Dora</span><div class="indicators" aria-label="dora indicators">
        {#each view.dora_indicators as indicator, slot (slot)}<Tile tile={indicator} size="small" />{/each}
      </div></div>
      {#if uraIndicators.length}
        <div class="indicator-line"><span>Ura-dora</span><div class="ura-indicators" aria-label="ura-dora indicators">
          {#each uraIndicators as indicator, slot (slot)}<Tile tile={indicator} size="small" />{/each}
        </div></div>
      {/if}
    </div>
    {#if view.counters || view.riichi_sticks}
      <div class="table-extras">
        {#if view.counters}<span>{view.counters} honba (+{view.counters * 300})</span>{/if}
        {#if view.riichi_sticks}<span>{view.riichi_sticks} riichi bet{view.riichi_sticks === 1 ? '' : 's'}</span>{/if}
      </div>
    {/if}
    <button class="inspect" onclick={inspectTable} aria-label="Inspect all discards and called sets">
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="3" y="3" width="5" height="6" rx="1" /><rect x="12" y="3" width="5" height="6" rx="1" /><rect x="3" y="12" width="5" height="5" rx="1" /><rect x="12" y="12" width="5" height="5" rx="1" /></svg>
      Discards
    </button>
  </div>
  <div class="place right"><Seat seat={right} side="right" dealer={right.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === right.player} /></div>
</div>

<dialog class="table-dialog" bind:this={tableDialog} aria-labelledby="table-dialog-title">
  <header><h2 id="table-dialog-title">All discards and called sets</h2><button onclick={() => tableDialog.close()}>Close</button></header>
  <div class="inspection-grid">
    {#each view.seats as seat, index (index)}
      <section><h3>{index === 0 ? 'You' : NAMES[seat.seat]} · {NAMES[seat.seat]} · {seat.score.toLocaleString()}</h3>
        <Discards discards={seat.discards} dora={shownDora} />
        {#if seat.melds.length}<Melds melds={seat.melds} dora={shownDora} />{/if}
        {#if seat.discards.length === 0}<p>No discards yet.</p>{/if}
      </section>
    {/each}
  </div>
</dialog>

<style>
  button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 12px; font: inherit; }
  button { cursor: pointer; touch-action: manipulation; }
  button:hover:not(:disabled) { background: #0007; }
  button:disabled { opacity: .55; cursor: default; }
  .board { display: grid; grid-template-columns: minmax(0,1fr) minmax(190px,auto) minmax(0,1fr); gap: 10px; align-items: start; }
  .across { grid-area: 1 / 2; }
  .left { grid-area: 2 / 1; justify-self: start; }
  .right { grid-area: 2 / 3; justify-self: end; }
  .centre { grid-area: 2 / 2; display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px 18px; max-width: 350px; border-radius: 12px; padding: 12px; background: #0004; }
  .round-details { display: contents; }
  .round, .wall { display: grid; text-align: center; }
  .round strong, .wall strong { font-size: 1.3rem; }
  .round span, .wall span { font-size: .7rem; }
  .table-indicators { display: grid; gap: 6px; min-width: 0; }
  .indicator-line { display: grid; gap: 4px; font-size: .65rem; }
  .ura-indicators { display: flex; gap: 4px; flex-wrap: wrap; align-items: flex-end; }
  .indicators { display: flex; gap: 4px; flex-wrap: wrap; align-items: flex-end; }
  .table-extras { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: .8rem; color: var(--gold); }
  .inspect { display: inline-flex; align-items: center; justify-content: center; gap: 6px; font-size: .8rem; }
  .inspect svg { width: 17px; height: 17px; flex: none; stroke: currentColor; stroke-width: 1.25; }
  .table-dialog { max-width: min(900px, calc(100vw - 20px)); width: 100%; max-height: 90dvh; background: var(--felt-deep); color: var(--ivory); border: 1px solid #d8a12a99; border-radius: 12px; padding: 14px; }
  .table-dialog::backdrop { background: #0009; }
  .table-dialog header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .table-dialog h2 { margin: 0; font-size: 1rem; }
  .inspection-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,280px),1fr)); gap: 16px; --tile-width: 60px; }
  .inspection-grid section { min-width: 0; display: grid; gap: 8px; align-content: start; }
  .inspection-grid h3 { font-size: .85rem; }
  @media (max-width: 760px) {
    .board { grid-template-columns: repeat(3,minmax(0,1fr)); gap: 6px; }
    .centre { grid-area: 1 / 1 / 2 / -1; max-width: none; justify-content: space-between; gap: 6px 10px; padding: 8px; }
    .round strong, .wall strong { font-size: 1rem; }
    .round span, .wall span { font-size: .65rem; }
    .indicators, .ura-indicators { --tile-width: 36px; }
    .left { grid-area: 2 / 1; justify-self: stretch; }
    .across { grid-area: 2 / 2; }
    .right { grid-area: 2 / 3; justify-self: stretch; }
    .place { --tile-width: 28px; }
  }
  @media (max-width: 359px) {
    .place { --tile-width: 23px; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .board { grid-column: 1; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 4px; }
    .centre { grid-area: 1 / 1 / 2 / -1; max-width: none; gap: 4px 8px; padding: 6px; }
    .left { grid-area: 2 / 1; justify-self: stretch; }
    .across { grid-area: 2 / 2; }
    .right { grid-area: 2 / 3; justify-self: stretch; }
    .place { --tile-width: 23px; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }

  @media (min-width: 761px) and (min-height: 501px) {
    .board {
      grid-template-columns: minmax(260px, 1fr) minmax(330px, .92fr) minmax(260px, 1fr);
      min-height: 390px;
      padding: 18px 20px;
      gap: 14px 20px;
      border: 1px solid rgba(247,242,228,.10);
      border-radius: 28px;
      background:
        radial-gradient(ellipse at center, rgba(78,150,109,.18) 0 26%, transparent 57%),
        linear-gradient(145deg, rgba(255,255,255,.025), rgba(0,0,0,.13));
      box-shadow: inset 0 0 50px rgba(0,0,0,.13), 0 16px 34px rgba(0,0,0,.10);
    }
    .place { --tile-width: 54px; align-self: stretch; }
    .across { width: min(450px, 100%); justify-self: center; }
    .left, .right { width: min(340px, 100%); align-self: center; }
    .centre {
      width: min(340px, 100%);
      min-height: 190px;
      align-self: center;
      justify-self: center;
      padding: 18px;
      border: 1px solid rgba(216,161,42,.28);
      border-radius: 20px;
      background: radial-gradient(circle at 50% 35%, rgba(47,111,80,.54), rgba(3,25,16,.48));
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 12px 28px rgba(0,0,0,.14);
    }
    .inspect { border-radius: 999px; }
  }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {

    .board { margin-top: 3px; }
    .centre {
      display: grid;
      grid-template-columns: minmax(78px, 1fr) minmax(0, auto) auto;
      gap: 8px;
      padding: 11px 12px;
      border: 0;
      border-radius: 14px;
      background: #0002;
      box-shadow: none;
    }
    .round-details { display: grid; grid-area: 1 / 1; gap: 1px; }
    .round { text-align: left; }
    .round strong { font-size: 1.18rem; font-weight: 650; line-height: 1.35; letter-spacing: -.02em; }
    .round span { display: none; }
    .wall { display: flex; gap: 4px; align-items: baseline; text-align: left; color: color-mix(in srgb, var(--ivory) 68%, transparent); }
    .wall strong, .wall span { font-size: .73rem; font-weight: 400; line-height: 1.5; }
    .wall strong { font-variant-numeric: tabular-nums; }
    .table-indicators { grid-area: 1 / 2; max-width: 140px; }
    .indicator-line { gap: 3px; font-size: .6rem; }
    .indicator-line > span { color: color-mix(in srgb, var(--ivory) 68%, transparent); }
    .indicators, .ura-indicators { --tile-width: 42px; gap: 3px; }
    .table-extras { grid-column: 1 / -1; gap: 4px 12px; padding-top: 5px; border-top: 1px solid #ffffff14; font-size: .7rem; }
    .inspect { grid-area: 1 / 3; padding: 8px 10px; border-color: transparent; border-radius: 10px; background: #ffffff0d; font-size: .76rem; }
  }
</style>
