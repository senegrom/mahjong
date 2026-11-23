<script lang="ts">
  import type { MatchTableProps } from './types';
  import Tile from '../Tile.svelte';
  import Seat from '../Seat.svelte';
  import Discards from '../Discards.svelte';
  import Melds from '../Melds.svelte';
  import { SEAT_NAMES as NAMES } from '../tiles.js';
  import { OPPONENT_POSITIONS } from '../opponents.js';

  // A review draws the table as it stood at a past decision: `hands` puts
  // each seat's concealed tiles face up, in the order of view.seats, `notes`
  // says what each seat was doing, and `review` names each opponent by where
  // they sat, lays the table out for four hands in view and keeps the seats
  // from saying a call, which is an event of play (Seat). `concealed` turns
  // the three opponents' tiles face down again where they lie, until the
  // player asks to see them.
  let { view, hints, thinking, pendingOpponent, hands = null, notes = null, review = false, concealed = false }: MatchTableProps = $props();
  // A review replays a hand on a second table while the match's own stays
  // on the page, so the dialog's title is named for its own table.
  const uid = $props.id();
  let tableDialog = $state<HTMLDialogElement | null>(null);
  let shownDora = $derived(hints ? (view?.dora_types ?? []) : []);
  let right = $derived(view?.seats[1]);
  let across = $derived(view?.seats[2]);
  let left = $derived(view?.seats[3]);
  let uraIndicators = $derived(view?.outcome?.wins?.find(win => win.ura_indicators?.length)?.ura_indicators ?? []);
  // At a review's decision that answers another player's discard, the seat
  // that let it go, whose row outlines it: the engine puts the tile on offer
  // at the end of that seat's discards, as it is thrown. A quad offered to be
  // robbed holds all four of its tile, so none ends a row, and nothing is
  // outlined. Play's table names the tile on offer in the call itself
  // (TurnChoices), and outlines nothing.
  let offeredBy = $derived.by(() => {
    if (!review || view?.phase !== 'call' || !view.pending_discard) return null;
    const discards = view.seats.find(seat => seat.seat === view.pending_from)?.discards ?? [];
    return discards[discards.length - 1]?.tile === view.pending_discard ? view.pending_from : null;
  });
  function inspectTable() { tableDialog?.showModal(); }
</script>

<div class="board" class:review>
  <div class="place across"><Seat seat={across} dealer={across.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === across.player}
    hand={hands?.[2] ?? null} position={review ? OPPONENT_POSITIONS[1] : ''} note={notes?.[2] ?? ''} offered={across.seat === offeredBy} {concealed} {review} /></div>
  <div class="place left"><Seat seat={left} dealer={left.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === left.player}
    hand={hands?.[3] ?? null} position={review ? OPPONENT_POSITIONS[2] : ''} note={notes?.[3] ?? ''} offered={left.seat === offeredBy} {concealed} {review} /></div>
  <div class="centre" role="group" aria-label="the table">
    <div class="round-details">
      <div class="round"><strong>{NAMES[view.round]} {view.kyoku}</strong><span>Round</span></div>
      <div class="wall"><strong>{view.wall}</strong><span>tiles left</span></div>
    </div>
    <div class="table-indicators">
      <div class="indicator-line"><span>Dora indicators</span><div class="indicators" role="group" aria-label="dora indicators">
        {#each view.dora_indicators as indicator, slot (slot)}<Tile tile={indicator} size="small" />{/each}
      </div></div>
      {#if uraIndicators.length}
        <div class="indicator-line"><span>Ura-dora indicators</span><div class="ura-indicators" role="group" aria-label="ura-dora indicators">
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
    <button class="app-control inspect" onclick={inspectTable} aria-label="Inspect all discards and called sets">
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="3" y="3" width="5" height="6" rx="1" /><rect x="12" y="3" width="5" height="6" rx="1" /><rect x="3" y="12" width="5" height="5" rx="1" /><rect x="12" y="12" width="5" height="5" rx="1" /></svg>
      Discards
    </button>
  </div>
  <div class="place right"><Seat seat={right} dealer={right.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === right.player}
    hand={hands?.[1] ?? null} position={review ? OPPONENT_POSITIONS[0] : ''} note={notes?.[1] ?? ''} offered={right.seat === offeredBy} {concealed} {review} /></div>
</div>

<dialog class="table-dialog" bind:this={tableDialog} aria-labelledby="{uid}-title">
  <header><h2 id="{uid}-title">All discards and called sets</h2><button class="app-control" onclick={() => tableDialog?.close()}>Close</button></header>
  <div class="inspection-grid">
    {#each view.seats as seat, index (index)}
      <section><h3>{index === 0 ? 'You' : OPPONENT_POSITIONS[index - 1]} · {NAMES[seat.seat]} · {seat.score.toLocaleString()}</h3>
        <Discards discards={seat.discards} dora={shownDora} />
        {#if seat.melds.length}<Melds melds={seat.melds} dora={shownDora} />{/if}
        {#if seat.discards.length === 0}<p>No discards yet.</p>{/if}
      </section>
    {/each}
  </div>
</dialog>

<style>
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
  /* The honba and the bets in ivory: the middle of the centre is the table's
     lightest felt, where even the lighter gold of words is under 4 to 1. */
  .table-extras { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: .8rem; font-weight: 600; color: var(--ivory); }
  .inspect { display: inline-flex; align-items: center; justify-content: center; gap: 6px; font-size: .8rem; }
  .inspect svg { width: 17px; height: 17px; flex: none; stroke: currentColor; stroke-width: 1.25; }
  .table-dialog { max-width: min(900px, calc(100vw - 20px - env(safe-area-inset-left, 0px) - env(safe-area-inset-right, 0px))); width: 100%; max-height: 90dvh; background: var(--felt-deep); color: var(--ivory); border: 1px solid #d8a12a99; border-radius: 12px; padding: 14px; }
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
  /* On a phone each opponent has a third of the table's width, and in play
     their discards are drawn as large as six to a row fit there (Discards):
     the one size of every phone, 14 pixels wide and 11.5 on the narrowest,
     was hard to tell apart. A discard is half of --tile-width. A row is six
     of them, one a riichi's, which lies a third wider, with five gaps, here
     of a pixel, inside the seat's edge and its padding, here 4 pixels a
     side: 3.1667 times --tile-width and 15 pixels. So the seat is measured,
     as a container, and each width it can have names the size that fits it
     with half a pixel to spare: discards 12.5 pixels wide on a 320-pixel
     phone, 15 at 360, 16.5 at 390, 17.5 at 412 and 18.5 at 430, and never
     over 22. The sizes are steps, not a share of the seat (cqi), which a
     tile's own parts would take of the nearest container instead: an
     unpainted tile's face is one (DecisionReplay does the same). With one
     pixel between the tiles a ring around one is a pixel wide, a
     fourteenth of the tile as on a wide screen's rows, so that it covers
     no neighbour (Tile). A browser without container queries keeps the
     sizes above, which fit with the same pixel and padding: 28 takes 104
     pixels of a seat 111 wide at 360, and 23 takes 88 of 97 at 320. A seat
     under 88 pixels wide, on a screen under 292, has discards 10.5 wide,
     which fit one of 82. A review's table keeps its own (below), and its
     rows their two pixels, which the outline of a tile on offer needs
     (Discards): play outlines none. */
  @media (max-width: 760px) {
    /* Four pixels between the seats, as held sideways, for the discards. */
    .board:not(.review) { column-gap: 4px; }
  }
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .place { container: seat / inline-size; }
    .board:not(.review) .place :global(.seat) { padding-inline: 4px; }
    .board:not(.review) .place :global(.pool.compact) { column-gap: 1px; }
    .board:not(.review) .place :global(.pool.compact .tile) { --ring-width: 1px; }
    @container seat (max-width: 87.83px) { .board:not(.review) .place :global(.seat) { --tile-width: 21px; } }
    @container seat (min-width: 94.7px) { .board:not(.review) .place :global(.seat) { --tile-width: 25px; } }
    @container seat (min-width: 101px) { .board:not(.review) .place :global(.seat) { --tile-width: 27px; } }
    @container seat (min-width: 110.5px) { .board:not(.review) .place :global(.seat) { --tile-width: 30px; } }
    @container seat (min-width: 113.7px) { .board:not(.review) .place :global(.seat) { --tile-width: 31px; } }
    @container seat (min-width: 120px) { .board:not(.review) .place :global(.seat) { --tile-width: 33px; } }
    @container seat (min-width: 126.4px) { .board:not(.review) .place :global(.seat) { --tile-width: 35px; } }
    @container seat (min-width: 132.7px) { .board:not(.review) .place :global(.seat) { --tile-width: 37px; } }
    @container seat (min-width: 142.2px) { .board:not(.review) .place :global(.seat) { --tile-width: 40px; } }
    @container seat (min-width: 154.9px) { .board:not(.review) .place :global(.seat) { --tile-width: 44px; } }
  }
  /* Held sideways, a phone under 640 pixels wide has one column for the
     table and the hand, and every pixel of the table's height is one the
     hand lies lower, so there the discards stay small. */
  @media (max-width: 639.98px) and (max-height: 500px) and (orientation: landscape) {
    .board:not(.review) .place :global(.seat) { --tile-width: 28px; }
  }

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
    .wall { display: flex; gap: 4px; align-items: baseline; text-align: left; color: color-mix(in srgb, var(--ivory) 84%, transparent); }
    .wall strong, .wall span { font-size: .73rem; font-weight: 400; line-height: 1.5; }
    .wall strong { font-variant-numeric: tabular-nums; }
    .table-indicators { grid-area: 1 / 2; max-width: 140px; }
    .indicator-line { gap: 3px; font-size: .7rem; }
    .indicator-line > span { color: color-mix(in srgb, var(--ivory) 84%, transparent); }
    .indicators, .ura-indicators { --tile-width: 42px; gap: 3px; }
    .table-extras { grid-column: 1 / -1; gap: 4px 12px; padding-top: 5px; border-top: 1px solid #ffffff14; font-size: .7rem; }
    .inspect { grid-area: 1 / 3; padding: 8px 10px; border-color: transparent; border-radius: 10px; background: #ffffff0d; font-size: .76rem; }
  }
  /* The dense table: the centre a bar across the top, and the three
     opponents in one row under it, Left, Opposite and Right, each seat as
     tall as what it holds. A tablet has it, since its tracks fit before the
     three-wide table's minimum (968px). So does any wider window under 900
     pixels high, where the opponents around the centre left the player's
     hand no room: in a 1366 by 768 laptop's browser, 650 pixels high, the
     hand lay wholly under the screen as the turn began, and at 1536 by 730
     half of it did. The dense table is some 160 pixels shorter, and with it
     the hand is on the screen from 600 pixels high (App closes the page's
     rows up under 700). From 900 the table has its room around the centre.
     The discards are 22 pixels wide, as a review's dense table draws them
     (below), where a tablet's were 14. */
  @media (min-width: 761px) and (min-height: 501px) and (max-width: 967.98px), (min-width: 761px) and (min-height: 501px) and (max-height: 899.98px) {
    .board { grid-template-columns: repeat(3, minmax(0, 1fr)); min-height: 0; gap: 8px; padding: 12px; }
    .centre { grid-area: 1 / 1 / 2 / -1; width: 100%; max-width: none; min-height: 0; padding: 8px 12px; align-self: start; justify-self: stretch; }
    .place { --tile-width: 44px; width: 100%; min-width: 0; align-self: start; justify-self: stretch; container: seat / inline-size; }
    .left { grid-area: 2 / 1; }
    .across { grid-area: 2 / 2; }
    .right { grid-area: 2 / 3; }
    .board :global(.seat) { min-height: 0; }
    /* A seat 300 pixels wide or more, as a third of a wide window is, has
       room for its discards beside its name, score and called sets rather
       than under them, and is then only as tall as its three rows of
       discards: with them underneath, a riichi's line or a called set in
       one seat put the hand's foot under a 650-pixel screen again. A review
       shows each seat's hand across its whole width, so its seats keep
       their one column. */
    @container seat (min-width: 300px) {
      .board:not(.review) :global(.seat) { grid-template-columns: minmax(0, 1fr) auto; column-gap: 14px; align-items: start; }
      .board:not(.review) :global(.seat > *) { grid-column: 1; }
      .board:not(.review) :global(.seat > .pool) { grid-column: 2; grid-row: 1 / span 4; margin-block-start: 0; }
    }
  }

  /* A review shows every hand face up (Seat), each concealed tile up to half
     of `--hand-tile` wide where its seat has room: 24 pixels, so that only a
     narrow seat would draw a row of fourteen under 18 pixels, and there they
     take two rows instead (Seat). On a phone held upright three columns would
     leave each hand a few tiles to a row, so there each opponent takes the
     width of the screen instead, in the order they play after the player,
     their discards a size larger than in play; held sideways the columns
     stay, their discards a little larger too. */
  .board.review .place { --hand-tile: 48px; }
  /* From 761 pixels wide, at every height over 500, a review's table is the
     dense one the tablet's widths have, at every width: the centre as a bar
     across the top, and the three opponents in one row under it, Left,
     Opposite and Right, each seat as tall as what it holds rather than a
     height kept for it, so that the table and the player's seat fit the
     screen together. A wide screen's table, the opponents around the centre,
     left most of it empty felt. The discards are 22 pixels wide, a size
     between a tablet's and a wide screen's in play. Play's table keeps its
     own layouts. */
  @media (min-width: 761px) and (min-height: 501px) {
    .board.review { grid-template-columns: repeat(3, minmax(0, 1fr)); min-height: 0; gap: 8px; padding: 12px; }
    .board.review .centre { grid-area: 1 / 1 / 2 / -1; width: 100%; max-width: none; min-height: 0; padding: 12px;
      align-self: start; justify-self: stretch; }
    .board.review .place { --tile-width: 44px; width: 100%; min-width: 0; align-self: start; justify-self: stretch; }
    .board.review .left { grid-area: 2 / 1; }
    .board.review .across { grid-area: 2 / 2; }
    .board.review .right { grid-area: 2 / 3; }
    .board.review :global(.seat) { min-height: 0; }
  }
  @media (max-width: 760px) and (orientation: portrait) {
    .board.review { grid-template-columns: minmax(0, 1fr); }
    .board.review .right { grid-area: 2 / 1; }
    .board.review .across { grid-area: 3 / 1; }
    .board.review .left { grid-area: 4 / 1; }
    .board.review .place { --tile-width: 36px; }
  }
  @media (max-width: 359px) and (orientation: portrait) {
    .board.review .place { --tile-width: 32px; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .board.review .place { --tile-width: 28px; }
  }
</style>
