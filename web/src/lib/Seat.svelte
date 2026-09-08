<script>
  import { onDestroy } from 'svelte';

  import { OPPONENT_LABELS } from './opponents.js';
  import { calledSet } from './ui.js';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';

  /**
   * One opponent, placed around the table. `side` is where they sit from the
   * player's chair, which decides whether their tiles run across or down.
   */
  let { seat, side = 'across', dealer = false, dora = [], thinking = false } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };

  // How long a call is said for: long enough to read from across the
  // table, gone well before the next one is made.
  const SAID_FOR = 2500;

  // Riichi is a live state of the table, so it stays for as long as it
  // lasts. A call is an event, so it is said as it happens and then goes
  // quiet: the meld itself goes on showing what was called, and a badge
  // that stayed would be announcing something ten turns old.
  let called = $state('');
  let melds = null;
  let seated = null;
  // The seat's view is a new object on every event, so this effect runs
  // many times a turn. The timer is held here rather than returned as the
  // effect's teardown, which each of those runs would cancel.
  let timer;

  $effect(() => {
    const now = seat.melds.map((meld) => meld.kind);
    const person = seat.player;
    // Nobody remembered in this chair yet, or a different person in it,
    // is the first look or the next deal, and neither is a call.
    const was = seated === person ? melds : null;
    melds = now;
    seated = person;
    const word = calledSet(was, now);
    if (!word) return;
    called = word;
    clearTimeout(timer);
    timer = setTimeout(() => { called = ''; }, SAID_FOR);
  });

  onDestroy(() => clearTimeout(timer));

  let announcement = $derived(seat.riichi ? 'Riichi' : called);
  const shortScore = score => `${(score / 1000).toFixed(1)}k`;
</script>

<section class="seat {side}" class:turn={seat.turn} class:thinking aria-label="{NAMES[seat.seat]} seat">
  <header>
    <span class="wind" class:dealer>{NAMES[seat.seat]}</span>
    {#if dealer}<span class="dealer-badge" aria-label="dealer">Dealer</span>{/if}
    <span class="score score-full">{seat.score.toLocaleString()}</span>
    <span class="score score-short" title={seat.score.toLocaleString()}>{shortScore(seat.score)}</span>
    {#if OPPONENT_LABELS[seat.controller]}
      <span class="opponent-type" data-controller={seat.controller} data-player={seat.player}
        aria-label={`${OPPONENT_LABELS[seat.controller]} opponent`}>{OPPONENT_LABELS[seat.controller]}</span>
    {/if}
    {#if seat.riichi}<span class="stick" title="declared riichi"></span>{/if}
    {#if announcement}
      <span class="called" aria-live="polite">{announcement}</span>
    {/if}
  </header>

  <div class="held" aria-label="{seat.hand_size} tiles in hand">
    {#each Array(Math.min(seat.hand_size, 14)) as _, index (index)}
      <span class="back"></span>
    {/each}
  </div>

  {#if seat.melds.length}
    <Melds melds={seat.melds} size="tiny" {dora} />
  {/if}

  <Discards discards={seat.discards} compact {dora} />
</section>

<style>
  .seat {
    display: grid;
    gap: 6px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.18);
    border: 1px solid transparent;
    min-width: 0;
    align-content: start;
  }

  .turn {
    border-color: var(--gold);
    background: rgba(0, 0, 0, 0.28);
  }

  header {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.82rem;
  }

  .opponent-type { font-size: .66rem; line-height: 1.2; opacity: .9; }
  .seat :global(.melds + .pool) { margin-block-start: 4px; }

  .wind {
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .dealer::after {
    content: ' ●';
    color: var(--gold);
    font-size: 0.7em;
    vertical-align: 0.2em;
  }

  .score {
    font-variant-numeric: tabular-nums;
    opacity: 0.8;
  }

  /* A riichi stick, drawn rather than written: it is what sits on the table. */
  .stick {
    width: 26px;
    height: 5px;
    border-radius: 3px;
    background: var(--ivory);
    position: relative;
    align-self: center;
  }

  .stick::after {
    content: '';
    position: absolute;
    inset: 1px 11px;
    background: var(--accent);
    border-radius: 50%;
  }

  /* Riichi while it lasts, and a call as it is made, said where it happened. */
  .called {
    margin-left: auto;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--gold);
    animation: settle 0.5s ease-out forwards;
  }

  /* It arrives with a little emphasis, and a call fades again on its own. */
  @keyframes settle {
    0% {
      opacity: 0;
      transform: translateY(-3px) scale(1.15);
    }
    100% {
      opacity: 1;
      transform: none;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .called {
      animation: none;
    }
  }

  /* Concealed tiles are shown as edges rather than faces: enough to count
     and to see a call, without a wall of colour competing with the board. */
  .held {
    display: flex;
    gap: 1px;
    flex-wrap: wrap;
  }

  .back {
    width: 7px;
    height: 15px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--rail) 0%, var(--rail-dark) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18);
    flex: none;
  }
  @media (max-width: 760px), (max-height: 500px) and (orientation: landscape) {
    .seat { padding: 6px; gap: 4px; }
    header { font-size: .7rem; gap: 2px 5px; }
    .back { width: 4px; height: 9px; }
    .called { font-size: .65rem; margin-left: 0; }
  }


  .dealer::after { content: none; }
  .dealer-badge {
    padding: 1px 6px;
    border: 1px solid rgba(216,161,42,.48);
    border-radius: 999px;
    color: var(--gold);
    font-size: .56rem;
    font-weight: 750;
    letter-spacing: .07em;
    text-transform: uppercase;
  }
  .score-short { display: none; }
  .opponent-type {
    padding: 1px 6px;
    border-radius: 999px;
    background: rgba(255,255,255,.06);
  }
  .seat { position: relative; background: rgba(2,25,16,.32); box-shadow: inset 0 1px 0 rgba(255,255,255,.025); }
  .turn {
    border-color: rgba(216,161,42,.88);
    background: rgba(3,31,20,.48);
    box-shadow: 0 0 0 1px rgba(216,161,42,.12), 0 0 22px rgba(216,161,42,.13), inset 0 1px 0 rgba(255,255,255,.04);
  }
  .thinking::before {
    content: '';
    position: absolute;
    inset: -3px;
    pointer-events: none;
    border: 1px solid rgba(239,196,94,.72);
    border-radius: 13px;
    animation: thinking-pulse 1.35s ease-in-out infinite;
  }
  @keyframes thinking-pulse { 0%,100% { opacity: .22; transform: scale(.995); } 50% { opacity: .9; transform: scale(1.008); } }

  @media (min-width: 761px) {
    .seat { min-height: 150px; padding: 12px 14px; border-radius: 14px; }
    .back { width: 8px; height: 17px; }
  }
  @media (max-width: 760px), (max-height: 500px) and (orientation: landscape) {
    .score-full { display: none; }
    .score-short { display: inline; }
    .dealer-badge { padding-inline: 4px; font-size: .5rem; }
  }
  @media (prefers-reduced-motion: reduce) { .thinking::before { animation: none; opacity: .7; } }
</style>
