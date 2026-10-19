<script>
  import { onDestroy } from 'svelte';

  import { OPPONENT_LABELS, opponentName } from './opponents.js';
  import { playedNetwork } from './trained-network.js';
  import { calledSet, seatAnnouncement } from './ui.js';
  import { SEAT_NAMES as NAMES } from './tiles.js';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';
  import Tile from './Tile.svelte';

  /**
   * One opponent at the table. Where they sit is MatchTable's to lay out:
   * it places each seat in a wrapper named for its side.
   *
   * A review shows the table as it stood at a past decision, this seat's
   * concealed tiles face up: `hand` is them, `{ hand, drawn }` with the tile
   * just drawn held apart; in play it is null, and they stay face down.
   * There `position` names where the opponent sat, such as "Right", and
   * `note` says what the seat was doing, such as "Discarded".
   */
  let { seat, dealer = false, dora = [], thinking = false, hand = null, position = '', note = '' } = $props();

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
  // The badge comes and goes with its word, which is for the eye. A screen
  // reader listens to a live region only once it exists, and JAWS and NVDA
  // often miss one inserted along with its words, so the same news is said
  // by a region that is always here, naming the seat it is about.
  let spoken = $derived(seatAnnouncement(NAMES[seat.seat], seat.riichi, called));
  const shortScore = score => `${(score / 1000).toFixed(1)}k`;
  // Face-up tiles are a list named for the seat and the player in it, as a
  // screen reader moves from hand to hand: "South's tiles, Right (Club)".
  let heldName = $derived(`${NAMES[seat.seat]}'s tiles${position ? `, ${position}` : ''}`
    + `${OPPONENT_LABELS[seat.controller] ? ` (${OPPONENT_LABELS[seat.controller]})` : ''}`);
</script>

<section class="seat" class:turn={seat.turn} class:thinking aria-label="{NAMES[seat.seat]} seat">
  <header>
    <span class="wind">{NAMES[seat.seat]}</span>
    {#if position}<span class="position">{position}</span>{/if}
    {#if dealer}<span class="dealer-badge">Dealer</span>{/if}
    <span class="score score-full">{seat.score.toLocaleString()}</span>
    <span class="score score-short" title={seat.score.toLocaleString()}>{shortScore(seat.score)}</span>
    {#if OPPONENT_LABELS[seat.controller]}
      <!-- On a phone the version pushed a seat's header onto a third line: the
        badge says Trained alone there, the page's header shows the version, and
        a screen reader still hears it. -->
      <span class="opponent-type" data-controller={seat.controller} data-player={seat.player} role="img"
        aria-label={`${opponentName(seat.controller)} opponent`}>{OPPONENT_LABELS[seat.controller]}{#if seat.controller === 'neural'}<span
          class="opponent-version">{` ${playedNetwork().version}`}</span>{/if}</span>
    {/if}
    {#if seat.riichi}<span class="stick" title="declared riichi"></span>{/if}
    {#if announcement}
      <span class="called" aria-hidden="true">{announcement}</span>
    {/if}
    {#if note}<span class="note">{note}</span>{/if}
    <span class="visually-hidden" role="status">{spoken}</span>
  </header>

  {#if hand}
    <ul class="revealed" aria-label={heldName} data-seat-hand={seat.seat}>
      {#each hand.hand as tile, index (index)}<li><Tile {tile} size="tiny" dora={dora.includes(tile)} /></li>{/each}
      {#if hand.drawn}<li class="drawn"><Tile tile={hand.drawn} size="tiny" drawn dora={dora.includes(hand.drawn)} /></li>{/if}
    </ul>
  {:else}
    <div class="held" role="img" aria-label="{seat.hand_size} tiles in hand">
      {#each Array(Math.min(seat.hand_size, 14)) as _, index (index)}
        <span class="back"></span>
      {/each}
    </div>
  {/if}

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
    border: 1px solid transparent;
    min-width: 0;
    align-content: start;
    position: relative;
    background: rgba(2,25,16,.32);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
  }

  .turn {
    border-color: rgba(216,161,42,.88);
    background: rgba(3,31,20,.48);
    box-shadow: 0 0 0 1px rgba(216,161,42,.12), 0 0 22px rgba(216,161,42,.13), inset 0 1px 0 rgba(255,255,255,.04);
  }

  header {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.82rem;
  }

  .opponent-type {
    padding: 1px 6px;
    border-radius: 999px;
    background: rgba(255,255,255,.06);
    font-size: .66rem;
    line-height: 1.2;
    opacity: .9;
  }
  .seat :global(.melds + .pool) { margin-block-start: 4px; }

  .wind {
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .score {
    font-variant-numeric: tabular-nums;
    opacity: 0.8;
  }
  .score-short { display: none; }

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

  /* A review's face-up tiles: fourteen to a row, the tile just drawn apart as
     in the player's own hand, as large as the seat's width allows up to the
     size of its discards, or `--hand-tile` where the table sets a larger one
     (MatchTable). The row is measured as a container; where it is too narrow
     for tiles of 12 pixels, they wrap instead, so four hands fit any screen
     without its page scrolling sideways. 18 pixels are the thirteen gaps
     between the tiles and the drawn tile's own. */
  .revealed {
    --hand-cap: var(--hand-tile, var(--tile-width));
    container-type: inline-size;
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 1px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .revealed li { display: flex; --tile-width: max(24px, min(var(--hand-cap), calc((100cqi - 18px) / 7))); }
  .revealed .drawn { margin-inline-start: 5px; }

  .position { opacity: .8; }

  /* What the seat was doing at the decision a review shows: a word, never
     a colour alone, beside the gold edge of the seat whose turn it is. */
  .note {
    padding: 1px 6px;
    border-radius: 999px;
    background: rgba(216,161,42,.16);
    color: var(--gold);
    font-size: .66rem;
    font-weight: 650;
    line-height: 1.2;
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

  /* Before the wide-screen sizes: a short landscape window matches both. */
  @media (max-width: 760px), (max-height: 500px) and (orientation: landscape) {
    .seat { padding: 6px; gap: 4px; }
    header { font-size: .7rem; gap: 2px 5px; }
    .back { width: 4px; height: 9px; }
    .called { font-size: .65rem; margin-left: 0; }
    .note { padding-inline: 4px; font-size: .56rem; }
    .score-full { display: none; }
    .score-short { display: inline; }
    .dealer-badge { padding-inline: 4px; font-size: .5rem; }
    .opponent-version { display: none; }
  }
  @media (min-width: 761px) {
    .seat { min-height: 150px; padding: 12px 14px; border-radius: 14px; }
    .back { width: 8px; height: 17px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .called { animation: none; }
    .thinking::before { animation: none; opacity: .7; }
  }
</style>
