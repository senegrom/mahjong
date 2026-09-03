<script>
  import Tile from './Tile.svelte';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';

  /**
   * One opponent, placed around the table. `side` is where they sit from the
   * player's chair, which decides whether their tiles run across or down.
   */
  let { seat, side = 'across', dealer = false } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };
  let vertical = $derived(side === 'left' || side === 'right');
</script>

<section class="seat {side}" class:turn={seat.turn} aria-label="{NAMES[seat.seat]} seat">
  <header>
    <span class="wind" class:dealer>{NAMES[seat.seat]}</span>
    <span class="score">{seat.score.toLocaleString()}</span>
    {#if seat.riichi}<span class="stick" title="declared riichi"></span>{/if}
  </header>

  <div class="held" class:vertical aria-label="{seat.hand_size} tiles in hand">
    {#each Array(Math.min(seat.hand_size, 14)) as _, index (index)}
      <span class="back"></span>
    {/each}
  </div>

  {#if seat.melds.length}
    <Melds melds={seat.melds} size="tiny" />
  {/if}

  <Discards discards={seat.discards} compact />
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
    gap: 8px;
    font-size: 0.82rem;
  }

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

  /* Concealed tiles are shown as edges rather than faces: enough to count
     and to see a call, without a wall of colour competing with the board. */
  .held {
    display: flex;
    gap: 1px;
    flex-wrap: wrap;
  }

  .held.vertical {
    max-width: calc(var(--tile-width) * 1.6);
  }

  .back {
    width: 7px;
    height: 15px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--rail) 0%, var(--rail-dark) 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18);
    flex: none;
  }
</style>
