<script>
  import Tile from './Tile.svelte';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';

  /** One opponent: their name plate, called sets, face-down hand and discards. */
  let { seat, dealer = false, compact = true } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };
</script>

<section class="seat" class:turn={seat.turn} aria-label="{NAMES[seat.seat]} seat">
  <header>
    <span class="wind" class:dealer>{NAMES[seat.seat]}</span>
    <span class="score">{seat.score.toLocaleString()}</span>
    {#if seat.riichi}<span class="riichi">riichi</span>{/if}
  </header>

  <div class="tiles" aria-label="concealed tiles">
    {#each Array(Math.min(seat.hand_size, 14)) as _, index (index)}
      <Tile facedown size="tiny" />
    {/each}
  </div>

  {#if seat.melds.length}
    <Melds melds={seat.melds} size="tiny" />
  {/if}

  <Discards discards={seat.discards} {compact} />
</section>

<style>
  .seat {
    display: grid;
    gap: 6px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.16);
    border: 1px solid transparent;
    min-width: 0;
  }

  .turn {
    border-color: var(--gold);
    box-shadow: 0 0 0 1px rgba(216, 161, 42, 0.35);
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: 0.85rem;
  }

  .wind {
    font-weight: 600;
    letter-spacing: 0.04em;
  }

  .dealer::after {
    content: ' ●';
    color: var(--gold);
  }

  .score {
    font-variant-numeric: tabular-nums;
    opacity: 0.85;
  }

  .riichi {
    color: var(--accent);
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .tiles {
    display: flex;
    gap: 1px;
    flex-wrap: wrap;
  }
</style>
