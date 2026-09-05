<script>
  import Tile from './Tile.svelte';

  /**
   * A discard row, six to a line as at a real table, with the riichi
   * declaration turned sideways and claimed tiles greyed out.
   */
  let { discards = [], compact = false, dora = [] } = $props();
</script>

<div class="pool" class:compact aria-label="discards">
  {#each discards as discard (discard.tile + '-' + discards.indexOf(discard))}
    <Tile
      tile={discard.tile}
      rotated={discard.riichi}
      dimmed={discard.claimed}
      dora={dora.includes(discard.tile)}
      size={compact ? 'tiny' : 'small'}
      title={discard.claimed ? `${discard.tile}, claimed` : discard.tile}
    />
  {/each}
</div>

<style>
  .pool {
    display: grid;
    grid-template-columns: repeat(6, calc(var(--tile-width) * 0.62));
    gap: 2px;
    justify-content: start;
    align-content: start;
    /* The player's own row sits directly above their hand, so it grows
       rather than reserving space that would push the hand down. */
    min-height: calc(var(--tile-width) * 0.62 * 1.35);
  }

  .compact {
    grid-template-columns: repeat(6, calc(var(--tile-width) * 0.5));
    min-height: calc(var(--tile-width) * 0.5 * 1.35 * 3);
  }

  /* Down a phone the seats are stacked, so reserved space costs scrolling
     rather than steadiness. The rows grow as the discards come. */
  @media (max-width: 760px) {
    .compact {
      min-height: calc(var(--tile-width) * 0.5 * 1.35);
    }
  }
</style>
