<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  /**
   * A discard row, six to a line as at a real table, with the riichi
   * declaration turned sideways, tiles thrown straight from the draw
   * shaded darker, and claimed tiles bordered in dark green. The tile's
   * name says the same, for the pointer and for screen readers.
   */
  let { discards = [], compact = false, dora = [] } = $props();
</script>

<div class="pool" class:compact role="group" aria-label="discards">
  {#each discards as discard, index (index)}
    <Tile
      tile={discard.tile}
      rotated={discard.riichi}
      claimed={discard.claimed}
      fromDraw={discard.drawn}
      dora={dora.includes(discard.tile)}
      size={compact ? 'tiny' : 'small'}
      title={`${tileWords(discard.tile)}${discard.claimed ? ', claimed' : ''}${discard.riichi ? ', riichi declaration' : ''}${discard.drawn ? ', discarded from the draw' : ''}${dora.includes(discard.tile) ? ', dora' : ''}`}
    />
  {/each}
</div>

<style>
  .pool {
    display: grid;
    grid-template-columns: repeat(6, max-content);
    align-items: end;
    gap: 2px;
    justify-content: start;
    align-content: start;
    /* The player's own row sits directly above their hand, so it grows
       rather than reserving space that would push the hand down. */
    min-height: calc(var(--tile-width) * 0.62 * 1.35);
  }

  /* Keep the caption visible, without reserving a blank row before any discard. */
  .pool:not(.compact):empty { min-height: 0; }

  .compact {
    grid-template-columns: repeat(6, max-content);
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
