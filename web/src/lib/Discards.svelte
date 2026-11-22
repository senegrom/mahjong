<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  /**
   * A discard row, six to a line as at a real table, with the riichi
   * declaration turned sideways, tiles thrown straight from the draw
   * shaded darker, and claimed tiles bordered in dark green. The tile's
   * name says the same, for the pointer and for screen readers.
   *
   * `marked` is the index of a tile outlined in gold: in a review, the tile
   * on offer at a decision that answers it, in the row of the seat that let
   * it go, whose edge is the same gold (Seat). Its name says so too.
   */
  let { discards = [], compact = false, dora = [], marked = null } = $props();
</script>

{#snippet thrown(discard, offered)}
  <Tile
    tile={discard.tile}
    rotated={discard.riichi}
    claimed={discard.claimed}
    fromDraw={discard.drawn}
    dora={dora.includes(discard.tile)}
    size={compact ? 'tiny' : 'small'}
    title={`${tileWords(discard.tile)}${discard.claimed ? ', claimed' : ''}${discard.riichi ? ', riichi declaration' : ''}${discard.drawn ? ', discarded from the draw' : ''}${dora.includes(discard.tile) ? ', dora' : ''}${offered ? ', the tile on offer' : ''}`}
  />
{/snippet}

<div class="pool" class:compact role="group" aria-label="discards">
  {#each discards as discard, index (index)}
    {#if index === marked}<span class="offered">{@render thrown(discard, true)}</span>{:else}{@render thrown(discard, false)}{/if}
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

  /* The tile on offer: a gold outline around the tile, in the gap the row
     leaves between tiles, so it covers none of its neighbours and moves
     nothing as it comes and goes. It lies over a dora's ring, which the
     tile's shine and its name still give. */
  .offered {
    display: inline-flex;
    border-radius: 4px;
    outline: 2px solid var(--gold);
  }

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
