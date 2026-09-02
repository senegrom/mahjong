<script>
  import Tile from './Tile.svelte';

  /**
   * Called sets, shown to the right of a player's tiles. The claimed tile is
   * turned sideways on the side the player it came from sits, which is how a
   * table shows who fed the set (EMA 2025 section 3.3.6). The two middle
   * tiles of a concealed quad are face down.
   */
  let { melds = [], size = 'small' } = $props();

  function rotatedIndex(meld) {
    if (meld.kind === 'concealed-kan') return -1;
    if (meld.from === 'left') return 0;
    if (meld.from === 'across') return 1;
    return meld.tiles.length - 1;
  }

  function facedown(meld, index) {
    return meld.kind === 'concealed-kan' && (index === 1 || index === 2);
  }
</script>

<div class="melds">
  {#each melds as meld, meldIndex (meldIndex)}
    <div class="meld" aria-label="{meld.kind} of {meld.tiles[0]}">
      {#each meld.tiles as tile, index (index)}
        <Tile
          {tile}
          {size}
          rotated={index === rotatedIndex(meld)}
          facedown={facedown(meld, index)}
        />
      {/each}
    </div>
  {/each}
</div>

<style>
  .melds {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .meld {
    display: flex;
    align-items: flex-end;
    gap: 1px;
  }
</style>
