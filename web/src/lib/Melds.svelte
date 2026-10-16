<script>
  import Tile from './Tile.svelte';
  import { meldLayout, meldWords } from './ui.js';

  /**
   * Called sets, shown to the right of a player's tiles. The claimed tile is
   * turned sideways on the side the player it came from sits, which is how a
   * table shows who fed the set (EMA 2025 section 3.3.6). The two middle
   * tiles of a concealed quad are face down.
   */
  let { melds = [], size = 'small', dora = [], shape = null } = $props();

  function facedown(meld, index) {
    return meld.kind === 'concealed-kan' && (index === 1 || index === 2);
  }
</script>

<div class="melds">
  {#each melds as meld, meldIndex (meldIndex)}
    {@const layout = meldLayout(meld)}
    <div class="meld" role="group" aria-label={meldWords(meld)}>
      {#each layout.tiles as tile, index (index)}
        <Tile
          {tile}
          {size}
          rotated={index === layout.turned}
          facedown={facedown(meld, index)}
          dora={!facedown(meld, index) && dora.includes(tile)}
          inShape={Boolean(shape?.has(`meld:${meldIndex}:${index}`))}
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
