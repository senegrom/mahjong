<script>
  import Tile from './Tile.svelte';

  /**
   * A discard row, six to a line as at a real table, with the riichi
   * declaration turned sideways and claimed tiles greyed out.
   */
  let { discards = [], compact = false } = $props();
</script>

<div class="pool" class:compact aria-label="discards">
  {#each discards as discard (discard.tile + '-' + discards.indexOf(discard))}
    <Tile
      tile={discard.tile}
      rotated={discard.riichi}
      dimmed={discard.claimed}
      size={compact ? 'tiny' : 'small'}
      title={discard.claimed ? `${discard.tile}, claimed` : discard.tile}
    />
  {/each}
</div>

<style>
  .pool {
    display: grid;
    grid-template-columns: repeat(6, auto);
    gap: 2px;
    justify-content: start;
    align-content: start;
    min-height: 1px;
  }

  .compact {
    grid-template-columns: repeat(6, auto);
  }
</style>
