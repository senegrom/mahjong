<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  let {
    tile,
    remaining = null,
    showRemaining = false,
    handIndex = null,
    onclick = null,
    disabled = false,
    muted = disabled,
    selected = false,
    drawn = false,
    discardShanten = null,
    safe = false,
    dora = false,
  } = $props();
</script>

<span class="hand-tile" data-hand-drawn={drawn ? 'true' : undefined}>
  <Tile {tile} {handIndex} {onclick} {disabled} {muted} {selected} {drawn}
    {discardShanten} {safe} {dora} />
  {#if showRemaining && remaining !== null}
    <span class="copy-count" class:dead={remaining === 0} class:thin={remaining === 1}
      title={`${remaining} unseen ${tileWords(tile)} ${remaining === 1 ? 'remains' : 'remain'}`}
      aria-label={`${remaining} unseen ${tileWords(tile)} ${remaining === 1 ? 'remains' : 'remain'}`}>
      {remaining}
    </span>
  {/if}
</span>

<style>
  .hand-tile {
    display: inline-flex;
    width: var(--tile-width);
    min-width: 0;
    flex: none;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
  }

  .copy-count {
    width: 72%;
    min-height: 12px;
    margin-top: 4px;
    padding-top: 2px;
    border-top: 1px solid rgba(247, 242, 228, .25);
    color: rgba(247, 242, 228, .72);
    font-size: .64rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    text-align: center;
  }

  .copy-count.thin {
    border-color: rgba(216, 161, 42, .62);
    color: #efc86c;
  }

  .copy-count.dead {
    border-color: rgba(255, 184, 164, .55);
    color: var(--warning-text);
  }
</style>
