<script>
  import Tile from './Tile.svelte';
  import { TILES, parseTiles } from './physical-position.js';
  let { label, tiles = [], onchange = null, onadd = null, limit = 14, notation = true } = $props();
  let text = $state('');
  let error = $state('');
  function add(values) {
    error = '';
    try {
      if (onadd) { for (const tile of values) onadd(tile); }
      else {
        if (tiles.length + values.length > limit) throw new Error(`At most ${limit} tiles here`);
        onchange([...tiles, ...values]);
      }
      text = '';
    } catch (e) { error = e.message ?? String(e); }
  }
  function enter() { try { add(parseTiles(text)); } catch (e) { error = e.message; } }
</script>

<div class="tile-entry">
  <span class="entry-label">{label}</span>
  {#if tiles.length}<div class="entered">{#each tiles as tile, index (index)}<Tile {tile} size="small" title={`Remove ${tile}`} onclick={onchange ? () => onchange(tiles.filter((_, i) => i !== index)) : null} />{/each}</div>{/if}
  <details>
    <summary>{onadd ? 'Choose a tile' : `Add tiles · ${tiles.length}/${limit}`}</summary>
    <div class="palette" aria-label={label}>
      {#each TILES as tile (tile)}<Tile {tile} size="small" title={`Add ${tile} to ${label}`} onclick={() => add([tile])} disabled={!onadd && tiles.length >= limit} muted={false} />{/each}
    </div>
    {#if notation && !onadd}<div class="notation"><input aria-label={`${label} in tile notation`} bind:value={text} placeholder="123m456p789s11z" onkeydown={event => { if (event.key === 'Enter') { event.preventDefault(); enter(); } }} /><button onclick={enter}>Add</button></div>{/if}
  </details>
  {#if error}<p role="alert">{error}</p>{/if}
</div>

<style>
  .tile-entry { display: grid; gap: 8px; min-width: 0; }
  .entry-label { font-size: .8rem; font-weight: 600; }
  .entered { display: flex; flex-wrap: wrap; gap: 5px; align-items: end; }
  summary { cursor: pointer; min-height: 32px; font-size: .8rem; color: var(--gold); }
  .palette { display: grid; grid-template-columns: repeat(9,minmax(0,1fr)); gap: 6px; max-width: 510px; padding: 7px 3px; }
  .palette :global(button.tile) { width: 100%; min-height: 40px; }
  .notation { display: flex; gap: 8px; margin-top: 8px; }
  input { min-width: 0; flex: 1; }
  input, button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 6px 10px; font: inherit; box-sizing: border-box; }
  p { color: var(--warning-text); font-size: .8rem; margin: 0; }
</style>
