<script>
  import { untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import { parseTiles } from './physical-position.js';
  import { TILE_TYPES, tileWords } from './tiles.js';
  // `onadd(tile, press)` takes one tile, and the click that chose it, by which
  // a caller can tell a keyboard's press (pressedByKeyboard); `onchange(tiles)`
  // takes the whole list. Either refuses by throwing, and the refusal is said
  // here, under the palette that was pressed. An `onchange` that says the
  // refusal itself answers false. Whichever way, what was typed stays in the
  // field to be corrected.
  let { label, tiles = [], onchange = null, onadd = null, limit = 14, expanded = false } = $props();
  // The palette is the player's to open and fold: `expanded` only says how it
  // starts. Set from the prop (`open={expanded}`), Svelte set it again whenever
  // anything drawn beside it changed, the count in its summary among them, so
  // it shut under the player after every tile.
  let open = $state(untrack(() => expanded));
  let text = $state('');
  let error = $state('');
  function add(values, press) {
    error = '';
    try {
      if (onadd) { for (const tile of values) onadd(tile, press); }
      else {
        if (tiles.length + values.length > limit) throw new Error(`At most ${limit} tiles here`);
        if (onchange([...tiles, ...values]) === false) return;
      }
      text = '';
    } catch (e) { error = e.message ?? String(e); }
  }
  function enter() { try { add(parseTiles(text)); } catch (e) { error = e.message; } }
</script>

<div class="tile-entry">
  <span class="entry-label">{label}</span>
  {#if tiles.length}<div class="entered">{#each tiles as tile, index (index)}<Tile {tile} size="small" title={`Remove ${tileWords(tile)}`} onclick={onchange ? () => onchange(tiles.filter((_, i) => i !== index)) : null} />{/each}</div>{/if}
  <details bind:open>
    <summary>{onadd ? 'Choose a tile' : `Add tiles · ${tiles.length}/${limit}`}</summary>
    <div class="palette" role="group" aria-label={label}>
      {#each TILE_TYPES as tile (tile)}<Tile {tile} size="small" title={`Add ${tileWords(tile)} to ${label}`} onclick={(_, press) => add([tile], press)} disabled={!onadd && tiles.length >= limit} muted={false} />{/each}
    </div>
    {#if !onadd}<div class="notation"><input aria-label={`${label} in tile notation`} bind:value={text} placeholder="123m456p789s11z" onkeydown={event => { if (event.key === 'Enter') { event.preventDefault(); enter(); } }} /><button onclick={enter}>Add</button></div>{/if}
  </details>
  {#if error}<p role="alert">{error}</p>{/if}
</div>

<style>
  /* The entry measures itself, since how wide it is follows from the panel it
     is in, not from the screen: a seat's panel beside another on a tablet is
     as narrow as a phone. */
  .tile-entry { display: grid; gap: 8px; min-width: 0; container-type: inline-size; }
  .entry-label { font-size: .8rem; font-weight: 600; }
  .entered { display: flex; flex-wrap: wrap; gap: 5px; align-items: end; }
  /* Pressed with a finger too: 44 px, its words in the middle. */
  summary { cursor: pointer; min-height: 44px; padding-block: 13px; font-size: .8rem; color: var(--gold-text); }
  .palette { display: grid; grid-template-columns: repeat(9,minmax(0,1fr)); gap: 6px; max-width: 510px; padding: 7px 3px; }
  .palette :global(button.tile) { width: 100%; min-height: 44px; }
  /* Nine columns leave a tile 44 px wide, the least a finger presses surely,
     only where the entry is 450 px wide; on a 390 px phone they left 32 by
     42. In less the palette has five: each suit as 1 to 5 over 6 to 9, then
     the winds and the dragons, a row each, so a tile is still where the eye
     looks for it. Each button is as wide as its column, 48 px or more, and
     its picture is drawn 36 px wide in the middle: pictures as wide as the
     columns would make the eight rows 640 px tall on that phone, and these
     make them 440. The tiles already entered, pressed to take one out, are
     44 px wide there too. */
  @container (max-width: 449px) {
    .palette { grid-template-columns: repeat(5,minmax(0,1fr)); max-width: 310px; }
    .palette :global(button.tile) { --face-width: 36px; padding-inline: calc((100% - 36px) / 2); }
    .palette :global(button.tile:is([data-tile^='1'], [data-tile='5z'])) { grid-column-start: 1; }
    .entered :global(button.tile) { --face-width: 44px; }
  }
  .notation { display: flex; gap: 8px; margin-top: 8px; }
  input { min-width: 0; flex: 1; }
  input, button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 6px 10px; font: inherit; box-sizing: border-box; }
  p { color: var(--warning-text); font-size: .8rem; margin: 0; }
</style>
