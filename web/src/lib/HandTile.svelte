<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  let {
    tile,
    size = 'normal',
    remaining = null,
    showRemaining = false,
    handIndex = null,
    onclick = null,
    toggle = false,
    disabled = false,
    muted = disabled,
    selected = false,
    drawn = false,
    discardShanten = null,
    safe = false,
    dora = false,
    /** One item of a hand shown as a list, as a review shows each hand. */
    listed = false,
    /** A review's words over the tile, such as ['You', 'AI']: an empty list
     * keeps their line all the same, so that a hand whose tags move from
     * tile to tile between steps does not move with them. Play passes none,
     * and draws no line. */
    tag = null,
  } = $props();

  // What each of a review's words says to a screen reader, after the tile's
  // own name: the word is for the eye, and too short to be heard alone.
  const SAID = { You: 'your move', Now: 'your pick', AI: "Trained AI's choice" };
  let note = $derived(tag ? tag.map(word => SAID[word] ?? word).join(', ') : '');
</script>

<span class="hand-tile" class:small={size === 'small'} class:tagged={tag !== null} data-hand-drawn={drawn ? 'true' : undefined} role={listed ? 'listitem' : undefined}>
  {#if tag !== null}
    <span class="tag" aria-hidden="true">{#each tag as word, index (word)}{#if index}{' · '}{/if}<b class:ai={word === 'AI'}>{word}</b>{/each}</span>
  {/if}
  <Tile {tile} {size} {handIndex} {onclick} {toggle} {disabled} {muted} {selected} {drawn}
    {discardShanten} {safe} {dora} {note} />
  {#if showRemaining && remaining !== null}
    <span class="copy-count" class:dead={remaining === 0} class:thin={remaining === 1} role="img"
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

  /* A review's words over the tile, Trained AI's in gold, as its view marks
     its choice, and the player's in the tiles' ivory. The line is kept on
     every tile of the hand, tagged or not. A word longer than its tile, as
     on the narrowest phones, runs over the untagged tiles beside it rather
     than taking a second line. */
  .tag {
    min-height: 1.15em;
    margin-bottom: 3px;
    font-size: .62rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: .03em;
    white-space: nowrap;
    color: var(--ivory);
  }
  .tag .ai { color: var(--gold); }
  /* A review marks a tile by its ring and its words, and nothing in its hand
     moves: a tile lifted as play lifts a selected one would rise into its
     words. Said as specifically as Tile's own lifts, so that it outranks
     them. */
  .hand-tile.tagged :global(button.tile:is(:hover, :focus-visible, .selected)) { transform: none; }

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
  .hand-tile.small { width: calc(var(--tile-width) * 0.62); }

  .copy-count.thin {
    border-color: rgba(216, 161, 42, .62);
    color: #efc86c;
  }

  .copy-count.dead {
    border-color: rgba(255, 184, 164, .55);
    color: var(--warning-text);
  }
</style>
