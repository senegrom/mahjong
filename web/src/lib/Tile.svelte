<script module>
  import dragonUrl from '../assets/white-dragon.webp';
  import { MATISSE_DRAGON_URL, TILE_IMAGE_URLS } from './tile-faces.js';
  const images = [];
  let preloading = null;
  // Decode EVERY face before the first hand, including the hidden dragon art.
  // Keep Image objects alive; the service worker also keeps the original bytes
  // across reloads, new games and browser restarts, independently of HTTP cache.
  export function preloadTiles(onProgress = () => {}) {
    if (preloading) return preloading;
    const urls = [...TILE_IMAGE_URLS, dragonUrl];
    let next = 0, complete = 0;
    preloading = Promise.all(Array.from({ length: 6 }, async () => {
      while (next < urls.length) {
        const url = urls[next++];
        const image = new Image();
        images.push(image);
        image.decoding = 'async';
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => done(new Error(`Tile download timed out: ${url}`)), 45000);
          const done = error => {
            clearTimeout(timer); image.onload = null; image.onerror = null;
            if (error) reject(error); else resolve();
          };
          image.onload = () => done();
          image.onerror = () => done(new Error(`Tile graphic could not load: ${url}`));
          image.src = url;
        });
        await image.decode();
        onProgress(++complete, urls.length);
      }
    })).catch(error => { preloading = null; throw error; });
    return preloading;
  }
</script>

<script>
  import { getContext } from 'svelte';
  import { tileWords } from './tiles.js';
  import { TILE_FACE_CONTEXT, tileImage } from './tile-faces.js';
  import { CUBIST_APPROVED } from './cubist-faces.js';

  const currentFace = getContext(TILE_FACE_CONTEXT) ?? (() => 'classic');
  let tileFace = $derived(currentFace());

  /**
   * One tile, using the selected face set; a face-down tile shows the back.
   * Every tile carries its name for screen readers, so
   * a hand can be read out without relying on the picture.
   *
   * A tile in the hand can carry marks, each a colour of ring around the
   * face: gold for a discard leaving tenpai, silver for one shanten, red
   * for dora, green for safety against declared riichi, blue for selection.
   * A newly drawn tile is identified by spacing, never by its own ring. One
   * mark is a solid ring; more than one is drawn as stripes of each colour
   * in turn, so no mark hides another.
   */
  let {
    tile = null,
    facedown = false,
    rotated = false,
    dimmed = false,
    selected = false,
    safe = false,
    dora: markedDora = false,
    drawn = false,
    discardShanten = null,
    size = 'normal',
    onclick = null,
    disabled = false,
    muted = disabled,
    title = '',
    handIndex = null,
  } = $props();

  // A hidden face must not disclose dora through its ring, name or effects.
  let dora = $derived(Boolean(markedDora && tile && !facedown));
  // Readiness describes a legal action on a visible, interactive hand tile.
  // Never let a stale preview mark a disabled tile, a meld or a hidden face.
  let readiness = $derived(tile && !facedown && onclick && !disabled
    ? discardShanten === 0 ? 'ready' : discardShanten === 1 ? 'one-away' : null
    : null);

  const COLOURS = {
    ready: 'var(--gold, #d8a12a)',
    'one-away': '#c5cbd3',
    dora: '#e2453d',
    safe: '#7fd1a0',
    selected: '#4ea3ff',
  };
  let marks = $derived(
    [readiness, dora && 'dora', safe && 'safe', selected && 'selected'].filter(Boolean),
  );
  let ring = $derived(
    marks.length === 0
      ? 'none'
      : marks.length === 1
        ? COLOURS[marks[0]]
        : `repeating-linear-gradient(45deg, ${marks
            .map((mark, index) => `${COLOURS[mark]} ${index * 6}px ${(index + 1) * 6}px`)
            .join(', ')})`,
  );

  let imageUrl = $derived(tileImage(tile, tileFace, facedown));
  let words = $derived(
    facedown ? 'face-down tile' : [tileWords(tile), dora && 'dora',
      drawn && 'just drawn', selected && 'selected',
      readiness === 'ready' && 'discard leaves a ready hand (tenpai)',
      readiness === 'one-away' && 'discard leaves one tile from ready (one shanten)',
      safe && 'safe against declared riichi, not guaranteed against undeclared hands']
      .filter(Boolean).join(', '),
  );
  // The white dragon's face is blank, which reads as a missing picture.
  // Sets that do not leave it plain frame it in blue; so does this one.
  let blank = $derived(tileFace === 'classic' && !facedown && tile === '5z');
  let whiteDragonDora = $derived(!facedown && tile === '5z' && dora);
  // The Cubist white dragon keeps its approved empty centre under the foil.
  let revealUrl = $derived(tileFace === 'cubist' ? null : tileFace === 'matisse' ? MATISSE_DRAGON_URL : dragonUrl);
</script>

{#if onclick}
  <button
    class="tile {size}"
    class:matisse={tileFace === 'matisse' && !facedown && Boolean(tile)}
    class:cubist={tileFace === 'cubist' && !facedown && CUBIST_APPROVED.includes(tile)}
    class:dali={tileFace === 'dali' && !facedown && Boolean(tile)}
    class:rotated
    class:dimmed
    class:muted
    class:selected
    class:safe
    class:dora
    class:drawn
    data-tile={tile}
    data-hand-index={handIndex ?? undefined}
    data-drawn={drawn ? 'true' : undefined}
    data-readiness={readiness ?? undefined}
    aria-pressed={selected}
    type="button"
    class:ringed={marks.length > 0}
    style:--ring={ring}
    {disabled}
    title={title || words}
    aria-label={title || words}
    onclick={() => onclick(tile)}
  >
    <span class="face" class:haku={whiteDragonDora}>
      <img src={imageUrl} alt="" draggable="false" class:blank />
      {#if dora && !facedown}
        {#key whiteDragonDora}
          {#if whiteDragonDora && revealUrl}
            <span class="haku-dragon-reveal" class:matisse={tileFace === 'matisse'} style:background-image={`url("${revealUrl}")`} aria-hidden="true"></span>
          {/if}
          <span class="foil" aria-hidden="true"></span>
        {/key}
      {/if}
    </span>
  </button>
{:else}
  <span
    class="tile {size}"
    class:matisse={tileFace === 'matisse' && !facedown && Boolean(tile)}
    class:cubist={tileFace === 'cubist' && !facedown && CUBIST_APPROVED.includes(tile)}
    class:dali={tileFace === 'dali' && !facedown && Boolean(tile)}
    class:rotated
    class:dimmed
    class:muted
    class:ringed={marks.length > 0}
    style:--ring={ring}
    role="img"
    aria-label={title || words}
    title={title || words}
  >
    <span class="face" class:haku={whiteDragonDora}>
      <img src={imageUrl} alt="" draggable="false" class:blank />
      {#if dora && !facedown}
        {#key whiteDragonDora}
          {#if whiteDragonDora && revealUrl}
            <span class="haku-dragon-reveal" class:matisse={tileFace === 'matisse'} style:background-image={`url("${revealUrl}")`} aria-hidden="true"></span>
          {/if}
          <span class="foil" aria-hidden="true"></span>
        {/key}
      {/if}
    </span>
  </span>
{/if}

<style>
  .tile {
    /* The ring and the sheen are placed against the tile's own box, and
       the box is its own stacking context so the ring, which sits behind
       the face, still sits in front of the table. */
    position: relative;
    isolation: isolate;
    display: inline-flex;
    align-items: flex-end;
    justify-content: center;
    --face-width: var(--tile-width);
    --face-radius: 4px;
    --ring-width: 3px;
    width: var(--face-width);
    padding: 0;
    border: none;
    background: none;
    line-height: 0;
    flex: none;
  }

  .tile.matisse, .tile.cubist, .tile.dali {
    /* Match the SVG's 26-unit corners at every tile size. */
    --face-radius: calc(var(--face-width) * 26 / 300);
  }

  .face {
    position: relative;
    display: block;
    width: 100%;
    aspect-ratio: 3 / 4;
    flex: none;
    isolation: isolate;
    /* One surface owns the outline, shadow and effect clipping. An image
       background beneath a differently rounded SVG makes a second edge. */
    border-radius: var(--face-radius);
    overflow: hidden;
    background: var(--ivory);
    box-shadow: 0 2px 3px rgba(0, 0, 0, 0.35);
    --sheen-from: 120%;
    --sheen-to: -20%;
    --sheen-duration: 5s;
  }

  /* Both layers start together, also when a keyed tile changes identity.
     Off-face endpoints leave Haku genuinely blank between passes. */
  .face.haku {
    --sheen-from: 160%;
    --sheen-to: -60%;
  }

  .tile img {
    width: 100%;
    /* The faces are 300 by 400. Said here as well, so a tile keeps its
       box while its face is still on the way: an image with no bytes yet
       has no height, and a loading tile collapsed to a bar. */
    aspect-ratio: 3 / 4;
    height: 100%;
    display: block;
    border-radius: inherit;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.55) inset;
  }

  .tile img.blank {
    box-shadow:
      0 1px 0 rgba(255, 255, 255, 0.55) inset,
      0 0 0 2px #4a7fb5 inset;
  }

  .small {
    --face-width: calc(var(--tile-width) * 0.62);
    --ring-width: 2px;
  }

  .tiny {
    --face-width: calc(var(--tile-width) * 0.5);
    --ring-width: 2px;
  }

  .rotated .face {
    transform: rotate(90deg);
    transform-origin: center;
  }

  /* A tile turned on its side, which is how a riichi declaration is shown.
     It is the same tile, so it keeps its size and the box is given the room
     the turn needs: the faces are 300 by 400, so a tile lying down is four
     thirds as wide as one standing up and just as long the other way.
     Sizing the box as a square instead made the picture hang over both
     edges and its neighbours. */
  .rotated {
    width: calc(var(--face-width) * 4 / 3);
    height: var(--face-width);
    align-items: center;
    justify-content: center;
  }

  .rotated .face {
    width: var(--face-width);
  }

  .dimmed .face {
    filter: grayscale(0.55) brightness(0.82);
  }

  /* Whatever a tile does, it does as a whole. The lift on hover, on focus
     and under the keyboard marker moves the button, and the ring, the
     sheen and the focus outline are all children of it, so nothing is
     left behind. The ring used to sit on a wrapper around the button and
     the lift moved only the picture, so the picture rose out of its ring. */
  button.tile {
    cursor: pointer;
    transition:
      transform 0.12s ease,
      filter 0.12s ease;
  }

  button.tile:hover:not(:disabled),
  button.tile:focus-visible,
  button.tile.selected {
    transform: translateY(-6px);
  }

  button.tile:hover:not(:disabled) .face,
  button.tile:focus-visible .face,
  button.tile.selected .face {
    box-shadow: 0 8px 10px rgba(0, 0, 0, 0.4);
  }

  button.tile:disabled {
    cursor: default;
  }

  button.tile:disabled.muted .face {
    filter: grayscale(0.7) brightness(0.75);
  }

  /* The ring: one colour, or stripes of several, behind the face. On the
     small tiles of a discard row or a called set it is thinner, in
     proportion. */
  .ringed::before {
    content: '';
    position: absolute;
    inset: calc(-1 * var(--ring-width));
    border-radius: calc(var(--face-radius) + var(--ring-width));
    background: var(--ring);
    z-index: -1;
  }

  /* A dora shines, as a foil card does: a sheen that crosses the face
     slowly, over the picture and under the pointer. */
  /* The approved artwork is a decorative layer, never a replacement for
     the tile name. Hide it entirely if CSS masking is unsupported. */
  .haku-dragon-reveal {
    display: none;
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background-size: 92% 94%;
    background-position: center;
    background-repeat: no-repeat;
    mix-blend-mode: multiply;
    /* The black brush master becomes a silver impression through the shine. */
    opacity: 0.25;
    -webkit-mask-image: linear-gradient(115deg, transparent 30%, black 44%, black 56%, transparent 70%);
    mask-image: linear-gradient(115deg, transparent 30%, black 44%, black 56%, transparent 70%);
    -webkit-mask-size: 250% 100%;
    mask-size: 250% 100%;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: var(--sheen-from) 0;
    mask-position: var(--sheen-from) 0;
    animation: dragon-reveal var(--sheen-duration) linear infinite;
  }

  /* Matched quiet/lit exports share a canvas. Reveal the lit state directly
     so its ivory face and silver cut-outs stay aligned with the base. */
  .haku-dragon-reveal.matisse {
    background-size: 100% 100%;
    mix-blend-mode: normal;
    opacity: 1;
  }

  @supports (mask-image: linear-gradient(black, transparent)) or (-webkit-mask-image: linear-gradient(black, transparent)) {
    .haku-dragon-reveal { display: block; }
  }

  @keyframes dragon-reveal {
    from {
      -webkit-mask-position: var(--sheen-from) 0;
      mask-position: var(--sheen-from) 0;
    }
    to {
      -webkit-mask-position: var(--sheen-to) 0;
      mask-position: var(--sheen-to) 0;
    }
  }

  .foil {
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background: linear-gradient(
      115deg,
      rgba(255, 255, 255, 0) 30%,
      rgba(255, 255, 255, 0.5) 44%,
      rgba(255, 214, 130, 0.4) 50%,
      rgba(160, 220, 255, 0.35) 56%,
      rgba(255, 255, 255, 0) 70%
    );
    background-size: 250% 100%;
    background-repeat: no-repeat;
    mix-blend-mode: screen;
    animation: sheen var(--sheen-duration) linear infinite;
  }

  @keyframes sheen {
    from {
      background-position: var(--sheen-from) 0;
    }
    to {
      background-position: var(--sheen-to) 0;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .haku-dragon-reveal {
      animation: none;
      -webkit-mask-position: 40% 0;
      mask-position: 40% 0;
    }
    .foil {
      animation: none;
      background-position: 40% 0;
    }
  }
</style>
