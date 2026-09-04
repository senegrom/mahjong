<script module>
  // Every face, fetched once when the page loads. A face is otherwise
  // fetched the first time a tile of that kind is shown, and until it
  // arrives the tile is blank, or still wears the face it had before,
  // which on a slow connection or a busy machine was up to a second.
  const FACES = [
    'Back',
    'Front',
    ...['Man', 'Pin', 'Sou'].flatMap((suit) => [1, 2, 3, 4, 5, 6, 7, 8, 9].map((rank) => suit + rank)),
    'Ton',
    'Nan',
    'Shaa',
    'Pei',
    'Haku',
    'Hatsu',
    'Chun',
  ];
  if (typeof Image !== 'undefined') {
    for (const face of FACES) {
      const image = new Image();
      image.src = `tiles/${face}.svg`;
    }
  }
</script>

<script>
  import { tileWords } from './tiles.js';

  /**
   * One tile. Faces are the public-domain drawings in /tiles; a face-down
   * tile shows the back. Every tile carries its name for screen readers, so
   * a hand can be read out without relying on the picture.
   *
   * A tile in the hand can carry marks, each a colour of ring around the
   * face: gold for the tile just drawn, red for a dora, green for a tile
   * that cannot deal in, blue for the one under the keyboard marker. One
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
    dora = false,
    drawn = false,
    size = 'normal',
    onclick = null,
    disabled = false,
    title = '',
  } = $props();

  const SUIT_FILES = { m: 'Man', p: 'Pin', s: 'Sou' };
  const HONOURS = ['Ton', 'Nan', 'Shaa', 'Pei', 'Haku', 'Hatsu', 'Chun'];
  function fileFor(name) {
    if (!name) return 'Back';
    const rank = Number(name[0]);
    const suit = name[1];
    if (suit === 'z') return HONOURS[rank - 1] ?? 'Blank';
    return `${SUIT_FILES[suit] ?? 'Man'}${rank}`;
  }

  const COLOURS = {
    drawn: 'var(--gold, #d8a12a)',
    dora: '#e2453d',
    safe: '#7fd1a0',
    selected: '#4ea3ff',
  };
  let marks = $derived(
    [drawn && 'drawn', dora && 'dora', safe && 'safe', selected && 'selected'].filter(Boolean),
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

  let file = $derived(facedown ? 'Back' : fileFor(tile));
  let words = $derived(
    facedown ? 'face-down tile' : dora ? `${tileWords(tile)}, dora` : tileWords(tile),
  );
  // The white dragon's face is blank, which reads as a missing picture.
  // Sets that do not leave it plain frame it in blue; so does this one.
  let blank = $derived(!facedown && tile === '5z');
</script>

{#if onclick}
  <button
    class="tile {size}"
    class:rotated
    class:dimmed
    class:selected
    class:ringed={marks.length > 0}
    style:--ring={ring}
    {disabled}
    title={title || words}
    aria-label={words}
    onclick={() => onclick(tile)}
  >
    <img src="tiles/{file}.svg" alt="" draggable="false" class:blank />
    {#if dora}<span class="foil" aria-hidden="true"></span>{/if}
  </button>
{:else}
  <span class="tile {size}" class:rotated class:dimmed role="img" aria-label={words} title={title || words}>
    <img src="tiles/{file}.svg" alt="" draggable="false" class:blank />
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
    width: var(--tile-width);
    padding: 0;
    border: none;
    background: none;
    line-height: 0;
    flex: none;
  }

  .tile img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
    box-shadow:
      0 1px 0 rgba(255, 255, 255, 0.55) inset,
      0 2px 3px rgba(0, 0, 0, 0.35);
    background: var(--ivory);
  }

  .tile img.blank {
    box-shadow:
      0 1px 0 rgba(255, 255, 255, 0.55) inset,
      0 0 0 2px #4a7fb5 inset,
      0 2px 3px rgba(0, 0, 0, 0.35);
  }

  .small {
    width: calc(var(--tile-width) * 0.62);
  }

  .tiny {
    width: calc(var(--tile-width) * 0.5);
  }

  .rotated img {
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
    width: calc(var(--tile-width) * 0.62 * 4 / 3);
    height: calc(var(--tile-width) * 0.62);
    align-items: center;
    justify-content: center;
  }

  .rotated img {
    width: calc(var(--tile-width) * 0.62);
  }

  .dimmed img {
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

  button.tile:hover:not(:disabled) img,
  button.tile:focus-visible img,
  button.tile.selected img {
    box-shadow:
      0 1px 0 rgba(255, 255, 255, 0.6) inset,
      0 8px 10px rgba(0, 0, 0, 0.4);
  }

  button.tile:disabled {
    cursor: default;
  }

  button.tile:disabled img {
    filter: grayscale(0.7) brightness(0.75);
  }

  /* The ring: one colour, or stripes of several, behind the face. */
  .ringed::before {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 6px;
    background: var(--ring);
    z-index: -1;
  }

  /* A dora shines, as a foil card does: a sheen that crosses the face
     slowly, over the picture and under the pointer. */
  .foil {
    position: absolute;
    inset: 0;
    border-radius: 4px;
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
    mix-blend-mode: screen;
    animation: sheen 5s linear infinite;
  }

  @keyframes sheen {
    from {
      background-position: 120% 0;
    }
    to {
      background-position: -20% 0;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .foil {
      animation: none;
      background-position: 40% 0;
    }
  }
</style>
