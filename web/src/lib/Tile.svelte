<script module>
  import dragonUrl from '../assets/white-dragon.webp';
  import { MATISSE_DRAGON_URL } from './tile-faces.js';
</script>

<script>
  import { getContext } from 'svelte';
  import { tileShorthand, tileWords } from './tiles.js';
  import { TILE_FACE_CONTEXT, tileImage } from './tile-faces.js';

  const currentFace = getContext(TILE_FACE_CONTEXT) ?? (() => 'classic');
  let tileFace = $derived(currentFace());

  /**
   * One tile, using the selected face set; a face-down tile shows the back.
   * Every tile carries its name for screen readers, so
   * a hand can be read out without relying on the picture.
   *
   * A face set still being painted has no picture for some tiles. Such a
   * tile shows its name in text on the ivory instead, so it plainly waits
   * for its artwork rather than borrowing another set's, and its picture
   * takes over as soon as the artwork is approved.
   *
   * A tile in the hand can carry marks, each a colour of ring around the
   * face: gold for a discard leaving tenpai, silver for one shanten, red
   * for dora, green for safety against declared riichi, blue for selection.
   * A newly drawn tile is identified by spacing, never by its own ring. One
   * mark is a solid ring; more than one is drawn as stripes of each colour
   * in turn, so no mark hides another.
   *
   * A discard row marks two things about a tile, each in its own way so a
   * tile can carry both. One thrown straight from the draw (tsumogiri) has
   * its face shaded darker, as Tenhou and Mahjong Soul show it; one thrown
   * from the hand keeps its face. One that another player claimed for a call
   * has a dark green border on its own edge, inside any ring.
   */
  let {
    tile = null,
    facedown = false,
    rotated = false,
    /** A discard another player took into a called set. */
    claimed: markedClaimed = false,
    selected = false,
    safe = false,
    dora: markedDora = false,
    drawn = false,
    /** A discard that was the tile just drawn, rather than one from the hand. */
    fromDraw: markedFromDraw = false,
    discardShanten = null,
    size = 'normal',
    /** Part of the shape a yaku is being explained by. Dimming the rest
     * would hide the artwork, so the tiles that belong are lifted instead. */
    inShape = false,
    onclick = null,
    /** A button whose press selects it: only then is `selected` announced as
     * pressed. Palette, remove and play-this buttons are plain buttons. */
    toggle = false,
    disabled = false,
    muted = disabled,
    title = '',
    handIndex = null,
  } = $props();

  // A hidden face must not disclose dora through its ring, name or effects.
  let dora = $derived(Boolean(markedDora && tile && !facedown));
  // Nor is a hidden face marked as thrown from the draw or claimed: its name
  // says only that it is face down, and the picture says the same.
  let fromDraw = $derived(Boolean(markedFromDraw && tile && !facedown));
  let claimed = $derived(Boolean(markedClaimed && tile && !facedown));
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
    facedown ? 'face-down tile' : [tileWords(tile), claimed && 'claimed', fromDraw && 'discarded from the draw', dora && 'dora',
      drawn && 'just drawn', selected && 'selected',
      readiness === 'ready' && 'discard leaves a ready hand (tenpai)',
      readiness === 'one-away' && 'discard leaves one tile from ready (one shanten)',
      safe && 'safe against declared riichi, not guaranteed against undeclared hands']
      .filter(Boolean).join(', '),
  );
  // A tile without a picture is written out in the words it is announced
  // with, the number over the suit or an honour's first word over its
  // second. Where the face is too small for the second word, its short name
  // stands in: a suit's letter under the number, which the short name
  // begins with, or an honour's letters in place of its words.
  let named = $derived.by(() => {
    if (imageUrl) return null;
    const [lead, rest] = tileWords(tile).split(' ');
    const short = tileShorthand(tile);
    return { lead, rest, letters: tile[1] === 'z' ? short : short.slice(lead.length) };
  });
  // The white dragon's face is blank, which reads as a missing picture.
  // Sets that do not leave it plain frame it in blue; so does this one.
  let blank = $derived(tileFace === 'classic' && !facedown && tile === '5z');
  // The dragon revealed under the foil belongs to a picture: a white dragon
  // written out has the foil alone, as any other dora has.
  let whiteDragonDora = $derived(!facedown && tile === '5z' && dora && !named);
  // Van Gogh retains its approved white-dragon artwork under the foil.
  let revealUrl = $derived(tileFace === 'van-gogh' ? null : tileFace === 'matisse' ? MATISSE_DRAGON_URL : dragonUrl);
</script>

{#if onclick}
  <button
    class="tile {size}"
    class:matisse={tileFace === 'matisse' && !facedown && Boolean(tile)}
    class:van-gogh={tileFace === 'van-gogh' && !facedown && Boolean(tile)}
    class:dali={tileFace === 'dali' && !facedown && Boolean(tile)}
    class:rotated
    class:claimed
    class:from-draw={fromDraw}
    class:muted
    class:selected
    class:safe
    class:dora
    class:drawn
    data-tile={tile}
    data-hand-index={handIndex ?? undefined}
    data-drawn={drawn ? 'true' : undefined}
    data-readiness={readiness ?? undefined}
    aria-pressed={toggle ? selected : undefined}
    type="button"
    class:ringed={marks.length > 0}
    class:in-shape={inShape}
    style:--ring={ring}
    {disabled}
    title={title || words}
    aria-label={title || words}
    onclick={() => onclick(tile)}
  >
    <span class="face" class:haku={whiteDragonDora} class:unpainted={Boolean(named)}>
      {#if named}
        <span class="name" aria-hidden="true"><b class="lead">{named.lead}</b><span class="rest">{named.rest}</span><b class="letters">{named.letters}</b></span>
      {:else}
        <img src={imageUrl} alt="" draggable="false" class:blank />
      {/if}
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
    class:van-gogh={tileFace === 'van-gogh' && !facedown && Boolean(tile)}
    class:dali={tileFace === 'dali' && !facedown && Boolean(tile)}
    class:rotated
    class:claimed
    class:from-draw={fromDraw}
    class:muted
    class:ringed={marks.length > 0}
    class:in-shape={inShape}
    style:--ring={ring}
    data-tile={tile}
    role="img"
    aria-label={title || words}
    title={title || words}
  >
    <span class="face" class:haku={whiteDragonDora} class:unpainted={Boolean(named)}>
      {#if named}
        <span class="name" aria-hidden="true"><b class="lead">{named.lead}</b><span class="rest">{named.rest}</span><b class="letters">{named.letters}</b></span>
      {:else}
        <img src={imageUrl} alt="" draggable="false" class:blank />
      {/if}
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
  /* A tile that makes the yaku being explained: lifted and lit, so the
     shape reads at a glance without dimming the rest of the hand. */
  .tile.in-shape {
    transform: translateY(-6px);
    box-shadow: 0 6px 14px rgba(216, 161, 42, .45);
    outline: 2px solid var(--gold, #d8a12a);
    outline-offset: 1px;
    border-radius: 6px;
    z-index: 2;
  }
  .tile.in-shape.rotated { transform: translateY(-6px) rotate(90deg); }
  @media (prefers-reduced-motion: no-preference) {
    .tile { transition: transform 120ms ease, box-shadow 120ms ease; }
  }

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
    /* A claimed tile's border, in proportion to the tile: one pixel on a
       phone's smallest rows, where more would cover the picture, two on the
       table's rows and three on the largest. */
    --claimed-width: clamp(1px, calc(var(--face-width) / 11), 3px);
    width: var(--face-width);
    padding: 0;
    border: none;
    background: none;
    line-height: 0;
    flex: none;
  }

  .tile.matisse, .tile.dali, .tile.van-gogh {
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

  /* A claimed white dragon is framed by its green border instead. On a
     phone's rows the border is thinner than the blue, which would show
     inside it as a second frame. */
  .claimed img.blank {
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.55) inset;
  }

  /* A tile its set has not painted yet: only its name, on the ivory. The
     type is measured against the face, so the name keeps its proportions
     whatever size a row or a grid gives the tile, and a phone's text
     enlarging may not push it past the edge. The sizes in tile widths are
     for a browser without container units. */
  .face.unpainted {
    container-type: inline-size;
  }

  .name {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #26302c;
    font-size: calc(var(--face-width) * 0.17);
    font-size: 17cqi;
    font-weight: 600;
    line-height: 1;
    white-space: nowrap;
    -webkit-text-size-adjust: 100%;
    text-size-adjust: 100%;
    -webkit-user-select: none;
    user-select: none;
  }

  /* The number, large, over its suit. */
  .name b {
    font-size: 3.3em;
    font-weight: 700;
    line-height: 0.9;
  }

  /* An honour's first word is what tells it apart, so it leads, as large as
     the longest of them fits. */
  [data-tile$='z'] .name b {
    font-size: 1.65em;
    line-height: 1.1;
  }

  /* The letters stand in for the words only where those cannot be read. */
  .name .letters {
    display: none;
  }

  /* A face smaller than a tile in the hand, as in a discard row, is too
     small for a second word to be read. A suited tile gives its suit by the
     letter tile notation writes it with, under a larger number: m for
     characters, p for circles and s for bamboo. The colour says the same,
     but never alone, so a red 8 and a blue 8 are told apart by their
     letters as well. An honour's first word stands alone. Each leaves room
     at the edges for a claimed tile's border. */
  @container (max-width: 40px) {
    .name .rest {
      display: none;
    }

    .name .lead {
      font-size: 3.8em;
    }

    /* A small letter's ink sits low in its line, so it is drawn up to its
       number, which also centres the pair on the face as it is seen. */
    .name .letters {
      display: block;
      position: relative;
      top: -0.1em;
      font-size: 3em;
      line-height: 0.8;
    }

    [data-tile$='z'] .name .lead {
      font-size: 1.6em;
    }

    [data-tile$='z'] .name .letters {
      display: none;
    }

    /* A tile turned on its side is wide and low, so its number and letter
       sit side by side on one baseline, as the notation writes them.
       Wrapping lets the pair be centred on the face. */
    .rotated .name {
      flex-flow: row wrap;
      align-content: center;
      align-items: baseline;
    }

    .rotated .name .letters {
      top: 0;
    }
  }

  /* On the table's own rows even an honour's word is too small to read at
     a glance, so from here down it is given in its letters, as large as
     they fit: E, S, W and N for the winds, Wh, G and R for the dragons. */
  @container (max-width: 34px) {
    [data-tile$='z'] .name .lead {
      display: none;
    }

    [data-tile$='z'] .name .letters {
      display: block;
      top: 0;
      font-size: 4em;
      line-height: 0.9;
    }

    [data-tile='5z'] .name .letters {
      font-size: 2.9em;
    }
  }

  /* Each suit keeps the colour its pictures are known by, as a second sign
     of the suit beside its word or letter: red characters, blue circles
     and green bamboo. The winds are in ink and each dragon in its colour,
     the white one in the blue that frames it on the Classic face. */
  [data-tile$='m'] .name, [data-tile='7z'] .name { color: #b3261e; }
  [data-tile$='p'] .name, [data-tile='5z'] .name { color: #1d4f91; }
  [data-tile$='s'] .name, [data-tile='6z'] .name { color: #1e6b3f; }

  /* On a tile turned on its side the name stays upright, so it reads as on
     every other tile and a 6 is never taken for a 9: its box is laid out
     lying down and turned back against the face's own turn. */
  .rotated .name {
    inset: auto;
    top: 50%;
    left: 50%;
    width: calc(100% * 4 / 3);
    height: 75%;
    transform: translate(-50%, -50%) rotate(-90deg);
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

  /* A discard thrown straight from the draw: the face is shaded to about
     half the light of a plain one but keeps its colours, so the picture
     still reads. Nothing else in a discard row darkens a face, so a darker
     face means only this. */
  .from-draw .face {
    filter: var(--from-draw-shade, brightness(0.72));
  }

  /* A claimed discard: a solid dark green border on the tile's own edge,
     over the rim of its face. It lies inside the tile's box, so it never
     reaches into the gap between tiles or meets a ring, which lies outside;
     a turned riichi tile's box is its face's footprint, so the border turns
     with it; and it is not part of the face, so the shade of a tile from the
     draw leaves it as it is. The face stays solid: letting the felt show
     through would darken it just as the shade does. */
  .claimed::after {
    content: '';
    position: absolute;
    inset: 0;
    border: var(--claimed-width) solid var(--claimed-border, #0e6e33);
    border-radius: var(--face-radius);
    pointer-events: none;
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
    /* A name has no picture around it to read instead, so a shine held
       still across it would wash it out. It rests on the corner instead. */
    .unpainted .foil {
      background-position: 100% 0;
    }
  }
</style>
