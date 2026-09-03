<script>
  /**
   * One tile. Faces are the public-domain drawings in /tiles; a face-down
   * tile shows the back. Every tile carries its name for screen readers, so
   * a hand can be read out without relying on the picture.
   */
  let {
    tile = null,
    facedown = false,
    rotated = false,
    dimmed = false,
    selected = false,
    size = 'normal',
    onclick = null,
    disabled = false,
    title = '',
  } = $props();

  const SUIT_FILES = { m: 'Man', p: 'Pin', s: 'Sou' };
  const HONOURS = ['Ton', 'Nan', 'Shaa', 'Pei', 'Haku', 'Hatsu', 'Chun'];
  const SUIT_WORDS = { m: 'characters', p: 'circles', s: 'bamboo' };
  const HONOUR_WORDS = [
    'east wind',
    'south wind',
    'west wind',
    'north wind',
    'white dragon',
    'green dragon',
    'red dragon',
  ];

  function fileFor(name) {
    if (!name) return 'Back';
    const rank = Number(name[0]);
    const suit = name[1];
    if (suit === 'z') return HONOURS[rank - 1] ?? 'Blank';
    return `${SUIT_FILES[suit] ?? 'Man'}${rank}`;
  }

  function wordsFor(name) {
    if (!name) return 'face-down tile';
    const rank = Number(name[0]);
    const suit = name[1];
    if (suit === 'z') return HONOUR_WORDS[rank - 1] ?? 'honour tile';
    return `${rank} ${SUIT_WORDS[suit]}`;
  }

  let file = $derived(facedown ? 'Back' : fileFor(tile));
  let words = $derived(facedown ? 'face-down tile' : wordsFor(tile));
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
    {disabled}
    title={title || words}
    aria-label={words}
    onclick={() => onclick(tile)}
  >
    <img src="tiles/{file}.svg" alt="" draggable="false" class:blank />
  </button>
{:else}
  <span class="tile {size}" class:rotated class:dimmed role="img" aria-label={words} title={title || words}>
    <img src="tiles/{file}.svg" alt="" draggable="false" class:blank />
  </span>
{/if}

<style>
  .tile {
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

  .rotated {
    width: calc(var(--tile-width) * 0.62);
    align-items: center;
    height: calc(var(--tile-width) * 0.62);
  }

  .dimmed img {
    filter: grayscale(0.55) brightness(0.82);
  }

  button.tile {
    cursor: pointer;
    transition:
      transform 0.12s ease,
      filter 0.12s ease;
  }

  button.tile:hover:not(:disabled) img,
  button.tile:focus-visible img {
    transform: translateY(-6px);
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

  .selected img {
    transform: translateY(-8px);
    box-shadow:
      0 0 0 2px var(--gold),
      0 8px 10px rgba(0, 0, 0, 0.45);
  }
</style>
