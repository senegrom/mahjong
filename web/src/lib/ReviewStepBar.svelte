<script>
  import { tick } from 'svelte';
  import Tile from './Tile.svelte';
  import { decisionInBrief, moveToken, moveWeight, stepSaid } from './game-review.js';
  import { percent, pressedByKeyboard, quadOnOffer } from './ui.js';
  import { SEAT_NAMES } from './tiles.js';

  /**
   * The review's step bar: Previous and Next, where the review is, and the
   * move made there beside Trained AI's, with the weight Trained AI gave
   * each. It stays at the top of the screen, under the game's bar where that
   * stays too, so that the answer and the way on stay in view however far
   * the table, the hand and Trained AI's view below are scrolled. On a screen
   * 500 pixels high or less it would take a third of the screen, so there it
   * stays where it is.
   *
   * What the bar shows is said in one line as the step changes (stepSaid),
   * from a status line in the page from the start, since a screen reader
   * hears a status only once it knows it. The place is not a status of its
   * own, which would say it twice.
   *
   * `previous` and `next` are the places Previous and Next go to, across
   * hands, or null at either end, and Next while the hand on show is being
   * reviewed, which `waiting` says (GameReview); `filter` is the decisions
   * they step through, which their names say; `onstep(place)` goes there.
   * `height` is the bar's own, for whatever lies under it to keep clear of
   * it.
   *
   * Where the player's settings keep Trained AI's move back (`moveHidden`),
   * the line gives the player's move without its weight, which would say how
   * Trained AI rated it, and "Trained AI ?" with Show answer in its place;
   * where they keep only the other players' tiles back (`tilesHidden`), Show
   * tiles. Either is the bar's one reveal, `onreveal(byKeyboard)`, told
   * whether a keyboard pressed it (pressedByKeyboard). A move the player
   * picked before the answer was shown (`pick`) is given as Now, between the
   * two.
   */
  let { hand, step, decision = null, previous = null, next = null, waiting = false, filter = 'every', moveHidden = false, tilesHidden = false,
    pick = null, onstep, onreveal = () => {}, height = $bindable(0) } = $props();

  // What Previous and Next step to, after the word each shows, which their
  // names begin with, so that a player who says the word reaches the button.
  const STEPS = { flagged: 'flagged decision', differed: 'decision where Trained AI differed', every: 'decision' };

  let frame = $derived(hand.frames[step]);
  let brief = $derived(decisionInBrief(frame, decision, { hidden: moveHidden }));
  let said = $derived(stepSaid({ hand, step, decision, hidden: moveHidden, pick }));
  // The weights the line gives where Trained AI was asked and its answer is
  // shown: the player's move beside Trained AI's choice, given riichi where
  // both declared it, and the move the player picked, if any.
  let yours = $derived(decision && !decision.forced && !moveHidden ? moveWeight(decision, frame.played) : null);
  let trained = $derived(brief.ai ? moveWeight(decision, { kind: decision.advised_kind, tile: decision.advised_tile ?? null }) : null);
  let now = $derived(pick && decision && !decision.forced && !moveHidden
    ? { token: moveToken(pick.kind, pick.tile ?? null), ...moveWeight(decision, pick) } : null);
  let discarder = $derived(SEAT_NAMES[frame.view.pending_from] ?? 'An opponent');
  let robbing = $derived(Boolean(quadOnOffer(frame.view)));
  let previousButton = $state(null);
  let nextButton = $state(null);
  let bar = $state(null);
  // The bar's height when last measured, to tell how far a step grew it.
  let measured = 0;

  // A step to either end disables the button pressed, which would drop its
  // focus to the page, so focus goes on to the other. Any other press keeps
  // it where it is. The other button is beside it, on the screen, so the
  // page is not scrolled to it, which would move the page under the player
  // wherever the bar is stuck at the top of the screen.
  async function press(place, pressed, other) {
    if (!place) return;
    const focused = document.activeElement === pressed;
    onstep(place);
    await tick();
    if (focused && pressed.disabled && !other.disabled) other.focus({ preventScroll: true });
  }

  // A step can make the bar taller, where the moves it gives take another
  // line. While the bar is stuck at the top of the screen, the page beneath
  // it stays where it was and the bar grows down over it: a control focused
  // just under it, such as a decision's line in the list of hands, would end
  // up beneath it. Such a control is brought back into view 8 pixels under
  // the bar, where scroll-margin brings a control focused below it
  // (GameReview). One that was beneath the bar already, or further down the
  // page, stays where the player left it.
  $effect(() => {
    const grown = measured ? height - measured : 0;
    measured = height;
    const focused = document.activeElement;
    if (grown <= 0 || !bar || !focused || focused === document.body || bar.contains(focused)) return;
    const edge = bar.getBoundingClientRect(), box = focused.getBoundingClientRect();
    if (box.left < edge.right && box.right > edge.left && box.top >= edge.bottom - grown && box.top < edge.bottom + 8) {
      scrollBy(0, box.top - edge.bottom - 8);
    }
  });
</script>

<div class="steps" role="group" aria-label="Steps" data-steps bind:offsetHeight={height} bind:this={bar}>
  <div class="buttons">
    <button class="app-control" data-previous aria-label={`Previous ${STEPS[filter]}`} disabled={!previous} bind:this={previousButton}
      onclick={() => press(previous, previousButton, nextButton)}>Previous</button>
    <!-- Waiting for the answers of the hand on show, Next is drawn and named
         as disabled but stays focusable, and a press does nothing: disabled
         outright, it would lose focus to Previous as the hand's review
         began, and leave it there once the answers came. -->
    <button class="app-control" data-next aria-label={`Next ${STEPS[filter]}`} disabled={!next && !waiting}
      aria-disabled={waiting ? 'true' : undefined} bind:this={nextButton}
      onclick={() => press(next, nextButton, previousButton)}>Next</button>
  </div>
  <!-- "decision" goes on the narrowest screens, as "Trained" does below. -->
  <p class="place" data-step>{hand.name} · <span class="long">{'decision '}</span>{step + 1} of {hand.frames.length}</p>
  <p class="moves" data-move-line>
    {#if brief.offered}<span class="on">On {discarder}'s{robbing ? ' kan' : ''} <Tile tile={brief.offered} size="tiny" />:</span>{/if}
    <span class="part">You {@render token(brief.you)}{#if yours?.weight != null}{' '}{percent(yours.weight)}{#if yours.given && !brief.ai}, given riichi{/if}{/if}</span>
    {#if now}
      <span class="dot" aria-hidden="true">·</span>
      <span class="part">Now {@render token(now.token)}{#if now.weight != null}{' '}{percent(now.weight)}{#if now.given && !brief.ai}, given riichi{/if}{/if}</span>
    {/if}
    <span class="dot" aria-hidden="true">·</span>
    <span class="part">
      {#if !brief.reviewed}
        <span class="long">{'Trained '}</span>AI …
      {:else if brief.forced}
        Only move
      {:else if brief.hidden}
        <span class="long">{'Trained '}</span>AI ?
      {:else if brief.agreed}
        <span class="long">{'Trained '}</span>AI agreed
      {:else}
        <span class="long">{'Trained '}</span>AI {@render token(brief.ai)}{#if trained?.weight != null}{' '}{percent(trained.weight)}{#if trained.given}, given riichi{/if}{/if}
      {/if}
    </span>
    <!-- Where Trained AI's move would be: Show answer turns over what the
         decision keeps back, the move and with it the other players' tiles
         where they are hidden too; Show tiles the tiles alone, where only
         they are. -->
    {#if moveHidden || tilesHidden}
      <button type="button" class="app-control reveal" data-reveal onclick={event => onreveal(pressedByKeyboard(event))}>{moveHidden ? 'Show answer' : 'Show tiles'}</button>
    {/if}
    {#if brief.flagged}<span class="dot" aria-hidden="true">·</span> <strong class="flag">Flagged</strong>{/if}
  </p>
  <p class="visually-hidden" role="status" data-step-said>{said}</p>
</div>

{#snippet token(move)}
  <span class="token">{#if move.word}<span>{move.word}</span>{/if}{#each move.tiles as tile, index (index)}<Tile {tile} size="tiny" />{/each}</span>
{/snippet}

<style>
  /* Opaque, so that nothing scrolled under it shows through, and over the
     table but under the game's bar, which comes first (z-index 18). */
  .steps {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 14px;
    padding: 6px 8px;
    border-radius: 10px;
    background: var(--felt-deep);
    box-shadow: 0 6px 14px #0004;
  }
  .buttons { display: flex; gap: 8px; }
  p { margin: 0; }
  .place { font-size: .85rem; font-variant-numeric: tabular-nums; }
  /* The player's move, the one they picked, Trained AI's and the flag,
     parted by dots that only the eye is given: a screen reader hears the
     moves. The reveal sits where Trained AI's move would be, wrapping with
     the rest. */
  .moves { display: flex; flex-wrap: wrap; align-items: center; gap: 2px 8px; font-size: .9rem; font-variant-numeric: tabular-nums; }
  .on, .part, .token { display: inline-flex; align-items: center; gap: 4px; }
  .dot { opacity: .7; }
  .flag { font-weight: 650; color: var(--gold); }
  /* Flush under the game's bar where that stays at the top of the screen
     (AppSettings): on a phone held upright.

     The bar's own buttons are on the screen wherever the bar is, so a focus
     that lands on them, handed on at either end or tabbed back to, keeps
     clear only of what stays above the bar, never of the bar itself, which
     would scroll the page whenever the bar is stuck: of the game's bar on a
     phone held upright, and where the bar stays in its place, of the game's
     bar as the headings keep clear of it. */
  @media (max-width: 760px) {
    .steps { top: 56px; }
    .steps button { scroll-margin-top: 56px; }
  }
  @media (max-height: 500px) {
    .steps { position: static; }
    .steps button { scroll-margin-top: 72px; }
  }
  @media (max-width: 480px) { .long { display: none; } }
</style>
