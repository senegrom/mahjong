<script>
  import { untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import DecisionFilter from './DecisionFilter.svelte';
  import { decisionInBrief, decisionKey, handEntry, listedIn } from './game-review.js';

  /**
   * The review's list of hands and decisions, from which the player goes to
   * any decision or hand: whatever opens it (`children`, the trend), the
   * filter, then every hand of the game in the order played, each headed by
   * its name, what it changed for the player and how it ended, with a line
   * for each decision the filter keeps, the player's move and Trained AI's in
   * tiles. A hand not yet reviewed says how far its review has got, and
   * offers to be reviewed next.
   *
   * `place` is the decision on show, whose line is current; where the filter
   * does not keep it, its line is drawn in its place all the same, dim, until
   * the player moves on. `onplace(place)` goes to a place, and
   * `onreview(hand)` reviews a hand next. `counts` is how many decisions each
   * filter keeps, and `empty` what the list says where its filter keeps none,
   * if anything. `open` is whether the list is on the screen, as a folded one
   * is not. `bar` is the height of the step bar, which stays over the top of
   * the screen where the list is folded under it, and which a step can make
   * taller.
   */
  let { hands, place = null, filter = $bindable('flagged'), counts, hints = true, trainedAvailable = false, empty = '', open = true,
    bar = 0, onplace, onreview, children } = $props();
  let root = $state(null);

  const on = (handIndex, step) => place?.hand === handIndex && place.step === step;

  // The list keeps the line of the decision on show in view by scrolling its
  // own box, the column it is on a wide screen or its box of hands where it
  // is folded above the decision, and never the page, which scrollIntoView
  // would move too. A step to another decision brings its line into view by
  // as little as will, and so does the list coming on the screen, drawn or
  // opened: the toggle that opens a folded list arrives once the browser has
  // opened it, so the list goes by whether it was on the screen before.
  //
  // Anything else that draws the list again, the hands reviewed meanwhile or
  // another filter, keeps still what the player is looking at, as a
  // browser's scroll anchoring does: the control in the box that has focus,
  // where it is in view; else the line on show, where that is; else the
  // first line or heading in view, unless the box is at its top, where it
  // stays. A line kept still whatever the player looks at would move all
  // they read above it by the lines arriving in between. Safari has no
  // scroll anchoring, so the boxes have none (GameReview), and the list does
  // it itself, alike in every browser. The line on show, kept still, stays
  // in view too, clear of the step bar as that grows over it. A folded list
  // keeps nothing: the browser may still lay out what it hides, at a cost.
  let held = null;
  let wasOpen = false;
  $effect.pre(() => {
    void hands; void filter; void place; void open; void bar;
    const opening = open && !wasOpen;
    wasOpen = open;
    held = opening || !open ? null : untrack(anchored);
  });
  $effect(() => {
    void hands; void filter; void place; void open; void bar;
    const before = held;
    held = null;
    if (open) untrack(() => keep(before));
  });

  const CURRENT = '[data-step-to][aria-current="step"]';
  // What the player reads the list by, in the order it runs down the box:
  // the trend's line and the filter, where the column holds them, each
  // hand's heading and each decision's line.
  const ROWS = 'summary, [data-filter-option], .head, [data-step-to]';
  // How far apart a line brought into view is kept from the edges of the
  // part of the box the player can see.
  const ROOM = 8;

  /** What the player is looking at in the list's box, the likeliest first,
   * each with how far below the box's top it is; the line on show, and
   * whether all of it is in view; null where the box is not on the screen. */
  function anchored() {
    const box = scroller();
    if (!box?.getClientRects().length) return null;
    const part = shownPart(box), line = box.querySelector(CURRENT), focused = document.activeElement;
    const seen = element => {
      const edge = element.getBoundingClientRect();
      return edge.bottom > part.top && edge.top < part.bottom;
    };
    const drawn = line?.getBoundingClientRect(), whole = Boolean(drawn) && drawn.top >= part.top && drawn.bottom <= part.bottom;
    const looked = [focused !== box && box.contains(focused) && seen(focused) && focused, line && seen(line) && line];
    // Rows not drawn, inside a folded trend, measure nothing and are passed.
    if (box.scrollTop > 0) {
      for (const element of box.querySelectorAll(ROWS)) {
        const { top, height } = element.getBoundingClientRect();
        if (!height) continue;
        if (top >= part.bottom) break;
        if (top >= part.top) looked.push(element);
      }
    }
    return { box, current: line?.dataset.stepTo ?? null, whole,
      anchors: looked.filter(Boolean).map(element => ({ element, offset: below(box, element) })) };
  }
  /** Once the list is drawn again: what the player was looking at goes back
   * where it was in the box, the first of it still in the list; then the
   * line on show is brought into view where the step is another, the list
   * has just come on the screen, or that line, all of it in view, was what
   * was kept still, for whatever has come over it meanwhile, such as the
   * step bar grown taller. A box scrolled by nothing is not set, which would
   * stop a scroll the player's touch has set going. */
  function keep(before) {
    const box = scroller();
    if (!box?.getClientRects().length) return;
    const anchor = before?.box === box ? before.anchors.find(({ element }) => element.isConnected && box.contains(element)) : null;
    const moved = anchor ? below(box, anchor.element) - anchor.offset : 0;
    if (Math.abs(moved) >= 0.5) box.scrollTop += moved;
    const line = box.querySelector(CURRENT);
    if (line && (line.dataset.stepTo !== before?.current || (before.whole && anchor?.element === line))) reveal(line, box);
  }
  /** How far below the top of the box an element of it is drawn. */
  const below = (box, element) => element.getBoundingClientRect().top - box.getBoundingClientRect().top;
  /** The box the list's hands scroll in: their own, where the list is folded
   * above the decision, or the column the list is, on a wide screen
   * (GameReview); never the page. */
  function scroller() {
    for (let node = root?.querySelector('.hands'); node && node !== document.body; node = node.parentElement) {
      if (/^(auto|scroll)$/.test(getComputedStyle(node).overflowY)) return node;
    }
    return null;
  }
  /** The part of the box the player can see: the part on the screen below
   * whatever stays over the top of the screen there, the step bar and on a
   * phone the game's bar too, as far down as a control of the box is
   * brought into view when focused (its scroll-margin-top, GameReview), less
   * the room it keeps; nothing stays over the column on a wide screen.
   * Where none of the box is on the screen, all of it, as the player will
   * find it. */
  function shownPart(box) {
    const edge = box.getBoundingClientRect(), control = box.querySelector('button');
    const covered = control ? Math.max(0, (parseFloat(getComputedStyle(control).scrollMarginTop) || 0) - ROOM) : 0;
    return edge.bottom > covered && edge.top < innerHeight
      ? { top: Math.max(edge.top, covered), bottom: Math.min(edge.bottom, innerHeight) }
      : { top: edge.top, bottom: edge.bottom };
  }
  /** Brings the line into the part of its box the player can see, by as
   * little as will. */
  function reveal(line, box) {
    const part = shownPart(box), shown = line.getBoundingClientRect();
    const top = part.top + ROOM, bottom = part.bottom - ROOM;
    if (shown.top < top) box.scrollTop -= top - shown.top;
    else if (shown.bottom > bottom) box.scrollTop += shown.bottom - bottom;
  }
  // The decisions of a hand that have a line: those the filter keeps, and
  // the one on show wherever the filter leaves it out.
  const steps = hand => hand.frames.flatMap((_, step) => (listedIn(hand.decisions?.[step] ?? null, filter) || on(hand.index, step) ? [step] : []));
  // What a hand's change of points reads, as a score sheet writes it with a
  // true minus; none where it changed nothing.
  const signed = value => `${value > 0 ? '+' : '−'}${Math.abs(value).toLocaleString()}`;

  function handState(hand) {
    if (!hand.frames.length) return 'No decisions';
    if (hand.status === 'reviewing') return `Reviewing, ${hand.done ?? 0} of ${hand.frames.length}`;
    if (hand.status !== 'done') return `${hand.frames.length} decision${hand.frames.length === 1 ? '' : 's'}, not reviewed`;
    return '';
  }
</script>

<div class="index" bind:this={root}>
  {@render children?.()}
  <DecisionFilter bind:value={filter} {counts} label="Which decisions to step through" />
  {#if empty}<p class="empty">{empty}</p>{/if}
  <!-- The hands and each hand's decisions are drawn without markers, so each
       list says it is one in so many words: Safari does not tell VoiceOver
       that a list without them is one. Each line is keyed by its hand and
       number, so that lines of other hands arriving as they are reviewed
       never take a line, or focus on it, away. -->
  <ol class="hands" role="list">
    {#each hands as hand (hand.index)}
      {@const lines = steps(hand)}
      {@const said = handState(hand)}
      <li class="hand" data-hand={hand.index} data-status={hand.status}>
        <div class="head">
          <!-- A hand's name goes to its first decision the filter keeps, or
               to its first decision if the filter keeps none there. -->
          {#if hand.frames.length}
            <button type="button" class="name" data-hand-row={hand.index} onclick={() => onplace(handEntry(hands, filter, hand.index))}>{hand.name}</button>
          {:else}
            <span class="name">{hand.name}</span>
          {/if}
          {#if hand.change}<span class="change">{signed(hand.change)}</span>{/if}
          <span class="outcome">{hand.outcome}</span>
          {#if said}<span class="state">{said}</span>{/if}
          <!-- The header names the hand, so the button need only say what it
               does; its accessible name names the hand too, after that word. -->
          {#if hand.status === 'waiting' && hand.frames.length && trainedAvailable}
            <button type="button" class="app-control review" data-review-hand={hand.index} onclick={() => onreview(hand.index)}
              aria-label={`Review: ${hand.name}`}>Review</button>
          {/if}
        </div>
        {#if lines.length}
          <ol class="lines" role="list">
            {#each lines as step (decisionKey(hand.index, step))}
              {@const decision = hand.decisions?.[step] ?? null}
              {@const line = decisionInBrief(hand.frames[step], decision)}
              {@const key = decisionKey(hand.index, step)}
              {@const dora = hints ? hand.frames[step].view.dora_types ?? [] : []}
              <!-- Its number, the player's move in tiles and words, then
                   Trained AI's after an arrow, a tick where it agreed, that
                   the move was the only one, or that Trained AI has not
                   answered yet; Flagged where the filter keeps more than the
                   flagged decisions. Its name says the same in the engine's
                   words. -->
              <li>
                <button type="button" class="line" class:flag={line.flagged} class:outside={!listedIn(decision, filter)}
                  data-step-to={key} data-flagged={line.flagged ? key : undefined} aria-current={on(hand.index, step) ? 'step' : undefined}
                  aria-label={`Decision ${step + 1}. ${line.said}`} onclick={() => onplace({ hand: hand.index, step })}>
                  <span class="number">{step + 1}</span>
                  {#if line.offered}<span class="token">on <Tile tile={line.offered} size="tiny" dora={dora.includes(line.offered)} /></span>{/if}
                  {@render token(line.you, dora)}
                  {#if !line.reviewed}<span class="verdict">…</span>
                  {:else if line.forced}<span class="verdict">Only move</span>
                  {:else if line.agreed}<span class="verdict">✓</span>
                  {:else}<span class="arrow">→</span>{@render token(line.ai, dora)}{/if}
                  {#if line.flagged && filter !== 'flagged'}<strong class="flag-mark">Flagged</strong>{/if}
                </button>
              </li>
            {/each}
          </ol>
        {/if}
      </li>
    {/each}
  </ol>
</div>

{#snippet token(move, dora)}
  <span class="token">{#if move.word}<span>{move.word}</span>{/if}{#each move.tiles as tile, index (index)}<Tile {tile} size="tiny" dora={dora.includes(tile)} />{/each}</span>
{/snippet}

<style>
  .index { display: grid; gap: 8px; min-width: 0; font-size: .84rem; }
  p { margin: 0; }
  .empty { opacity: .82; }
  .hands, .lines { list-style: none; margin: 0; padding: 0; display: grid; min-width: 0; }
  .hands { gap: 10px; }
  .lines { gap: 2px; margin-top: 4px; }
  /* A hand's name and its change on a line, the outcome and its state under
     them, and Review at the end of the line where it is offered. */
  .head { display: flex; flex-wrap: wrap; align-items: center; gap: 0 10px; min-width: 0; }
  .name { min-height: 32px; padding: 4px 6px; margin-left: -6px; border: 0; border-radius: 6px; background: none; color: inherit;
    font: inherit; font-weight: 650; text-align: left; cursor: pointer; }
  span.name { display: inline-flex; align-items: center; cursor: default; }
  .name:hover { background: #ffffff14; }
  .change { margin-left: auto; font-variant-numeric: tabular-nums; opacity: .85; }
  .outcome, .state { flex-basis: 100%; min-width: 0; opacity: .8; font-size: .78rem; }
  .review { min-height: 36px; padding: 4px 12px; margin-top: 4px; }
  /* A decision's line: a flagged decision has the accent's edge, the one on
     show is gold, and one shown though its filter leaves it out is dim. */
  .line { width: 100%; min-height: 36px; display: flex; flex-wrap: wrap; align-items: center; gap: 2px 6px; padding: 3px 8px;
    border: 1px solid transparent; border-left: 3px solid transparent; border-radius: 6px; background: none; color: inherit;
    font: inherit; text-align: left; cursor: pointer; }
  .line:hover { background: #ffffff10; }
  .line.flag { border-left-color: var(--accent); }
  .line[aria-current='step'] { border-color: var(--gold); background: rgba(216, 161, 42, .2); }
  .line.outside { opacity: .6; }
  .number { min-width: 1.6em; font-weight: 650; font-variant-numeric: tabular-nums; }
  .token { display: inline-flex; align-items: center; gap: 3px; }
  .verdict { opacity: .85; }
  .arrow { opacity: .7; }
  .flag-mark { margin-left: auto; font-weight: 650; color: var(--gold); }
  /* Tapped, a line or a name keeps a touch's room. */
  @media (pointer: coarse) {
    .line, .name { min-height: 44px; }
  }
</style>
