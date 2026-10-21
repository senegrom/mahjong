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
   * is not.
   */
  let { hands, place = null, filter = $bindable('flagged'), counts, hints = true, trainedAvailable = false, empty = '', open = true,
    onplace, onreview, children } = $props();
  let root = $state(null);

  const on = (handIndex, step) => place?.hand === handIndex && place.step === step;

  // The list keeps the line of the decision on show in view by scrolling its
  // own box, the column it is on a wide screen or its box of hands where it
  // is folded above the decision, and never the page, which scrollIntoView
  // would move too. Another decision's line is brought into view by as
  // little as will, and so is the line as the list comes on the screen. Where
  // hands reviewed meanwhile, or another filter, draw lines above it, the line
  // stays where it was in the box: Chromium's scroll anchoring would keep it
  // so and Safari's would not, so the boxes have none (GameReview), and the
  // list does it itself, alike in every browser.
  let held = null;
  $effect.pre(() => {
    void hands; void filter; void place; void open;
    held = untrack(measured);
  });
  $effect(() => {
    void hands; void filter; void place; void open;
    const now = untrack(measured);
    if (!now) return;
    if (held?.key === now.key && held.box === now.box) now.box.scrollTop += now.offset - held.offset;
    else reveal(now);
  });

  /** The line on show, the box it scrolls in, and how far below the box's
   * top it is, or null where no box scrolls it. */
  function measured() {
    const line = root?.querySelector('[data-step-to][aria-current="step"]');
    const box = line && scroller(line);
    return box ? { key: line.dataset.stepTo, line, box, offset: line.getBoundingClientRect().top - box.getBoundingClientRect().top } : null;
  }
  /** The nearest box around an element that scrolls, short of the page. */
  function scroller(element) {
    for (let node = element.parentElement; node && node !== document.body; node = node.parentElement) {
      const { overflowY } = getComputedStyle(node);
      if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) return node;
    }
    return null;
  }
  /** Brings the line into the part of its box that is on the screen, as the
   * column runs on below the screen's foot at the top of the page; or into
   * the box, where none of it is on the screen. */
  function reveal({ line, box }) {
    const edge = box.getBoundingClientRect(), shown = line.getBoundingClientRect(), room = 8;
    const onScreen = edge.bottom > 0 && edge.top < innerHeight;
    const top = (onScreen ? Math.max(edge.top, 0) : edge.top) + room;
    const bottom = (onScreen ? Math.min(edge.bottom, innerHeight) : edge.bottom) - room;
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
