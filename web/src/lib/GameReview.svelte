<script>
  import { onMount, tick, untrack } from 'svelte';
  import DecisionReplay from './DecisionReplay.svelte';
  import ReviewIndex from './ReviewIndex.svelte';
  import ReviewStepBar from './ReviewStepBar.svelte';
  import ReviewTrend from './ReviewTrend.svelte';
  import { FILTER_LABELS, agreement, filterCounts, handEntry, landing, neighbour, playlist } from './game-review.js';
  import { REVIEW_WIDE, reviewKey, waitingNote } from './ui.js';

  /**
   * The game, looked at again, on one screen: from any hand's result the game
   * so far, and once the match is over the whole game. Every decision the
   * player made, in every hand, is put to Trained AI, the newest hand first,
   * and the screen opens on that hand: the hand just played, or the final
   * one. A step bar at the top of the screen goes from one decision to the
   * next that its filter keeps, the flagged ones, those where Trained AI
   * differed or every one, across hands, and gives the move made there beside
   * Trained AI's. Under it is the decision: the table as it stood, every
   * player's tiles face up, and Trained AI's view. The list of hands and
   * decisions, which goes to any decision or hand, is a column of its own on
   * the left where the screen is wide enough (REVIEW_WIDE), and elsewhere
   * folded between the step bar and the decision.
   *
   * The review itself (game-review.js) belongs to the match and goes on while
   * this screen is closed, until the next deal; the screen only shows it.
   * Whether the screen is open or not, the review waits before each batch
   * while the hand's own notes ask Trained AI, then goes on where it was, so
   * that a note never waits behind more than the batch already sent (App).
   * `trend` is the games this browser has reviewed, oldest first.
   *
   * `filter` is which decisions the step bar steps through and the list
   * shows, and `place` where the player is, `{ hand, step }`. App keeps both,
   * the filter with the preferences and the place for the match, so that the
   * review opens where the player left it; without a place it opens where it
   * lands (landing), which is kept as the place. The keys step and go back
   * where the player keeps keyboard shortcuts on (`shortcuts`), as play's do,
   * and not on a touch screen (`touch`).
   */
  let { review, trend = [], hints = true, trainedAvailable = false, offline = null, onclose, shortcuts = true, touch = false,
    filter = $bindable('flagged'), place = $bindable(null) } = $props();

  const read = owner => ({
    hands: owner.hands.map(hand => ({ ...hand })), busy: owner.busy, failure: owner.failure, held: owner.held,
    slow: owner.slow, estimate: owner.estimate, complete: owner.complete, requested: owner.requested,
    positionMs: owner.positionMs, over: owner.over,
  });
  // The review tells this screen when it changes; the screen reads it again.
  let version = $state(0);
  let status = $derived.by(() => { void version; return read(review); });
  let root = $state(null);
  let heading = $state(null);
  let controls = $state(null);
  // The step bar's height, which a control below it is kept clear of.
  let barHeight = $state(0);

  $effect(() => {
    const owner = review;
    owner.onChange = () => { version++; };
    return () => { owner.onChange = () => {}; };
  });

  // Whether the list of hands and decisions, the trend and the trend's table
  // of games are open. Each is the player's to open and fold, and stays as
  // they leave it across steps and hands. The states are held here rather
  // than set on the lists from the review's: the page sets that afresh at
  // each hand reviewed, which would undo what the player did. The list
  // starts folded, the decision taking the screen; a slow device holds after
  // each hand, and the list then offers any hand to review next, so it opens
  // by itself once, when the review turns out slow.
  let listOpen = $state(untrack(() => status.slow && !status.complete));
  let trendOpen = $state(false);
  let gamesOpen = $state(false);
  let openedForSlow = untrack(() => listOpen);
  $effect(() => {
    if (openedForSlow || !status.slow || status.complete) return;
    openedForSlow = true;
    listOpen = true;
  });

  // Whether the screen has room for the list in a column of its own beside
  // the decision (REVIEW_WIDE). A server render measures nothing, and folds
  // the list above the decision. The list is drawn afresh in its other place
  // as the screen changes, so a control of it that had focus has it again
  // there, the folded list opened for it, rather than focus dropping to the
  // page.
  let wide = $state(globalThis.matchMedia?.(REVIEW_WIDE).matches ?? false);
  const KEPT = ['data-step-to', 'data-hand-row', 'data-review-hand', 'data-filter-option'];
  onMount(() => {
    const media = matchMedia(REVIEW_WIDE);
    const changed = async () => {
      const active = document.activeElement, inList = Boolean(active?.closest?.('[data-review-index]'));
      const name = inList ? KEPT.find(attribute => active.hasAttribute(attribute)) : null;
      wide = media.matches;
      if (!inList) return;
      if (!wide) listOpen = true;
      await tick();
      const again = name && root?.querySelector(`[data-review-index] [${name}="${active.getAttribute(name)}"]`);
      (again || root?.querySelector('[data-review-index] [aria-current="step"]') || stepButton() || heading)?.focus();
    };
    media.addEventListener('change', changed);
    return () => media.removeEventListener('change', changed);
  });

  // Opening the review asks for it, once Trained AI can be reached; after
  // that the player decides when it stops and goes on.
  $effect(() => {
    if (trainedAvailable && !review.requested && !review.closed) untrack(() => review.review());
  });

  let decisions = $derived(status.hands.flatMap(hand => hand.decisions ?? []));
  let total = $derived(status.hands.reduce((sum, hand) => sum + hand.frames.length, 0));
  let done = $derived(status.hands.reduce((sum, hand) => sum
    + (hand.status === 'done' ? hand.frames.length : hand.status === 'reviewing' ? hand.done ?? 0 : 0), 0));
  // How many decisions each filter keeps: the flagged ones, and those where
  // Trained AI differed, of the hands reviewed, and every decision of every
  // hand, which the table and the player's move give before Trained AI
  // answers.
  let counts = $derived({ ...filterCounts(decisions), every: total });
  let figure = $derived(agreement(decisions));
  let list = $derived(playlist(status.hands, filter));
  let reviewing = $derived(status.hands.find(hand => hand.status === 'reviewing') ?? null);
  let reviewed = $derived(status.hands.filter(hand => hand.status === 'done').length);
  let download = $derived(waitingNote('', offline));

  // The decision on show: the player's place, or where the review lands
  // while they have none for this match.
  const valid = entry => Boolean(entry && status.hands[entry.hand]?.frames[entry.step]);
  let at = $derived(valid(place) ? place : landing(status.hands, filter));
  let shown = $derived(at ? status.hands[at.hand] : null);
  let current = $derived(shown?.decisions?.[at.step] ?? null);
  // Where the review lands is kept as the player's place, so that nothing but
  // the player moves it on: not the hands reviewed later, nor another filter.
  $effect(() => {
    if (at && at !== place) place = { hand: at.hand, step: at.step };
  });
  // Previous and Next go to the decision the filter keeps before or after
  // the one on show, across hands, wherever the one on show is. A hand being
  // reviewed holds no decision the flagged or the differing ones keep until
  // its answers arrive, so Next waits for them there, rather than passing
  // over decisions about to be kept; every decision is held already.
  let previous = $derived(at ? neighbour(list, at, -1) : null);
  let waiting = $derived(Boolean(at) && shown.status === 'reviewing' && filter !== 'every');
  let next = $derived(!at || waiting ? null : neighbour(list, at, 1));

  // A control that had focus and has gone, or can no longer have it.
  const lost = element => !element || element === document.body || !element.isConnected || Boolean(element.disabled);
  // Where focus goes when the control that had it goes: Next, or Previous at
  // the end, which is where the player steps from.
  const stepButton = () => root?.querySelector('[data-steps] [data-next]:not(:disabled)')
    ?? root?.querySelector('[data-steps] [data-previous]:not(:disabled)') ?? null;

  // Goes to a place, from the step bar, the list or a key. Focus stays where
  // it is, unless the step takes away the control that had it, such as the
  // line of a decision the filter leaves out, or a button in Trained AI's
  // view the next decision has none of: then Next has it, or Previous at the
  // end, rather than the page.
  async function go(target) {
    if (!target) return;
    const had = root?.contains(document.activeElement);
    place = { hand: target.hand, step: target.step };
    await tick();
    if (had && lost(document.activeElement)) (stepButton() ?? heading)?.focus();
  }

  // The hand before or after the one on show that the player made a decision
  // in, at its first decision the filter keeps, or its first (handEntry).
  function handStep(direction) {
    for (let index = (at?.hand ?? 0) + direction; index >= 0 && index < status.hands.length; index += direction) {
      const entry = handEntry(status.hands, filter, index);
      if (entry) return entry;
    }
    return null;
  }

  // The review's keys (reviewKey): ← and → step as Previous and Next do,
  // with Shift to the hand before or after, and Esc goes back. They work
  // where the player keeps keyboard shortcuts on, as play's do, and not on a
  // touch screen; never while a dialog is open over the page, such as the
  // settings' or the table's discards, nor in a field.
  function press(event) {
    if (!shortcuts || touch) return;
    const action = reviewKey(event, { dialogOpen: Boolean(document.querySelector('dialog[open]')) });
    if (action === 'back') onclose?.();
    else if (action === 'previous' || action === 'next') void go(action === 'next' ? next : previous);
    else if (action === 'previous-hand' || action === 'next-hand') void go(handStep(action === 'next-hand' ? 1 : -1));
    else return;
    event.preventDefault();
  }

  // The review's buttons change under the player's hand: Continue becomes
  // Stop and Stop Continue, a slow device's hold offers the next hand, and a
  // hand's own Review goes once that hand is under review. Focus that would
  // drop to the page goes where the button was instead: to the controls the
  // review has now, or to the name of the hand whose Review went; and when
  // nothing is left to press, to Next, or Previous at the end.
  let focusWas = null;
  $effect.pre(() => {
    void status;
    const active = document.activeElement;
    focusWas = !root?.contains(active) ? null : controls?.contains(active) ? { to: 'controls' }
      : active.dataset.reviewHand !== undefined ? { to: 'hand', hand: active.dataset.reviewHand } : { to: 'step' };
  });
  $effect(() => {
    void status;
    const was = focusWas;
    focusWas = null;
    if (!was || !lost(document.activeElement)) return;
    const target = was.to === 'controls' ? controls?.querySelector('button')
      : was.to === 'hand' ? root?.querySelector(`[data-hand-row="${was.hand}"]`) : null;
    (target ?? stepButton() ?? heading)?.focus();
  });

  $effect(() => {
    heading?.focus({ preventScroll: true });
    heading?.scrollIntoView({ block: 'start' });
  });

  function duration(seconds) {
    if (seconds < 60) return `${Math.max(5, Math.round(seconds / 5) * 5)} seconds`;
    const minutes = Math.round(seconds / 60);
    return `${minutes} minute${minutes === 1 ? '' : 's'}`;
  }

  // Said once a hand is done or the review stops, not at every batch: the
  // decision count beside it changes too often to be read out. While the
  // match is under way, the heading says the review is of the game so far,
  // so the headline need not. The whole screen is Trained AI's review, so the
  // headline names Trained AI only where it cannot be reached: a review not
  // yet asked for starts by itself once it can be (above), and one asked for
  // offers Continue again then. Once every hand is reviewed the headline is
  // said, and not shown: there is nothing more to watch.
  let headline = $derived.by(() => {
    if (status.failure) return `Review stopped: ${status.failure}`;
    if (status.complete) return `All hands reviewed: ${total} decision${total === 1 ? '' : 's'}.`;
    if (!trainedAvailable && !status.busy) {
      return status.requested ? 'Trained AI cannot be reached right now.' : 'Waiting until Trained AI can be reached.';
    }
    if (status.busy) return reviewing ? `Reviewing ${reviewing.name}…` : 'Reviewing your game…';
    if (status.held) {
      return `Reviewed ${reviewed} of ${status.hands.length} hands; about ${(status.positionMs / 1000).toFixed(1)} seconds a decision on this device.`;
    }
    return `Reviewed ${reviewed} of ${status.hands.length} hands.`;
  });
  let detail = $derived(status.complete || !total ? ''
    : `${done} of ${total} decisions${status.estimate ? ` · about ${duration(status.estimate)} left` : ''}${status.busy && download ? ` · ${download}` : ''}`);
  // What the list says where its filter keeps no decision, once every hand
  // is reviewed; before that, the hands' own states say how far the review
  // has got.
  let empty = $derived(counts[filter] || !status.complete ? ''
    : { flagged: 'None flagged.', differed: 'Trained AI agreed at every decision.', every: status.over ? 'You made no decisions.' : 'No decisions yet.' }[filter]);
</script>

<svelte:window onkeydown={press} />

<!-- Not marked busy while it reviews: that takes long enough for a screen
     reader to miss every status it says, and the decisions reviewed are
     there to look at meanwhile. -->
<section class="game-review" aria-labelledby="game-review-title" data-game-review bind:this={root}>
  <header>
    <h2 id="game-review-title" tabindex="-1" bind:this={heading}>{status.over ? 'Whole game' : 'Game so far'}</h2>
    <!-- Back to where the review was opened: the hand's result while the
         match is under way, and the standings once it is over. -->
    <button class="app-control" data-close-review onclick={onclose}>{status.over ? 'Back to the standings' : 'Back to the result'}</button>
  </header>

  <div class="progress" class:finished={status.complete}>
    <p class="headline" class:visually-hidden={status.complete} role="status">{headline}</p>
    {#if detail}<p class="detail">{detail}</p>{/if}
    {#if status.busy || (done && !status.complete)}<progress max={total} value={done} aria-label="Decisions reviewed"></progress>{/if}
    <div class="controls" bind:this={controls}>
      {#if status.busy}
        <button class="app-control" data-stop-review onclick={() => review.stop()}>Stop</button>
      {:else if !status.complete && trainedAvailable}
        {#if status.held}
          <!-- Not "Next hand", which is the result's button that deals one. -->
          <button class="app-control primary" data-next-hand onclick={() => review.resume()}>Review next hand</button>
          <button class="app-control" data-rest onclick={() => review.resume({ patient: true })}>Review the rest</button>
        {:else}
          <!-- Continue answers Stop, in its place. Before the review is asked
               for, the button takes the name of the result's and the
               standings' buttons that start it. -->
          <button class="app-control primary" data-continue onclick={() => review.review(undefined, { patient: true })}>
            {status.requested ? 'Continue' : status.over ? 'Review whole game' : 'Review game so far'}</button>
        {/if}
      {/if}
    </div>
  </div>

  <div class="panes" class:wide>
    <!-- The step bar, the list where it is folded, and the decision, in the
         order they are read and tabbed through. The step bar stays at the
         top of the screen while the rest of the stage is scrolled under it. -->
    <div class="stage" data-stage style:--steps-height={barHeight ? `${barHeight}px` : undefined}>
      {#if at}
        <ReviewStepBar hand={shown} step={at.step} decision={current} {previous} {next} {waiting} {filter} onstep={go} bind:height={barHeight} />
      {/if}
      {#if !wide}
        <details class="list" data-review-index bind:open={listOpen}>
          <summary>Hands · {FILTER_LABELS[filter]} ({counts[filter]})</summary>
          {@render index(listOpen)}
        </details>
      {/if}
      {#if at}
        <DecisionReplay hand={shown} step={at.step} {hints} />
      {:else}
        <p class="empty">{status.over ? 'You made no decisions.' : 'No decisions yet.'}</p>
      {/if}
    </div>
    {#if wide}
      <!-- After the stage in the page, so that Tab reaches Previous and Next
           first; the grid draws it on the left. -->
      <aside class="rail" data-review-index aria-labelledby="review-hands">
        <h3 id="review-hands" class="visually-hidden">Hands</h3>
        {@render index(true)}
      </aside>
    {/if}
  </div>
</section>

<!-- The list, opened by the trend once a decision is reviewed. The game so far
     is not yet a game for the trend, which it joins once the match is over
     and every hand is reviewed. `open` is whether the list is on the screen,
     for it to keep its line in view, and `bar` the step bar's height, which
     the line keeps clear of as a step makes the bar taller. -->
{#snippet index(open)}
  <ReviewIndex hands={status.hands} place={at} bind:filter {counts} {hints} {trainedAvailable} {empty} {open} bar={barHeight} onplace={go}
    onreview={hand => review.review([hand])}>
    {#if figure.decisions}
      <ReviewTrend compact {figure} complete={status.complete && status.over} over={status.over} games={trend} current={review.id}
        bind:open={gamesOpen} bind:expanded={trendOpen} />
    {/if}
  </ReviewIndex>
{/snippet}

<style>
  .game-review { display: grid; gap: 12px; min-width: 0; padding: 14px; border-radius: 14px; background: rgba(0, 0, 0, .28); border: 1px solid rgba(216, 161, 42, .4); }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 12px; }
  h2 { margin: 0; font-size: 1.15rem; font-weight: 600; }
  h2:focus { outline: none; }
  h2:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  p { margin: 0; }
  .progress { display: grid; gap: 6px; }
  /* Once every hand is reviewed nothing of the progress is shown, and it
     takes no room: its headline is only said. */
  .progress.finished { display: contents; }
  .headline { font-size: .9rem; }
  .detail { font-size: .8rem; opacity: .8; font-variant-numeric: tabular-nums; }
  progress { width: 100%; max-width: 420px; height: 8px; accent-color: var(--gold); }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; }
  .controls:empty { display: none; }
  .panes { display: grid; min-width: 0; }
  /* The stage is the container the decision sizes the player's tiles by
     (DecisionReplay), as it narrows beside the list. */
  .stage { display: grid; gap: 10px; min-width: 0; container: stage / inline-size; }
  .empty { font-size: .86rem; opacity: .82; }
  .list { font-size: .84rem; min-width: 0; }
  .list > summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  /* Open above the decision, the list of hands scrolls in a box of its own,
     at most 60% of the screen tall, so that the decision is never far
     below; the box leaves room inside it for a line's focus ring. The list
     keeps its line in view, and what the player is looking at still, itself
     (ReviewIndex), in every browser alike, rather than by the browser's
     scroll anchoring, which Safari lacks. */
  .list :global(.hands) { max-height: 60dvh; overflow-y: auto; overflow-anchor: none; padding: 4px 6px; border-radius: 10px; background: #0002; }
  /* On a wide screen the list is a column of its own on the left, beside the
     decision, which comes first in the page. It stays on the screen as the
     decision is scrolled, and scrolls on its own, leaving room inside it for
     a focus ring; the filter's buttons are drawn smaller to keep one row. */
  .panes.wide { grid-template-columns: 300px minmax(0, 1fr); gap: 16px; align-items: start; }
  .panes.wide .stage { grid-column: 2; grid-row: 1; }
  .rail { grid-column: 1; grid-row: 1; position: sticky; top: 8px; max-height: calc(100dvh - 16px); overflow-y: auto; overflow-anchor: none;
    padding: 6px 8px 8px; border-radius: 10px; background: #0002; min-width: 0; }
  .rail :global([data-filter-option]) { padding: 4px 9px; font-size: .8rem; }
  /* A control focused below the step bar, which stays at the top of the
     screen, is brought into view below it rather than under it: the bar's
     own height, and on a phone held upright the game's bar above it too.
     Where the screen is 500 pixels high or less the step bar stays in its
     place, and only the game's bar, which stays there in landscape, is kept
     clear of, as the heading keeps clear of it. The bar's own buttons are
     left out: they are in the bar, and keep clear only of what stays above
     it (ReviewStepBar). The list folded above the decision takes the same
     margin, less the 8 pixels a control keeps, for how far what stays at the
     top of the screen reaches over it, and keeps its line on show below that
     (ReviewIndex). */
  .stage :global(:is(button, summary):not([data-steps] *)) { scroll-margin-top: calc(var(--steps-height, 56px) + 8px); }
  @media (max-width: 760px) {
    .game-review { padding: 10px; }
    .stage :global(:is(button, summary):not([data-steps] *)) { scroll-margin-top: calc(56px + var(--steps-height, 84px) + 8px); }
  }
  @media (max-height: 500px) {
    .stage :global(:is(button, summary):not([data-steps] *)) { scroll-margin-top: 72px; }
  }
  /* Where the game's bar stays at the top of the screen (AppSettings), the
     heading scrolled to must land below it, not under it. */
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h2 { scroll-margin-top: 72px; }
  }
</style>
