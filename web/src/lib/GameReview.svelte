<script>
  import { onMount, tick, untrack } from 'svelte';
  import DecisionReplay from './DecisionReplay.svelte';
  import ReviewIndex from './ReviewIndex.svelte';
  import ReviewStepBar from './ReviewStepBar.svelte';
  import ReviewTrend from './ReviewTrend.svelte';
  import { FILTER_LABELS, agreement, decisionKey, filterCounts, handEntry, hiddenAt, landing, neighbour, playlist, revealing } from './game-review.js';
  import { REVIEW_WIDE, heldOnSteps, moveHandFocus, reviewKey, waitingNote } from './ui.js';

  /**
   * The game, looked at again, on one screen: from any hand's result the game
   * so far, and once the match is over the whole game. Every decision the
   * player made, in every hand, is put to Trained AI, the newest hand first,
   * and the screen opens on that hand: the hand just played, or the final
   * one. A step bar at the top of the screen goes from one decision to the
   * next that its filter keeps, the flagged ones, those where Trained AI
   * differed or every one, across hands, and gives the move made there beside
   * Trained AI's. Under it is the decision: the table as it stood, every
   * player's tiles face up unless the player keeps them back (below), and
   * Trained AI's view. The list of hands and decisions, which goes to any
   * decision or hand, is a column of its own on the left where the screen is
   * wide enough (REVIEW_WIDE), and elsewhere folded between the step bar and
   * the decision.
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
   *
   * The player can have the review keep Trained AI's move back (`hideMove`),
   * and the other players' tiles (`hideTiles`), until they ask for them at
   * each decision, by Show answer or Show tiles, or by picking a move of
   * their own, which is kept as Now. Both settings are stored with the
   * preferences, and offered in the review's header as in the settings.
   * `revealed` is what the player has revealed, by decision ("hand:step",
   * revealing), which App keeps for the match, so that no decision asks
   * again, whatever is closed, opened or stepped through meanwhile.
   */
  let { review, trend = [], hints = true, trainedAvailable = false, offline = null, onclose, shortcuts = true, touch = false,
    filter = $bindable('flagged'), place = $bindable(null), revealed = $bindable({}), hideMove = $bindable(false),
    hideTiles = $bindable(false) } = $props();

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
  // What the player has revealed of the decision on show, and what it keeps
  // back from them now (hiddenAt): Trained AI's move, the other players'
  // tiles, both or neither. A decision never holds Next back, however much
  // it keeps back.
  let reveal = $derived(at ? revealed[decisionKey(at.hand, at.step)] ?? null : null);
  let hidden = $derived(hiddenAt(current, reveal, { hideMove, hideTiles }));
  let pick = $derived(reveal?.pick ?? null);

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

  // Reveals what the decision on show keeps back: its answer, and the other
  // players' tiles where those are kept back too, or the tiles alone where
  // only they are; and keeps the move the player picked there to answer it,
  // `pick`, which reveals the decision as Show answer does. It comes from
  // Show answer or Show tiles, a move picked in the hand or under it, or a
  // key. App keeps the reveal for the match, so the decision never asks
  // again. A reveal takes away the control that made it, the button or the
  // live hand's tile, and focus then goes on to Next, or to Previous at the
  // end, where the player steps from, rather than to the page; a key leaves
  // focus where it was, unless that went too. Only a keyboard's press
  // (`byKeyboard`) moves the page to show where focus went, as App's land()
  // does: on a screen 500 pixels high or less the step bar stays in its
  // place, often far above, and a scroll there would take the hand the
  // player just pressed, and the answer it brought, off the screen.
  //
  // The other players' racks turn over where they lie, and the table stays
  // where the player was looking: what goes with the question can still move
  // it. On a phone held upright the line under the hand and the moves offered
  // there lie between the player's seat and the table, and the step bar's
  // line of moves can take another row once it gives the answer. So the page
  // is moved by as far as the table moved, before focus is handed on, alike
  // in every browser, as Safari has no scroll anchoring; unless the table
  // lies wholly below the screen, where it is not what the player is looking
  // at, and what they are looking at, above it, has not moved.
  async function revealHere(move, byKeyboard) {
    if (!at || (!hidden.move && !hidden.tiles) || (move && !hidden.move)) return;
    const had = root?.contains(document.activeElement);
    const table = () => root?.querySelector('[data-decision] .board')?.getBoundingClientRect().top ?? null;
    const before = table();
    revealed = { ...revealed, [decisionKey(at.hand, at.step)]: revealing(current, move) };
    await tick();
    const after = table();
    if (before !== null && after !== null && before < innerHeight && Math.abs(after - before) >= 0.5) scrollBy(0, after - before);
    if (had && lost(document.activeElement)) (stepButton() ?? heading)?.focus({ preventScroll: !byKeyboard });
  }

  // The player's hand while it is live, Trained AI's move kept back: its
  // tiles are what the digits pick, and ← and → move along it there.
  const liveHand = () => (hidden.move ? root?.querySelector('[data-decision] .mine .hand') ?? null : null);

  // The tile a digit picks while the hand is live, as play's digits throw
  // one: 1 to 9 the hand's first nine tiles, 0 the tile just drawn; nothing
  // where that tile is no legal discard there.
  function digitTile(action) {
    const frame = shown.frames[at.step], me = frame.view.seats[0];
    const tile = action === 'pick-drawn' ? me.drawn : me.hand[Number(action.slice('pick-'.length)) - 1];
    return tile && frame.choices.some(choice => choice.kind === 'discard' && choice.tile === tile) ? tile : null;
  }

  // The review's keys (reviewKey): ← and → step as Previous and Next do,
  // with Shift to the hand before or after; s shows what the decision keeps
  // back, as the step bar's button does; while the hand is live, the digits
  // pick a tile of it and ← and → move along it, as play's keys do; and Esc
  // goes back. They work where the player keeps keyboard shortcuts on, as
  // play's do, and not on a touch screen; never while a dialog is open over
  // the page, such as the settings' or the table's discards, nor in a field.
  // Whatever the shortcuts, a held Enter presses the step bar's buttons only
  // once (heldOnSteps): held on Show answer or a tile of the live hand, it
  // would press Next as the reveal handed focus there, and step past the
  // answer it had just shown.
  function press(event) {
    if (heldOnSteps(event)) {
      event.preventDefault();
      return;
    }
    if (!shortcuts || touch) return;
    const hand = liveHand(), action = reviewKey(event, { hand, dialogOpen: Boolean(document.querySelector('dialog[open]')) });
    if (action === 'back') onclose?.();
    else if (action === 'previous' || action === 'next') void go(action === 'next' ? next : previous);
    else if (action === 'previous-hand' || action === 'next-hand') void go(handStep(action === 'next-hand' ? 1 : -1));
    else if (action === 'previous-tile' || action === 'next-tile') moveHandFocus(hand, action === 'next-tile' ? 1 : -1);
    else if (action === 'reveal' && (hidden.move || hidden.tiles)) void revealHere(null, true);
    else if (action?.startsWith('pick-') && digitTile(action)) void revealHere({ kind: 'discard', tile: digitTile(action) }, true);
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
    <div class="actions">
      <!-- The two settings, here as in Options, where the player quizzes
           themselves; the same settings, so the two places agree. On a phone
           they are in the settings alone, which leave the review more room. -->
      <div class="hiding">
        <label><input type="checkbox" bind:checked={hideMove} data-hide-move /> Hide Trained AI's move</label>
        <label><input type="checkbox" bind:checked={hideTiles} data-hide-tiles /> Hide other players' tiles</label>
      </div>
      <!-- Back to where the review was opened: the hand's result while the
           match is under way, and the standings once it is over. -->
      <button class="app-control" data-close-review onclick={onclose}>{status.over ? 'Back to the standings' : 'Back to the result'}</button>
    </div>
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
        <ReviewStepBar hand={shown} step={at.step} decision={current} {previous} {next} {waiting} {filter} moveHidden={hidden.move}
          tilesHidden={hidden.tiles} hiding={hideMove || hideTiles} {pick} onstep={go} onreveal={byKeyboard => revealHere(null, byKeyboard)}
          bind:height={barHeight} />
      {/if}
      {#if !wide}
        <details class="list" data-review-index bind:open={listOpen}>
          <summary>Hands · {FILTER_LABELS[filter]} ({counts[filter]})</summary>
          {@render index(listOpen)}
        </details>
      {/if}
      {#if at}
        <DecisionReplay hand={shown} step={at.step} {hints} moveHidden={hidden.move} tilesHidden={hidden.tiles} {pick} onpick={revealHere} />
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
  <ReviewIndex hands={status.hands} place={at} bind:filter {counts} {hints} {trainedAvailable} {empty} {open} bar={barHeight} {hideMove} {revealed}
    onplace={go} onreview={hand => review.review([hand])}>
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
  /* The two settings and Back at the right of the header, Back last; where
     they have no room beside the heading they take a row of their own, and
     Back one under them, still at the right. The settings are drawn as
     Options draws its checkboxes, a finger's height where a finger presses
     them. On a phone they are left to the settings. */
  .actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 4px 16px; margin-left: auto; }
  .hiding { display: flex; flex-wrap: wrap; align-items: center; gap: 0 16px; }
  .hiding label { display: inline-flex; align-items: center; gap: 8px; min-height: 36px; font-size: .85rem; cursor: pointer; }
  .hiding input { width: 20px; height: 20px; margin: 0; accent-color: var(--gold); }
  @media (pointer: coarse) { .hiding label { min-height: 44px; } }
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
    .hiding { display: none; }
    .stage :global(:is(button, summary):not([data-steps] *)) { scroll-margin-top: calc(56px + var(--steps-height, 84px) + 8px); }
  }
  @media (max-height: 500px) {
    .stage :global(:is(button, summary):not([data-steps] *)) { scroll-margin-top: 72px; }
  }
  /* Where the game's bar stays at the top of the screen (AppSettings), the
     heading scrolled to must land below it, not under it, and so must the
     header's controls beside it, focused from further down the page. */
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h2, header :is(input, button) { scroll-margin-top: 72px; }
  }
</style>
