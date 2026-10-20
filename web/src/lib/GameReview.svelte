<script>
  import { tick, untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import DecisionFilter from './DecisionFilter.svelte';
  import DecisionReplay from './DecisionReplay.svelte';
  import ReviewTrend from './ReviewTrend.svelte';
  import { LEADING, agreement, filterCounts, filterDecisions, movedNote, weightsSaid } from './game-review.js';
  import { waitingNote } from './ui.js';

  /**
   * The game, looked at again: from any hand's result the game so far, led by
   * the hand just played, and once the match is over the whole game. Every
   * decision the player made, in every hand, is put to Trained AI, led by the
   * few where the player's move departed furthest from its choice, each of
   * which can be replayed and chosen again. A filter lists every decision
   * where Trained AI chose differently instead, or every decision, and any of
   * them, or any hand reviewed, can be replayed with every player's tiles in
   * view. The review itself (game-review.js) belongs to the match and goes on
   * while this screen is closed, until the next deal; the screen only shows
   * it. Whether the screen is open or not, the review waits before each batch
   * while the hand's own notes ask Trained AI, then goes on where it was, so
   * that a note never waits behind more than the batch already sent (App).
   * `trend` is the games this browser has reviewed, oldest first.
   * `filter` is which decisions the list shows: the flagged ones as it opens,
   * those where Trained AI chose differently, or every decision.
   */
  let { review, trend = [], hints = true, trainedAvailable = false, offline = null, onclose, filter = $bindable('flagged') } = $props();

  const read = owner => ({
    hands: owner.hands.map(hand => ({ ...hand })), busy: owner.busy, failure: owner.failure, held: owner.held,
    slow: owner.slow, estimate: owner.estimate, complete: owner.complete, requested: owner.requested,
    positionMs: owner.positionMs, over: owner.over,
  });
  // The review tells this screen when it changes; the screen reads it again.
  let version = $state(0);
  let status = $derived.by(() => { void version; return read(review); });
  // What is open for replay: a decision from the list, at its step, or a
  // hand from the list of hands, from its deal.
  let opened = $state.raw(null);
  // Where the decision opened ranked in the list when it was opened.
  let openedRank = -1;
  let everyShown = $state(false);
  // The listed decision whose button has keyboard focus, and what the list
  // says when the hands reviewed since have put it lower.
  let focusedItem = $state(null);
  let moved = $state('');
  let heading = $state(null);
  let list = $state(null);
  let controls = $state(null);
  let handList = $state(null);
  let flagsHeading = $state(null);
  // The row of the hand just played, which a review of the game so far
  // leads with.
  let lead = $state(null);

  $effect(() => {
    const owner = review;
    owner.onChange = () => { version++; };
    return () => { owner.onChange = () => {}; };
  });

  // Whether the list of hands, and the trend's table of games, are open.
  // Both are the player's to open and fold, and are kept while a replay
  // takes the list's place, so that Back to the list finds them as they
  // were, a replayed hand's button among them. A slow device holds after
  // each hand, and the list of hands then offers any hand to review next,
  // so it opens by itself once, when the review turns out slow. The states
  // are held here rather than set on the lists from the review's: the page
  // sets that afresh at each hand reviewed, which would undo what the player
  // did.
  let handsOpen = $state(untrack(() => status.slow && !status.complete));
  let trendOpen = $state(false);
  let openedForSlow = untrack(() => handsOpen);
  $effect(() => {
    if (openedForSlow || !status.slow || status.complete) return;
    openedForSlow = true;
    handsOpen = true;
  });

  // The review's buttons change under the player's hand: Continue becomes
  // Stop and Stop Continue, a slow device's hold offers the next hand, and a
  // hand's own button goes once that hand is under review. Focus that would
  // drop to the page goes where the button was instead: to the controls the
  // review has now, or to the list of hands; and when nothing is left to
  // press, to the listed decisions. A listed decision's button is not
  // taken away while it has focus (below); should focus drop from the list
  // all the same, the list's heading takes it.
  let focusWas = null;
  $effect.pre(() => {
    void status;
    const active = document.activeElement;
    focusWas = controls?.contains(active) ? 'controls' : handList?.contains(active) ? 'hands' : list?.contains(active) ? 'flags' : null;
  });
  $effect(() => {
    void status;
    const was = focusWas;
    focusWas = null;
    const active = document.activeElement;
    if (!was || (active && active !== document.body && active.isConnected)) return;
    (was === 'hands' ? handList?.querySelector('summary') : was === 'flags' ? flagsHeading : controls?.querySelector('button') ?? flagsHeading)?.focus();
  });

  // Opening the review asks for it, once Trained AI can be reached; after
  // that the player decides when it stops and goes on.
  $effect(() => {
    if (trainedAvailable && !review.requested && !review.closed) untrack(() => review.review());
  });

  const key = decision => `${decision.hand}:${decision.index}`;
  let decisions = $derived(status.hands.flatMap(hand => hand.decisions ?? []));
  let counts = $derived(filterCounts(decisions));
  let listed = $derived(filterDecisions(decisions, filter));
  /** Where a listed decision now stands, from 0, or -1. */
  const rankOf = wanted => listed.findIndex(decision => key(decision) === wanted);
  let figure = $derived(agreement(decisions));
  // The leading few, or every decision the filter holds: asked for, or
  // because the hands reviewed since have put the decision the player is on
  // out of the leading few. Its button is then kept, not taken from under
  // their focus.
  let shown = $derived(everyShown || (focusedItem !== null && rankOf(focusedItem) >= LEADING) ? listed : listed.slice(0, LEADING));
  let total = $derived(status.hands.reduce((sum, hand) => sum + hand.frames.length, 0));
  let done = $derived(status.hands.reduce((sum, hand) => sum
    + (hand.status === 'done' ? hand.frames.length : hand.status === 'reviewing' ? hand.done ?? 0 : 0), 0));
  let reviewing = $derived(status.hands.find(hand => hand.status === 'reviewing') ?? null);
  let reviewed = $derived(status.hands.filter(hand => hand.status === 'done').length);
  let download = $derived(waitingNote('', offline));
  let openedHand = $derived(opened ? status.hands[opened.hand] : null);
  // A decision, or a hand, is open for replay, in place of the list.
  let replaying = $derived(Boolean(opened && openedHand?.decisions));
  // The hand just played, whose result the player came from while the match
  // is under way: the review of the game so far asks about it first, and
  // leads with it. A finished game has none.
  let latest = $derived(status.over ? null : status.hands.at(-1) ?? null);

  // What the list is headed, and what the button that shows the rest of it
  // calls them, for each filter. A heading names its list as the pressed
  // filter does, in the review's verdicts; focus lands on it, so each is a
  // phrase of its own rather than a word.
  const HEADINGS = { flagged: 'Flagged decisions', differed: 'Where Trained AI differed', every: 'Every decision' };
  const REST = { flagged: count => `Show all ${count} flagged`, differed: count => `Show all ${count} that differ`, every: count => `Show all ${count} decisions` };

  $effect(() => {
    heading?.focus({ preventScroll: true });
    heading?.scrollIntoView({ block: 'start' });
  });

  // Each hand reviewed ranks the list afresh, and can put the decision the
  // player is on lower, or out of the leading few. Its button keeps focus
  // wherever it goes: once it leaves the leading few, every decision the
  // filter holds is shown and stays shown, rather than the list taking it
  // away. A list should not be reshuffled under someone reading it without
  // their being told, so the status then says where it is now. A place, once
  // said, holds until the next change of place, which says it again.
  let heard = null;
  $effect(() => {
    const at = focusedItem === null ? -1 : rankOf(focusedItem);
    untrack(() => {
      if (at >= 0 && heard?.key === focusedItem && at > heard.rank) {
        const widened = at >= LEADING && !everyShown;
        if (widened) everyShown = true;
        moved = movedNote(at, widened, filter);
      }
      heard = at >= 0 ? { key: focusedItem, rank: at } : null;
    });
  });

  // The note is about the decision the player was on, so it goes when focus
  // moves to another or leaves the list. A window or tab that loses focus
  // sends a focusout too, but focus stays on the decision, which has it again
  // when the window comes back, so its button is kept meanwhile. Focus that
  // moves on has left the decision by the time its focusout is heard.
  function focusItem(event) {
    const entry = event.target.closest('[data-listed]')?.dataset.listed ?? null;
    if (entry !== focusedItem) moved = '';
    focusedItem = entry;
  }
  function leaveList(event) {
    if (list?.contains(event.relatedTarget) || list?.contains(document.activeElement)) return;
    focusedItem = null;
    moved = '';
  }

  // Another filter lists another set of decisions, the leading few again.
  function filtered() {
    everyShown = false;
    moved = '';
  }

  function openDecision(decision) {
    openedRank = rankOf(key(decision));
    focusedItem = null;
    moved = '';
    opened = { key: key(decision), hand: decision.hand, start: decision.index, retry: Boolean(decision.judgement?.flagged), whole: false };
  }

  // `from` is the button that opened the hand, for Back to the list to
  // return to: its own in the list of hands, or the hand just played's.
  function openHand(hand, from = 'hands') {
    focusedItem = null;
    moved = '';
    opened = { key: `hand ${hand.index}`, hand: hand.index, start: 0, retry: false, whole: true, from };
  }

  function duration(seconds) {
    if (seconds < 60) return `${Math.max(5, Math.round(seconds / 5) * 5)} seconds`;
    const minutes = Math.round(seconds / 60);
    return `${minutes} minute${minutes === 1 ? '' : 's'}`;
  }

  /** How a hand ended, and what it changed for the player. */
  const handOutcome = hand => `${hand.outcome}${hand.change ? ` · you ${hand.change > 0 ? '+' : '−'}${Math.abs(hand.change).toLocaleString()}` : ''}`;

  function handState(hand) {
    if (!hand.frames.length) return 'No decisions';
    if (hand.status === 'reviewing') return `Reviewing, ${hand.done ?? 0} of ${hand.frames.length}`;
    if (hand.status !== 'done') return `${hand.frames.length} decision${hand.frames.length === 1 ? '' : 's'}, not reviewed`;
    const { matched, decisions: counted } = agreement(hand.decisions);
    return counted ? `${matched} of ${counted} matched` : 'No choices to compare';
  }

  // Said once a hand is done or the review stops, not at every batch: the
  // decision count beside it changes too often to be read out. While the
  // match is under way, the heading says the review is of the game so far,
  // so the headline need not. The whole screen is Trained AI's review, so the
  // headline names Trained AI only where it cannot be reached: a review not
  // yet asked for starts by itself once it can be (above), and one asked for
  // offers Continue again then.
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
  // What the list says when its filter holds nothing yet. While the review
  // goes on, the headline and the progress above say how far it has got;
  // once it is done, the list says it in the words its filter says it in
  // (filterSaid).
  let empty = $derived.by(() => {
    if (!status.complete) return 'None yet.';
    return { flagged: 'None flagged.', differed: 'Trained AI agreed at every decision.',
      every: status.over ? 'You made no decisions.' : 'No decisions yet.' }[filter];
  });

  // Back to what was opened. A hand goes back to the button that opened it:
  // its own in the list of hands, kept open for it, or the one in the row of
  // the hand just played. A decision goes back to its button wherever it now
  // stands: the review goes on while a decision is open, and the hands it
  // reviews meanwhile can put the decision lower, or out of the leading few.
  // Then every decision the filter holds is shown, so the player comes back
  // to it all the same, and the status says where it went. Should the button
  // be missing all the same, the list's heading takes focus rather than the
  // page.
  async function closeDecision() {
    const was = opened;
    opened = null;
    if (!was) return;
    if (was.whole) {
      await tick();
      const button = was.from === 'latest' ? lead?.querySelector('[data-replay-latest]')
        : handList?.querySelector(`[data-replay-hand="${was.hand}"]`);
      (button ?? flagsHeading)?.focus();
      return;
    }
    const rankWas = openedRank, at = rankOf(was.key);
    const widened = at >= LEADING && !everyShown;
    if (widened) everyShown = true;
    await tick();
    (list?.querySelector(`[data-open="${was.key}"]`) ?? flagsHeading)?.focus();
    if (at > rankWas && rankWas >= 0) moved = movedNote(at, widened, filter);
  }
</script>

<!-- Not marked busy while it reviews: that takes long enough for a screen
     reader to miss every status it says, and the list is usable meanwhile. -->
<section class="game-review" aria-labelledby="game-review-title" data-game-review>
  <header>
    <h2 id="game-review-title" tabindex="-1" bind:this={heading}>{status.over ? 'Your game, looked at again' : 'Your game so far, looked at again'}</h2>
    <!-- Back to where the review was opened: the hand's result while the
         match is under way, and the standings once it is over. -->
    <button class="app-control" data-close-review onclick={onclose}>{status.over ? 'Back to the standings' : 'Back to the result'}</button>
  </header>

  {#if replaying}
    {#key opened.key}<DecisionReplay hand={openedHand} start={opened.start} retry={opened.retry} whole={opened.whole} {hints} onback={closeDecision} />{/key}
  {:else}
    {#if latest}
      <!-- The hand the player came from, which the review of the game so far
           asks about first. It has no heading of its own, which would come
           before the list's as a second, for one row, and its state is no
           status: the headline already says which hand is under review, and
           the count beside it changes too often to be read out. -->
      <div class="latest" bind:this={lead} data-latest-hand={latest.index}>
        <span class="label">Just played:</span>
        <span class="name">{latest.name}</span>
        <span class="outcome">{handOutcome(latest)}</span>
        <span class="state">{handState(latest)}</span>
        <!-- The row names its hand, so its button need only say what it
             does; its accessible name names the hand too, after that word.
             It does what the list of hands' button for the hand does, and
             has its name. -->
        {#if latest.status === 'done' && latest.frames.length}
          <button class="app-control" data-replay-latest onclick={() => openHand(latest, 'latest')}
            aria-label={`Replay: ${latest.name}`}>Replay</button>
        {/if}
      </div>
    {/if}

    <div class="progress">
      <p class="headline" role="status">{headline}</p>
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

    <!-- The game so far is not yet a game for the trend: it joins once the
         match is over and every hand is reviewed. -->
    {#if figure.decisions}
      <ReviewTrend {figure} complete={status.complete && status.over} over={status.over} games={trend} current={review.id} bind:open={trendOpen} />
    {/if}

    <h3 tabindex="-1" bind:this={flagsHeading}>{HEADINGS[filter]}</h3>
    <DecisionFilter bind:value={filter} {counts} label="Which decisions to list" said={{ leading: LEADING }} onchange={filtered} />
  {/if}
  <!-- Under the heading, and in the page while a decision is open as well: a
       screen reader hears a status only once it is in the accessibility tree,
       and Back to the list rebuilds the list around this line as it speaks.
       Out of the layout while it is silent. -->
  <p class="moved" class:visually-hidden={!moved} data-moved role="status">{moved}</p>
  {#if !replaying}
    {#if shown.length}
      <!-- This list and the list of hands are drawn without markers, so each
           says it is a list in so many words: Safari does not tell
           VoiceOver that a list without them is one. -->
      <ol class="flagged" role="list" bind:this={list} onfocusin={focusItem} onfocusout={leaveList}>
        {#each shown as decision (key(decision))}
          {@const flagged = Boolean(decision.judgement?.flagged)}
          {@const name = status.hands[decision.hand].name}
          <li data-listed={key(decision)} data-flagged={flagged ? key(decision) : undefined} class:flag={flagged}>
            <p class="where">{name} · decision {decision.index + 1}{#if flagged && filter !== 'flagged'}{' · '}<strong class="flag-mark">Flagged</strong>{/if}</p>
            {#if decision.call_tile}<p class="context">Response to <Tile tile={decision.call_tile} size="tiny" dora={hints && (decision.dora_types ?? []).includes(decision.call_tile)} /> from {decision.call_from}</p>{/if}
            <p class="move">{#if decision.played_tile}<Tile tile={decision.played_tile} size="small" dora={hints && (decision.dora_types ?? []).includes(decision.played_tile)} />{/if}<span>You: {decision.played}</span></p>
            {#if decision.forced}
              <p class="weight">Only move.</p>
            {:else if decision.agreed}
              <p class="weight">Trained AI agreed.</p>
            {:else}
              <!-- Trained AI's own choice waits behind a flagged decision's
                   retry; any other says it here. -->
              {#if !flagged}<p class="move">{#if decision.advised_tile}<Tile tile={decision.advised_tile} size="small" dora={hints && (decision.dora_types ?? []).includes(decision.advised_tile)} />{/if}<span>Trained AI: {decision.advised}</span></p>{/if}
              {#if decision.judgement}
                <p class="weight">{weightsSaid(decision.judgement)}</p>
              {/if}
            {/if}
            <!-- A flagged decision opens to be chosen again, as the replay's
                 own button for it says; any other opens to be looked at. -->
            <button class="app-control" data-open={key(decision)} onclick={() => openDecision(decision)}
              aria-label={`${flagged ? 'Choose again' : 'Replay'}: ${name}, decision ${decision.index + 1}`}>{flagged ? 'Choose again' : 'Replay'}</button>
          </li>
        {/each}
      </ol>
      {#if listed.length > LEADING}
        <button class="app-control quiet" data-every-flag onclick={() => { everyShown = !everyShown; }}>{everyShown ? `Show the first ${LEADING}` : REST[filter](listed.length)}</button>
      {/if}
    {:else}
      <p class="empty">{empty}</p>
    {/if}

    <details class="hands" bind:open={handsOpen} bind:this={handList}>
      <summary>Hands · {reviewed} of {status.hands.length} reviewed</summary>
      <ol role="list">
        {#each status.hands as hand (hand.index)}
          <li>
            <span class="name">{hand.name}</span>
            <span class="outcome">{handOutcome(hand)}</span>
            <span class="state">{handState(hand)}</span>
            <!-- As in the row of the hand just played: the row names the hand,
                 and the button's accessible name names it after its word. -->
            {#if hand.status === 'waiting' && hand.frames.length && trainedAvailable}
              <button class="app-control" data-review-hand={hand.index} onclick={() => review.review([hand.index])}
                aria-label={`Review: ${hand.name}`}>Review</button>
            {:else if hand.status === 'done' && hand.frames.length}
              <button class="app-control" data-replay-hand={hand.index} onclick={() => openHand(hand)}
                aria-label={`Replay: ${hand.name}`}>Replay</button>
            {/if}
          </li>
        {/each}
      </ol>
    </details>
  {/if}
</section>

<style>
  .game-review { display: grid; gap: 12px; min-width: 0; padding: 14px; border-radius: 14px; background: rgba(0, 0, 0, .28); border: 1px solid rgba(216, 161, 42, .4); }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 12px; }
  h2 { margin: 0; font-size: 1.15rem; font-weight: 600; }
  h2:focus, h3:focus { outline: none; }
  h2:focus-visible, h3:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  h3 { margin: 6px 0 0; font-size: 1rem; font-weight: 600; }
  p { margin: 0; }
  .progress { display: grid; gap: 6px; }
  .headline { font-size: .9rem; }
  .detail { font-size: .8rem; opacity: .8; font-variant-numeric: tabular-nums; }
  progress { width: 100%; max-width: 420px; height: 8px; accent-color: var(--gold); }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; }
  .controls:empty { display: none; }
  .quiet { justify-self: start; background: transparent; }
  .flagged { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
  /* A flagged decision has the accent's edge, and says so in words outside
     the list of flagged decisions; any other a plain one. */
  .flagged li { display: grid; gap: 6px; min-width: 0; padding: 10px 12px; border-radius: 10px; background: rgba(0, 0, 0, .24); border-left: 3px solid #ffffff33; }
  .flagged li.flag { border-left-color: var(--accent); }
  .flag-mark { font-weight: 650; color: var(--gold); }
  .where { font-size: .78rem; letter-spacing: .04em; opacity: .75; }
  .context, .move { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 8px; font-size: .9rem; }
  .context { font-size: .8rem; }
  .weight { font-size: .84rem; }
  .flagged button { justify-self: start; }
  .empty { font-size: .86rem; opacity: .82; }
  .moved { font-size: .84rem; }
  .hands { font-size: .84rem; }
  .hands summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .hands ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
  .hands li { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding: 6px 8px; border-radius: 8px; background: #0002; }
  /* The hand just played reads as a row of the list of hands, which it is,
     brought up to lead the review of the game so far: edged in gold, as the
     seat whose move it was is edged in a replay, under a line that says why
     it leads. */
  .latest { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding: 8px 10px; border-radius: 8px; background: #0002;
    border-left: 3px solid var(--gold); font-size: .84rem; }
  .latest .label { flex-basis: 100%; font-size: .78rem; letter-spacing: .04em; opacity: .75; }
  .name { font-weight: 600; min-width: 6em; }
  .outcome { flex: 1 1 14em; min-width: 0; opacity: .85; }
  .state { font-size: .78rem; opacity: .8; }
  @media (max-width: 760px) { .game-review { padding: 10px; } }
  /* Where the game's bar stays at the top of the screen (AppSettings), a
     heading scrolled to must land below it, not under it. */
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h2, h3 { scroll-margin-top: 72px; }
  }
</style>
