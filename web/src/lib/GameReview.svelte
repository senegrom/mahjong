<script>
  import { tick, untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import DecisionReplay from './DecisionReplay.svelte';
  import ReviewTrend from './ReviewTrend.svelte';
  import { FLAG_PLAYED_BELOW, FLAG_PREFERRED_FROM, LEADING, agreement, rankDecisions } from './game-review.js';
  import { percent, waitingNote } from './ui.js';
  import { trainedName } from './trained-network.js';

  /**
   * The whole game, looked at again once it is over: every decision the player
   * made, in every hand, put to Trained AI, and the few where the player's move
   * departed furthest from its choice, each of which can be replayed and
   * chosen again. The review itself (game-review.js) belongs to the finished
   * match and goes on while this screen is closed; the screen only shows it.
   * `trend` is the games this browser has reviewed, oldest first.
   */
  let { review, trend = [], hints = true, trainedAvailable = false, offline = null, onclose } = $props();

  const read = owner => ({
    hands: owner.hands.map(hand => ({ ...hand })), busy: owner.busy, failure: owner.failure, held: owner.held,
    slow: owner.slow, estimate: owner.estimate, complete: owner.complete, requested: owner.requested,
    positionMs: owner.positionMs,
  });
  // The review tells this screen when it changes; the screen reads it again.
  let version = $state(0);
  let status = $derived.by(() => { void version; return read(review); });
  let opened = $state.raw(null);
  let everyFlag = $state(false);
  let heading = $state(null);
  let list = $state(null);
  let controls = $state(null);
  let handList = $state(null);
  let flagsHeading = $state(null);

  $effect(() => {
    const owner = review;
    owner.onChange = () => { version++; };
    return () => { owner.onChange = () => {}; };
  });

  // The review's buttons change under the player's hand: Continue becomes
  // Stop and Stop Continue, a slow device's hold offers the next hand, and a
  // hand's own button goes once that hand is under review. Focus that would
  // drop to the page goes where the button was instead: to the controls the
  // review has now, or to the list of hands; and when nothing is left to
  // press, to the flagged decisions.
  let focusWas = null;
  $effect.pre(() => {
    void status;
    const active = document.activeElement;
    focusWas = controls?.contains(active) ? 'controls' : handList?.contains(active) ? 'hands' : null;
  });
  $effect(() => {
    void status;
    const was = focusWas;
    focusWas = null;
    const active = document.activeElement;
    if (!was || (active && active !== document.body && active.isConnected)) return;
    (was === 'hands' ? handList?.querySelector('summary') : controls?.querySelector('button') ?? flagsHeading)?.focus();
  });

  // Opening the review asks for it, once Trained AI can be reached; after
  // that the player decides when it stops and goes on.
  $effect(() => {
    if (trainedAvailable && !review.requested && !review.closed) untrack(() => review.review());
  });

  let decisions = $derived(status.hands.flatMap(hand => hand.decisions ?? []));
  let flagged = $derived(rankDecisions(decisions));
  let figure = $derived(agreement(decisions));
  let shown = $derived(everyFlag ? flagged : flagged.slice(0, LEADING));
  let total = $derived(status.hands.reduce((sum, hand) => sum + hand.frames.length, 0));
  let done = $derived(status.hands.reduce((sum, hand) => sum
    + (hand.status === 'done' ? hand.frames.length : hand.status === 'reviewing' ? hand.done ?? 0 : 0), 0));
  let reviewing = $derived(status.hands.find(hand => hand.status === 'reviewing') ?? null);
  let reviewed = $derived(status.hands.filter(hand => hand.status === 'done').length);
  let download = $derived(waitingNote('', offline));
  let openedHand = $derived(opened ? status.hands[opened.hand] : null);

  $effect(() => {
    heading?.focus({ preventScroll: true });
    heading?.scrollIntoView({ block: 'start' });
  });

  const key = decision => `${decision.hand}:${decision.index}`;

  function duration(seconds) {
    if (seconds < 60) return `${Math.max(5, Math.round(seconds / 5) * 5)} seconds`;
    const minutes = Math.round(seconds / 60);
    return `${minutes} minute${minutes === 1 ? '' : 's'}`;
  }

  function handState(hand) {
    if (!hand.frames.length) return 'No decisions';
    if (hand.status === 'reviewing') return `Reviewing, ${hand.done ?? 0} of ${hand.frames.length}`;
    if (hand.status !== 'done') return `${hand.frames.length} decision${hand.frames.length === 1 ? '' : 's'}, not reviewed yet`;
    const { matched, decisions: counted } = agreement(hand.decisions);
    return counted ? `${matched} of ${counted} matched` : 'No choices to compare';
  }

  // Said once a hand is done or the review stops, not at every batch: the
  // decision count beside it changes too often to be read out.
  let headline = $derived.by(() => {
    if (status.failure) return `The review stopped: ${status.failure}`;
    if (status.complete) return `Every hand is reviewed: ${total} decision${total === 1 ? '' : 's'}.`;
    if (!trainedAvailable && !status.busy) {
      return `The review needs Trained AI, which cannot be reached right now.${status.requested ? '' : ' It starts as soon as Trained AI can be reached.'}`;
    }
    if (status.busy) return reviewing ? `Trained AI is reviewing ${reviewing.name}.` : 'Trained AI is reviewing your game.';
    if (status.held) {
      return `Reviewed ${reviewed} of ${status.hands.length} hands. A decision takes about ${(status.positionMs / 1000).toFixed(1)} seconds on this device.`;
    }
    return `Reviewed ${reviewed} of ${status.hands.length} hands.`;
  });
  let detail = $derived(status.complete || !total ? ''
    : `${done} of ${total} decisions${status.estimate ? ` · the rest would take about ${duration(status.estimate)}` : ''}${status.busy && download ? ` · ${download}` : ''}`);

  // Back to the decision opened. Wider gaps found by the review meanwhile can
  // have moved it out of the leading few, so then every flag is shown, and
  // the player comes back to it all the same.
  async function closeDecision() {
    const opener = opened && key(opened);
    opened = null;
    if (opener && !shown.some(decision => key(decision) === opener)) everyFlag = true;
    await tick();
    list?.querySelector(`[data-open="${opener}"]`)?.focus();
  }
</script>

<!-- Not marked busy while it reviews: that takes long enough for a screen
     reader to miss every status it says, and the list is usable meanwhile. -->
<section class="game-review" aria-labelledby="game-review-title" data-game-review>
  <header>
    <h2 id="game-review-title" tabindex="-1" bind:this={heading}>Your game, looked at again</h2>
    <button class="app-control" data-close-review onclick={onclose}>Back to the standings</button>
  </header>

  {#if opened && openedHand}
    {#key key(opened)}<DecisionReplay hand={openedHand} decision={opened} {hints} onback={closeDecision} />{/key}
  {:else}
    <p class="intro">{trainedName('Trained AI')} is asked about every decision you made, hand by hand. A decision is flagged when your move had
      under {Math.round(FLAG_PLAYED_BELOW * 100)}% of Trained AI's weight while its own choice had {Math.round(FLAG_PREFERRED_FROM * 100)}% or more.
      Matching Trained AI is not the same as playing well: it has habits of its own.</p>

    <div class="progress">
      <p class="headline" role="status">{headline}</p>
      {#if detail}<p class="detail">{detail}</p>{/if}
      {#if status.busy || (done && !status.complete)}<progress max={total} value={done} aria-label="Decisions reviewed"></progress>{/if}
      <div class="controls" bind:this={controls}>
        {#if status.busy}
          <button class="app-control" data-stop-review onclick={() => review.stop()}>Stop</button>
        {:else if !status.complete && trainedAvailable}
          {#if status.held}
            <button class="app-control primary" data-next-hand onclick={() => review.resume()}>Review the next hand</button>
            <button class="app-control" data-rest onclick={() => review.resume({ patient: true })}>Review the rest of the game</button>
          {:else}
            <button class="app-control primary" data-continue onclick={() => review.review(undefined, { patient: true })}>
              {status.requested ? 'Continue the review' : 'Review the whole game'}</button>
          {/if}
        {/if}
      </div>
    </div>

    {#if figure.decisions}
      <ReviewTrend {figure} complete={status.complete} games={trend} current={review.id} />
    {/if}

    <h3 tabindex="-1" bind:this={flagsHeading}>Where you departed most from Trained AI</h3>
    {#if shown.length}
      <ol class="flagged" bind:this={list}>
        {#each shown as decision (key(decision))}
          <li data-flagged={key(decision)}>
            <p class="where">{status.hands[decision.hand].name} · your decision {decision.index + 1}</p>
            {#if decision.call_tile}<p class="context">Response to <Tile tile={decision.call_tile} size="tiny" dora={hints && (decision.dora_types ?? []).includes(decision.call_tile)} /> from {decision.call_from}</p>{/if}
            <p class="move">{#if decision.played_tile}<Tile tile={decision.played_tile} size="small" dora={hints && (decision.dora_types ?? []).includes(decision.played_tile)} />{/if}<span>You: {decision.played}</span></p>
            <p class="weight">Trained AI gave {decision.judgement.stage === 'tile' ? 'this tile, once riichi was declared,' : 'this move'} {percent(decision.judgement.played)}, and its own choice {percent(decision.judgement.preferred)}.</p>
            <button class="app-control" data-open={key(decision)} onclick={() => { opened = decision; }}
              aria-label={`Replay and choose again: ${status.hands[decision.hand].name}, your decision ${decision.index + 1}`}>Replay and choose again</button>
          </li>
        {/each}
      </ol>
      {#if flagged.length > LEADING}
        <button class="app-control quiet" onclick={() => { everyFlag = !everyFlag; }}>{everyFlag ? `Show the first ${LEADING}` : `Show all ${flagged.length} flagged`}</button>
      {/if}
    {:else if status.complete}
      <p class="empty">No decision departed that far from Trained AI.</p>
    {:else}
      <p class="empty">{reviewed ? 'None in the hands reviewed so far.' : 'They appear here as each hand is reviewed.'}</p>
    {/if}

    <details class="hands" open={status.slow && !status.complete} bind:this={handList}>
      <summary>Hands · {reviewed} of {status.hands.length} reviewed</summary>
      <ol>
        {#each status.hands as hand (hand.index)}
          <li>
            <span class="name">{hand.name}</span>
            <span class="outcome">{hand.outcome}{hand.change ? ` · you ${hand.change > 0 ? '+' : '−'}${Math.abs(hand.change).toLocaleString()}` : ''}</span>
            <span class="state">{handState(hand)}</span>
            {#if hand.status === 'waiting' && hand.frames.length && trainedAvailable}
              <button class="app-control" data-review-hand={hand.index} onclick={() => review.review([hand.index])}
                aria-label={`Review this hand: ${hand.name}`}>Review this hand</button>
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
  .intro { font-size: .84rem; opacity: .85; max-width: 70ch; }
  .progress { display: grid; gap: 6px; }
  .headline { font-size: .9rem; }
  .detail { font-size: .8rem; opacity: .8; font-variant-numeric: tabular-nums; }
  progress { width: 100%; max-width: 420px; height: 8px; accent-color: var(--gold); }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; }
  .controls:empty { display: none; }
  .quiet { justify-self: start; background: transparent; }
  .flagged { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
  .flagged li { display: grid; gap: 6px; min-width: 0; padding: 10px 12px; border-radius: 10px; background: rgba(0, 0, 0, .24); border-left: 3px solid var(--accent); }
  .where { font-size: .78rem; letter-spacing: .04em; opacity: .75; }
  .context, .move { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 8px; font-size: .9rem; }
  .context { font-size: .8rem; }
  .weight { font-size: .84rem; }
  .flagged button { justify-self: start; }
  .empty { font-size: .86rem; opacity: .82; }
  .hands { font-size: .84rem; }
  .hands summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .hands ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
  .hands li { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding: 6px 8px; border-radius: 8px; background: #0002; }
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
