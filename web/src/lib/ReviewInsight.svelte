<script>
  import { untrack } from 'svelte';
  import { decisionKey } from './game-review.js';
  import { keptTime, reasonedAbout, searchTime } from './review-insight-policy.js';

  /**
   * Why Trained AI threw its tile and not the player's, and what would have
   * changed its mind, at a reviewed decision where the player threw one tile
   * and Trained AI chose another (reasonedAbout): the last part of Trained
   * AI's view (DecisionReplay), which is drawn only once the decision's
   * answer is shown, so that nothing here names Trained AI's tile while the
   * player's settings keep it back. `insight` is the match's reasoning
   * (MatchInsight), which keeps what it has had for the match; `step` of
   * `hand` the decision on show.
   *
   * The Why line is worked out as the decision is shown, in the page's
   * engine and at most one forward pass, and drawn once it is had, with the
   * button under it, so that the button never moves under a pointer about to
   * press it. It names the engine's fact that leads, with its verdict: Why,
   * Partly, Not explained, or Not checked where its check could not be run.
   * Where a check failed or was not run, the verdict is a button whose title
   * is the check's workings, a tooltip, and which shows them under the line
   * when pressed, for a keyboard, a finger or a screen reader. A line that
   * could not be worked out at all says why, as Not checked.
   *
   * "What would change its mind?" asks Trained AI about the table changed in
   * one thing the player could see, a few seconds' work: it says about how
   * long from the review's own time a position (`positionMs`), as it stood
   * when the decision came on show, and is offered wherever the reasoning
   * searches (MatchInsight.at) and Trained AI can be reached
   * (`trainedAvailable`). Pressed, it gives way to the search's progress and
   * Cancel, and those to what was found: the changes that make Trained AI
   * throw the player's tile, a line for each kind of change, or the pair that
   * does where no single change does, or that nothing asked did. A step to
   * another decision stops a search under way (MatchInsight.leave), and the
   * panel draws only the decision on show's own, so a result never shows on a
   * decision it was not asked for. A search had is kept, and shown again
   * whenever its decision is.
   *
   * Focus goes where the control that had it went: from the button to
   * Cancel, from Cancel back to the button, and once the search ends to the
   * line that leads what it found. A status line, in the page from the
   * start, says politely that the search began and what it found, unless
   * focus went to that line, which says it.
   */
  let { insight = null, hand, step, positionMs = null, trainedAvailable = true } = $props();

  // The reasoning tells the panel when it changes; the panel reads it again.
  let version = $state(0);
  $effect(() => {
    const owner = insight;
    if (!owner) return;
    owner.onChange = () => { version++; };
    return () => { owner.onChange = () => {}; };
  });

  // The decision the panel speaks of, by its key, or null at one the
  // reasoning says nothing of. A key rather than the decision, so that the
  // review's later batches, which hand the panel its hand afresh, ask nothing
  // again. As it changes, whatever was under way at the decision before is
  // stopped, and the Why line of the one now on show is worked out.
  let subject = $derived(reasonedAbout(hand.decisions?.[step] ?? null) ? decisionKey(hand.index, step) : null);
  $effect(() => {
    const at = subject, owner = insight;
    if (!at || !owner) return;
    untrack(() => owner.explain(at));
    return () => untrack(() => owner.leave(at));
  });
  let shown = $derived.by(() => { void version; return subject && insight ? insight.at(subject) : null; });
  // Whether the Why line is had, or failed: until then neither it nor the
  // button is drawn.
  let settled = $derived(['done', 'failed'].includes(shown?.why?.state));
  // The Why line: what the reasoning had, which a line that failed has too
  // where it got as far as its fact, or else the failure, said as not
  // checked.
  let why = $derived(shown?.why?.result ?? (shown?.why?.state === 'failed'
    ? { verdict: 'failed', label: 'Not checked', text: String(shown.why.failure).replace(/\.?$/, '.'), check: '' } : null));
  let search = $derived(shown?.search ?? null);
  let offered = $derived(Boolean(shown?.offered) && trainedAvailable);

  // About how long the search takes, beside the button in a few characters
  // and in its name in words (searchTime), at the review's time a position as
  // it stood when the decision came on show, kept while it is on show
  // (keptTime), so that neither follows the review's later batches.
  let timed = null;
  let speed = $derived.by(() => {
    timed = keptTime(timed, subject, positionMs);
    return timed.ms;
  });
  let estimate = $derived(shown ? searchTime(shown, speed) : null);

  // Whether the check's workings are shown under the Why line, which stays as
  // the player leaves it across steps, as the view's other folded parts do.
  let checkOpen = $state(false);

  let panel = $state(null);
  let said = $state('');
  // A control that had focus and has gone, or can no longer have it.
  const lost = element => !element || element === document.body || !element.isConnected || Boolean(element.disabled);
  // What had focus in the panel before it changed, and the decision and the
  // state of the search it last drew, to tell what a change did.
  let focusWas = null;
  let drawn = null;
  $effect.pre(() => {
    void shown;
    const active = document.activeElement;
    focusWas = panel?.contains(active) ? active : null;
  });
  $effect(() => {
    const at = subject, state = search?.state ?? null, before = drawn, had = focusWas;
    drawn = { at, state };
    focusWas = null;
    // Another decision: nothing it says is about the one before.
    if (before?.at !== at) {
      said = '';
      return;
    }
    if (before.state === state) return;
    const gone = Boolean(had) && lost(document.activeElement);
    const go = selector => { if (gone) panel?.querySelector(selector)?.focus({ preventScroll: true }); };
    if (state === 'searching') {
      said = 'Trying changes…';
      go('[data-insight-cancel]');
    } else if (before.state !== 'searching') {
      return;
    } else if (state === 'done') {
      go('[data-insight-lead]');
      said = gone ? '' : search.said.lead.replace(/:$/, '.');
    } else {
      // Cancelled, or failed: the button is back, with the failure under it.
      go('[data-insight-search]');
      said = state === 'failed' ? `Search stopped: ${search.failure.replace(/\.?$/, '.')}` : '';
    }
  });
</script>

<!-- The Why line and the button under it, then the search's progress or what
     it found, in the order they are read; the status line waits in the page,
     empty, for its words. -->
{#if shown}
  <div class="insight" data-insight data-settled={settled ? '' : undefined} bind:this={panel}>
    {#if why}
      <p class="why" data-why={why.verdict}>
        {#if why.check}<button type="button" class="verdict" data-why-check aria-expanded={checkOpen} title={why.check}
          onclick={() => { checkOpen = !checkOpen; }}>{why.label}</button>{:else}<strong class="verdict">{why.label}</strong>{/if}: {why.text}
      </p>
      {#if why.check && checkOpen}<p class="workings" data-why-workings>{why.check}</p>{/if}
    {/if}
    {#if settled}
      {#if search?.state === 'done'}
        <div class="found" data-insight-result>
          <p class="lead" tabindex="-1" data-insight-lead><strong>{search.said.lead}</strong>{#if search.said.detail}{' '}{search.said.detail}{/if}</p>
          {#if search.said.items.length}<ul>{#each search.said.items as item, index (index)}<li>{item}</li>{/each}</ul>{/if}
          {#if search.said.still}<p>{search.said.still}</p>{/if}
        </div>
      {:else if search?.state === 'searching'}
        <div class="searching" data-insight-searching>
          <progress max={Math.max(1, search.planned)} value={search.asked} aria-label="Changes tried" data-insight-progress></progress>
          <span class="count">Trying changes · {search.asked} of {search.planned}</span>
          <button type="button" class="app-control" data-insight-cancel onclick={() => insight.cancel(subject)}>Cancel</button>
        </div>
      {:else if offered}
        <button type="button" class="app-control ask" data-insight-search
          aria-label={estimate ? `What would change its mind? About ${estimate.said}` : undefined} onclick={() => insight.search(subject)}
          >What would change its mind?{#if estimate}<span class="estimate"> · about {estimate.short}</span>{/if}</button>
        {#if search?.state === 'failed'}<p class="failed" data-insight-failure>Search stopped: {search.failure}</p>{/if}
      {/if}
    {/if}
    <p class="visually-hidden" role="status" data-insight-said>{said}</p>
  </div>
{/if}

<style>
  /* As wide as the view's moves, read across in one go, and never wider than
     the screen: a long tile name or change wraps where it must. */
  .insight { display: grid; gap: 8px; max-width: 560px; font-size: .84rem; overflow-wrap: anywhere; }
  p { margin: 0; }
  .verdict { font-weight: 650; }
  /* A verdict a check failed is a button that reads as the word it is, its
     dotted line saying that there is more to it. */
  button.verdict { padding: 0; border: 0; background: none; color: inherit; font: inherit; font-weight: 650; text-decoration: underline dotted;
    text-underline-offset: 3px; cursor: help; }
  .workings { font-size: .8rem; opacity: .85; }
  .ask { justify-self: start; text-align: left; }
  .estimate { opacity: .8; font-variant-numeric: tabular-nums; }
  .searching { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
  .searching progress { flex: 1 1 120px; max-width: 240px; height: 8px; accent-color: var(--gold); }
  .count { font-variant-numeric: tabular-nums; }
  .found { display: grid; gap: 6px; }
  .found ul { margin: 0; padding-left: 18px; display: grid; gap: 4px; }
  /* The line focus goes to as the search ends, where Cancel was: brought
     into view clear of the step bar that stays at the top of the screen, as
     the stage's buttons are (GameReview). */
  .lead { scroll-margin-top: calc(var(--steps-height, 56px) + 8px); }
  .lead:focus { outline: none; }
  .lead:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  .failed { color: var(--warning-text); }
  @media (max-width: 760px) { .lead { scroll-margin-top: calc(56px + var(--steps-height, 84px) + 8px); } }
  @media (max-height: 500px) { .lead { scroll-margin-top: 72px; } }
</style>
