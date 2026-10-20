<script>
  import { onMount, untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import ReviewPreference from './ReviewPreference.svelte';
  import { tileWords } from './tiles.js';
  import { analyzePolicy } from './policy.js';
  import { reviewWithStrong } from './review-policy.js';
  import { reviewFallbackNote, waitingNote } from './ui.js';
  import { trainedName } from './trained-network.js';

  /**
   * What the hand looked like afterwards. A review that only marks moves
   * right or wrong teaches nothing, so each line shows what the choice
   * traded: how far the hand was left from complete, how many tiles would
   * have improved it, and whether the tile could have dealt in.
   */
  // Trained AI is the default adviser, as readSettings has it. The heading
  // is bound so that the result's button, which goes as it opens the review,
  // can hand focus to it (App). `onask` is told as the review begins to ask
  // Trained AI, and answers what to call once it stops asking: the review of
  // the game so far, which asks the same worker, waits for it meanwhile
  // (App). Without a listener there is nothing to tell. `hideAnswers` is the
  // player's setting that Trained AI's reviews keep its move back until they
  // ask for it (App): these notes are a second way into the hand, so with it
  // on they keep back what the adviser would have done, whichever adviser it
  // is, rather than give away what the review of the game asks the player to
  // find.
  let { notes = [], hints = true, engine = null, trainedAvailable = false, offline = null, adviser = $bindable('strong'),
    heading = $bindable(null), onask = () => () => {}, hideAnswers = false } = $props();
  let trainedNotes = $state.raw(null);
  let reviewing = $state(false);
  let completed = $state(0);
  let failure = $state('');
  let retry = $state(0);
  let cached = null;
  // Trained AI needs the network. While it cannot be had (offline before it
  // was downloaded, or no hand left to ask about), Club reviews in its place
  // and says so; the chosen adviser is kept, and takes over once it can.
  let trainedPossible = $derived(trainedAvailable && Boolean(engine));
  let reviewer = $derived(adviser === 'strong' && trainedPossible ? 'strong' : 'club');
  let activeNotes = $derived(reviewer === 'strong' ? trainedNotes ?? [] : notes);
  // A first Trained review can be what downloads the network.
  let download = $derived(waitingNote('', offline));

  // The review's two status lines come into the page with it, and a screen
  // reader hears a status line only once it knows it. So the lines wait in
  // the review, empty, until it has been drawn and a tenth of a second more,
  // the time a screen reader may take to see what came into the page, before
  // any words arrive; from then on their words change as the review does.
  let linesKnown = $state(false);
  onMount(() => {
    let timer = 0;
    const frame = requestAnimationFrame(() => { timer = setTimeout(() => { linesKnown = true; }, 100); });
    return () => { cancelAnimationFrame(frame); clearTimeout(timer); };
  });
  // Why Club reviews in Trained AI's place, while it does, and how far
  // Trained AI has got, while it reviews.
  let fallback = $derived(linesKnown && adviser === 'strong' && reviewer === 'club' && notes.length ? reviewFallbackNote(engine) : '');
  let progress = $derived(linesKnown && reviewing
    ? `Trained AI is reviewing your decisions… ${completed} of ${notes.length}${download ? ` · ${download}` : ''}` : '');

  let disputed = $derived(activeNotes.filter((note) => !note.agreed));
  let shown = $state('disputed');

  // A whole hand played the way the adviser would have is worth saying.
  let clean = $derived(activeNotes.length > 0 && disputed.length === 0);
  let listed = $derived(shown === 'all' || clean ? activeNotes : disputed);

  // Whether the player has asked to see the adviser's answers, which the
  // setting keeps back until they do: what each decision's adviser would have
  // done instead, with its preference, or Club's reason and numbers. The
  // count of decisions matched, the filter and its counts stay, since they
  // say how far the player went wrong, not what was better. Once asked, the
  // answers stay for as long as the notes are open, through a change of
  // filter or of adviser.
  let answersShown = $state(false);
  let hiding = $derived(hideAnswers && !answersShown);

  $effect(() => {
    const source = notes, owner = engine;
    void retry;
    reviewing = false;
    failure = '';
    if (reviewer !== 'strong' || !source.length) return;
    if (cached?.source === source && cached.engine === owner) { trainedNotes = cached.rows; return; }
    const abort = new AbortController();
    trainedNotes = null;
    completed = 0;
    reviewing = true;
    // Said before the first question goes to the worker, so that the review
    // of the game so far sends nothing more until this review stops asking,
    // and said again, once, as it stops, whichever way comes first: as its
    // answers settle, whether they came, failed or ran out of time, or
    // sooner, as it is closed, as the player leaves Play or changes its
    // adviser, or as it is asked again, which the next run says afresh. Told
    // untracked, so that nothing the listener reads becomes a reason to ask
    // again.
    let asking = untrack(() => onask());
    const stopAsking = () => { asking?.(); asking = null; };
    void reviewWithStrong(owner, source, analyzePolicy, {
      signal: abort.signal, onProgress: count => { if (!abort.signal.aborted) completed = count; },
    }).then(rows => {
      if (abort.signal.aborted) return;
      cached = { source, engine: owner, rows };
      trainedNotes = rows;
    }).catch(error => {
      if (!abort.signal.aborted) failure = error?.message ?? String(error);
    }).finally(() => {
      stopAsking();
      if (!abort.signal.aborted) reviewing = false;
    });
    return () => {
      abort.abort();
      stopAsking();
    };
  });

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }
</script>

<section class="review" aria-label="your decisions this hand" aria-busy={reviewing}>
  <header>
    <h3 tabindex="-1" bind:this={heading}>Your hand, looked at again</h3>
    <label class="adviser">Review adviser
      <select bind:value={adviser} aria-label="Review adviser">
        <option value="club">Club</option>
        <option value="strong" disabled={!trainedPossible}>{trainedName('Trained AI')}</option>
      </select>
    </label>
    {#if reviewer === 'strong'}<p class="policy-help">Percentages describe policy preferences, not win probabilities. Riichi's declaration weight and the discard weight given riichi are separate decisions, not a joint move probability.</p>{/if}
    <!-- This status line, and the one after the header, wait empty and out
         of sight for their words (linesKnown). -->
    <p class="policy-help" class:visually-hidden={!fallback} data-review-fallback role="status">{fallback}</p>
    {#if activeNotes.length && !reviewing && !failure}
      <p class="summary">
        {activeNotes.length - disputed.length} of {activeNotes.length}
        {activeNotes.length === 1 ? 'decision' : 'decisions'} matched {reviewer === 'strong' ? trainedName('Trained AI') : 'Club'}.
      </p>
    {/if}
  </header>

  <p class:visually-hidden={!progress} data-review-progress role="status">{progress}</p>
  {#if !notes.length}
    <p class="empty">You made no decisions this hand.</p>
  {:else if reviewing}<!-- The status line above says how far it has got. -->
  {:else if failure}
    <p role="alert">{failure} <button class="retry" onclick={() => retry++}>Retry Trained review</button>
      <button class="retry" data-review-club onclick={() => adviser = 'club'}>Review with Club</button></p>
  {:else}
    {#if clean}<p class="clean">Every move was the one the adviser would have made.</p>{/if}
    {#if !clean}
    <div class="tabs" role="group" aria-label="which decisions to show">
      <button class:on={shown === 'disputed'} aria-pressed={shown === 'disputed'} onclick={() => (shown = 'disputed')}>
        Where it differs ({disputed.length})
      </button>
      <button class:on={shown === 'all'} aria-pressed={shown === 'all'} onclick={() => (shown = 'all')}>
        Every decision ({activeNotes.length})
      </button>
    </div>
    {/if}

    {#if !clean || reviewer === 'strong'}
    {#if hideAnswers}
      <!-- One button for every decision's answer. It stays in its place as
           it is pressed, its words saying what the next press does, so
           keyboard focus stays on it. -->
      <button class="answers" data-show-answers onclick={() => { answersShown = !answersShown; }}>{answersShown ? 'Hide answers' : 'Show answers'}</button>
    {/if}
    <ol>
      {#each listed as note, index (index)}
        <li class:agreed={note.agreed}>
          {#if note.call_tile}<p class="call-context">Response to <Tile tile={note.call_tile} size="tiny" dora={hints && (note.dora_types ?? []).includes(note.call_tile)} /> from {note.call_from}</p>{/if}
          <div class="moves">
            <span class="turn">{note.turn}</span>
            <span class="played">
              {#if note.played_tile}
                <Tile tile={note.played_tile} size="small" dora={hints && (note.dora_types ?? []).includes(note.played_tile)} />
              {/if}
              <span class="what">{note.played}</span>
            </span>
            {#if !note.agreed && !hiding}
              <span class="instead">instead of</span>
              <span class="advised">
                {#if note.advised_tile}
                  <Tile tile={note.advised_tile} size="small" dora={hints && (note.dora_types ?? []).includes(note.advised_tile)} />
                {/if}
                <span class="what">{note.advised}</span>
              </span>
            {/if}
          </div>

          {#if hiding}<!-- The adviser's answer waits for Show answers. -->
          {:else if reviewer === 'strong'}
            <ReviewPreference {note} />
          {:else if !note.agreed}
            <p class="why">{note.reason}</p>
            {#if note.shanten_played != null && note.shanten_advised != null}
            <table class="numbers">
              <thead>
                <tr>
                  <th scope="col"></th>
                  <th scope="col">{note.played_tile ? tileWords(note.played_tile) : 'played'}</th>
                  <th scope="col">
                    {note.advised_tile ? tileWords(note.advised_tile) : 'advised'}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Hand left</th>
                  <td>{distance(note.shanten_played)}</td>
                  <td>{distance(note.shanten_advised)}</td>
                </tr>
                <tr>
                  <th scope="row">Tiles that improve it</th>
                  <td class:worse={note.acceptance_played < note.acceptance_advised}>
                    {note.acceptance_played}
                  </td>
                  <td>{note.acceptance_advised}</td>
                </tr>
                {#if note.danger_played !== 'quiet' || note.danger_advised !== 'quiet'}
                  <tr>
                    <th scope="row">Against the riichi</th>
                    <td class:worse={note.danger_played === 'live'}>
                      {note.danger_played === 'live' ? 'could deal in' : 'could not deal in'}
                    </td>
                    <td>
                      {note.danger_advised === 'live' ? 'could deal in' : 'could not deal in'}
                    </td>
                  </tr>
                {/if}
                {#if note.dora_played || note.dora_advised}
                  <tr>
                    <th scope="row">Worth as dora</th>
                    <td class:worse={note.dora_played > note.dora_advised}>
                      {note.dora_played === 0
                        ? 'nothing'
                        : `${note.dora_played} han`}
                    </td>
                    <td>
                      {note.dora_advised === 0 ? 'nothing' : `${note.dora_advised} han`}
                    </td>
                  </tr>
                {/if}
              </tbody>
            </table>
            {/if}
          {:else}
            <p class="why quiet">{note.reason}</p>
          {/if}
        </li>
      {/each}
    </ol>
    {/if}
  {/if}
</section>

<style>
  .review {
    display: grid;
    gap: 12px;
    padding: 12px;
    min-width: 0;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.28);
    border: 1px solid rgba(255, 255, 255, 0.12);
  }

  h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  /* The heading takes focus as the review opens, and a press by keyboard, or
     any press where the result is not a sheet, scrolls to it with as little
     movement as will show it and the focus ring drawn just outside it for a
     keyboard (App). So it lands a ring's width and a little from
     the edge of the screen, below the game's bar where that stays at the top
     of the screen (AppSettings), and on a phone clear of the result's chip at
     the foot of the screen (ScoreScreen). */
  h3:focus { outline: none; }
  h3:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  h3 { scroll-margin: 8px 0; }
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h3 { scroll-margin-top: 72px; }
  }
  @media (max-width: 760px) {
    h3 { scroll-margin-bottom: calc(72px + env(safe-area-inset-bottom, 0px)); }
  }

  .adviser { display: flex; align-items: center; flex-wrap: wrap; gap: 8px 14px; margin-top: 10px; font-size: .85rem; }
  select, .retry { min-height: 44px; padding: 8px 12px; background: #0004; color: inherit; border: 1px solid #ffffff55; border-radius: 8px; font: inherit; }
  option { background: #17241f; color: var(--ivory); }
  .retry { cursor: pointer; }
  .policy-help { font-size: .8rem; opacity: .75; margin: 8px 0 0; }
  .call-context { display: flex; align-items: center; gap: 6px; margin: 0; font-size: .8rem; }

  .summary,
  .empty,
  .clean {
    margin: 4px 0 0;
    font-size: 0.88rem;
    opacity: 0.82;
  }

  .clean {
    color: #7fd1a0;
    opacity: 1;
  }

  .tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tabs button {
    padding: 8px 12px;
    min-height: 44px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
  }

  .tabs button.on {
    border-color: var(--gold);
    color: var(--gold);
  }

  /* Sized as the filter's buttons are, a finger's width tall, but
     square-cornered and never gold, since it is no filter: it shows or hides
     the answers in whichever list the filter keeps. */
  .answers {
    justify-self: start;
    padding: 8px 12px;
    min-height: 44px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
  }

  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 10px;
    max-height: 42vh;
    overflow-y: auto;
  }

  li {
    display: grid;
    min-width: 0;
    gap: 6px;
    padding: 8px 10px;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.24);
    border-left: 3px solid var(--accent);
  }

  li.agreed {
    border-left-color: rgba(127, 209, 160, 0.7);
  }

  .moves {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 0.9rem;
  }

  .turn {
    font-variant-numeric: tabular-nums;
    opacity: 0.5;
    min-width: 1.4em;
    text-align: right;
  }

  .played,
  .advised {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .instead {
    opacity: 0.6;
    font-size: 0.82rem;
  }

  .advised .what {
    color: var(--gold);
  }

  .why {
    margin: 0;
    font-size: 0.84rem;
    opacity: 0.85;
  }

  .why.quiet {
    opacity: 0.6;
  }

  .numbers {
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
    overflow-wrap: anywhere;
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
  }

  .numbers th {
    text-align: left;
    font-weight: 500;
    opacity: 0.7;
    padding: 1px 14px 1px 0;
  }

  .numbers thead th {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.5;
  }

  .numbers td {
    padding: 1px 14px 1px 0;
    text-align: right;
  }

  .worse {
    color: var(--warning-text);
  }
</style>
