<script>
  import Tile from './Tile.svelte';
  import DecisionFilter from './DecisionFilter.svelte';
  import ReviewStepBar from './ReviewStepBar.svelte';
  import MatchTable from './app/MatchTable.svelte';
  import PlayerHand from './app/PlayerHand.svelte';
  import { groupPolicyChoices } from './action-weights.js';
  import { decisionInBrief, differed, filterCounts, moveToken, sameMove, seatNotes } from './game-review.js';
  import { TAG_SAID, percent } from './ui.js';
  import { SEAT_NAMES as NAMES } from './tiles.js';

  /**
   * A reviewed hand, replayed a decision of the player's at a time from the
   * deal, with what the others did in between, and the whole table as it
   * stood at each step: every player's tiles face up, the tile just drawn
   * apart, the sets, the discards and the board. The replay's own engine
   * read the tiles once the hand was over (scanMatch), never the match.
   *
   * Every step shows the move made there and, where Trained AI was asked,
   * its own: in the step bar at the top of the screen, over its tile in the
   * hand, which is ringed in blue as Agent watch and Guided play ring the
   * network's choice, and in its view of every move open there. `start` is
   * the step opened at, and `whole` says the hand was opened from its deal
   * rather than at a decision. The saved match is never touched, and nothing
   * is played on from any step.
   */
  let { hand, start = 0, whole = false, hints = true, onback } = $props();

  let step = $derived(start);
  let everyMove = $state(false);
  let listFilter = $state('every');
  let heading = $state(null);
  // The step bar's height, which a control below it is kept clear of.
  let barHeight = $state(0);
  // The list of the hand's decisions starts open where the screen has room
  // for it above the table; after that it is the player's to open and fold.
  // So are the list of what happened since the previous decision and the
  // table of what each move leaves, which start folded, read less than the
  // table and the weights, and stay as the player leaves them across a step
  // that takes them from the page. Each is held here rather than set on the
  // list from the step: the page sets that afresh at every step, and as
  // later hands are reviewed, which folded a list under the player's press
  // and dropped keyboard focus on a step's button to the page.
  const roomy = !(globalThis.matchMedia?.('(max-width: 760px), (max-height: 500px) and (orientation: landscape)').matches ?? true);
  let listOpen = $state(roomy);
  let linesOpen = $state(false);
  let numbersOpen = $state(false);

  let frame = $derived(hand.frames[step]);
  let current = $derived(hand.decisions?.[step] ?? null);
  let notes = $derived(seatNotes(frame.view));
  let me = $derived(frame.view.seats[0]);
  let shownDora = $derived(hints ? frame.view.dora_types ?? [] : []);
  let brief = $derived(decisionInBrief(frame, current));
  // Previous and Next step through every decision of the hand.
  let previous = $derived(step > 0 ? { hand: hand.index, step: step - 1 } : null);
  let next = $derived(step < hand.frames.length - 1 ? { hand: hand.index, step: step + 1 } : null);

  // The decision whose view of every move this step shows: any that was put
  // to Trained AI, which a move with no alternative never was.
  let shown = $derived(current?.weighted && !current.forced ? current : null);
  let advised = $derived(shown ? { kind: shown.advised_kind, tile: shown.advised_tile ?? null } : null);
  let played = $derived(shown ? { kind: shown.played_kind, tile: shown.played_tile ?? null } : null);
  let stage = $derived(shown?.judgement?.stage);

  // The tiles in the hand that the step's moves throw, by their place in it:
  // the player's, and Trained AI's where it was asked. A move that throws no
  // tile from the hand, a call, a win or a pass, has none. Where the tile
  // just drawn is one of them, it is that copy, as play throws the draw.
  let tiles = $derived([...me.hand, ...(me.drawn ? [me.drawn] : [])]);
  function placeOf(move) {
    if (!move || !['discard', 'riichi'].includes(move.kind)) return null;
    const index = move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile);
    return index >= 0 ? index : null;
  }
  let yours = $derived(placeOf(frame.played));
  let trained = $derived(placeOf(advised));
  // The words over those tiles, in a line kept on every tile (HandTile):
  // "You" over the player's, "AI" over Trained AI's, both over one tile
  // where they agreed. Trained AI's tile is ringed as well (`selected`).
  let tags = $derived(tiles.map((_, index) => [index === yours && 'You', index === trained && 'AI'].filter(Boolean)));

  // The moves by weight, riichi's discards together under the declaration.
  let ordered = $derived(shown ? [...shown.weighted]
    .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) : []);
  // The leading moves and every move the player or the network picked; the
  // rest wait behind a button, which says what a press does: how many more it
  // shows, or that it shows fewer. The button stays once they are shown, to
  // hide them again, so the focus on it stays too rather than dropping to the
  // page.
  let rows = $derived(groupPolicyChoices(everyMove ? ordered : ordered.filter((choice, index) => index < 5 || marked(choice))));
  let extra = $derived(ordered.filter((choice, index) => index >= 5 && !marked(choice)).length);

  // The moves the numbers compare, each once and only where the review
  // measured it: the player's and Trained AI's.
  let compared = $derived(!shown ? [] : [
    { move: played, title: 'You' },
    ...(!sameMove(advised, played.kind, played.tile) ? [{ move: advised, title: 'Trained AI' }] : []),
  ].map(entry => ({ ...entry, measure: shown.measures?.find(measure => sameMove(measure, entry.move.kind, entry.move.tile)) ?? null }))
    .filter(entry => entry.measure));

  // Every decision of the hand, each a step of its own, for the list, which
  // shows those its filter keeps in the order played.
  let reached = $derived(hand.frames.map((entry, index) => ({ index, frame: entry, decision: hand.decisions?.[index] ?? null })));
  let listCounts = $derived(filterCounts(reached.flatMap(entry => (entry.decision ? [entry.decision] : []))));
  let listed = $derived(reached.filter(({ decision }) => listFilter === 'every'
    || (decision && (listFilter === 'flagged' ? Boolean(decision.judgement?.flagged) : differed(decision)))));

  function marked(choice) {
    return [advised, played].some(move => move && sameMove(choice, move.kind, move.tile));
  }

  // The words a move of the view is tagged with, as the hand tags its tiles.
  const tagsOf = choice => [sameMove(choice, played.kind, played.tile) && 'You', sameMove(choice, advised.kind, advised.tile) && 'AI']
    .filter(Boolean);

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }

  // A step's own button, in the list, keeps focus as it shows its step, and
  // the step bar hands its own on where a step disables the one pressed.
  function goTo(target) {
    if (target !== step) everyMove = false;
    step = target;
  }

  $effect(() => { heading?.focus({ preventScroll: true }); heading?.scrollIntoView({ block: 'start' }); });
</script>

<section class="decision" aria-labelledby="decision-title" data-decision style:--steps-height={barHeight ? `${barHeight}px` : undefined}>
  <header>
    <h3 id="decision-title" tabindex="-1" bind:this={heading}>{whole ? `${hand.name}, whole hand` : `${hand.name}, decision ${start + 1}`}</h3>
    <button class="app-control" data-back onclick={onback}>Back to the list</button>
  </header>

  <ReviewStepBar {hand} {step} decision={current} {previous} {next} onstep={place => goTo(place.step)} bind:height={barHeight} />

  <!-- Every decision of the hand, each a button to its step, which keeps
       focus as the step it shows changes. -->
  <details class="decisions" bind:open={listOpen} data-decisions>
    <summary>Decisions in this hand ({reached.length})</summary>
    <div class="decision-list">
      <DecisionFilter bind:value={listFilter} counts={listCounts} label="Which of this hand's decisions to list"
        said={{ scope: ' in this hand', ranked: false }} />
      {#if listed.length}
        <!-- Drawn without markers, so said to be a list in so many words:
             Safari does not tell VoiceOver that a list without them is one.
             Each line shows the player's move and Trained AI's in tiles and
             words; its name says them in the engine's words. -->
        <ol role="list">
          {#each listed as entry (entry.index)}
            {@const line = decisionInBrief(entry.frame, entry.decision)}
            <li>
              <button class="app-control" class:flag={line.flagged} data-step-to={entry.index}
                aria-current={entry.index === step ? 'step' : undefined} aria-label={`Decision ${entry.index + 1}. ${line.said}`}
                onclick={() => goTo(entry.index)}>
                <span class="number">{entry.index + 1}</span>
                {#if line.offered}<span class="token">on <Tile tile={line.offered} size="tiny" dora={shownDora.includes(line.offered)} /></span>{/if}
                {@render token(line.you)}
                {#if !line.reviewed}<span class="verdict">…</span>
                {:else if line.forced}<span class="verdict">Only move</span>
                {:else if line.agreed}<span class="verdict">✓</span>
                {:else}<span class="arrow">→</span>{@render token(line.ai)}{/if}
                {#if line.flagged && listFilter !== 'flagged'}<strong class="flag-mark">Flagged</strong>{/if}
              </button>
            </li>
          {/each}
        </ol>
      {:else}
        <p class="empty">{listFilter === 'flagged' ? 'None flagged.' : 'Trained AI agreed at every decision.'}</p>
      {/if}
    </div>
  </details>

  <!-- The table as it stood at this step, all of it, every player's tiles
       face up: the board and the other players around it, what happened
       since the player's previous decision, then the player's own seat. The
       step bar says it as the step changes. -->
  <div class="overview" data-overview>
    <MatchTable view={frame.view} {hints} thinking={false} pendingOpponent={null} hands={frame.hands ?? null} {notes} review />

    {#if frame.lines.length}
      <details class="lines" bind:open={linesOpen}>
        <!-- How many events, in brackets as the review's other summaries
             give a count; the items below are the events. -->
        <summary>{step === 0 ? 'Since the deal' : 'Since the previous decision'} ({frame.lines.length})</summary>
        <ol>{#each frame.lines as line, index (index)}<li>{line}</li>{/each}</ol>
      </details>
    {/if}

    <PlayerHand view={frame.view} engine={null} closed={true} {hints} busy={false} blocked={false}
      discardChoices={[]} confirmDiscards={false} picked={null} selected={trained}
      canDiscard={() => false} selectTile={() => {}} syncHandFocus={() => {}}
      handLabel={`${NAMES[me.seat]}'s tiles, yours`} listed note={notes[0]} {tags} />
    <!-- What the hand is described by: the two moves, in words. -->
    <p id="hand-help" class="visually-hidden">{brief.said}</p>
  </div>

  {#if shown}
    <!-- Trained AI's view of the move made at the step, which the review
         holds for every decision it was asked about: each move open there by
         its weight, tagged as the hand tags its tiles. -->
    <section class="view" aria-labelledby="view-title" data-view>
      <h4 id="view-title">Trained AI's view</h4>
      {#if stage === 'tile'}<p class="note">Both declared riichi: tiles compared given riichi.</p>{/if}

      <div class="weights" role="list" aria-label="Trained AI's weight for each move">
        {#each rows as row, index (index)}
          {#if row.choice}
            {@render weightRow(row.choice, row.choice.weight)}
          {:else}
            <div class="reach" role="listitem">
              <div class="row reach-heading"><span class="move">Declare riichi</span><meter min="0" max="1" value={row.weight} aria-hidden="true"></meter><strong>{percent(row.weight)}</strong><span class="tag"></span></div>
              <div role="list" aria-label="Discards given riichi">
                {#each row.choices as choice, slot (slot)}{@render weightRow(choice, choice.conditionalWeight, true)}{/each}
              </div>
            </div>
          {/if}
        {/each}
      </div>
      {#if extra}<button class="app-control quiet" data-every-move onclick={() => { everyMove = !everyMove; }}>{everyMove ? 'Show fewer moves' : `Show ${extra} more move${extra === 1 ? '' : 's'}`}</button>{/if}

      {#if compared.length > 1}
        <!-- Counted as the hand review counts it (docs/UI.md); a player reads
             the table without being told how. Folded, as it is read less than
             the weights, and named for a screen reader's list of tables. -->
        <details class="numbers" bind:open={numbersOpen}>
          <summary>What each move leaves</summary>
          <table>
            <caption class="visually-hidden">What each move leaves</caption>
            <thead><tr><th scope="col"></th>{#each compared as entry (entry.title)}<th scope="col">{entry.title}{#if entry.move.tile}<br /><Tile tile={entry.move.tile} size="tiny" />{/if}</th>{/each}</tr></thead>
            <tbody>
              <tr><th scope="row">Hand left</th>{#each compared as entry (entry.title)}<td>{distance(entry.measure.shanten)}</td>{/each}</tr>
              <tr><th scope="row">Tiles that improve it</th>{#each compared as entry (entry.title)}<td>{entry.measure.acceptance}</td>{/each}</tr>
              {#if compared.some(entry => entry.measure.danger !== 'quiet')}
                <!-- A move that discards nothing, a win or a quad, is not read against the riichi. -->
                <tr><th scope="row">Against the riichi</th>{#each compared as entry (entry.title)}<td class:worse={entry.measure.danger === 'live'}>{{ live: 'could deal in', safe: 'could not deal in' }[entry.measure.danger] ?? '—'}</td>{/each}</tr>
              {/if}
              {#if compared.some(entry => entry.measure.dora)}
                <tr><th scope="row">Worth as dora</th>{#each compared as entry (entry.title)}<td>{entry.measure.dora ? `${entry.measure.dora} han` : 'nothing'}</td>{/each}</tr>
              {/if}
            </tbody>
          </table>
        </details>
      {/if}
    </section>
  {/if}
</section>

{#snippet token(move)}
  <span class="token">{#if move.word}<span>{move.word}</span>{/if}{#each move.tiles as tile, index (index)}<Tile {tile} size="tiny" dora={shownDora.includes(tile)} />{/each}</span>
{/snippet}

<!-- A move of the view: its tile, or a word and its tiles, then its weight
     and the words it is tagged with. A riichi given riichi is its tile, under
     the declaration's own row. A screen reader hears the move in the
     engine's words in place of the pictures, and each tag in full; the bar
     says no more than the percentage beside it, so it is not read. -->
{#snippet weightRow(choice, weight, conditional = false)}
  {@const words = tagsOf(choice)}
  <div class="row" role="listitem" class:best={sameMove(choice, advised.kind, advised.tile)} data-move={choice.label}>
    <span class="move"><span aria-hidden="true">{@render token(conditional ? { word: '', tiles: [choice.tile] } : moveToken(choice.kind, choice.tile))}</span><span
      class="visually-hidden">{choice.label}{conditional ? ', given riichi' : ''}</span></span>
    {#if weight == null}<strong>{conditional ? 'Not evaluated' : 'Unscored'}</strong>
    {:else}<meter min="0" max="1" value={weight} aria-hidden="true"></meter><strong>{percent(weight)}</strong>{/if}
    <span class="tag" aria-hidden="true">{#each words as word, index (word)}{#if index}{' · '}{/if}<b class:ai={word === 'AI'}>{word}</b>{/each}</span>
    {#if words.length}<span class="visually-hidden">, {words.map(word => TAG_SAID[word]).join(', ')}</span>{/if}
  </div>
{/snippet}

<style>
  .decision { display: grid; gap: 10px; min-width: 0; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 12px; }
  h3 { margin: 0; font-size: 1rem; }
  h4 { margin: 0; font-size: .95rem; }
  h3:focus { outline: none; }
  h3:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  .decisions, .lines { font-size: .82rem; }
  .decisions summary, .lines summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .decision-list { display: grid; gap: 8px; }
  /* As many to a row as fit, two on the narrowest phone, so a hand's dozen
     or more decisions take a few rows rather than a column. */
  .decisions ol { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(136px, 1fr)); gap: 6px; }
  .decisions li { display: flex; min-width: 0; }
  .decisions button { width: 100%; display: flex; flex-wrap: wrap; align-items: center; gap: 2px 6px; padding: 5px 8px; text-align: left; font-size: .8rem; }
  .decisions .number { min-width: 1.3em; font-weight: 650; font-variant-numeric: tabular-nums; }
  .token { display: inline-flex; align-items: center; gap: 3px; }
  .decisions .verdict { opacity: .85; }
  .arrow { opacity: .7; }
  /* A flagged decision is marked by an edge as the whole-game list marks it,
     and outside the flagged ones says so in words; the step shown is gold. */
  .decisions .flag { border-left: 3px solid var(--accent); }
  .flag-mark { font-weight: 650; color: var(--gold); }
  .decisions [aria-current='step'] { border-color: var(--gold); background: rgba(216, 161, 42, .2); }
  .empty { margin: 0; opacity: .82; }
  .overview { display: grid; gap: 10px; min-width: 0; }
  .lines ol { margin: 0; padding: 6px 12px 6px 28px; display: grid; gap: 2px; background: #0003; border-radius: 8px; }
  .quiet { justify-self: start; background: transparent; }
  .view { display: grid; gap: 10px; padding: 12px; border: 1px solid #d8a12a66; border-radius: 12px; background: #0003; min-width: 0; }
  .note { margin: 0; font-size: .78rem; opacity: .8; }
  /* A move a row, no wider than it needs to be read across: the move, its
     bar, its percentage and its tags, Trained AI's choice edged in the blue
     its tile is ringed with in the hand. */
  .weights { display: grid; gap: 4px; max-width: 560px; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; padding: 4px 6px; border: 2px solid transparent; border-top-color: #ffffff15; border-radius: 8px; font-size: .82rem; }
  .row.best { border-color: #4ea3ff; }
  .move { display: flex; align-items: center; flex: 1 1 6em; min-width: 0; }
  .row strong { min-width: 52px; text-align: right; font-variant-numeric: tabular-nums; }
  meter { flex-shrink: 0; width: 90px; accent-color: var(--gold); }
  .tag { min-width: 4.5em; font-size: .72rem; font-weight: 700; white-space: nowrap; }
  .tag .ai { color: var(--gold); }
  .reach { border: 1px solid #ffffff28; border-radius: 8px; padding: 6px; display: grid; gap: 4px; }
  .reach-heading { border-top-color: transparent; }
  .numbers { font-size: .82rem; }
  .numbers summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .numbers table { border-collapse: collapse; width: 100%; max-width: 560px; table-layout: fixed; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
  .numbers th { text-align: left; font-weight: 500; opacity: .75; padding: 2px 10px 2px 0; vertical-align: bottom; }
  .numbers td { padding: 2px 10px 2px 0; }
  .worse { color: var(--warning-text); }
  @media (max-width: 400px) { meter { width: 56px; } }
  /* A control focused below the step bar, which stays at the top of the
     screen, is brought into view below it rather than under it: the bar's
     own height, and on a phone held upright the game's bar above it too.
     Where the screen is 500 pixels high or less the step bar stays in its
     place, and only the game's bar, which stays there in landscape, is kept
     clear of, as the headings keep clear of it. */
  .decision :global(:is(button, summary)) { scroll-margin-top: calc(var(--steps-height, 56px) + 8px); }
  @media (max-width: 760px) {
    .decision :global(:is(button, summary)) { scroll-margin-top: calc(56px + var(--steps-height, 84px) + 8px); }
  }
  @media (max-height: 500px) {
    .decision :global(:is(button, summary)) { scroll-margin-top: 72px; }
  }
  /* Below the game's bar where it stays at the top of the screen (AppSettings). */
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h3 { scroll-margin-top: 72px; }
  }
</style>
