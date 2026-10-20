<script>
  import { tick, untrack } from 'svelte';
  import Tile from './Tile.svelte';
  import DecisionFilter from './DecisionFilter.svelte';
  import MatchTable from './app/MatchTable.svelte';
  import PlayerHand from './app/PlayerHand.svelte';
  import { groupPolicyChoices } from './action-weights.js';
  import { decisionInBrief, differed, filterCounts, sameMove, seatNotes, tableSaid, weightsSaid } from './game-review.js';
  import { callLabel, percent, quadOnOffer } from './ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from './tiles.js';

  /**
   * A reviewed hand, replayed a decision of the player's at a time from the
   * deal, with what the others did in between, and the whole table as it
   * stood at each step: every player's tiles face up, the tile just drawn
   * apart, the sets, the discards and the board. The replay's own engine
   * read the tiles once the hand was over (scanMatch), never the match.
   *
   * A flagged decision is one to choose again before Trained AI's answer is
   * shown. Opened at one (`retry`), the replay asks for the choice there, and
   * goes no further than that decision until it is answered; any other
   * flagged decision it reaches offers the same, or the answer outright.
   * Every other decision shows the move made and Trained AI's view of it.
   * `start` is the step opened at, and `whole` says the hand was opened from
   * its deal rather than at a decision. The saved match is never touched,
   * and nothing is played on from any step.
   */
  let { hand, start = 0, retry = false, whole = false, hints = true, onback } = $props();

  let step = $derived(start);
  const flaggedAt = index => Boolean(hand.decisions?.[index]?.judgement?.flagged);
  // Opened to choose again at a flagged decision.
  let retrying = $derived(retry && flaggedAt(start));
  // How each flagged decision stands, by step: 'offered' until the player
  // asks to choose again there or to see the answer, 'choosing' while they
  // choose, 'answered' once Trained AI's answer is shown, with the move
  // chosen again if they chose one. The decision opened to choose again
  // starts out being chosen.
  let asked = $state({});
  const standing = index => asked[index] ?? { mode: retrying && index === start ? 'choosing' : 'offered', retried: null };
  let everyMove = $state(false);
  let listFilter = $state('every');
  let heading = $state(null);
  let answerHeading = $state(null);
  let handElement = $state(null);
  let nextButton = $state(null);
  let previousButton = $state(null);
  // The list of the hand's decisions starts open where the screen has room
  // for it above the table, unless the replay was opened to choose again;
  // after that it is the player's to open and fold. So is the list of what
  // happened since the previous decision, which starts open, and stays as
  // the player leaves it across a step where nothing happened, which takes
  // it from the page. Each is held here rather than set on the list from
  // the step: the page sets that afresh at every step, and as later hands
  // are reviewed, which folded a list under the player's press and dropped
  // keyboard focus on a step's button to the page.
  const roomy = !(globalThis.matchMedia?.('(max-width: 760px), (max-height: 500px) and (orientation: landscape)').matches ?? true);
  let listOpen = $state(untrack(() => !retrying && roomy));
  let linesOpen = $state(true);

  let frame = $derived(hand.frames[step]);
  let current = $derived(hand.decisions?.[step] ?? null);
  // 'shown' for a decision that was not flagged; otherwise how it stands.
  let mode = $derived(flaggedAt(step) ? standing(step).mode : 'shown');
  let retried = $derived(mode === 'shown' ? null : standing(step).retried);
  let choosing = $derived(mode === 'choosing');
  // The player is choosing again here, or has, so the move made in the game
  // waits for Trained AI's answer, which says it.
  let engaged = $derived(mode === 'choosing' || mode === 'answered');
  // The decision opened to choose again, while it is still to be answered,
  // is as far as the replay goes; then the rest of the hand follows.
  let pending = $derived(retrying && standing(start).mode !== 'answered');
  let last = $derived(pending ? start : hand.frames.length - 1);
  let theOne = $derived(retrying && step === start);
  // Said with the table as the step changes: a flagged decision says so too,
  // but for the one opened to choose again, whose step already says it.
  let tableLine = $derived(`${tableSaid(frame.view)}${flaggedAt(step) && !theOne ? ' Trained AI flagged your move here.' : ''}`);
  let notes = $derived(seatNotes(frame.view));
  let me = $derived(frame.view.seats[0]);
  let discardChoices = $derived(choosing ? frame.choices.filter(choice => choice.kind === 'discard') : []);
  let otherChoices = $derived(frame.choices.filter(choice => choice.kind !== 'discard'));
  // Where the choice is made. The hand takes focus at the decision and this
  // prompt describes it, but an answer to another's discard leaves no tile to
  // throw, so the prompt then sends the player to the moves below it.
  let choiceHelp = $derived(discardChoices.length ? `: select a tile to discard${otherChoices.length ? ', or a move below' : ''}`
    : otherChoices.length ? ': choose a move below' : '');
  let robbing = $derived(Boolean(quadOnOffer(frame.view)));
  let offered = $derived(frame.view.phase === 'call' ? frame.view.pending_discard : null);
  let discarder = $derived(NAMES[frame.view.pending_from] ?? 'An opponent');
  // What the player did at a step, and whether Trained AI agreed: shown under
  // the hand, and said with the step, so stepping through the hand by ear
  // tells the moves as well as where it is. The verdict is the one the hand's
  // list gives the step (decisionInBrief), since it is said at every step. A
  // move with no alternative was never put to Trained AI.
  let stepMove = $derived(engaged ? ''
    : `Your move: ${frame.choices.find(choice => sameMove(choice, frame.played.kind, frame.played.tile))?.label ?? frame.played.kind}.`);
  let stepVerdict = $derived(engaged || !current ? '' : current.forced ? 'Only move.'
    : current.agreed ? 'Trained AI agreed.' : 'Trained AI differed.');
  let shownDora = $derived(hints ? frame.view.dora_types ?? [] : []);
  // The tile thrown at a step, or chosen again there, is marked in the hand
  // as a selected tile is in play.
  let thrown = $derived.by(() => {
    const move = engaged ? retried : frame.played;
    if (!move || !['discard', 'riichi'].includes(move.kind) || !me) return null;
    const tiles = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
    const index = move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile);
    return index >= 0 ? index : null;
  });

  // The decision whose answer, or Trained AI's view, this step shows: a
  // flagged one once answered, any other that was put to Trained AI.
  let shown = $derived(mode === 'answered' || (mode === 'shown' && current?.weighted && !current.forced) ? current : null);
  let tried = $derived(mode === 'answered' ? retried : null);
  let advised = $derived(shown ? { kind: shown.advised_kind, tile: shown.advised_tile ?? null } : null);
  let played = $derived(shown ? { kind: shown.played_kind, tile: shown.played_tile ?? null } : null);
  let stage = $derived(shown?.judgement?.stage);
  const labelOf = move => shown?.weighted.find(choice => sameMove(choice, move.kind, move.tile))?.label ?? '';

  // The moves by weight, riichi's discards together under the declaration.
  let ordered = $derived(shown ? [...shown.weighted]
    .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) : []);
  // The leading moves and every move the player or the network picked; the
  // rest wait behind a button. The button stays once they are shown, to hide
  // them again, so the focus on it stays too rather than dropping to the page.
  let rows = $derived(groupPolicyChoices(everyMove ? ordered : ordered.filter((choice, index) => index < 5 || marked(choice))));
  let extra = $derived(ordered.filter((choice, index) => index >= 5 && !marked(choice)).length);

  // The moves the numbers compare, each once: the game's, the new one, the network's.
  let compared = $derived(!shown ? [] : [
    { move: played, title: 'In the game' },
    ...(tried && !sameMove(tried, played.kind, played.tile) ? [{ move: tried, title: 'This time' }] : []),
    ...(!sameMove(advised, played.kind, played.tile) && !(tried && sameMove(advised, tried.kind, tried.tile))
      ? [{ move: advised, title: 'Trained AI' }] : []),
  ].map(entry => ({ ...entry, measure: shown.measures?.find(measure => sameMove(measure, entry.move.kind, entry.move.tile)) ?? null })));
  let measured = $derived(compared.every(entry => entry.measure));

  // The hand's decisions as far as the replay goes, each a step of its own,
  // for the list, which shows those its filter keeps in the order played.
  let reached = $derived(hand.frames.slice(0, last + 1)
    .map((entry, index) => ({ index, frame: entry, decision: hand.decisions?.[index] ?? null })));
  let listCounts = $derived(filterCounts(reached.flatMap(entry => (entry.decision ? [entry.decision] : []))));
  let listed = $derived(reached.filter(({ decision }) => listFilter === 'every'
    || (decision && (listFilter === 'flagged' ? Boolean(decision.judgement?.flagged) : differed(decision)))));

  function marked(choice) {
    return [advised, played, ...(tried ? [tried] : [])].some(move => move && sameMove(choice, move.kind, move.tile));
  }

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }

  // Where focus goes when a step disables the button pressed, which would
  // drop it to the page: to the hand where a choice is to be made there, as
  // play gives the hand focus when a turn comes; otherwise on to Next, or at
  // the hand's last decision back to Previous.
  function landing() {
    if (choosing) return handElement;
    if (step < last) return nextButton;
    return step > 0 ? previousButton : heading;
  }

  // A step to either end of the replay disables the button pressed. Every
  // other button stays, and focus with it, among them a step's own button in
  // the list of the hand's decisions.
  async function goTo(next) {
    const target = Math.max(0, Math.min(last, next));
    if (target !== step) everyMove = false;
    step = target;
    await tick();
    const active = document.activeElement;
    if (active && active !== document.body && active.isConnected && !active.matches(':disabled')) return;
    const land = landing();
    land?.focus(land === handElement ? { preventScroll: true } : undefined);
  }

  async function answer(retriedMove) {
    asked[step] = { mode: 'answered', retried: retriedMove };
    await tick();
    answerHeading?.focus({ preventScroll: true });
    answerHeading?.scrollIntoView({ block: 'nearest' });
  }

  function choose(choice) {
    if (choosing) void answer({ kind: choice.kind, tile: choice.tile ?? null });
  }

  // The answer outright: before choosing again, or at a flagged decision
  // the player would rather not choose again.
  function reveal() { void answer(null); }

  // To be chosen again here: the answer goes, if it was shown, with the button
  // pressed, and the hand takes focus, as it does in play when a turn comes,
  // and says what to do through its description, the prompt; it is brought
  // into view to be chosen from.
  async function again() {
    asked[step] = { mode: 'choosing', retried: null };
    everyMove = false;
    await tick();
    handElement?.focus({ preventScroll: true });
    handElement?.scrollIntoView({ block: 'nearest' });
  }

  // A riichi has two weights, as ReviewPreference shows them: the
  // declaration's, and the tile's once the declaration is made, which is
  // asked only where the network declared.
  function weightOf(move) {
    const choice = shown?.weighted.find(entry => sameMove(entry, move.kind, move.tile));
    if (!choice || choice.weight == null) return 'unscored';
    if (choice.kind !== 'riichi') return percent(choice.weight);
    return `${percent(choice.weight)} to declare riichi · discard given riichi: ${choice.conditionalWeight == null ? 'not evaluated' : percent(choice.conditionalWeight)}`;
  }

  $effect(() => { heading?.focus({ preventScroll: true }); heading?.scrollIntoView({ block: 'start' }); });
</script>

<section class="decision" aria-labelledby="decision-title" data-decision>
  <header>
    <h3 id="decision-title" tabindex="-1" bind:this={heading}>{whole ? `${hand.name}, whole hand` : `${hand.name}, decision ${start + 1}`}</h3>
    <button class="app-control" data-back onclick={onback}>Back to the list</button>
  </header>

  <div class="steps" role="group" aria-label="Replay of the hand">
    <button class="app-control" onclick={() => goTo(0)} disabled={step === 0}>From the deal</button>
    <button class="app-control" onclick={() => goTo(step - 1)} disabled={step === 0} aria-label="Previous decision" bind:this={previousButton}>Previous</button>
    <button class="app-control" onclick={() => goTo(step + 1)} disabled={step >= last} aria-label="Next decision" bind:this={nextButton}>Next</button>
    {#if !whole}<button class="app-control" onclick={() => goTo(start)} disabled={step === start}>This decision</button>{/if}
    <p class="where" role="status"><span data-step>Decision {step + 1} of {last + 1}{theOne ? ', flagged' : ''}</span>{#if stepMove}<span class="visually-hidden">. {stepMove} {stepVerdict}</span>{/if}</p>
  </div>

  <!-- Every decision of the hand the replay reaches, each a button to its
       step, which keeps focus as the step it shows changes. -->
  <details class="decisions" bind:open={listOpen} data-decisions>
    <summary>{pending ? 'Decisions up to the flagged one' : 'Decisions in this hand'} ({reached.length})</summary>
    <div class="decision-list">
      <DecisionFilter bind:value={listFilter} counts={listCounts} label="Which of this hand's decisions to list"
        said={{ scope: ' in this hand', ranked: false }} />
      {#if listed.length}
        <!-- Drawn without markers, so said to be a list in so many words:
             Safari does not tell VoiceOver that a list without them is one. -->
        <ol role="list">
          {#each listed as entry (entry.index)}
            {@const brief = decisionInBrief(entry.frame, entry.decision)}
            <li>
              <button class="app-control" class:flag={entry.decision?.judgement?.flagged} data-step-to={entry.index}
                aria-current={entry.index === step ? 'step' : undefined} onclick={() => goTo(entry.index)}>
                <span class="number"><span class="visually-hidden">{'Decision '}</span>{entry.index + 1}</span>
                <span class="move">{brief.move}{#if brief.tile} <Tile tile={brief.tile} size="tiny" dora={shownDora.includes(brief.tile)} />{/if}</span>
                <span class="verdict">{brief.verdict}</span>
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
       since the player's previous decision, then the player's own seat. -->
  <section class="overview" aria-labelledby="overview-title" data-overview>
    <h4 id="overview-title">The table at decision {step + 1}</h4>
    <!-- Said as the step changes, from a line in the page from the start. -->
    <p class="visually-hidden" role="status" data-table-said>{tableLine}</p>

    <MatchTable view={frame.view} {hints} thinking={false} pendingOpponent={null} hands={frame.hands ?? null} {notes} review />

    {#if frame.lines.length}
      <details class="lines" bind:open={linesOpen}>
        <summary>{step === 0 ? 'Since the deal' : 'Since your previous decision'} · {frame.lines.length} event{frame.lines.length === 1 ? '' : 's'}</summary>
        <ol>{#each frame.lines as line, index (index)}<li>{line}</li>{/each}</ol>
      </details>
    {/if}

    <PlayerHand view={frame.view} engine={null} closed={true} {hints} busy={false} blocked={false}
      {discardChoices} confirmDiscards={false} picked={null} selected={thrown}
      canDiscard={tile => discardChoices.some(choice => choice.tile === tile)}
      selectTile={tile => { const choice = discardChoices.find(entry => entry.tile === tile); if (choice) choose(choice); }}
      syncHandFocus={() => {}} handLabel={`${NAMES[me.seat]}'s tiles, yours`} listed note={notes[0]} bind:handElement />
  </section>

  {#if offered}
    <p class="offered"><Tile tile={offered} size="small" dora={shownDora.includes(offered)} />
      <span>{discarder} {robbing ? 'declared a kan of' : 'discarded'} the {tileWords(offered)}</span></p>
  {/if}

  {#if choosing}
    <p id="hand-help" class="prompt">What would you play here? Choose again before you see Trained AI's answer{choiceHelp}.</p>
    {#if otherChoices.length}
      <div class="moves">
        {#each otherChoices as choice, index (index)}
          <button class="app-control" data-retry={choice.kind} onclick={() => choose(choice)}>
            {callLabel(choice, robbing)}{#if choice.tile && choice.kind !== 'pass'} <Tile tile={choice.tile} size="tiny" />{/if}
          </button>
        {/each}
      </div>
    {/if}
    <button class="app-control quiet" data-reveal onclick={reveal}>Show answer</button>
  {:else if mode === 'answered'}
    <p id="hand-help" class="visually-hidden">Trained AI's answer is below.</p>
  {:else}
    <p id="hand-help" class="prompt">{stepMove} {stepVerdict}</p>
    {#if mode === 'offered'}
      <!-- Another flagged decision than the one opened: the player may choose
           again here too, before the answer is shown, or see it now. -->
      <div class="offer" data-offer>
        <p>{weightsSaid(current.judgement, 'Flagged: ')}</p>
        <div class="after">
          <button class="app-control" data-choose-here onclick={again}>Choose again</button>
          <button class="app-control quiet" data-reveal onclick={reveal}>Show answer</button>
        </div>
      </div>
    {/if}
  {/if}

  {#if shown}
    <!-- At a flagged decision Trained AI's answer, once asked for; at any
         other step its view of the move made there, which the review holds. -->
    <section class="answer" aria-labelledby="answer-title" data-answer={mode === 'answered' ? '' : undefined} data-view={mode === 'shown' ? '' : undefined}>
      <h4 id="answer-title" tabindex="-1" bind:this={answerHeading}>{mode === 'answered' ? "Trained AI's answer" : "Trained AI's view"}</h4>
      <ul class="verdicts">
        {#if tried}
          <li><strong>This time:</strong> {labelOf(tried)} · {weightOf(tried)}{sameMove(tried, advised.kind, advised.tile) ? ' — Trained AI’s choice' : ''}</li>
        {/if}
        <li><strong>In the game:</strong> {shown.played} · {weightOf(played)}</li>
        <li><strong>Trained AI:</strong> {shown.advised} · {weightOf(advised)}</li>
      </ul>
      <p class="help">Percentages are the network's preferences, not chances of winning.{stage === 'tile' ? ' Both declared riichi, so the tiles are compared as the discard given riichi.' : ''}</p>

      <div class="weights" role="list" aria-label="Trained AI's weight for each move">
        {#each rows as row, index (index)}
          {#if row.choice}
            {@render weightRow(row.choice, row.choice.weight)}
          {:else}
            <div class="reach" role="listitem">
              <div class="row reach-heading"><span class="label">Declare riichi</span><meter min="0" max="1" value={row.weight} aria-label="Riichi declaration weight"></meter><strong>{percent(row.weight)}</strong></div>
              <div role="list" aria-label="Discards given riichi">
                {#each row.choices as choice, slot (slot)}{@render weightRow(choice, choice.conditionalWeight, true)}{/each}
              </div>
            </div>
          {/if}
        {/each}
      </div>
      {#if extra}<button class="app-control quiet" data-every-move onclick={() => { everyMove = !everyMove; }}>{everyMove ? 'Show the leading moves' : `Show every move (${extra} more)`}</button>{/if}

      {#if compared.length > 1 && measured}
        <table class="numbers">
          <caption>What each move leaves, counted as the hand review counts it</caption>
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
      {/if}
      {#if mode === 'answered'}
        <div class="after">
          <button class="app-control" data-again onclick={again}>Choose again</button>
          <button class="app-control" onclick={onback}>Back to the list</button>
        </div>
      {/if}
    </section>
  {/if}
</section>

{#snippet weightRow(choice, weight, conditional = false)}
  <div class="row" role="listitem" class:best={sameMove(choice, advised.kind, advised.tile)} data-move={choice.label}>
    <span class="label">{#if choice.tile}<Tile tile={choice.tile} size="tiny" dora={shownDora.includes(choice.tile)} />{/if}{choice.label}</span>
    {#if weight == null}<strong>{conditional ? 'Not evaluated' : 'Unscored'}</strong>
    {:else}<meter min="0" max="1" value={weight} aria-label={conditional ? `${choice.label}, given riichi` : choice.label}></meter><strong>{percent(weight)}</strong>{/if}
    <span class="marks">
      {#if sameMove(choice, advised.kind, advised.tile)}<span class="mark trained">Trained AI's choice</span>{/if}
      {#if sameMove(choice, played.kind, played.tile)}<span class="mark">In the game</span>{/if}
      {#if tried && sameMove(choice, tried.kind, tried.tile)}<span class="mark">This time</span>{/if}
    </span>
  </div>
{/snippet}

<style>
  .decision { display: grid; gap: 10px; min-width: 0; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px 12px; }
  h3 { margin: 0; font-size: 1rem; }
  h4 { margin: 0; font-size: .95rem; }
  h3:focus, h4:focus { outline: none; }
  h3:focus-visible, h4:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  .steps { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; }
  .where { margin: 0; font-size: .85rem; opacity: .85; }
  .decisions, .lines { font-size: .82rem; }
  .decisions summary, .lines summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .decision-list { display: grid; gap: 8px; }
  /* As many to a row as fit, two on the narrowest phone, so a hand's dozen
     or more decisions take a few rows rather than a column. */
  .decisions ol { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(136px, 1fr)); gap: 6px; }
  .decisions li { display: flex; min-width: 0; }
  .decisions button { width: 100%; display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 2px 8px; padding: 5px 8px; text-align: left; font-size: .8rem; }
  .decisions .number { grid-row: span 2; font-weight: 650; font-variant-numeric: tabular-nums; }
  .decisions .move { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
  .decisions .verdict { font-size: .72rem; opacity: .8; }
  /* A flagged decision is marked by an edge as the whole-game list marks it,
     and says so in words; the step shown is gold. */
  .decisions .flag { border-left: 3px solid var(--accent); }
  .decisions [aria-current='step'] { border-color: var(--gold); background: rgba(216, 161, 42, .2); }
  .empty { margin: 0; opacity: .82; }
  .overview { display: grid; gap: 10px; min-width: 0; }
  .lines ol { margin: 0; padding: 6px 12px 6px 28px; display: grid; gap: 2px; background: #0003; border-radius: 8px; }
  .offered { display: flex; align-items: center; gap: 8px; margin: 0; font-size: .9rem; }
  .prompt { margin: 0; font-size: .9rem; }
  .offer { display: grid; gap: 8px; padding: 10px 12px; border-radius: 10px; background: #0003; border-left: 3px solid var(--accent); font-size: .86rem; }
  .offer p { margin: 0; }
  .moves, .after { display: flex; flex-wrap: wrap; gap: 8px; }
  .moves button { display: inline-flex; align-items: center; gap: 6px; text-align: left; }
  .quiet { justify-self: start; background: transparent; }
  .answer { display: grid; gap: 10px; padding: 12px; border: 1px solid #d8a12a66; border-radius: 12px; background: #0003; min-width: 0; }
  .verdicts { margin: 0; padding-left: 18px; display: grid; gap: 4px; font-size: .88rem; }
  .help { margin: 0; font-size: .78rem; opacity: .8; }
  .weights { display: grid; gap: 4px; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; padding: 5px 6px; border: 2px solid transparent; border-top-color: #ffffff15; border-radius: 8px; font-size: .82rem; }
  .row.best { border-color: #4ea3ff; }
  .label { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; overflow-wrap: anywhere; }
  .row strong { min-width: 56px; text-align: right; font-variant-numeric: tabular-nums; }
  meter { flex-shrink: 0; width: 90px; accent-color: var(--gold); }
  .marks { flex-basis: 100%; display: flex; flex-wrap: wrap; gap: 4px 10px; font-size: .72rem; }
  .marks:empty { display: none; }
  .mark { opacity: .85; }
  .mark.trained { color: var(--gold); opacity: 1; }
  .reach { border: 1px solid #ffffff28; border-radius: 8px; padding: 6px; display: grid; gap: 4px; }
  .reach-heading { border-top-color: transparent; }
  .numbers { border-collapse: collapse; width: 100%; table-layout: fixed; overflow-wrap: anywhere; font-size: .82rem; font-variant-numeric: tabular-nums; }
  .numbers caption { text-align: left; font-size: .78rem; opacity: .8; padding-bottom: 4px; }
  .numbers th { text-align: left; font-weight: 500; opacity: .75; padding: 2px 10px 2px 0; vertical-align: bottom; }
  .numbers td { padding: 2px 10px 2px 0; }
  .worse { color: var(--warning-text); }
  @media (max-width: 400px) { meter { width: 56px; } }
  /* Below the game's bar where it stays at the top of the screen (AppSettings). */
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h3, h4, .decision :global(.hand) { scroll-margin-top: 72px; }
  }
</style>
