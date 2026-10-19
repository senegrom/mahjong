<script>
  import { tick } from 'svelte';
  import Tile from './Tile.svelte';
  import MatchTable from './app/MatchTable.svelte';
  import PlayerHand from './app/PlayerHand.svelte';
  import { groupPolicyChoices } from './action-weights.js';
  import { decisionInBrief, sameMove, seatNotes, tableSaid } from './game-review.js';
  import { callLabel, percent, quadOnOffer } from './ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from './tiles.js';

  /**
   * A reviewed hand, replayed a decision of the player's at a time from the
   * deal, with what the others did in between and the table as it stood at
   * each: every player's tiles, the discards, the sets and the board.
   *
   * Opened with `retry` at a flagged decision, the player chooses again there
   * before Trained AI's answer is shown. Until it is, the other players'
   * tiles stay face down at every step, unless the player asks to see them,
   * and the hand is replayed only as far as the decision, since what came
   * after would give the answer away as well. Opened without, to look through
   * the hand, every step shows every hand, the move made and Trained AI's
   * view of it. The saved match is never touched, and nothing is played on
   * from any step.
   */
  let { hand, decision, retry = true, hints = true, onback } = $props();

  // Opened at the decision; the hand before it is a step back each.
  let step = $derived(decision.index);
  let retried = $state(null);
  let revealed = $state(false);
  let everyMove = $state(false);
  // Whether the player asked to see the other players' tiles before
  // choosing again.
  let peek = $state(false);
  let heading = $state(null);
  let answerHeading = $state(null);
  let handElement = $state(null);
  let nextButton = $state(null);
  let previousButton = $state(null);
  // The list of the hand's decisions starts open for a hand opened to be
  // looked through, where the screen has room for it above the table.
  const roomy = !(globalThis.matchMedia?.('(max-width: 760px)').matches ?? true);

  // The answer to the decision chosen again is still to come.
  let pending = $derived(retry && !revealed);
  // The last step that can be shown: while the answer is to come, the
  // decision itself.
  let last = $derived(pending ? decision.index : hand.frames.length - 1);
  let frame = $derived(hand.frames[step]);
  let current = $derived(hand.decisions?.[step] ?? null);
  let atDecision = $derived(step === decision.index);
  // This step is the decision chosen again.
  let retrying = $derived(retry && atDecision);
  let choosing = $derived(retrying && !revealed);
  let hidden = $derived(pending && !peek);
  // Every player's tiles as the review's own replay read them, face up
  // unless they are hidden.
  let hands = $derived(hidden ? null : current?.hands ?? null);
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
  // tells the moves as well as where it is. A move with no alternative was
  // never put to Trained AI.
  let stepMove = $derived(retrying ? ''
    : `Your move: ${frame.choices.find(choice => sameMove(choice, frame.played.kind, frame.played.tile))?.label ?? frame.played.kind}.`);
  let stepVerdict = $derived(!current || retrying ? '' : current.forced ? 'It was the only move.'
    : current.agreed ? 'Trained AI chose the same.' : 'Trained AI chose differently.');
  let shownDora = $derived(hints ? frame.view.dora_types ?? [] : []);
  // The tile thrown at a step, or chosen again at the decision, is marked in
  // the hand as a selected tile is in play.
  let thrown = $derived.by(() => {
    const move = retrying ? retried : frame.played;
    if (!move || !['discard', 'riichi'].includes(move.kind) || !me) return null;
    const tiles = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
    const index = move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile);
    return index >= 0 ? index : null;
  });

  // The decision whose answer, or Trained AI's view of it, this step shows:
  // none while it is being chosen again, and none for a move with no
  // alternative, which says so under the hand.
  let shown = $derived(choosing || !current?.weighted || current.forced ? null : current);
  let tried = $derived(retrying ? retried : null);
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

  function marked(choice) {
    return [advised, played, ...(tried ? [tried] : [])].some(move => move && sameMove(choice, move.kind, move.tile));
  }

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }

  // Where focus goes when a step disables the button pressed, which would
  // drop it to the page: at the decision being chosen again to the hand,
  // where the choice is made, as play gives the hand focus when a turn comes;
  // at that decision answered, to the answer; otherwise on to Next, or at
  // the last step back to Previous.
  function landing() {
    if (choosing) return handElement;
    if (retrying && revealed) return answerHeading;
    if (step < last) return nextButton;
    return step > 0 ? previousButton : heading;
  }

  // A step to either end of the hand disables the button pressed. Every
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

  async function choose(choice) {
    if (!choosing) return;
    retried = { kind: choice.kind, tile: choice.tile ?? null };
    revealed = true;
    await tick();
    answerHeading?.focus({ preventScroll: true });
    answerHeading?.scrollIntoView({ block: 'nearest' });
  }

  async function reveal() {
    revealed = true;
    await tick();
    answerHeading?.focus({ preventScroll: true });
  }

  // The answer goes, and the button pressed with it, and the other players'
  // tiles go face down again. The hand takes focus, as it does in play when
  // a turn comes, and says what to do through its description, the prompt;
  // it is brought into view to be chosen from.
  async function again() {
    retried = null;
    revealed = false;
    everyMove = false;
    peek = false;
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
    <h3 id="decision-title" tabindex="-1" bind:this={heading}>{retry ? `${hand.name}, your decision ${decision.index + 1}` : `${hand.name}, decision by decision`}</h3>
    <button class="app-control" data-back onclick={onback}>Back to the list</button>
  </header>

  <div class="steps" role="group" aria-label="Replay of the hand">
    <button class="app-control" onclick={() => goTo(0)} disabled={step === 0}>From the deal</button>
    <button class="app-control" onclick={() => goTo(step - 1)} disabled={step === 0} aria-label="Previous decision" bind:this={previousButton}>Previous</button>
    <button class="app-control" onclick={() => goTo(step + 1)} disabled={step >= last} aria-label="Next decision" bind:this={nextButton}>Next</button>
    {#if retry}<button class="app-control" onclick={() => goTo(decision.index)} disabled={atDecision}>This decision</button>{/if}
    <p class="where" role="status"><span data-step>Your decision {step + 1} of {last + 1}{retrying ? ', the one flagged' : ''}</span>{#if stepMove}<span class="visually-hidden">. {stepMove} {stepVerdict}</span>{/if}</p>
  </div>

  <!-- Every decision of the hand that can be shown, each a step of its own.
       A button keeps focus when pressed, as the step it shows changes. -->
  <details class="decisions" open={!retry && roomy} data-decisions>
    <summary>{pending ? 'Your decisions up to the flagged one' : 'Your decisions in this hand'} · {last + 1}</summary>
    <ol>
      {#each hand.frames.slice(0, last + 1) as entry, index (index)}
        {@const brief = decisionInBrief(entry, hand.decisions?.[index], pending && index === decision.index)}
        <li>
          <button class="app-control" data-step-to={index} aria-current={index === step ? 'step' : undefined} onclick={() => goTo(index)}>
            <span class="number"><span class="visually-hidden">Decision </span>{index + 1}</span>
            <span class="move">{brief.move}{#if brief.tile}<Tile tile={brief.tile} size="tiny" />{/if}</span>
            <span class="verdict">{brief.verdict}</span>
          </button>
        </li>
      {/each}
    </ol>
  </details>

  <!-- The table as it stood at this step, all of it: each player's tiles,
       face up unless the decision is still to be chosen again, then the
       discards, the sets and the board, which were always in view. -->
  <section class="overview" aria-labelledby="overview-title" data-overview>
    <div class="overview-head">
      <h4 id="overview-title">{hidden ? 'The table' : 'Every hand'} at your decision {step + 1}</h4>
      {#if pending}
        <label class="peek"><input type="checkbox" bind:checked={peek} aria-describedby="peek-note" data-peek /> Show the other players' tiles</label>
        <p id="peek-note" class="help">You could not see them when you chose, so they give away more than you knew then.</p>
      {/if}
    </div>
    <!-- Said as the step changes, from a line that is in the page from the start. -->
    <p class="visually-hidden" role="status" data-table-said>{tableSaid(frame.view, !hidden)}</p>

    <MatchTable view={frame.view} {hints} thinking={false} pendingOpponent={null} {hands} {notes} review />

    {#if frame.lines.length}
      <details class="lines" open>
        <summary>{step === 0 ? 'Since the deal' : 'Since your previous decision'} · {frame.lines.length} event{frame.lines.length === 1 ? '' : 's'}</summary>
        <ol>{#each frame.lines as line, index (index)}<li>{line}</li>{/each}</ol>
      </details>
    {/if}

    <PlayerHand view={frame.view} engine={null} closed={true} {hints} busy={false} blocked={false}
      {discardChoices} confirmDiscards={false} picked={null} selected={thrown}
      canDiscard={tile => discardChoices.some(choice => choice.tile === tile)}
      selectTile={tile => { const choice = discardChoices.find(entry => entry.tile === tile); if (choice) void choose(choice); }}
      syncHandFocus={() => {}} handLabel={`${NAMES[me.seat]}'s tiles, yours`} listed note={notes[0]} bind:handElement />
  </section>

  {#if offered}
    <p class="offered"><Tile tile={offered} size="small" dora={shownDora.includes(offered)} />
      <span>{discarder} {robbing ? 'declared a kan of' : 'discarded'} the {tileWords(offered)}</span></p>
  {/if}

  {#if !retrying}
    <p id="hand-help" class="prompt">{stepMove} {stepVerdict}</p>
  {:else if choosing}
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
    <button class="app-control quiet" data-reveal onclick={reveal}>Show Trained AI's answer</button>
  {:else}
    <p id="hand-help" class="visually-hidden">Trained AI's answer is below.</p>
  {/if}

  {#if shown}
    <!-- At the decision chosen again, Trained AI's answer; at any other step
         its view of the move made there, which the review already holds. -->
    <section class="answer" aria-labelledby="answer-title" data-answer={retrying ? '' : undefined} data-view={retrying ? undefined : ''}>
      <h4 id="answer-title" tabindex="-1" bind:this={answerHeading}>{retrying ? "Trained AI's answer" : `Trained AI's view${shown.judgement?.flagged ? ' · flagged' : ''}`}</h4>
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
      {#if retrying}
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
      {#if sameMove(choice, played.kind, played.tile)}<span class="mark">Your move in the game</span>{/if}
      {#if tried && sameMove(choice, tried.kind, tried.tile)}<span class="mark">Your choice now</span>{/if}
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
  .decisions { font-size: .82rem; }
  .decisions summary, .lines summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  /* As many to a row as fit, two on the narrowest phone, so a hand's dozen
     or more decisions take a few rows rather than a column. */
  .decisions ol { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 6px; }
  .decisions button { width: 100%; height: 100%; display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 2px 8px; padding: 5px 8px; text-align: left; font-size: .8rem; }
  .decisions .number { grid-row: span 2; font-weight: 650; font-variant-numeric: tabular-nums; }
  .decisions .move { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
  .decisions .verdict { font-size: .72rem; opacity: .8; }
  .decisions [aria-current='step'] { border-color: var(--gold); background: rgba(216, 161, 42, .2); }
  .overview { display: grid; gap: 10px; min-width: 0; }
  .overview-head { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 16px; }
  .peek { display: inline-flex; align-items: center; gap: 8px; min-height: 36px; font-size: .86rem; cursor: pointer; }
  .peek input { width: 18px; height: 18px; margin: 0; accent-color: var(--gold); }
  .overview-head .help { flex-basis: 100%; }
  .lines { font-size: .82rem; }
  .lines ol { margin: 0; padding: 6px 12px 6px 28px; display: grid; gap: 2px; background: #0003; border-radius: 8px; }
  .offered { display: flex; align-items: center; gap: 8px; margin: 0; font-size: .9rem; }
  .prompt { margin: 0; font-size: .9rem; }
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
