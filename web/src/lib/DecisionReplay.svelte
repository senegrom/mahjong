<script>
  import { tick } from 'svelte';
  import Tile from './Tile.svelte';
  import MatchTable from './app/MatchTable.svelte';
  import PlayerHand from './app/PlayerHand.svelte';
  import { groupPolicyChoices } from './action-weights.js';
  import { sameMove } from './game-review.js';
  import { callLabel, quadOnOffer } from './ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from './tiles.js';

  /**
   * One flagged decision, opened at the position it was made from. The hand
   * can be stepped through from the deal, a decision of the player's at a
   * time, with what the others did in between. At the decision the player
   * chooses again before Trained AI's answer is shown; the saved match is
   * never touched, and nothing is played on from there.
   */
  let { hand, decision, hints = true, onback } = $props();

  // Opened at the decision; the hand before it is a step back each.
  let step = $derived(decision.index);
  let retried = $state(null);
  let revealed = $state(false);
  let everyMove = $state(false);
  let heading = $state(null);
  let answerHeading = $state(null);
  let handElement = $state(null);
  let nextButton = $state(null);

  let frame = $derived(hand.frames[step]);
  let atDecision = $derived(step === decision.index);
  let choosing = $derived(atDecision && !revealed);
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
  let earlier = $derived(hand.decisions?.[step] ?? null);
  // What the player did at an earlier step, and whether Trained AI agreed:
  // shown under the hand, and said with the step, so stepping through the
  // hand by ear tells the moves as well as where it is. A move with no
  // alternative was never put to Trained AI.
  let earlierMove = $derived(atDecision ? ''
    : `Your move: ${frame.choices.find(choice => sameMove(choice, frame.played.kind, frame.played.tile))?.label ?? frame.played.kind}.`);
  let earlierVerdict = $derived(!earlier || atDecision ? '' : earlier.forced ? 'It was the only move.'
    : earlier.agreed ? 'Trained AI chose the same.' : 'Trained AI chose differently.');
  let shownDora = $derived(hints ? frame.view.dora_types ?? [] : []);
  let stage = $derived(decision.judgement?.stage);
  // The tile thrown at an earlier step, or chosen again at the decision, is
  // marked in the hand as a selected tile is in play.
  let thrown = $derived.by(() => {
    const move = atDecision ? retried : frame.played, me = frame.view.seats[0];
    if (!move || !['discard', 'riichi'].includes(move.kind) || !me) return null;
    const tiles = [...me.hand, ...(me.drawn ? [me.drawn] : [])];
    const index = move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile);
    return index >= 0 ? index : null;
  });

  const percent = weight => weight > 0 && weight < .001 ? '<0.1%' : `${(weight * 100).toFixed(1)}%`;
  let advised = $derived({ kind: decision.advised_kind, tile: decision.advised_tile ?? null });
  let played = $derived({ kind: decision.played_kind, tile: decision.played_tile ?? null });
  const labelOf = move => decision.weighted.find(choice => sameMove(choice, move.kind, move.tile))?.label ?? '';

  // The moves by weight, riichi's discards together under the declaration.
  let ordered = $derived([...decision.weighted]
    .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)));
  // The leading moves and every move the player or the network picked; the
  // rest wait behind a button. The button stays once they are shown, to hide
  // them again, so the focus on it stays too rather than dropping to the page.
  let rows = $derived(groupPolicyChoices(everyMove ? ordered : ordered.filter((choice, index) => index < 5 || marked(choice))));
  let extra = $derived(ordered.filter((choice, index) => index >= 5 && !marked(choice)).length);

  // The moves the numbers compare, each once: the game's, the new one, the network's.
  let compared = $derived([
    { move: played, title: 'In the game' },
    ...(retried && !sameMove(retried, played.kind, played.tile) ? [{ move: retried, title: 'This time' }] : []),
    ...(!sameMove(advised, played.kind, played.tile) && !(retried && sameMove(advised, retried.kind, retried.tile))
      ? [{ move: advised, title: 'Trained AI' }] : []),
  ].map(entry => ({ ...entry, measure: decision.measures?.find(measure => sameMove(measure, entry.move.kind, entry.move.tile)) ?? null })));
  let measured = $derived(compared.every(entry => entry.measure));

  function marked(choice) {
    return [advised, played, ...(retried ? [retried] : [])].some(move => sameMove(choice, move.kind, move.tile));
  }

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }

  // A step to either end of the hand disables the button pressed, which would
  // drop focus to the page. It is handed on instead: at the deal to Next,
  // which goes on from there, and at the decision to the hand, where the
  // choice is made. In between every button stays, and focus with it.
  async function goTo(next) {
    step = Math.max(0, Math.min(decision.index, next));
    await tick();
    if (step > 0 && !atDecision) return;
    const active = document.activeElement;
    if (active && active !== document.body && active.isConnected && !active.matches(':disabled')) return;
    if (atDecision) handElement?.focus({ preventScroll: true });
    else nextButton?.focus();
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

  // The answer goes, and the button pressed with it. The hand takes focus, as
  // it does in play when a turn comes, and says what to do through its
  // description, the prompt; it is brought into view to be chosen from.
  async function again() {
    retried = null;
    revealed = false;
    everyMove = false;
    await tick();
    handElement?.focus({ preventScroll: true });
    handElement?.scrollIntoView({ block: 'nearest' });
  }

  // A riichi has two weights, as ReviewPreference shows them: the
  // declaration's, and the tile's once the declaration is made, which is
  // asked only where the network declared.
  function weightOf(move) {
    const choice = decision.weighted.find(entry => sameMove(entry, move.kind, move.tile));
    if (!choice || choice.weight == null) return 'unscored';
    if (choice.kind !== 'riichi') return percent(choice.weight);
    return `${percent(choice.weight)} to declare riichi · discard given riichi: ${choice.conditionalWeight == null ? 'not evaluated' : percent(choice.conditionalWeight)}`;
  }

  $effect(() => { heading?.focus({ preventScroll: true }); heading?.scrollIntoView({ block: 'start' }); });
</script>

<section class="decision" aria-labelledby="decision-title" data-decision>
  <header>
    <h3 id="decision-title" tabindex="-1" bind:this={heading}>{hand.name}, your decision {decision.index + 1}</h3>
    <button class="app-control" data-back onclick={onback}>Back to the list</button>
  </header>

  <div class="steps" role="group" aria-label="Replay of the hand">
    <button class="app-control" onclick={() => goTo(0)} disabled={step === 0}>From the deal</button>
    <button class="app-control" onclick={() => goTo(step - 1)} disabled={step === 0} aria-label="Previous decision">Previous</button>
    <button class="app-control" onclick={() => goTo(step + 1)} disabled={atDecision} aria-label="Next decision" bind:this={nextButton}>Next</button>
    <button class="app-control" onclick={() => goTo(decision.index)} disabled={atDecision}>This decision</button>
    <p class="where" role="status"><span data-step>Your decision {step + 1} of {decision.index + 1}{atDecision ? ', the one flagged' : ''}</span>{#if earlierMove}<span class="visually-hidden">. {earlierMove} {earlierVerdict}</span>{/if}</p>
  </div>

  <MatchTable view={frame.view} {hints} thinking={false} pendingOpponent={null} />

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
    syncHandFocus={() => {}} bind:handElement />

  {#if offered}
    <p class="offered"><Tile tile={offered} size="small" dora={shownDora.includes(offered)} />
      <span>{discarder} {robbing ? 'declared a kan of' : 'discarded'} the {tileWords(offered)}</span></p>
  {/if}

  {#if !atDecision}
    <p id="hand-help" class="prompt">{earlierMove} {earlierVerdict}</p>
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

  {#if atDecision && revealed}
    <section class="answer" aria-labelledby="answer-title" data-answer>
      <h4 id="answer-title" tabindex="-1" bind:this={answerHeading}>Trained AI's answer</h4>
      <ul class="verdicts">
        {#if retried}
          <li><strong>This time:</strong> {labelOf(retried)} · {weightOf(retried)}{sameMove(retried, advised.kind, advised.tile) ? ' — Trained AI’s choice' : ''}</li>
        {/if}
        <li><strong>In the game:</strong> {decision.played} · {weightOf(played)}</li>
        <li><strong>Trained AI:</strong> {decision.advised} · {weightOf(advised)}</li>
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
      <div class="after">
        <button class="app-control" data-again onclick={again}>Choose again</button>
        <button class="app-control" onclick={onback}>Back to the list</button>
      </div>
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
      {#if retried && sameMove(choice, retried.kind, retried.tile)}<span class="mark">Your choice now</span>{/if}
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
  .lines { font-size: .82rem; }
  .lines summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
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
