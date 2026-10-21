<script>
  import { onMount, tick } from 'svelte';
  import Tile from './Tile.svelte';
  import MatchTable from './app/MatchTable.svelte';
  import PlayerHand from './app/PlayerHand.svelte';
  import { groupPolicyChoices } from './action-weights.js';
  import { decisionInBrief, decisionKey, moveToken, sameMove, seatNotes } from './game-review.js';
  import { TAG_SAID, callLabel, callTiles, percent, pressedByKeyboard, quadOnOffer } from './ui.js';
  import { SEAT_NAMES as NAMES } from './tiles.js';

  /**
   * The decision the review is on, `step` of `hand`: the table as it stood
   * there, every player's tiles face up, the tile just drawn apart, the sets,
   * the discards and the board; the player's own seat; what the others did
   * since the player's previous decision; and, where Trained AI was asked,
   * its view of every move open there. The replay's own engine read the
   * tiles once the hand was over (scanMatch), never the match.
   *
   * The move made there and Trained AI's are tagged over their tiles in the
   * hand, Trained AI's ringed in blue as Agent watch and Guided play ring the
   * network's choice; the step bar above gives both (ReviewStepBar). The
   * decision stays on the page as the review steps, across hands too, so
   * that what the player opened or folded here stays as they left it. The
   * saved match is never touched, and nothing is played on from any step.
   *
   * The player's settings can keep back what the decision would give away
   * until they ask for it (GameReview). With Trained AI's move kept back
   * (`moveHidden`) there is no view, no ring and no "AI" over a tile, and the
   * decision is put to the player: every legal discard in the hand is a
   * button, as in play, the other moves are buttons under it in play's words,
   * and whichever they press is their answer, `onpick(move, byKeyboard)`,
   * told whether a keyboard pressed it (pressedByKeyboard). Once shown, a
   * move they picked (`pick`) is tagged "Now" wherever the moves are tagged.
   * With the other players' tiles kept back (`tilesHidden`) the opponents'
   * hands lie face down in their racks.
   */
  let { hand, step, hints = true, moveHidden = false, tilesHidden = false, pick = null, onpick = () => {} } = $props();

  // The events since the previous decision and the table of what each move
  // leaves start folded, read less than the table and the weights, and stay
  // as the player leaves them across a step that takes them from the page.
  // Each is held here rather than set on the list from the step: the page
  // sets that afresh at every step, and as later hands are reviewed, which
  // folded a list under the player's press.
  let linesOpen = $state(false);
  let numbersOpen = $state(false);
  // Every move of the view, where the player asked for them at this
  // decision; the next decision leads with the few again.
  let everyMoveAt = $state(null);
  // On a phone held upright the player's seat comes before the table, so that
  // the player's tiles are the first thing under the step bar; held sideways,
  // and on every larger screen, it comes after, as in play. It is drawn in
  // one place or the other rather than moved by CSS, so that a screen reader
  // and the keyboard go through it in the order it is seen. A server render
  // measures nothing, and draws it after.
  //
  // As the screen turns from one to the other, the seat is drawn afresh in
  // its other place, and a control of it that had focus, the hand, a tile of
  // the live hand or a move under it, would go with the seat it was in,
  // dropping focus to the page. The same control in the seat as now drawn
  // has it instead, brought into view as focus is anywhere on the stage,
  // clear of the bars (GameReview). The seat is drawn from the same decision,
  // so the control is there.
  const UPRIGHT = '(max-width: 760px) and (orientation: portrait)';
  const SEAT = '.mine :is(.hand, button), .picks button';
  let upright = $state(globalThis.matchMedia?.(UPRIGHT).matches ?? false);
  let root = $state(null);
  onMount(() => {
    const media = matchMedia(UPRIGHT);
    upright = media.matches;
    const changed = async () => {
      const kept = [...(root?.querySelectorAll(SEAT) ?? [])].indexOf(document.activeElement);
      upright = media.matches;
      if (kept < 0) return;
      await tick();
      root?.querySelectorAll(SEAT)[kept]?.focus();
    };
    media.addEventListener('change', changed);
    return () => media.removeEventListener('change', changed);
  });

  let frame = $derived(hand.frames[step]);
  let current = $derived(hand.decisions?.[step] ?? null);
  let notes = $derived(seatNotes(frame.view));
  let me = $derived(frame.view.seats[0]);
  let shownDora = $derived(hints ? frame.view.dora_types ?? [] : []);
  let brief = $derived(decisionInBrief(frame, current, { hidden: moveHidden }));
  let key = $derived(decisionKey(hand.index, step));
  let everyMove = $derived(everyMoveAt === key);

  // The decision whose view of every move this step shows: any that was put
  // to Trained AI, which a move with no alternative never was, once its
  // answer is shown.
  let shown = $derived(current?.weighted && !current.forced && !moveHidden ? current : null);
  let advised = $derived(shown ? { kind: shown.advised_kind, tile: shown.advised_tile ?? null } : null);
  let played = $derived(shown ? { kind: shown.played_kind, tile: shown.played_tile ?? null } : null);
  // The move the player picked here before the answer was shown, if any.
  let now = $derived(shown && pick ? { kind: pick.kind, tile: pick.tile ?? null } : null);
  let stage = $derived(shown?.judgement?.stage);

  // While Trained AI's move is kept back the player answers the decision as
  // in play: a legal discard by its tile in the hand, and any other move,
  // a riichi, a call, a win or a pass, by its button under the hand, in the
  // words play's buttons give it (callLabel), with its tiles beside them.
  let discards = $derived(moveHidden ? frame.choices.filter(choice => choice.kind === 'discard') : []);
  let others = $derived(moveHidden ? frame.choices.filter(choice => choice.kind !== 'discard') : []);
  let robbing = $derived(Boolean(quadOnOffer(frame.view)));
  const canPick = tile => discards.some(choice => choice.tile === tile);

  // The tiles in the hand that the step's moves throw, by their place in it:
  // the player's, the one they picked, and Trained AI's where it was asked
  // and its answer is shown. A move that throws no tile from the hand, a
  // call, a win or a pass, has none. Where the tile just drawn is one of
  // them, it is that copy, as play throws the draw.
  let tiles = $derived([...me.hand, ...(me.drawn ? [me.drawn] : [])]);
  function placeOf(move) {
    if (!move || !['discard', 'riichi'].includes(move.kind)) return null;
    const index = move.tile === me.drawn ? tiles.length - 1 : tiles.indexOf(move.tile);
    return index >= 0 ? index : null;
  }
  let yours = $derived(placeOf(frame.played));
  let picked = $derived(placeOf(now));
  let trained = $derived(placeOf(advised));
  // The words over those tiles, in a line kept on every tile (HandTile):
  // "You" over the player's, "Now" over the one they picked, "AI" over
  // Trained AI's, joined over one tile that more than one chose. Trained
  // AI's tile is ringed as well (`selected`).
  let tags = $derived(tiles.map((_, index) => [index === yours && 'You', index === picked && 'Now', index === trained && 'AI']
    .filter(Boolean)));

  // The moves by weight, riichi's discards together under the declaration.
  let ordered = $derived(shown ? [...shown.weighted]
    .sort((a, b) => (b.weight ?? -1) - (a.weight ?? -1) || (a.index ?? 99) - (b.index ?? 99)) : []);
  // The leading moves and every move the player, their pick or the network
  // chose; the rest wait behind a button, which says what a press does: how
  // many more it shows, or that it shows fewer. The button stays once they
  // are shown, to hide them again, so the focus on it stays too rather than
  // dropping to the page.
  let rows = $derived(groupPolicyChoices(everyMove ? ordered : ordered.filter((choice, index) => index < 5 || marked(choice))));
  let extra = $derived(ordered.filter((choice, index) => index >= 5 && !marked(choice)).length);

  // The moves the numbers compare, each once and only where the review
  // measured it: the player's, the one they picked and Trained AI's. A pick
  // the review did not measure, as it measures every move only where the
  // decision is flagged, has no column, and the others keep theirs.
  let compared = $derived(!shown ? [] : [
    { move: played, title: 'You' },
    ...(now && !sameMove(now, played.kind, played.tile) && !sameMove(now, advised.kind, advised.tile) ? [{ move: now, title: 'Now' }] : []),
    ...(!sameMove(advised, played.kind, played.tile) ? [{ move: advised, title: 'Trained AI' }] : []),
  ].map(entry => ({ ...entry, measure: shown.measures?.find(measure => sameMove(measure, entry.move.kind, entry.move.tile)) ?? null }))
    .filter(entry => entry.measure));

  function marked(choice) {
    return [advised, played, now].some(move => move && sameMove(choice, move.kind, move.tile));
  }

  // The words a move of the view is tagged with, as the hand tags its tiles.
  const tagsOf = choice => [sameMove(choice, played.kind, played.tile) && 'You', now && sameMove(choice, now.kind, now.tile) && 'Now',
    sameMove(choice, advised.kind, advised.tile) && 'AI'].filter(Boolean);

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }
</script>

<!-- The table as it stood at this step, all of it, every player's tiles face
     up unless the player keeps the others' back: the player's own seat and
     the board with the other players around it, the seat after the board
     but on a phone held upright, then what happened since the player's
     previous decision. The step bar says it as the step changes. -->
<div class="decision" data-decision bind:this={root}>
  <div class="overview" data-overview>
    {#if upright}{@render seat()}{/if}
    <!-- Drawn afresh for another hand, so that a seat does not announce a
         call made in the hand before. The hands turn over in place, without
         being drawn afresh. -->
    {#key hand.index}
      <MatchTable view={frame.view} {hints} thinking={false} pendingOpponent={null} hands={frame.hands ?? null} {notes} review
        concealed={tilesHidden} />
    {/key}
    {#if !upright}{@render seat()}{/if}

    {#if frame.lines.length}
      <details class="lines" bind:open={linesOpen}>
        <!-- How many events, in brackets as the review's other summaries
             give a count; the items below are the events. -->
        <summary>{step === 0 ? 'Since the deal' : 'Since the previous decision'} ({frame.lines.length})</summary>
        <ol>{#each frame.lines as line, index (index)}<li>{line}</li>{/each}</ol>
      </details>
    {/if}
  </div>

  {#if shown}
    <!-- Trained AI's view of the move made at the step, which the review
         holds for every decision it was asked about: each move open there by
         its weight, tagged as the hand tags its tiles. -->
    <section class="view" aria-labelledby="view-title" data-view>
      <h3 id="view-title">Trained AI's view</h3>
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
      {#if extra}<button class="app-control quiet" data-every-move onclick={() => { everyMoveAt = everyMove ? null : key; }}>{everyMove ? 'Show fewer moves' : `Show ${extra} more move${extra === 1 ? '' : 's'}`}</button>{/if}

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
</div>

<!-- The player's own seat, with the words over the tiles the moves concern,
     and what the hand is described by: the two moves, in words, or while
     Trained AI's move is kept back, what the player can do, followed by the
     moves other than a discard, as play puts its calls under the hand. -->
{#snippet seat()}
  <PlayerHand view={frame.view} engine={null} closed={true} {hints} busy={false} blocked={false}
    discardChoices={discards} confirmDiscards={false} picked={null} selected={trained}
    canDiscard={canPick} selectTile={(tile, _index, event) => onpick({ kind: 'discard', tile }, pressedByKeyboard(event))} syncHandFocus={() => {}}
    handLabel={`${NAMES[me.seat]}'s tiles, yours`} listed note={notes[0]} {tags} />
  {#if moveHidden}
    <p id="hand-help" class="prompt">Pick a move, or Show answer.</p>
    {#if others.length}
      <div class="picks" role="group" aria-label="Other moves">
        {#each others as choice, index (index)}
          <button type="button" class="app-control" data-pick={choice.kind} aria-label={callLabel(choice, robbing)}
            onclick={event => onpick({ kind: choice.kind, tile: choice.tile ?? null }, pressedByKeyboard(event))}>
            <span>{callLabel(choice, robbing)}</span>
            {#if callTiles(choice, frame.view.pending_discard).length}
              <span class="preview" aria-hidden="true">
                {#each callTiles(choice, frame.view.pending_discard) as tile, slot (slot)}<Tile {tile} size="tiny" dora={shownDora.includes(tile)} />{/each}
              </span>
            {/if}
          </button>
        {/each}
      </div>
    {/if}
  {:else}
    <p id="hand-help" class="visually-hidden">{brief.said}</p>
  {/if}
{/snippet}

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
  /* The player's thirteen tiles and the one drawn keep one row wherever the
     stage, the review's column for the decision (GameReview), has room for
     them at all: from 761 pixels wide the hand draws its tiles --mine-tile
     wide (PlayerHand), and fourteen of them with their gaps, the drawn
     tile's and the seat's padding take fourteen tiles and 125 pixels, so 52
     pixels from a stage of 860, 42 from 715, 40 from 685 and 38 below: a
     window 761 pixels wide leaves a stage of 678 beside a classic 17-pixel
     scrollbar, which the media query counts as part of the width. The sizes
     are fixed steps rather than a share of the stage (cqi), which a tile's
     own parts would take of the nearest container instead: an unpainted
     tile's face is one. */
  .decision { --mine-tile: 38px; }
  @container stage (min-width: 685px) { .decision { --mine-tile: 40px; } }
  @container stage (min-width: 715px) { .decision { --mine-tile: 42px; } }
  @container stage (min-width: 860px) { .decision { --mine-tile: 52px; } }
  h3 { margin: 0; font-size: .95rem; }
  /* What the player can do while the answer is kept back, and the moves
     other than a discard, laid out as play lays out its calls (TurnChoices). */
  .prompt { margin: 0; font-size: .86rem; }
  .picks { display: flex; flex-wrap: wrap; gap: 8px; }
  .picks button { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 8px; text-align: left; overflow-wrap: anywhere; }
  .preview { display: inline-flex; align-items: center; gap: 4px; }
  .lines { font-size: .82rem; }
  .lines summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .token { display: inline-flex; align-items: center; gap: 3px; }
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
</style>
