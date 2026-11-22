<script lang="ts">
  import type { TurnChoicesProps } from './types';
  import Tile from '../Tile.svelte';
  import { RELOAD_TO_CONTINUE, callLabel, callTiles, partedNote, quadOnOffer, repeatedClick } from '../ui.js';
  import { tileWords, SEAT_NAMES as NAMES } from '../tiles.js';

  // All actions return to App's guarded match session. The controls do not
  // choose a fallback move or mutate a saved game on their own.
  let {
    view, shownDora, busy, thinking, failure, saveConflict, loadNote,
    pendingOpponent, myTurn, confirmDiscards, shortcuts, touch,
    selectedTile, callChoices, choose, discard, oncancel, onclub = null,
  }: TurnChoicesProps = $props();
  // The discard a call is offered on, and the tile chosen while its discard
  // waits to be confirmed. Each has a panel that comes and goes with it.
  let offered = $derived(view.phase === 'call' && callChoices.length ? view.pending_discard : null);
  let chosen = $derived(myTurn && !busy ? selectedTile : null);
  let discarder = $derived((view.pending_from ? NAMES[view.pending_from] : undefined) ?? 'An opponent');
  // A quad that may be robbed is offered as its tile, the way a discard is,
  // but nobody discarded it: the win takes the tile added to the quad.
  let robbing = $derived(Boolean(quadOnOffer(view)));
  // A screen reader listens to a live region only once it exists, and JAWS
  // and NVDA often miss one inserted along with its words, as the panels
  // are. So their news is said by one region that is always here.
  let announcement = $derived(offered ? `${discarder} ${robbing ? 'declared a kan of' : 'discarded'} the ${tileWords(offered)}`
    : chosen ? `Selected: ${tileWords(chosen)}` : '');
  // What a Trained opponent waits for, where the page or its worker says:
  // the note's words, and the share of a download it ends with. The prompt
  // says the words, which change a few times in a download, and shows the
  // share beside them, which changes a hundred times: a screen reader reads
  // out every change of a live region.
  let waiting = $derived(partedNote(busy && thinking && !failure && !saveConflict ? loadNote : ''));
  let said = $derived(saveConflict ? RELOAD_TO_CONTINUE
    : failure ? 'Choose a recovery option above to continue.'
    : busy ? (!thinking ? 'Playing the turn…'
      : waiting.words || (pendingOpponent ? `${pendingOpponent.position} Trained opponent is thinking…` : 'Trained opponents are thinking…'))
    : myTurn ? (confirmDiscards ? 'Your turn. Select a tile, then confirm the discard.' : 'Your turn. Choose a tile to discard.')
    : callChoices.length ? 'Choose a call, or pass.' : 'Waiting for the others…');
  // The hand takes focus as the turn comes (App) and the tile guide gives the
  // keys, so only a screen reader is told them, wherever they work: with the
  // shortcuts on, on the player's turn, and not on a touch screen.
  let keys = $derived(myTurn && !busy && !failure && !saveConflict && !touch && shortcuts ? 'Focus your hand to use arrows and Enter.' : '');
  // A press on a choice acts once. The next decision's buttons are drawn
  // where these were, and against opponents that answer at once they are
  // there before the second press of a double click lands, which would
  // answer the next decision too: it is let go (repeatedClick). Enter held
  // down on one presses it once as well, which App sees to for every button
  // in a `data-moves` row (heldMove).
  function press(event: MouseEvent, act: () => void) {
    if (!repeatedClick(event)) act();
  }
</script>

<p class="visually-hidden" role="status">{announcement}</p>
<!-- Only the prompt's words are a live region. How far a download has got is
     shown beside them, outside it (waiting). -->
<p class="prompt" class:staged={Boolean(offered)}><span role="status">{said}</span>{#if waiting.figure}{' '}<span data-figure>{waiting.figure}</span>{/if}</p>
<!-- What the hand names as its description, which a screen reader reads as
     the hand takes focus: the keys, and nothing the prompt has just said. It
     is no live region, so a turn is not heard twice. -->
<p id="hand-help" class="visually-hidden">{keys}</p>
<!-- Offered by App only while what a Trained opponent waits for is the
     network's download, never while it thinks. What it does is in its
     title, to keep the words on the table few. -->
{#if onclub}
  <div class="meanwhile">
    <button class="app-control" data-club-meanwhile onclick={onclub}
      title="Club plays the Trained opponents' seats for this match. The download goes on, and a new game goes back to Trained once it is done.">Play Club meanwhile</button>
  </div>
{/if}
{#if chosen}
  <div class="confirm-discard" data-moves>
    <span>Selected: {tileWords(chosen)}</span>
    <button class="app-control primary" onclick={event => press(event, () => { if (selectedTile) discard(selectedTile); })}>Discard {tileWords(chosen)}</button>
    <button class="app-control" onclick={oncancel}>Cancel</button>
  </div>
{/if}
{#if callChoices.length}
  <div class="calls">
    {#if offered}
      <div class="call-stage">
        <span class="call-kicker">{robbing ? 'Kan' : 'Discard'}</span>
        <Tile tile={offered} dora={shownDora.includes(offered)} />
        <span class="call-message"><strong>{discarder}</strong><span>{robbing ? 'declared a kan of' : 'discarded'} the {tileWords(offered)}</span></span>
      </div>
    {/if}
    <div class="call-options" data-moves>
      {#each callChoices as choice, index (index)}
        <button class="app-control" class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} class:win-call={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}
          aria-label={callLabel(choice, robbing)} disabled={busy || Boolean(failure || saveConflict)} onclick={event => press(event, () => choose(choice))}>
          <span class="call-label">{callLabel(choice, robbing)}</span>
          <span class="call-word">{callLabel(choice, robbing).split(' — ')[0]}</span>
          {#if callTiles(choice, view.pending_discard).length}
            <span class="call-preview" aria-hidden="true">
              {#each callTiles(choice, view.pending_discard) as tile, slot (slot)}<Tile {tile} size="tiny" />{/each}
            </span>
          {/if}
        </button>
      {/each}
    </div>
  </div>
{/if}

<style>
  .prompt { margin: 0; font-size: .9rem; }
  /* A call's staged discard and its buttons take a row each among the
     controls (App), except on a phone held sideways (below). */
  .calls { display: contents; }
  .call-options { display: flex; flex-wrap: wrap; gap: 8px; }
  .call-options button { display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; text-align: left; }
  .call-label { min-width: 0; overflow-wrap: anywhere; }
  /* The call's word alone, such as Chii, which its name begins with, shown
     in place of the name where there is room for no more (below). */
  .call-word { display: none; }
  .call-preview { display: inline-flex; align-items: center; gap: 4px; }
  .confirm-discard { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .confirm-discard > span { font-size: .85rem; }
  @media (max-width: 760.98px) {
    .prompt { font-size: .82rem; }
    .call-options { gap: 6px; }
    .call-options button { font-size: .8rem; padding: 6px 10px; }
  }
  @media (min-width: 640px) and (max-height: 500.98px) and (orientation: landscape) {
    .prompt { font-size: .8rem; }
  }

  .call-stage {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    align-items: center;
    gap: 9px 12px;
    padding: 10px 12px;
    border: 1px solid rgba(216, 161, 42, .45);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(216,161,42,.11), rgba(0,0,0,.23));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    animation: offer-in .24s ease-out both;
  }
  .call-kicker {
    grid-column: 1 / -1;
    margin-bottom: -5px;
    color: var(--gold-text);
    font-size: .68rem;
    font-weight: 750;
    letter-spacing: .14em;
    text-transform: uppercase;
  }
  .call-message { display: grid; min-width: 0; line-height: 1.25; }
  .call-message span { opacity: .78; font-size: .82rem; }
  .call-options { animation: choices-in .22s .06s ease-out both; }
  .call-options button.win-call {
    border-color: #efc45e;
    background: linear-gradient(180deg, #b93b24, #8e2414);
    box-shadow: 0 5px 16px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.16);
  }
  @keyframes offer-in { from { opacity: 0; transform: translateY(5px) scale(.985); } to { opacity: 1; transform: none; } }
  @keyframes choices-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  @media (max-width: 760.98px), (min-width: 640px) and (max-height: 500.98px) and (orientation: landscape) {
    .call-stage { grid-template-columns: auto auto minmax(0,1fr); padding: 9px 10px; }
  }

  /* Held sideways, a phone's screen leaves the hand and a call's buttons
     only the room under the game's bar, so the discard is staged in a card
     no taller than the buttons, which follow it in the same rows, as many
     to a row as fit: the tile, small, with the kicker over who let it go.
     Each button shows its call's word beside the call's tiles. The rest of
     the buttons' names and of the card's words, which the tiles say again,
     are still read out, as is the prompt, whose line the call takes. */
  @media (max-height: 500.98px) and (orientation: landscape) {
    .calls { display: flex; flex-wrap: wrap; gap: 6px; }
    .call-options { display: contents; }
    .call-options button { gap: 6px; min-width: 44px; justify-content: center; padding: 6px 8px; animation: choices-in .22s .06s ease-out both; }
    .call-label { display: none; }
    .call-word { display: inline; }
    .call-preview { --tile-width: 38px; gap: 3px; }
    .call-stage { --tile-width: 26px; grid-template-columns: auto auto; gap: 0 8px; padding: 3px 10px; }
    .call-kicker { grid-area: 1 / 2; margin: 0; }
    .call-stage > :global(.tile) { grid-area: 1 / 1 / 3; }
    .call-message { grid-area: 2 / 2; }
    .prompt.staged, .call-message span { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  }
</style>
