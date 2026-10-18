<script>
  import { tick } from 'svelte';
  import Tile from './Tile.svelte';
  import Melds from './Melds.svelte';
  import { tilesFor, noteFor, isCircumstance } from './yaku.js';
  import { RESULT_SHEET, signed, winValue } from './ui.js';
  import { SEAT_NAMES as NAMES } from './tiles.js';

  let {
    outcome,
    seats = [],
    onnext,
    // Opens the hand's review, and answers false when it could not be
    // built; reviewFailure then says why.
    onreview,
    reviewed = false,
    reviewFailure = '',
    onlog,
    ongame,
    // Opens the review of the whole game from the final hand.
    ongamereview,
    // Whether that review is open, from this sheet or from the standings.
    gameReviewOpen = false,
    // Whether the keyboard has sent focus on to the standings as the game
    // ended, pressing the last hand's Next hand (App).
    standingsFocused = false,
    finalHand = false,
    dora = [],
    bets = 0,
    busy = false,
    hints = true,
    // What the first seat, the one the view belongs to, is called: the
    // person playing, or in Watch the agent being followed.
    playerLabel = 'You',
  } = $props();

  let minimized = $state(false);
  // Which yaku the reader is asking about, and the tiles that make it. A
  // yaku about how the hand was won lights nothing, and says why instead.
  let asking = $state(null);
  let shape = $derived(asking ? new Set(tilesFor(asking.yaku, asking.win)) : null);
  const ask = (win, yaku) => { asking = { win, yaku }; };
  const stopAsking = () => { asking = null; };
  const showing = (win, yaku) => asking?.win === win && asking?.yaku === yaku;
  let primaryWin = $derived(outcome?.wins?.[0] ?? null);
  let primaryValue = $derived(primaryWin ? winValue(primaryWin) : '');

  function reveal(node) {
    const id = requestAnimationFrame(() => {
      const phone = matchMedia(RESULT_SHEET).matches;
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!phone) node.scrollIntoView({ block: 'nearest', behavior: reduced ? 'instant' : 'smooth' });
      node.focus({ preventScroll: true });
    });
    return { destroy: () => cancelAnimationFrame(id) };
  }

  // The hand's review opens below the result, and on a phone this sheet steps
  // aside for it and for the table and the hand that it covers. The button
  // goes once the review is open, so opening it hands focus on to the review
  // (App) rather than leaving it to drop to the page. The press goes along:
  // on a phone, whether the keyboard made it decides whether the page moves
  // to the review. A review that could not be built opens nothing, so the
  // sheet stays, and the button, with focus, and the line after the buttons
  // says why. Where the result is not a sheet, that line may fall below the
  // screen, so it is brought into view; on a phone it stays at the foot of
  // the sheet with the buttons.
  let failureLine = $state(null);
  function reviewTable(event) {
    if (onreview?.(event) === false) {
      void tick().then(() => { if (!matchMedia(RESULT_SHEET).matches) failureLine?.scrollIntoView({ block: 'nearest' }); });
      return;
    }
    minimized = true;
  }

  // The chip goes as it brings the result back, so focus goes with it to the
  // result, as when the result first appeared, rather than to the page.
  let sheet = $state(null);
  async function showResult() {
    minimized = false;
    await tick();
    sheet?.focus({ preventScroll: true });
  }

  // The whole game's review comes back to the standings, so on a phone this
  // sheet steps aside for it as it does for the hand's review, whichever
  // button opened it: its own, or the standings' own, which the sheet covers
  // but a keyboard or a screen reader still reaches. The standings are then
  // in view, and not under the sheet, either way. It steps aside for them as
  // well when the keyboard has sent focus on to them as the game ended, so
  // that focus is not hidden under it; a tap leaves the result up, as it was.
  $effect(() => { if (gameReviewOpen || standingsFocused) minimized = true; });
</script>

{#if minimized}
  <button class="result-chip" onclick={showResult}>Show result · {outcome.line}</button>
{/if}

<section class="screen" class:minimized aria-label="how the hand ended" tabindex="-1" bind:this={sheet} use:reveal>
  <header class="result-header">
    <div class="result-title">
      <span class="eyebrow">Hand result</span>
      <h2>{outcome.line}</h2>
    </div>
    {#if primaryWin}
      <div class="hero-score" role="img" aria-label={`${NAMES[primaryWin.seat]} ${primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}, ${primaryWin.limit ?? `${primaryWin.han} han`}, ${primaryValue}`}>
        <span>{NAMES[primaryWin.seat]} · {primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}</span>
        <strong>{primaryWin.limit ? primaryWin.limit.toUpperCase() : `${primaryWin.han} HAN`}</strong>
        <b>{primaryValue}</b>
      </div>
    {/if}
    {#if outcome.kind === 'draw' && outcome.tenpai.length}
      <p class="waiting">Waiting: {outcome.tenpai.map((seat) => NAMES[seat]).join(', ')}</p>
    {/if}
    {#if outcome.kind === 'draw' && bets > 0}
      <p class="waiting">{bets} riichi bet{bets > 1 ? 's' : ''} stay{bets > 1 ? '' : 's'} on the table for the next winner.</p>
    {/if}
  </header>

  {#each outcome.wins as win (win.seat)}
    <article class="win">
      <div class="win-heading">
        <h3>{NAMES[win.seat]} — {win.by === 'self-draw' ? 'Tsumo' : 'Ron'}</h3>
        <span>{win.han} han{#if win.fu} · {win.fu} minipoints{/if}</span>
      </div>
      <div class="tiles">
        {#each win.hand as tile, index (tile + index)}
          <Tile {tile} size="small" dora={hints && (win.dora_types ?? dora).includes(tile)}
            inShape={asking?.win === win && Boolean(shape?.has(`hand:${index}`))} />
        {/each}
        <span class="gap"></span>
        <span class="winning"><Tile tile={win.winning_tile} size="small" dora={hints && (win.dora_types ?? dora).includes(win.winning_tile)}
          inShape={asking?.win === win && Boolean(shape?.has('won'))} /></span>
        {#if win.melds.length}
          <span class="gap"></span>
          <Melds melds={win.melds} size="small" dora={hints ? (win.dora_types ?? dora) : []}
            shape={asking?.win === win ? shape : null} />
        {/if}
      </div>

      <div class="bonus-indicators">
        <div class="indicator-row" role="group" aria-label="winning hand dora indicators">
          <span>Dora indicators</span>
          <div class="indicator-tiles">{#each win.dora_indicators ?? [] as tile, slot (slot)}<Tile {tile} size="small" />{/each}</div>
        </div>
        {#if win.ura_indicators?.length}
          <div class="indicator-row ura" role="group" aria-label="winning hand ura-dora indicators">
            <span>Ura-dora indicators</span>
            <div class="indicator-tiles">{#each win.ura_indicators as tile, slot (slot)}<Tile {tile} size="small" />{/each}</div>
          </div>
          <details class="indicator-help">
            <summary>ⓘ Ura-dora</summary>
            <p>Revealed after a riichi win. Each indicator points to the next tile and repeated indicators count again.{win.limit === 'yakuman' ? ' Dora do not add han to yakuman.' : ''}</p>
          </details>
        {/if}
      </div>

      <div class="working">
        <ul class="yaku">
          {#each win.yaku as yaku, index (yaku.id ?? `${yaku.name}:${index}`)}
            <li>
              <button type="button" class="yaku-name" class:asking={showing(win, yaku)}
                aria-describedby={showing(win, yaku) ? 'yaku-note' : undefined}
                onmouseenter={() => ask(win, yaku)} onmouseleave={stopAsking}
                onfocus={() => ask(win, yaku)} onblur={stopAsking}
                onclick={() => ask(win, yaku)}>
                <span>{yaku.name}</span><b>{yaku.han}</b>
              </button>
              {#if showing(win, yaku)}
                <p class="yaku-note" id="yaku-note" role="status">
                  {noteFor(yaku.name)}
                  {#if shape?.size}<span class="lit">The tiles that make it are lifted above.</span>
                  {:else if isCircumstance(yaku.name)}<span class="lit">It is about how the hand was won, so no tile makes it.</span>
                  {:else}<span class="lit">Tile attribution is unavailable for this saved result.</span>{/if}
                </p>
              {/if}
            </li>
          {/each}
          {#if win.dora || win.ura_indicators?.length}<li class="dora-count"><span>Dora</span><b>{win.dora - (win.ura_dora ?? 0)}</b></li>{/if}
          {#if win.ura_indicators?.length}<li class="ura-count"><span>Ura-dora</span><b>{win.ura_dora ?? 0}</b></li>{/if}
        </ul>
        <p class="total"><b>{win.han}</b> han{#if win.fu}<span class="fu">, {win.fu} minipoints</span>{/if}{#if win.limit}<span class="limit">{win.limit}</span>{/if}</p>
        <p class="payment">{win.payment}{#if win.bets > 0}, and {win.bets.toLocaleString()} in bets from the table{/if}</p>
      </div>
    </article>
  {/each}

  <table class="changes">
    <thead><tr><th scope="col">Player</th><th scope="col">Change</th><th scope="col">Points</th></tr></thead>
    <tbody>
      {#each seats as seat, index (seat.seat)}
        <tr><th scope="row">{index === 0 ? playerLabel : NAMES[seat.seat]}</th>
          <td class:up={outcome.changes[index] > 0} class:down={outcome.changes[index] < 0}>{signed(outcome.changes[index] ?? 0)}</td>
          <td class="after">{seat.score.toLocaleString()}</td></tr>
      {/each}
    </tbody>
  </table>

  <div class="buttons">
    {#if finalHand}<span class="final-caption">Final hand</span>
    {:else}<button disabled={busy} class="primary" onclick={onnext}>Next hand</button>{/if}
    {#if onreview && !reviewed}<button class="quiet" onclick={reviewTable}>{#if finalHand}Review <span class="qualifier">final</span> hand{:else}View table / my hand{/if}</button>{/if}
    {#if onlog}<button class="quiet" onclick={onlog} title="The hand as an mjai event log, which replayers and other riichi programs read">Save <span class="qualifier">{finalHand ? 'final' : 'this'}</span> hand</button>{/if}
    <!-- At the end the standings offer the whole game, its review and its
         file, close and all. On a phone this sheet covers them, so it offers
         both there too. -->
    {#if ongamereview && finalHand}<button class="quiet whole-game" data-review-game onclick={() => ongamereview()}>Review <span class="whole">whole</span> game</button>{/if}
    {#if ongame && finalHand}<button class="quiet whole-game" data-save-game onclick={ongame} title="Every hand of this game from East 1 as one mjai event log">Save <span class="whole">whole</span> game</button>
    {:else if ongame}<button class="quiet" data-save-game onclick={ongame} title="Every finished hand of this game from East 1, this one included, as one mjai event log">Save game <span class="so-far">so far</span></button>{/if}
    <!-- Why the hand's review could not be built, said where its button was
         pressed. A screen reader hears a status line only once it knows it,
         so the line waits here, empty and out of sight, for its words. -->
    {#if onreview}<p class="review-failure" class:visually-hidden={!reviewFailure} data-review-failure role="status"
      bind:this={failureLine}>{reviewFailure}</p>{/if}
  </div>
</section>

<style>
  .result-chip { display: none; }
  .screen {
    display: grid;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
    gap: 14px;
    padding: 18px 20px;
    border: 1px solid rgba(216,161,42,.42);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(2,31,20,.62), rgba(0,0,0,.31));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 14px 34px rgba(0,0,0,.12);
  }
  .result-header { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 10px 20px; }
  .result-title { min-width: min(100%, 260px); }
  .eyebrow { display: block; margin-bottom: 3px; color: var(--gold); font-size: .64rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  h2 { margin: 0; font-size: 1.12rem; font-weight: 650; }
  h3 { margin: 0; font-size: .9rem; }
  .hero-score { display: grid; justify-items: end; line-height: 1.08; }
  .hero-score span { font-size: .72rem; opacity: .7; }
  .hero-score strong { margin-top: 3px; color: var(--gold); font-size: 1.02rem; letter-spacing: .08em; }
  .hero-score b { margin-top: 3px; font-size: 1.75rem; font-variant-numeric: tabular-nums; }
  .waiting { flex-basis: 100%; margin: 0; font-size: .83rem; opacity: .78; }
  .win { display: grid; gap: 10px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
  .win-heading { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 12px; }
  .win-heading span { font-size: .72rem; opacity: .7; }
  .tiles { display: flex; align-items: flex-end; gap: 2px; flex-wrap: wrap; }
  .gap { width: 12px; }
  .winning { display: inline-flex; border-radius: 6px; box-shadow: 0 0 0 2px var(--gold), 0 0 14px rgba(216,161,42,.18); }
  .bonus-indicators { display: flex; flex-wrap: wrap; gap: 10px 22px; min-width: 0; }
  .indicator-row { display: grid; gap: 6px; min-width: 0; font-size: .78rem; }
  .indicator-tiles { display: flex; flex-wrap: wrap; gap: 5px; }
  .indicator-help { flex-basis: 100%; font-size: .76rem; opacity: .82; }
  .indicator-help summary { min-height: 28px; padding: 2px 0; color: var(--gold); }
  .indicator-help p { margin: 2px 0 0; max-width: 62ch; }
  .final-caption { align-self: center; font-size: .85rem; }
  .working { display: grid; gap: 4px; }
  .yaku { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,210px),1fr)); gap: 1px 18px; font-size: .88rem; }
  .yaku li { display: block; padding: 1px 0; border-bottom: 1px dotted rgba(255,255,255,.16); }
  /* The dora rows are counts, not controls, so they keep the row shape the
     yaku rows had before their names became buttons: label left, count right. */
  .yaku li.dora-count, .yaku li.ura-count { display: flex; justify-content: space-between; gap: 12px; }
  /* The name is the control: hovering or tabbing to it explains the yaku and
     lifts the tiles that make it. It must not look like a button in a row of
     numbers, so it keeps the row's own shape. */
  .yaku-name { display: flex; width: 100%; justify-content: space-between; gap: 12px; padding: 1px 0;
    border: 0; background: none; color: inherit; font: inherit; text-align: left; cursor: help; }
  .yaku-name:hover, .yaku-name.asking { color: var(--gold, #d8a12a); }
  .yaku-name:focus-visible { outline: 2px solid var(--gold, #d8a12a); outline-offset: 2px; border-radius: 4px; }
  .yaku-note { margin: 2px 0 6px; font-size: .82rem; line-height: 1.35; opacity: .9; }
  .yaku-note .lit { display: block; opacity: .7; font-size: .78rem; }
  .total { margin: 4px 0 0; font-size: 1rem; }
  .fu { opacity: .85; }
  .limit { margin-left: 8px; color: var(--gold); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
  .payment { margin: 0; font-size: .9rem; opacity: .9; }
  .changes { width: 100%; max-width: 100%; table-layout: fixed; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; }
  .changes th { padding: 2px 16px 2px 0; text-align: left; font-weight: 500; opacity: .84; }
  .changes td { padding: 2px 16px 2px 0; }
  .up { color: #7fd1a0; }
  .down { color: var(--warning-text); }
  .after { opacity: .7; }
  .result-header, .win, .tiles, .bonus-indicators, .working, .buttons { min-width: 0; max-width: 100%; box-sizing: border-box; }
  .buttons { display: flex; flex-wrap: wrap; gap: 8px; }
  /* A row of its own below the buttons, kept a little clear of the foot of
     the screen when it is brought into view. */
  .review-failure { flex-basis: 100%; margin: 0; font-size: .82rem; scroll-margin: 8px 0; }
  .buttons button { min-width: 0; max-width: 100%; min-height: 44px; padding: 8px 18px; border: 1px solid var(--button-accent); border-radius: 999px; background: var(--button-accent); color: var(--button-text); font-weight: 650; cursor: pointer; }
  .buttons button.quiet { border-color: rgba(255,255,255,.25); background: transparent; color: inherit; font-weight: 500; }
  /* Beside the standings, which offer the same review and file, they are not
     repeated. */
  .buttons .whole-game { display: none; }

  @media (max-width: 760px) {
    .buttons .whole-game { display: block; }
    .screen {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      width: 100%;
      max-width: 100vw;
      box-sizing: border-box;
      z-index: 50;
      max-height: 88dvh;
      overflow: auto;
      padding: 18px 18px max(14px, env(safe-area-inset-bottom));
      border-width: 1px 0 0;
      border-radius: 22px 22px 0 0;
      background: color-mix(in srgb, var(--felt-deep) 97%, black 3%);
      box-shadow: 0 -100vh 0 100vh rgba(0,0,0,.60), 0 -18px 55px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.05);
      animation: result-up .28s cubic-bezier(.2,.8,.2,1) both;
      overscroll-behavior: contain;
    }
    .screen.minimized { display: none; }
    .result-chip {
      display: block;
      position: fixed;
      right: 10px;
      bottom: max(10px, env(safe-area-inset-bottom));
      z-index: 42;
      max-width: calc(100vw - 20px);
      min-height: 42px;
      padding: 7px 13px;
      border: 1px solid rgba(216,161,42,.65);
      border-radius: 999px;
      background: rgba(8,42,28,.94);
      box-shadow: 0 8px 25px rgba(0,0,0,.35);
      font-size: .78rem;
    }
    .result-header { position: relative; }
    .hero-score { flex-basis: 100%; justify-items: start; padding: 9px 0 3px; border-top: 1px solid rgba(255,255,255,.09); }
    .hero-score b { max-width: 100%; font-size: 2rem; overflow-wrap: anywhere; }
    .changes th, .changes td { min-width: 0; padding-right: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .buttons {
      position: sticky;
      bottom: calc(-1 * max(14px, env(safe-area-inset-bottom)));
      z-index: 2;
      margin: 2px -18px calc(-1 * max(14px, env(safe-area-inset-bottom)));
      padding: 10px 18px max(14px, env(safe-area-inset-bottom));
      background: linear-gradient(180deg, transparent, color-mix(in srgb,var(--felt-deep) 97%,black 3%) 20%);
    }
    .buttons .primary { flex: 1 1 145px; }
    @keyframes result-up { from { opacity: 0; transform: translateY(28px); } to { opacity: 1; transform: none; } }
  }
  /* On a phone the final hand's buttons read "Review hand", "Save hand",
     "Review game" and "Save game", and the other hands' saves "Save hand"
     and "Save game". The full labels take a third row of buttons over the
     result even on the largest phones, and a fourth on the narrowest, as the
     phone's fonts allow, so the short ones start above the largest phones. */
  @media (max-width: 480px) {
    .buttons button { padding-inline: 12px; }
    .qualifier, .so-far, .whole { display: none; }
  }

  @media (prefers-reduced-motion: reduce) { .screen { animation: none; } }
</style>
