<script>
  import Tile from './Tile.svelte';
  import Melds from './Melds.svelte';

  let {
    outcome,
    seats = [],
    onnext,
    ongame,
    onreview,
    reviewed = false,
    onlog,
    gameOver = false,
    finalHand = false,
    dora = [],
    bets = 0,
    busy = false,
    hints = true,
  } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };
  let minimized = $state(false);
  let primaryWin = $derived(outcome?.wins?.[0] ?? null);

  function reveal(node) {
    const id = requestAnimationFrame(() => {
      const phone = matchMedia('(max-width: 760px)').matches;
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!phone) node.scrollIntoView({ block: 'nearest', behavior: reduced ? 'instant' : 'smooth' });
      node.focus({ preventScroll: true });
    });
    return { destroy: () => cancelAnimationFrame(id) };
  }

  function signed(value) {
    if (value === 0) return '0';
    return value > 0 ? `+${value.toLocaleString()}` : value.toLocaleString();
  }

  function heroPayment(payment = '') {
    return payment.split(',')[0]
      .replace(/\b\d{4,6}\b/g, value => Number(value).toLocaleString())
      .replace(/\s+from.*$/, '');
  }

  function reviewTable() {
    onreview?.();
    minimized = true;
  }
</script>

{#if minimized}
  <button class="result-chip" onclick={() => minimized = false}>Show result · {outcome.line}</button>
{/if}

<section class="screen" class:minimized aria-label="how the hand ended" tabindex="-1" use:reveal>
  <header class="result-header">
    <div class="result-title">
      <span class="eyebrow">Hand result</span>
      <h2>{outcome.line}</h2>
    </div>
    {#if primaryWin}
      <div class="hero-score" aria-label={`${NAMES[primaryWin.seat]} ${primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}, ${primaryWin.limit ?? `${primaryWin.han} han`}, ${heroPayment(primaryWin.payment)}`}>
        <span>{NAMES[primaryWin.seat]} · {primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}</span>
        <strong>{primaryWin.limit ? primaryWin.limit.toUpperCase() : `${primaryWin.han} HAN`}</strong>
        <b>{heroPayment(primaryWin.payment)}</b>
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
          <Tile {tile} size="small" dora={hints && (win.dora_types ?? dora).includes(tile)} />
        {/each}
        <span class="gap"></span>
        <span class="winning"><Tile tile={win.winning_tile} size="small" dora={hints && (win.dora_types ?? dora).includes(win.winning_tile)} /></span>
        {#if win.melds.length}
          <span class="gap"></span>
          <Melds melds={win.melds} size="small" dora={hints ? (win.dora_types ?? dora) : []} />
        {/if}
      </div>

      <div class="bonus-indicators">
        <div class="indicator-row" aria-label="winning hand dora indicators">
          <span>Dora indicators</span>
          <div class="indicator-tiles">{#each win.dora_indicators ?? [] as tile, slot (slot)}<Tile {tile} size="small" />{/each}</div>
        </div>
        {#if win.ura_indicators?.length}
          <div class="indicator-row ura" aria-label="winning hand ura-dora indicators">
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
          {#each win.yaku as yaku (yaku.name)}<li><span>{yaku.name}</span><b>{yaku.han}</b></li>{/each}
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
        <tr><th scope="row">{index === 0 ? 'You' : NAMES[seat.seat]}</th>
          <td class:up={outcome.changes[index] > 0} class:down={outcome.changes[index] < 0}>{signed(outcome.changes[index] ?? 0)}</td>
          <td class="after">{seat.score.toLocaleString()}</td></tr>
      {/each}
    </tbody>
  </table>

  <div class="buttons">
    {#if finalHand}<span class="final-caption">Final hand</span>
    {:else if gameOver}<button disabled={busy} class="primary" onclick={ongame}>Play again</button>
    {:else}<button disabled={busy} class="primary" onclick={onnext}>Next hand</button>{/if}
    {#if onreview && !reviewed}<button class="quiet" onclick={reviewTable}>{finalHand ? 'Review final hand' : 'View table / my hand'}</button>{/if}
    {#if onlog}<button class="quiet" onclick={onlog} title="The hand as an mjai event log, which replayers and other riichi programs read">{finalHand ? 'Save final hand' : 'Save this hand'}</button>{/if}
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
  .yaku li { display: flex; justify-content: space-between; gap: 12px; padding: 1px 0; border-bottom: 1px dotted rgba(255,255,255,.16); }
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
  .buttons button { min-width: 0; max-width: 100%; min-height: 44px; padding: 8px 18px; border: 1px solid var(--button-accent); border-radius: 999px; background: var(--button-accent); color: var(--button-text); font-weight: 650; cursor: pointer; }
  .buttons button.quiet { border-color: rgba(255,255,255,.25); background: transparent; color: inherit; font-weight: 500; }

  @media (max-width: 760px) {
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

  @media (prefers-reduced-motion: reduce) { .screen { animation: none; } }
</style>
