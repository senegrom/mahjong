<script>
  import Tile from './Tile.svelte';
  import Melds from './Melds.svelte';

  /**
   * How the hand ended: the hand that won, why it scored what it did, and
   * what it moved. At a table this is the moment everyone looks at, so it
   * shows the working rather than a number.
   */
  let {
    outcome,
    seats = [],
    onnext,
    ongame,
    onreview,
    reviewed = false,
    onlog,
    gameOver = false,
    dora = [],
    bets = 0,
  } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };

  // The table is tall enough that the result can land below the fold.
  function reveal(node) {
    node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function signed(value) {
    if (value === 0) return '0';
    return value > 0 ? `+${value.toLocaleString()}` : value.toLocaleString();
  }
</script>

<section class="screen" aria-label="how the hand ended" use:reveal>
  <header>
    <h2>{outcome.line}</h2>
    {#if outcome.kind === 'draw' && outcome.tenpai.length}
      <p class="waiting">
        Waiting: {outcome.tenpai.map((seat) => NAMES[seat]).join(', ')}
      </p>
    {/if}
    {#if outcome.kind === 'draw' && bets > 0}
      <p class="waiting">
        {bets} riichi bet{bets > 1 ? 's' : ''} stay{bets > 1 ? '' : 's'} on the table for the next
        winner, which is why a player who declared shows less than they were paid.
      </p>
    {/if}
  </header>

  {#each outcome.wins as win (win.seat)}
    <article class="win">
      <div class="tiles">
        {#each win.hand as tile, index (tile + index)}
          <Tile {tile} size="small" dora={dora.includes(tile)} />
        {/each}
        <span class="gap"></span>
        <span class="winning">
          <Tile tile={win.winning_tile} size="small" dora={dora.includes(win.winning_tile)} />
        </span>
        {#if win.melds.length}
          <span class="gap"></span>
          <Melds melds={win.melds} size="small" {dora} />
        {/if}
      </div>

      <div class="working">
        <ul class="yaku">
          {#each win.yaku as yaku (yaku.name)}
            <li><span>{yaku.name}</span><b>{yaku.han}</b></li>
          {/each}
          {#if win.dora}
            <li><span>Dora</span><b>{win.dora}</b></li>
          {/if}
        </ul>
        <p class="total">
          <b>{win.han}</b> han{#if win.fu}<span class="fu">, {win.fu} minipoints</span
            >{/if}{#if win.limit}<span class="limit">{win.limit}</span>{/if}
        </p>
        <p class="payment">
          {win.payment}{#if win.bets > 0}, and {win.bets.toLocaleString()} in bets from the
            table{/if}
        </p>
      </div>
    </article>
  {/each}

  <table class="changes">
    <tbody>
      {#each seats as seat, index (seat.seat)}
        <tr>
          <th scope="row">{index === 0 ? 'You' : NAMES[seat.seat]}</th>
          <td class:up={outcome.changes[index] > 0} class:down={outcome.changes[index] < 0}>
            {signed(outcome.changes[index] ?? 0)}
          </td>
          <td class="after">{seat.score.toLocaleString()}</td>
        </tr>
      {/each}
    </tbody>
  </table>

  <div class="buttons">
    {#if gameOver}
      <button class="primary" onclick={ongame}>Play again</button>
    {:else}
      <button class="primary" onclick={onnext}>Next hand</button>
    {/if}
    {#if onreview && !reviewed}
      <button class="quiet" onclick={onreview}>Look at my hand again</button>
    {/if}
    {#if onlog}
      <button
        class="quiet"
        onclick={onlog}
        title="The hand as an mjai event log, which replayers and other riichi programs read"
      >
        Save this hand
      </button>
    {/if}
  </div>
</section>

<style>
  .screen {
    display: grid;
    gap: 14px;
    padding: 16px 18px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(216, 161, 42, 0.35);
  }

  h2 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
  }

  .waiting {
    margin: 4px 0 0;
    font-size: 0.85rem;
    opacity: 0.8;
  }

  .win {
    display: grid;
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.12);
  }

  .tiles {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    flex-wrap: wrap;
  }

  .gap {
    width: 12px;
  }

  .winning {
    display: inline-flex;
    border-radius: 6px;
    box-shadow: 0 0 0 2px var(--gold);
  }

  .working {
    display: grid;
    gap: 4px;
  }

  .yaku {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1px 18px;
    font-size: 0.9rem;
  }

  .yaku li {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    border-bottom: 1px dotted rgba(255, 255, 255, 0.16);
    padding: 1px 0;
  }

  .total {
    margin: 4px 0 0;
    font-size: 1rem;
  }

  .fu {
    opacity: 0.85;
  }

  .limit {
    margin-left: 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    color: var(--gold);
  }

  .payment {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
  }

  .changes {
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
    font-size: 0.9rem;
  }

  .changes th {
    text-align: left;
    font-weight: 500;
    padding: 2px 16px 2px 0;
    opacity: 0.85;
  }

  .changes td {
    padding: 2px 16px 2px 0;
  }

  .up {
    color: #7fd1a0;
  }

  .down {
    color: var(--accent);
  }

  .after {
    opacity: 0.7;
  }

  .buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .buttons button {
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid var(--accent);
    background: var(--accent);
    font-weight: 600;
    cursor: pointer;
  }

  .buttons button.quiet {
    background: transparent;
    border-color: rgba(255, 255, 255, 0.25);
    color: inherit;
    font-weight: 500;
  }
</style>
