<script>
  import { placeLabel } from './ui.js';
  /**
   * Where everybody finished. A game is played for placement rather than
   * points, so the places lead and the arithmetic follows: what was on the
   * table, the bonus the place earned or cost, and what the game came to.
   */
  let { standings = [], onagain } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };

  function signed(value) {
    if (value === 0) return '0';
    return value > 0 ? `+${value.toLocaleString()}` : value.toLocaleString();
  }

  let yours = $derived(standings.find((row) => row.you));
</script>

<section class="standings" aria-label="final standings">
  <h2>
    {#if yours}
      You finished {placeLabel(yours).toLowerCase()}
    {:else}
      The game is over
    {/if}
  </h2>

  <table>
    <thead>
      <tr>
        <th scope="col">Place</th>
        <th scope="col">Seat</th>
        <th scope="col" class="number">Points</th>
        <th
          scope="col"
          class="number"
          title="The winner bonus for the place, and any riichi bets still on the table, which go to the leader"
        >
          Bonus
        </th>
        <th scope="col" class="number">Result</th>
      </tr>
    </thead>
    <tbody>
      {#each standings as row (row.player)}
        <tr class:you={row.you}>
          <td>{placeLabel(row)}</td>
          <td>{row.you ? 'You' : NAMES[row.seat]}</td>
          <td class="number">{row.score.toLocaleString()}</td>
          <td class="number" class:up={row.uma > 0} class:down={row.uma < 0}>
            {signed(row.uma)}
          </td>
          <td class="number" class:up={row.total > 0} class:down={row.total < 0}>
            {signed(row.total)}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>

  <button class="primary" onclick={onagain}>Play again</button>
</section>

<style>
  .standings {
    display: grid;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    gap: 14px;
    padding: 16px 18px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.32);
    border: 1px solid rgba(216, 161, 42, 0.4);
    justify-items: start;
    min-width: 0;
    overflow-x: auto;
  }

  h2 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 600;
  }

  table {
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
    font-size: 0.92rem;
  }

  th {
    text-align: left;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.65;
    font-weight: 600;
    padding: 0 18px 4px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.16);
  }

  td {
    padding: 4px 18px 4px 0;
  }

  .number {
    text-align: right;
  }

  .you td {
    font-weight: 600;
  }

  .you {
    color: var(--gold);
  }

  .up {
    color: #7fd1a0;
  }

  .down {
    color: var(--warning-text);
  }

  /* Keep all five columns visible on phones; scrolling the whole panel hid
     the result column even when the document itself had no overflow. */
  @media (max-width: 600px) {
    .standings { padding: 12px 10px; }
    table { width: 100%; max-width: 100%; table-layout: fixed; font-size: .82rem; }
    th { font-size: .62rem; letter-spacing: .02em; }
    th, td { padding-right: 4px; }
    th:last-child, td:last-child { padding-right: 0; }
    .number { min-width: 0; overflow: hidden; text-overflow: clip; white-space: nowrap; }
  }

  button {
    min-height: 44px;
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid var(--button-accent);
    background: var(--button-accent);
    color: var(--button-text);
    font-weight: 600;
    cursor: pointer;
  }
</style>
