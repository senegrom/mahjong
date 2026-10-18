<script>
  import { placeLabel, signed } from './ui.js';
  import { SEAT_NAMES as NAMES } from './tiles.js';
  /**
   * Where everybody finished. A game is played for placement rather than
   * points, so the places lead and the arithmetic follows: what was on the
   * table, the bonus the place earned or cost, and what the game came to.
   * `onsave`, where given, saves the whole game as one mjai log, and
   * `onreview` opens the review of every decision the player made in it.
   * `playerLabel` is what the row the engine marks as `you` is called: the
   * person playing, or in Watch the agent being followed. `heading` is bound
   * so that focus can go to it as the game ends: the last hand's Next hand,
   * which goes as it brings the standings, hands focus on to it (App).
   */
  let { standings = [], onagain, onsave, onreview, playerLabel = 'You', heading = $bindable(null) } = $props();

  let yours = $derived(standings.find((row) => row.you));
</script>

<section class="standings" aria-label="final standings">
  <h2 tabindex="-1" bind:this={heading}>
    {#if yours}
      {playerLabel} finished {placeLabel(yours).toLowerCase()}
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
          <td>{row.you ? playerLabel : NAMES[row.seat]}</td>
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

  <div class="buttons">
    <button class="primary" onclick={onagain}>Play again</button>
    {#if onreview}<button class="quiet" data-review-game onclick={onreview}>Review whole game</button>{/if}
    {#if onsave}<button class="quiet" data-save-game onclick={onsave} title="Every hand of this game from East 1 as one mjai event log, which replayers and other riichi programs read">Save whole game</button>{/if}
  </div>
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

  /* The heading takes focus as the game ends, and a press by keyboard, or
     any press where the result is not a sheet, scrolls to it with as little
     movement as will show it and the focus ring drawn just outside it for a
     keyboard (App). So it lands a ring's width and a little from
     the edge of the screen, below the game's bar where that stays at the top
     of the screen (AppSettings), and on a phone clear of the result's chip at
     the foot of the screen (ScoreScreen). */
  h2:focus { outline: none; }
  h2:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
  h2 { scroll-margin: 8px 0; }
  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    h2 { scroll-margin-top: 72px; }
  }
  @media (max-width: 760px) {
    h2 { scroll-margin-bottom: calc(72px + env(safe-area-inset-bottom, 0px)); }
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

  .buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-width: 100%;
  }

  button {
    min-height: 44px;
    max-width: 100%;
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid var(--button-accent);
    background: var(--button-accent);
    color: var(--button-text);
    font-weight: 600;
    cursor: pointer;
  }

  button.quiet {
    border-color: rgba(255, 255, 255, 0.25);
    background: transparent;
    color: inherit;
    font-weight: 500;
  }
</style>
