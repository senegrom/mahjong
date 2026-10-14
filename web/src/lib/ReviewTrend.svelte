<script>
  import { TREND_WINDOW } from './review-trend.js';

  /**
   * How much of this game matched Trained AI, against the games reviewed
   * before it in this browser: the figure, its change from the earlier games'
   * average, and the last dozen games as a line, with a table of the same.
   * Matching Trained AI is not the same as playing well, so the change is
   * stated, not coloured as good or bad.
   */
  let { figure, complete = false, games = [], current = '' } = $props();
  const SHOWN = 12;
  // The line's box: a 4px end-dot and its 2px ring fit inside the margin.
  const WIDTH = 240, HEIGHT = 56, MARGIN = 7;

  const percent = share => `${Math.round(share * 100)}%`;
  const shareOf = list => {
    const decisions = list.reduce((sum, game) => sum + game.decisions, 0);
    return decisions ? list.reduce((sum, game) => sum + game.matched, 0) / decisions : null;
  };
  const when = at => new Date(at).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

  let earlier = $derived(games.filter(game => game.id !== current).slice(-TREND_WINDOW));
  let earlierShare = $derived(shareOf(earlier));
  let change = $derived(complete && figure.share != null && earlierShare != null
    ? Math.round(figure.share * 100) - Math.round(earlierShare * 100) : null);
  let shown = $derived(games.slice(-SHOWN));
  let points = $derived(shown.map((game, index) => ({
    game,
    x: shown.length === 1 ? WIDTH / 2 : MARGIN + index * (WIDTH - 2 * MARGIN) / (shown.length - 1),
    y: MARGIN + (1 - game.matched / game.decisions) * (HEIGHT - 2 * MARGIN),
  })));
  let summary = $derived(shown.length
    ? `Matched Trained AI in your last ${shown.length} reviewed game${shown.length === 1 ? '' : 's'}, on a scale of 0 to 100%: from ${percent(shown[0].matched / shown[0].decisions)} to ${percent(shown.at(-1).matched / shown.at(-1).decisions)}`
    : '');
</script>

<div class="trend" data-trend>
  <p class="label">Matched Trained AI</p>
  <p class="value">{figure.share == null ? '—' : percent(figure.share)}{#if !complete && figure.share != null}<span class="so-far"> so far</span>{/if}</p>
  <p class="count">{figure.matched} of {figure.decisions} decision{figure.decisions === 1 ? '' : 's'}{complete ? '' : ' in the hands reviewed'}. Moves with only one choice are left out.</p>
  {#if change !== null}
    <p class="change">{change === 0 ? 'The same as' : `${change > 0 ? '+' : '−'}${Math.abs(change)} points against`} your {earlier.length === 1 ? 'previous game' : `previous ${earlier.length} games`} ({percent(earlierShare)})</p>
  {:else if !complete && earlier.length}
    <p class="change">This game joins your trend once every hand is reviewed.</p>
  {/if}
  {#if points.length >= 2}
    <svg class="line" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label={summary}>
      <line class="base" x1={MARGIN} x2={WIDTH - MARGIN} y1={HEIGHT - MARGIN} y2={HEIGHT - MARGIN} />
      <polyline points={points.map(point => `${point.x},${point.y}`).join(' ')} />
      {#each points as point (point.game.id)}
        {#if point.game.id === current}<circle class="now" cx={point.x} cy={point.y} r="4" />{/if}
        <circle class="hit" cx={point.x} cy={point.y} r="12"><title>{when(point.game.at)}: {percent(point.game.matched / point.game.decisions)}</title></circle>
      {/each}
    </svg>
  {/if}
  {#if shown.length}
    <details class="games">
      <summary>Your reviewed games ({shown.length})</summary>
      <table>
        <thead><tr><th scope="col">Reviewed</th><th scope="col">Matched</th><th scope="col">Decisions</th></tr></thead>
        <tbody>
          {#each [...shown].reverse() as game (game.id)}
            <tr class:now={game.id === current}><td>{when(game.at)}{game.id === current ? ' (this game)' : ''}</td>
              <td>{percent(game.matched / game.decisions)}</td><td>{game.decisions}</td></tr>
          {/each}
        </tbody>
      </table>
    </details>
  {/if}
</div>

<style>
  .trend { display: grid; gap: 4px; padding: 12px 14px; border-radius: 12px; background: #0003; border: 1px solid #ffffff1f; min-width: 0; }
  p { margin: 0; }
  .label { font-size: .8rem; opacity: .8; }
  .value { font-size: 2rem; font-weight: 600; line-height: 1.1; }
  .so-far { font-size: .9rem; font-weight: 400; opacity: .8; }
  .count, .change { font-size: .8rem; opacity: .85; }
  .line { width: 100%; max-width: 240px; height: auto; margin-top: 4px; overflow: visible; }
  .base { stroke: #ffffff26; stroke-width: 1; }
  polyline { fill: none; stroke: color-mix(in srgb, var(--ivory) 50%, transparent); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .now { fill: var(--gold); stroke: var(--felt-deep); stroke-width: 2; }
  .hit { fill: transparent; }
  .games { font-size: .8rem; }
  .games summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th { text-align: left; font-weight: 500; opacity: .7; padding: 2px 14px 2px 0; }
  td { padding: 2px 14px 2px 0; }
  tr.now td { font-weight: 600; }
</style>
