<script>
  import { untrack } from 'svelte';
  import { settle_physical } from '../wasm/riichi.js';
  import { WINDS } from './agents.js';
  import { setTiles } from './guided-game.js';
  import TileEntry from './TileEntry.svelte';
  import Tile from './Tile.svelte';
  import Melds from './Melds.svelte';

  /** Physical facts consumed by the Rust settlement adapter.
   * @typedef {object} PhysicalSettlementInput
   * @property {boolean} confirmed_no_furiten
   * @property {number[]} winners
   * @property {string[][]} hands
   * @property {string | null} winning_tile
   * @property {string[]} ura
   * @property {boolean[]} tenpai
   */

  // Keep the public `state` prop without shadowing Svelte's $state rune.
  let { state: gameState, onsettle, onnext } = $props();
  let ending = $derived(gameState.ending);
  let p = $derived(ending.position);
  let selected = $state(untrack(() => WINDS.map((_, i) => ending.winners.includes(i))));
  let tenpai = $state(untrack(() => p.players.map(player => player.riichi !== 'none')));
  let hands = $state([[], [], [], []]);
  let winning = $state([]), ura = $state([]);
  let confirmedNoFuriten = $state(false);
  let unknownRon = $derived(ending.kind === 'ron' && p.players.some((player, i) => selected[i] && !player.hand.length));
  let preview = $state(null), failure = $state('');
  let draw = $derived(ending.kind === 'draw');
  let needsUra = $derived(!draw && p.players.some((player, i) => selected[i] && player.riichi !== 'none'));
  /** @type {PhysicalSettlementInput} */
  let input = $derived({
    confirmed_no_furiten: confirmedNoFuriten,
    winners: draw ? [] : selected.flatMap((yes, i) => yes ? [i] : []),
    hands: hands.map((tiles, i) => !p.players[i].hand.length && (draw ? tenpai[i] : selected[i]) ? tiles : []),
    winning_tile: ending.kind === 'tsumo' && p.phase === 'draw' ? winning[0] ?? null : null,
    ura: needsUra ? ura : [],
    tenpai: draw ? tenpai : [false, false, false, false],
  });
  let inputKey = $derived(JSON.stringify(input));
  let result = $derived(gameState.settlement ?? preview);
  $effect(() => { inputKey; preview = null; failure = ''; });
  function calculate() {
    try { preview = settle_physical(JSON.parse(JSON.stringify(ending)), JSON.parse(inputKey)); failure = ''; }
    catch (error) { failure = error.message ?? String(error); preview = null; }
  }
  const signed = value => `${value >= 0 ? '+' : ''}${value.toLocaleString()}`;
  const viewMelds = player => player.melds.map(m => ({ kind: m.kind, tiles: setTiles(m), from: ['self', 'right', 'across', 'left'][m.from] }));
</script>

<section class="guided-result" aria-label="Guided hand settlement">
  {#if !gameState.settlement}
    <h4>{draw ? 'Confirm revealed tenpai hands' : 'Confirm the winning hand'} </h4>
    <p>The Rust rules engine calculates yaku, han, fu and payments. Enter only tiles revealed at the table; called sets are already recorded.</p>
    {#if ending.kind === 'ron'}
      <p>Winning tile <Tile tile={p.pending} size="tiny" /> · from {WINDS[p.turn]}. Select all simultaneous ron winners.</p>
      <div class="flags">{#each WINDS as wind, i (wind)}{#if i !== p.turn}
        <label><input type="checkbox" bind:checked={selected[i]} disabled={ending.winners.includes(i)} /> {wind} wins</label>
      {/if}{/each}</div>
    {:else if ending.kind === 'tsumo'}
      <p>{WINDS[ending.winners[0]]} won by tsumo.</p>
      {#if p.phase === 'draw'}
        <TileEntry label="Winning drawn tile" tiles={winning} limit={1} expanded onchange={tiles => { winning = tiles; }} />
      {:else}<p>Recorded winning tile <Tile tile={p.drawn} size="tiny" /></p>{/if}
    {:else}
      <p>Riichi players must reveal tenpai. A player without riichi may declare noten. The engine checks every declared tenpai hand.</p>
      <div class="flags">{#each WINDS as wind, i (wind)}
        <label><input type="checkbox" bind:checked={tenpai[i]} disabled={p.players[i].riichi !== 'none'} /> {wind} declares tenpai</label>
      {/each}</div>
    {/if}
    {#each p.players as player, i (i)}
      {#if draw ? tenpai[i] : selected[i]}
        {#if player.hand.length}<p>{WINDS[i]}: using the recorded concealed hand.</p>
        {:else}<TileEntry label={`${WINDS[i]} revealed concealed hand`} tiles={hands[i]} limit={13 - 3 * player.melds.length}
          expanded onchange={tiles => { hands[i] = tiles; }} />
          <p>Enter {13 - 3 * player.melds.length} tiles, excluding the winning tile and all called sets.</p>
        {/if}
      {/if}
    {/each}
    {#if unknownRon}<label class="furiten-confirm"><input type="checkbox" bind:checked={confirmedNoFuriten} /> All selected ron winners confirm no temporary or riichi furiten from passing a win.</label>{/if}
    {#if needsUra}
      <TileEntry label="Revealed ura-dora indicators" tiles={ura} limit={p.indicators.length} expanded onchange={tiles => { ura = tiles; }} />
      <p>Enter all {p.indicators.length} ura indicator{p.indicators.length === 1 ? '' : 's'}, even when they add no han.</p>
    {/if}
    <button class="primary" onclick={calculate}>Calculate hand settlement</button>
    {#if failure}<p role="alert">{failure}</p>{/if}
  {/if}

  {#if result}
    <h4>{gameState.settlement ? 'Points applied' : 'Settlement preview'}</h4>
    {#each result.winners as winner (winner.seat)}
      <article aria-label={`${WINDS[winner.seat]} scored hand`}>
        <h4>{WINDS[winner.seat]} · {winner.limit ?? `${winner.han} han · ${winner.fu} fu`}</h4>
        <div class="tiles">{#each winner.hand as tile, i (i)}<Tile {tile} size="tiny" />{/each}</div>
        <Melds melds={viewMelds(p.players[winner.seat])} size="tiny" />
        <p>Winning tile <Tile tile={winner.winning_tile} size="tiny" /></p>
        <ul>{#each winner.yaku as yaku, i (i)}<li>{yaku.name} · {yaku.yakuman ? 'yakuman' : `${yaku.han} han`}</li>{/each}</ul>
        <p>Dora: {winner.dora} han · Ura-dora: {winner.ura_dora} han · Total: {winner.han} han</p>
        {#if winner.limit === 'yakuman'}<p>Dora do not increase this yakuman payment.</p>{/if}
        <div class="tiles" role="group" aria-label={`${WINDS[winner.seat]} dora indicators`}><span>Dora indicators</span>{#each winner.indicators as tile, i (i)}<Tile {tile} size="tiny" />{/each}</div>
        {#if winner.ura_indicators.length}<div class="tiles" role="group" aria-label={`${WINDS[winner.seat]} ura-dora indicators`}><span>Ura-dora indicators</span>{#each winner.ura_indicators as tile, i (i)}<Tile {tile} size="tiny" />{/each}</div>{/if}
        {#if winner.fu_detail.length}<details><summary>Fu calculation · {winner.fu} fu</summary><ul>{#each winner.fu_detail as [reason, fu], i (i)}<li>{reason}: {fu}</li>{/each}</ul></details>{/if}
      </article>
    {/each}
    {#if draw}<p>Tenpai: {result.tenpai.length ? result.tenpai.map(i => WINDS[i]).join(', ') : 'none'}.</p>{/if}
    <div class="ledger"><table aria-label="Guided score changes"><thead><tr><th>Seat</th><th>Settlement</th>{#if gameState.opening}<th>Whole hand</th>{/if}<th>New score</th></tr></thead><tbody>
      {#each WINDS as wind, i (wind)}<tr><th>{wind}</th><td>{signed(result.deltas[i])}</td>{#if gameState.opening}<td>{signed(result.after[i] - gameState.opening[i])}</td>{/if}<td>{result.after[i].toLocaleString()}</td></tr>{/each}
    </tbody></table></div>
    <p>Settlement includes honba and the riichi pot. Riichi bets were deducted when declared; whole-hand changes include those earlier deductions.</p>
    <p>Riichi sticks: {result.sticks_before} → {result.sticks_after}. Dealer {result.repeat ? 'repeats' : 'moves'}; next hand has {result.next_counters} honba.</p>
    {#if result.refunded_riichi != null}<p>{WINDS[result.refunded_riichi]}’s 1,000-point declaration bet is refunded because the declaration discard was won on.</p>{/if}
    {#if gameState.settlement}<button class="primary" onclick={() => onnext(result.repeat)}>Next hand · dealer {result.repeat ? 'repeats' : 'moves'}</button>
    {:else}<button class="primary" onclick={() => onsettle(JSON.parse(inputKey))}>Apply settlement</button>{/if}
  {/if}
</section>

<style>
  .guided-result, article { display: grid; gap: 12px; min-width: 0; }
  h4, p, ul { margin: 0; }
  h4 { font-size: 1rem; } p, li, label, table { font-size: .84rem; line-height: 1.5; }
  article { padding: 12px; border: 1px solid #ffffff33; border-radius: 10px; }
  .flags, .tiles { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .furiten-confirm, .flags label { display: flex; gap: 8px; align-items: center; min-height: 44px; }
  input { width: 20px; height: 20px; }
  .ledger { overflow-x: auto; } table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px; text-align: right; white-space: nowrap; border-bottom: 1px solid #ffffff33; }
  th:first-child { text-align: left; }
  button { min-height: 48px; border: 0; border-radius: 8px; padding: 10px 16px; font: inherit; cursor: pointer; justify-self: start; }
  .primary { background: var(--button-accent); color: var(--button-text); font-weight: 600; }
  [role=alert] { color: var(--warning-text); }
  summary { cursor: pointer; min-height: 36px; }
</style>
