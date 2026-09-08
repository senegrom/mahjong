<script>
  import { onDestroy } from 'svelte';
  import { Game } from '../wasm/riichi.js';
  import { chooseAction } from './policy.js';
  import { AGENTS, WINDS, evaluateAgent } from './agents.js';
  import { WatchSession } from './watch-session.js';
  import AgentWeights from './AgentWeights.svelte';
  import Tile from './Tile.svelte';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';
  import ScoreScreen from './ScoreScreen.svelte';
  import Standings from './Standings.svelte';

  let { ready, trainedAvailable, strongAvailable, opponents, trainedModel } = $props();
  let lineup = $state(['club', 'club', 'club', 'club']);
  let watch = $state.raw(null);
  let view = $state(null);
  let analysis = $state(null);
  let busy = $state(false);
  let failure = $state('');
  let auto = $state(false);
  let showWeights = $state(true);
  let speed = $state(1400);
  let standings = $state(null);
  let log = $state([]);
  let configured = false;
  const positions = ['Followed agent', 'Right', 'Opposite', 'Left'];
  $effect(() => {
    if (!configured && ready) {
      configured = true;
      const available = trainedModel === 'strong' && strongAvailable ? 'strong'
        : trainedAvailable ? 'quick' : strongAvailable ? 'strong' : 'club';
      lineup = [available, ...opponents.map(value => value === 'neural' ? available : value)];
    }
  });
  function update(owner) {
    if (watch !== owner) return;
    view = owner.match.view;
    analysis = owner.analysis;
    busy = owner.busy;
    failure = owner.failure || owner.match.failure;
    auto = owner.autoplay;
    log = owner.match.events;
    standings = owner.match.over ? owner.match.engine.standings() : null;
  }
  function start() {
    watch?.dispose();
    watch = new WatchSession(Game, Date.now() % 2 ** 31, lineup, {
      ai: (planes, mask, signal, model) => chooseAction(planes, mask, 0, 20000, signal, model),
      evaluate: evaluateAgent, onChange: update, delay: speed,
    });
    watch.autoplay = auto;
    void watch.prepare();
  }
  onDestroy(() => watch?.dispose());
</script>

<section class="agent-watch" aria-label="Agent watch">
  <div class="watch-heading"><h2>Agent watch</h2><span>Follow one seat through a full game.</span></div>
  <details open={!view} class="watch-setup">
    <summary>Agents at the table{#if watch} · following {AGENTS[watch.lineup[0]]}{/if}</summary>
    <div class="agent-fields">
      {#each positions as position, index (index)}
        <label>{position}<select bind:value={lineup[index]} aria-label={position}>
          {#each Object.entries(AGENTS) as [key, label] (key)}
            {#if key !== 'strong' || strongAvailable}{#if key !== 'quick' || trainedAvailable}<option value={key}>{label}</option>{/if}{/if}
          {/each}
        </select></label>
      {/each}
    </div>
    <button class="primary" onclick={start} disabled={!ready}>{view ? 'Start a new watched game' : 'Start watching'}</button>
  </details>
  <div class="watch-controls">
    <label><input type="checkbox" checked={auto} onchange={event => { auto = event.currentTarget.checked; watch?.setAutoplay(auto); }} /> Auto play</label>
    <label><input type="checkbox" bind:checked={showWeights} /> Show choice weights</label>
    <label>Pace<select value={speed} onchange={event => { speed = Number(event.currentTarget.value); if (watch) { watch.delay = speed; watch.schedule(); } }} aria-label="Watch pace"><option value={700}>Fast</option><option value={1400}>Normal</option><option value={3000}>Slow</option></select></label>
    {#if view}<button onclick={() => watch.step()} disabled={busy || Boolean(standings)}>{view.phase === 'over' ? 'Next hand' : 'Play this choice'}</button>{/if}
  </div>
  {#if failure}<div role="alert">{failure} <button disabled={busy} onclick={() => watch.prepare()}>Retry agent</button></div>{/if}
  {#if view}
    <div class="watch-round"><strong>{view.round} {view.kyoku}</strong><span>{view.wall} tiles left · {view.counters} honba · {view.riichi_sticks} riichi sticks</span><span class="tiles">{#each view.dora_indicators as tile, index (index)}<Tile {tile} size="tiny" />{/each}</span></div>
    <div class="watch-table">
      {#each view.seats as seat, index (index)}
        <section class:followed={index === 0} class:turn={seat.turn}>
          <header><strong>{index === 0 ? 'Following' : positions[index]} · {WINDS[['east','south','west','north'].indexOf(seat.seat)]}</strong><span>{AGENTS[watch.lineup[index]]} · {seat.score.toLocaleString()}{seat.riichi ? ' · Riichi' : ''}</span></header>
          {#if index === 0}<div class="tiles hand">{#each seat.hand as tile, slot (slot)}<Tile {tile} size="small" />{/each}{#if seat.drawn}<span class="drawn"><Tile tile={seat.drawn} size="small" /></span>{/if}</div>{/if}
          <Discards discards={seat.discards} compact={index !== 0} />
          {#if seat.melds.length}<Melds melds={seat.melds} size="small" />{/if}
        </section>
      {/each}
    </div>
    {#if busy}<p role="status">The agents are thinking…</p>{:else if analysis && !showWeights}<p role="status">{auto ? 'Auto play is running.' : 'Paused before the followed agent’s next choice.'}</p>{/if}
    {#if showWeights}<AgentWeights {analysis} />{/if}
    {#if standings}<Standings {standings} onagain={start} />{/if}
    {#if view.phase === 'over' && view.outcome}
      <ScoreScreen outcome={view.outcome} seats={view.seats} gameOver={Boolean(standings)} finalHand={Boolean(standings)} {busy}
        onnext={() => watch.step()} ongame={start} reviewed={true} />
    {/if}
    <details><summary>Hand history · {log.length} events</summary>{#each log as line, index (index)}<p class="log-line">{line}</p>{/each}</details>
  {/if}
</section>

<style>
  .agent-watch { display: grid; gap: 14px; grid-column: 1 / -1; }
  .watch-heading, .watch-controls, .watch-round { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 20px; }
  h2 { font-size: 1.15rem; margin: 0; }
  .watch-heading > span { font-size: .85rem; opacity: .7; }
  .watch-setup { padding: 12px; background: #0003; border-radius: 12px; }
  summary { cursor: pointer; min-height: 36px; }
  .agent-fields { display: grid; grid-template-columns: repeat(auto-fit,minmax(145px,1fr)); gap: 12px; margin: 8px 0 16px; }
  .agent-fields label { display: grid; gap: 6px; font-size: .8rem; }
  .watch-controls label { display: flex; align-items: center; gap: 6px; font-size: .82rem; }
  input[type=checkbox] { width: 20px; height: 20px; accent-color: var(--gold); }
  select, button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 12px; font: inherit; }
  select { max-width: 100%; }
  option { background: #17241f; color: var(--ivory); }
  button { cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .primary { background: var(--button-accent); color: var(--button-text); }
  .watch-round { font-size: .82rem; text-transform: capitalize; }
  .watch-table { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; }
  .watch-table section { min-width: 0; padding: 10px; border-radius: 10px; border: 1px solid #ffffff22; background: #0003; }
  .watch-table .followed { grid-column: 1 / -1; border-color: #d8a12a88; }
  .watch-table .turn { box-shadow: inset 0 2px var(--gold); }
  header { display: flex; flex-wrap: wrap; gap: 5px 14px; font-size: .82rem; margin-bottom: 12px; }
  header > span { opacity: .8; }
  .tiles { display: flex; align-items: end; flex-wrap: wrap; gap: 5px; }
  .hand { margin-bottom: 12px; }
  .drawn { margin-left: 10px; }
  .log-line { margin: 4px 0; font-size: .8rem; }
  @media (max-width: 640px) { .watch-table { grid-template-columns: 1fr; } .watch-table section { grid-column: 1; } .hand { --tile-width: 42px; } }
</style>
