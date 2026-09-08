<script>
  import { onDestroy } from 'svelte';
  import { PhysicalAnalysis } from '../wasm/riichi.js';
  import { AGENTS, WINDS, evaluateAgent } from './agents.js';
  import { TILES, PHYSICAL_KEY, emptyPosition, readPhysical, recordDraw, recordDiscard, recordChoice } from './physical-position.js';
  import { tileWords } from './tiles.js';
  import TileEntry from './TileEntry.svelte';
  import Tile from './Tile.svelte';
  import AgentWeights from './AgentWeights.svelte';

  let { ready, trainedAvailable, strongAvailable, storage } = $props();
  let position = $state(emptyPosition());
  let agent = $state('club');
  let analysis = $state(null);
  let analyzedKey = '';
  let busy = $state(false);
  let failure = $state('');
  let saveWarning = $state('');
  let history = $state([]);
  let eventSeat = $state(0);
  let eventKind = $state('draw');
  let request = null;
  let loaded = false;
  let revision = '';
  $effect(() => {
    if (!loaded) { loaded = true; position = readPhysical(storage); if (trainedAvailable) agent = 'quick'; }
  });
  $effect(() => {
    const key = JSON.stringify(position) + agent;
    if (key !== revision) {
      revision = key; request?.abort(); request = null; busy = false;
      analysis = null; failure = '';
    }
    if (loaded) {
      try { storage?.setItem(PHYSICAL_KEY, JSON.stringify({ version: 1, position })); }
      catch { saveWarning = 'This device could not save the physical table.'; }
    }
  });
  const snapshot = () => JSON.parse(JSON.stringify(position));
  function edit(change) {
    const before = snapshot();
    const next = structuredClone(before);
    try {
      const result = change(next) ?? next;
      history = [...history.slice(-29), before];
      position = result;
    } catch (error) { failure = error.message ?? String(error); }
  }
  function appendPastDiscard(index, tile) {
    edit(p => {
      const order = Math.max(-1, ...p.players.flatMap(player => player.discards.map(d => d.order))) + 1;
      p.players[index].discards.push({ tile, order, drawn: false, riichi: false, claimed: false });
      if (order >= 3) p.first_turns = false;
    });
  }
  function recordEvent(tile) {
    edit(p => eventKind === 'draw' ? recordDraw(p, eventSeat, tile) : recordDiscard(p, eventSeat, tile, eventKind === 'riichi'));
  }
  async function analyze() {
    request?.abort();
    const owner = new AbortController(); request = owner;
    const input = snapshot(), key = JSON.stringify(input) + agent;
    let engine;
    busy = true; failure = ''; analysis = null;
    try {
      engine = new PhysicalAnalysis(input);
      const result = await evaluateAgent(engine, agent, owner.signal);
      if (request !== owner || owner.signal.aborted || key !== JSON.stringify(position) + agent) return;
      analysis = result; analyzedKey = key;
    } catch (error) {
      if (request === owner && !owner.signal.aborted) failure = error.message ?? String(error);
    } finally {
      engine?.free();
      if (request === owner) { busy = false; request = null; }
    }
  }
  function record(choice) {
    if (!analysis || analyzedKey !== JSON.stringify(position) + agent) { failure = 'Analyse the current table before recording a choice.'; return; }
    const choices = analysis.choices;
    edit(p => recordChoice(p, choice, choices));
  }
  function setDecision(event) {
    const value = event.currentTarget.value;
    edit(p => {
      p.phase = value === 'act' ? 'act' : 'call';
      p.pending_kind = value === 'act' ? 'discard' : value;
      if (p.phase === 'act') { p.turn = p.seat; p.pending = null; }
      else { p.drawn = null; p.just_claimed = null; if (p.turn === p.seat) p.turn = (p.seat + 3) % 4; }
    });
  }
  onDestroy(() => request?.abort());
</script>

<section class="physical-play" aria-label="Physical agent play">
  <div class="physical-heading"><h2>Physical agent play</h2><button onclick={() => { if (history.length) { position = history.at(-1); history = history.slice(0, -1); } }} disabled={!history.length}>Undo edit / move</button><button onclick={() => edit(() => emptyPosition())}>Clear table</button></div>
  <p class="intro">Enter the table in front of you. Include the drawn tile in the concealed hand; leave unknown hands empty. Tap entered tiles to remove them. All draws and calls are recorded by you.</p>
  <div class="physical-toolbar">
    <label>Analyse seat<select bind:value={position.seat} onchange={() => { if (position.phase === 'act') position.turn = position.seat; }} aria-label="Analyse seat">{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label>
    <label>Agent<select bind:value={agent} aria-label="Physical play agent">{#each Object.entries(AGENTS) as [key, label] (key)}{#if key !== 'strong' || strongAvailable}{#if key !== 'quick' || trainedAvailable}<option value={key}>{label}</option>{/if}{/if}{/each}</select></label>
    <button class="primary" onclick={analyze} disabled={!ready || busy}>{busy ? 'Analysing…' : 'Show agent weights'}</button>
  </div>
  {#if failure}<p class="failure" role="alert">{failure}</p>{/if}
  {#if saveWarning}<p role="status">{saveWarning}</p>{/if}
  {#if analysis}
    <AgentWeights {analysis} />
    <div class="record-choices"><span>Record what was played:</span>{#each analysis.choices as choice, index (index)}<button class:primary={choice.kind === analysis.choice.kind && choice.tile === analysis.choice.tile} onclick={() => record(choice)}>{choice.label}</button>{/each}</div>
  {/if}
  <details class="table-fields" open>
    <summary>Round and current decision</summary>
    <div class="fields">
      <label>Round wind<select bind:value={position.round}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label>
      <label>Hand<input type="number" min="1" max="4" bind:value={position.kyoku} /></label>
      <label>Honba<input type="number" min="0" max="100" bind:value={position.counters} /></label>
      <label>Riichi sticks<input type="number" min="0" max="100" bind:value={position.riichi_sticks} /></label>
      <label>Live wall remaining<input type="number" min="0" max="70" bind:value={position.wall} /></label>
      <label>Decision<select value={position.phase === 'call' ? position.pending_kind : position.phase} onchange={setDecision}>
        <option value="act">After a draw / set call</option><option value="discard">Respond to discard</option><option value="extended-kan">Rob added kan</option><option value="concealed-kan">Rob concealed kan</option>
        {#if position.phase === 'draw'}<option value="draw">Waiting for a real draw</option>{/if}{#if position.phase === 'over'}<option value="over">Hand finished</option>{/if}
      </select></label>
      {#if position.phase === 'call'}
        <label>Offered by<select bind:value={position.turn}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label>
        <label>Pending tile<select bind:value={position.pending}><option value={null}>Choose tile</option>{#each TILES as tile (tile)}<option value={tile}>{tileWords(tile)}</option>{/each}</select></label>
      {:else if position.phase === 'act'}
        <label>Drawn tile · already in hand<select bind:value={position.drawn} onchange={() => { if (position.drawn) position.just_claimed = null; }}><option value={null}>No draw · after calling a set</option>{#each [...new Set(position.players[position.seat].hand)] as tile (tile)}<option value={tile}>{tileWords(tile)}</option>{/each}</select></label>
        {#if !position.drawn}<label>Tile just claimed<select bind:value={position.just_claimed}><option value={null}>Choose tile</option>{#each TILES as tile (tile)}<option value={tile}>{tileWords(tile)}</option>{/each}</select></label>{/if}
      {/if}
    </div>
    <div class="flags"><label><input type="checkbox" bind:checked={position.first_turns} /> First turns unbroken</label><label><input type="checkbox" bind:checked={position.after_quad} /> Replacement draw after kan</label></div>
    <TileEntry label="Dora indicators · one plus one per completed kan" tiles={position.indicators} limit={5} onchange={tiles => edit(p => { p.indicators = tiles; })} />
  </details>
  <details class="record-event" open>
    <summary>Record the next physical move</summary>
    <div class="fields"><label>Seat<select bind:value={eventSeat}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label><label>Move<select bind:value={eventKind}><option value="draw">Draw / replacement draw</option><option value="discard">Discard</option><option value="riichi">Declare riichi and discard</option></select></label></div>
    <TileEntry label={`${WINDS[eventSeat]} · ${eventKind}`} onadd={recordEvent} />
    <p class="help">Draw adds to a known hand. Discard removes from a known hand; for an unknown hand it also counts that player’s draw. After kan, enter the new indicator and the real replacement tile.</p>
  </details>
  <div class="physical-seats">
    {#each position.players as player, index (index)}
      <details class="physical-seat" open={index === position.seat}>
        <summary><strong>{WINDS[index]}</strong> · {player.hand.length ? `${player.hand.length} concealed tiles` : 'Unknown hand'} · {player.score?.toLocaleString()}</summary>
        <div class="seat-content">
          <div class="fields"><label>Points<input type="number" min="-100000" max="200000" step="100" bind:value={player.score} /></label>
            <label>Declaration<select bind:value={player.riichi}><option value="none">No riichi</option><option value="riichi">Riichi</option><option value="double">Double riichi</option></select></label></div>
          <div class="flags"><label><input type="checkbox" bind:checked={player.ippatsu} /> Ippatsu active</label><label><input type="checkbox" bind:checked={player.furiten} /> Passed ron · furiten</label></div>
          <TileEntry label={`${WINDS[index]} concealed hand · drawn tile included`} tiles={player.hand} onchange={tiles => edit(p => { p.players[index].hand = tiles; })} />
          <div class="meld-editor"><strong>Called sets and concealed kans</strong>
            {#each player.melds as meld, slot (slot)}
              <div class="meld-fields">
                <label>Set<select bind:value={meld.kind} onchange={() => { if (meld.kind === 'chii') meld.from = 3; else if (meld.kind === 'concealed-kan') meld.from = 0; else if (meld.from === 0) meld.from = 3; }}>
                  <option value="chii">Chii</option><option value="pon">Pon</option><option value="kan">Open kan</option><option value="extended-kan">Added kan</option><option value="concealed-kan">Concealed kan</option>
                </select></label>
                <label>{meld.kind === 'chii' ? 'Lowest tile' : 'Tile'}<select bind:value={meld.tile}>{#each TILES as tile (tile)}<option value={tile}>{tileWords(tile)}</option>{/each}</select></label>
                <label>From<select bind:value={meld.from}><option value={3}>Left</option><option value={2}>Opposite</option><option value={1}>Right</option><option value={0}>Self</option></select></label>
                <button onclick={() => edit(p => { p.players[index].melds.splice(slot, 1); })} aria-label={`Remove ${WINDS[index]} set ${slot + 1}`}>Remove</button>
              </div>
            {/each}
            <button onclick={() => edit(p => { p.first_turns = false; p.players[index].melds.push({ kind: 'pon', tile: '1m', from: 3 }); })} disabled={player.melds.length >= 4}>Add called set</button>
          </div>
          <div class="discard-editor"><strong>Discards · chronological order across the table</strong>
            {#each player.discards as discard, slot (slot)}
              <div class="discard-fields"><Tile tile={discard.tile} size="tiny" />
                <label>Order<input type="number" min="0" max="399" bind:value={discard.order} /></label>
                <label><input type="checkbox" bind:checked={discard.drawn} /> From draw</label>
                <label><input type="checkbox" bind:checked={discard.riichi} /> Riichi</label>
                <label><input type="checkbox" bind:checked={discard.claimed} /> Claimed</label>
                <button onclick={() => edit(p => { p.players[index].discards.splice(slot, 1); })} aria-label={`Remove ${WINDS[index]} discard ${slot + 1}`}>×</button>
              </div>
            {/each}
            <TileEntry label={`${WINDS[index]} · add an earlier discard`} onadd={tile => appendPastDiscard(index, tile)} />
            <p class="help">Order starts at 0 and counts all players’ discards. Mark the riichi declaration tile and tiles taken into calls.</p>
          </div>
        </div>
      </details>
    {/each}
  </div>
</section>

<style>
  .physical-play { display: grid; gap: 14px; grid-column: 1 / -1; min-width: 0; }
  h2 { margin: 0; font-size: 1.15rem; margin-right: auto; }
  .physical-heading, .physical-toolbar, .flags, .record-choices { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
  .intro, .help { font-size: .82rem; line-height: 1.5; opacity: .8; margin: 0; max-width: 85ch; }
  .physical-toolbar { padding: 12px; background: #0004; border-radius: 12px; }
  .physical-toolbar > label, .fields > label, .meld-fields > label { display: grid; gap: 5px; font-size: .78rem; min-width: 0; }
  .fields { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 12px; }
  select, input:not([type=checkbox]), button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 10px; font: inherit; box-sizing: border-box; min-width: 0; max-width: 100%; }
  select option { background: #17241f; color: var(--ivory); }
  button { cursor: pointer; font-size: .82rem; }
  button:disabled { opacity: .5; cursor: default; }
  .primary { background: var(--button-accent); color: var(--button-text); font-weight: 600; }
  .record-choices { font-size: .82rem; }
  .record-choices > span { width: 100%; }
  .table-fields, .record-event, .physical-seat { padding: 14px; background: #0003; border: 1px solid #ffffff22; border-radius: 12px; }
  summary { min-height: 36px; cursor: pointer; font-size: .9rem; }
  .table-fields > :not(summary), .record-event > :not(summary) { margin-top: 12px; }
  .flags label, .discard-fields label { display: flex; align-items: center; gap: 5px; font-size: .78rem; }
  input[type=checkbox] { width: 20px; height: 20px; accent-color: var(--gold); flex-shrink: 0; }
  .physical-seats { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; align-items: start; }
  .seat-content, .meld-editor, .discard-editor { display: grid; gap: 12px; }
  .meld-editor, .discard-editor { border-top: 1px solid #ffffff22; padding-top: 12px; font-size: .8rem; }
  .meld-fields { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; align-items: end; }
  .discard-fields { display: flex; flex-wrap: wrap; gap: 7px; align-items: center; padding: 8px 0; border-bottom: 1px solid #ffffff18; }
  .discard-fields input[type=number] { width: 66px; }
  .failure { margin: 0; padding: 12px; border-radius: 10px; background: var(--accent-soft); color: var(--accent); }
  @media (max-width: 740px) { .physical-seats { grid-template-columns: 1fr; } }
</style>
