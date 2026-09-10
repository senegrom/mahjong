<script>
  import { onMount, onDestroy, untrack } from 'svelte';
  import { PhysicalAnalysis, settle_physical } from '../wasm/riichi.js';
  import { AGENTS, WINDS, evaluateAgent, isTrained } from './agents.js';
  import { TILES } from './physical-position.js';
  import { PhysicalStore } from './physical-store.js';
  import { emptyGuided, guidedEvent, editGuided, undoGuided, GUIDED_FORMAT, parseGuided, visibleCounts, doraTiles, setTiles } from './guided-game.js';
  import { tileWords } from './tiles.js';
  import TileEntry from './TileEntry.svelte';
  import Tile from './Tile.svelte';
  import HandTile from './HandTile.svelte';
  import Discards from './Discards.svelte';
  import Melds from './Melds.svelte';
  import AgentWeights from './AgentWeights.svelte';
  import GuidedResult from './GuidedResult.svelte';

  let { ready, trainedAvailable, storage, hints = true } = $props();
  let game = $state(emptyGuided());
  let state = $derived(game.state);
  let position = $derived(state.position);
  let mine = $derived(position.players[position.seat]);
  let dora = $derived(hints ? doraTiles(position) : []);
  let counts = $derived(visibleCounts(position));
  let hand = $derived([...mine.hand].sort((a, b) => TILES.indexOf(a) - TILES.indexOf(b)));
  let mounted = $state(false);
  let loaded = $state(false), unreadable = $state(false), conflict = $state(''), warning = $state('');
  let failure = $state(''), busy = $state(false), analysis = $state(null);
  let discardRiichi = $state(false), discardDrawn = $state(false);
  let caller = $state(-1), callKind = $state('pon'), callTile = $state('');
  let kanKind = $state('concealed-kan');
  let resultSeat = $state(0), resultKind = $state('Tsumo');
  let loadedStore, request, analyzedKey = '', closed = false;
  const snapshot = () => JSON.parse(JSON.stringify(game));
  const decisionKey = () => JSON.stringify([state.stage, position, state.agent]);
  let blocked = $derived(!ready || !loaded || unreadable || Boolean(conflict));
  let ownTurn = $derived(state.nextSeat === position.seat);
  let inputPrompt = $derived(ownTurn
    ? position.after_quad ? 'What is your replacement tile?' : 'What did you draw?'
    : `${WINDS[state.nextSeat]} discard?`);
  let candidates = $derived(WINDS.map((name, seat) => ({ name, seat })).filter(s => s.seat !== position.turn && s.seat !== position.seat && s.seat !== state.claim?.seat));

  async function load() {
    request?.abort(); analysis = null; busy = false; loaded = false;
    const saved = await loadedStore.read();
    if (closed) return;
    game = saved; unreadable = loadedStore.unreadable; conflict = ''; failure = ''; loaded = true;
  }
  onMount(() => {
    loadedStore = new PhysicalStore(storage, navigator.locks, {
      format: { ...GUIDED_FORMAT, parse: text => parseGuided(text, settle_physical) },
      onWarning: message => { if (!closed) warning = message.replaceAll('physical table', 'guided game'); },
      onConflict: message => { if (!closed) { conflict = message.replaceAll('physical table', 'guided game'); request?.abort(); analysis = null; busy = false; } },
    });
    mounted = true;
    const changed = event => loadedStore.changed(event);
    window.addEventListener('storage', changed);
    return () => window.removeEventListener('storage', changed);
  });
  $effect(() => {
    if (ready && mounted) untrack(() => { void load(); });
  });
  $effect(() => {
    if (loaded && !unreadable && !conflict) void loadedStore.save(snapshot());
  });
  $effect(() => {
    const key = decisionKey(), enabled = !blocked;
    untrack(() => {
      request?.abort(); request = null; analysis = null; busy = false; failure = '';
      discardRiichi = false; discardDrawn = false; caller = -1; callTile = '';
      if (enabled && state.stage === 'decision') void analyze(key);
    });
  });
  function edit(change) {
    if (blocked) return;
    try { game = editGuided(snapshot(), change); failure = ''; }
    catch (error) { failure = error.message ?? String(error); }
  }
  function validate(p) { const engine = new PhysicalAnalysis(p); engine.free(); }
  function act(event) {
    if (blocked) return false;
    try { game = guidedEvent(snapshot(), event, validate, settle_physical); failure = ''; return true; }
    catch (error) { failure = error.message ?? String(error); return false; }
  }
  async function analyze(key = decisionKey()) {
    if (blocked || state.stage !== 'decision') return;
    request?.abort();
    const owner = new AbortController(); request = owner;
    const p = JSON.parse(JSON.stringify(position)), agent = state.agent;
    busy = true; failure = ''; analysis = null;
    let engine;
    try {
      engine = new PhysicalAnalysis(p);
      const result = await evaluateAgent(engine, agent, owner.signal);
      if (closed || request !== owner || owner.signal.aborted || key !== decisionKey()) return;
      analysis = result; analyzedKey = key;
    } catch (error) {
      if (!closed && request === owner && !owner.signal.aborted) failure = error.message ?? String(error);
    } finally {
      engine?.free();
      if (request === owner) { busy = false; request = null; }
    }
  }
  function choose(choice, confirm = true) {
    const source = analysis, key = analyzedKey;
    if (blocked || !source || key !== decisionKey()) return;
    if (confirm && !window.confirm(`Record this physical move: ${choice.label}?`)) return;
    if (closed || blocked || source !== analysis || key !== decisionKey()) return;
    act({ type: 'choice', choice, choices: JSON.parse(JSON.stringify(source.choices)) });
  }
  function undo() {
    if (!blocked) { game = undoGuided(snapshot()); failure = ''; }
  }
  async function restart() {
    if (!loaded || conflict || !window.confirm('Start a new guided game? This replaces its saved progress.')) return;
    if (unreadable) {
      if (await loadedStore.save(emptyGuided(), { clearUnreadable: true })) await load();
    } else { game = emptyGuided(); failure = ''; }
  }
  function finish() {
    const result = resultKind === 'Exhaustive draw' || resultKind === 'Other hand end' ? resultKind : `${WINDS[resultSeat]}: ${resultKind}`;
    if (window.confirm(`Finish this hand: ${result}?`)) act({ type: 'finish', result,
      kind: resultKind === 'Exhaustive draw' ? 'draw' : resultKind === 'Other hand end' ? 'manual' : resultKind.toLowerCase(), winner: resultSeat });
  }
  const viewMelds = player => player.melds.map(m => ({ kind: m.kind, tiles: setTiles(m), from: ['self', 'right', 'across', 'left'][m.from] }));
  onDestroy(() => { closed = true; request?.abort(); loadedStore?.close(); });
</script>

<section class="guided-play" aria-label="Agent guided physical game">
  <header class="guided-heading"><h2>Agent guided physical game</h2><div class="buttons"><button onclick={undo} disabled={blocked || !game.past.length}>Undo last step</button><button onclick={restart} disabled={!loaded || Boolean(conflict)}>New guided game</button></div></header>
  {#if !loaded}<p role="status">Loading your guided game…</p>{/if}
  {#if conflict}<p role="alert">{conflict} <button onclick={load}>Reload saved game</button></p>{/if}
  {#if unreadable}<p role="alert">The saved guided game could not be read. It has been preserved. Choose New guided game to replace it.</p>{/if}
  {#if warning}<p role="status">{warning}</p>{/if}
  <fieldset class="guided-controls" disabled={blocked}>
    <div class="guide-meta">
      <span>{WINDS[position.round]} {position.kyoku} · You: {WINDS[position.seat]} · {mine.score?.toLocaleString()} points</span>
      <label>Adviser<select aria-label="Guided game adviser" value={state.agent} onchange={e => edit(s => { s.agent = e.currentTarget.value; })}>
        {#each Object.entries(AGENTS) as [key, name] (key)}{#if !isTrained(key) || trainedAvailable || state.agent === key}<option value={key}>{name}</option>{/if}{/each}
      </select></label>
    </div>
    {#if !['setup', 'hand', 'dora'].includes(state.stage)}
      <div class="your-hand" aria-label="Your guided hand">
        <div class="hand-caption"><strong>Your hand</strong><span>{mine.riichi !== 'none' ? 'Riichi · ' : ''}{mine.furiten ? 'Passed win · furiten · ' : ''}{position.wall} live tiles left</span></div>
        <div class="held-tiles">{#each hand as tile, i (i)}{@const choice = analysis?.choices.find(c => c.tile === tile && c.kind === 'discard')}
          <HandTile {tile} size="small" dora={dora.includes(tile)} remaining={4 - counts.get(tile)} showRemaining={hints}
            drawn={tile === position.drawn && hand.lastIndexOf(tile) === i}
            selected={Boolean(analysis && ['discard', 'riichi'].includes(analysis.choice.kind) && analysis.choice.tile === tile)}
            onclick={choice ? () => choose(choice) : null} />
        {/each}</div>
        <Melds melds={viewMelds(mine)} {dora} />
      </div>
    {/if}
    <div class="guide-prompt" data-stage={state.stage}>
      {#if state.stage === 'setup'}
        <p class="eyebrow">1 · Table setup</p><h3>Your points and seat?</h3>
        <p>Start with your points and seat wind. Then enter your first 13 tiles and the dora indicator.</p>
        <div class="fields">
          <label>Your seat<select aria-label="Your seat" value={position.seat} onchange={e => edit(s => { s.position.seat = Number(e.currentTarget.value); })}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label>
          <label>Your points<input aria-label="Your points" type="number" step="100" value={mine.score} oninput={e => edit(s => { s.position.players[s.position.seat].score = e.currentTarget.value === '' ? null : Number(e.currentTarget.value); })} /></label>
        </div>
        <details><summary>Round, other scores and sticks</summary><div class="fields">
          <label>Round wind<select value={position.round} onchange={e => edit(s => { s.position.round = Number(e.currentTarget.value); })}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label>
          {#each [['kyoku', 'Hand number', 1, 4], ['counters', 'Honba', 0, 100], ['riichi_sticks', 'Riichi sticks', 0, 100]] as [key, label, min, max] (key)}<label>{label}<input type="number" {min} {max} value={position[key]} oninput={e => edit(s => { s.position[key] = e.currentTarget.value === '' ? null : Number(e.currentTarget.value); })} /></label>{/each}
          {#each position.players as player, i (i)}{#if i !== position.seat}<label>{WINDS[i]} points<input type="number" step="100" value={player.score} oninput={e => edit(s => { s.position.players[i].score = e.currentTarget.value === '' ? null : Number(e.currentTarget.value); })} /></label>{/if}{/each}
        </div></details>
        <button class="primary" onclick={() => act({ type: 'setup' })}>Next · starting tiles</button>
      {:else if state.stage === 'hand'}
        <p class="eyebrow">2 · Starting hand</p><h3>Choose your first 13 tiles</h3>
        <p>Tap tiles below. Tap an entered tile to remove it. If you are East, enter the extra dealer tile at the draw prompt.</p>
        <TileEntry label="Your starting hand" tiles={mine.hand} limit={13} expanded onchange={tiles => edit(s => { s.position.players[s.position.seat].hand = tiles; })} />
        <button class="primary" disabled={mine.hand.length !== 13} onclick={() => act({ type: 'hand' })}>Next · dora indicator</button>
      {:else if state.stage === 'dora' || state.stage === 'indicator'}
        <p class="eyebrow">{state.stage === 'dora' ? '3 · Dora' : 'Kan · new indicator'}</p><h3>{state.stage === 'dora' ? 'Which dora indicator is showing?' : 'Which new dora indicator was revealed?'}</h3>
        <p>Enter the face-up indicator tile itself.</p>
        <TileEntry label={state.stage === 'dora' ? 'First dora indicator' : 'New kan indicator'} expanded onadd={tile => act({ type: 'indicator', tile })} />
      {:else if state.stage === 'turn'}
        <p class="eyebrow">{ownTurn ? 'Your turn' : `${WINDS[state.nextSeat]} · ${position.after_quad ? 'replacement draw' : state.needsDraw ? 'draw and discard' : 'discard after calling'}`}</p><h3>{inputPrompt}</h3>
        {#if ownTurn}<p>Enter the tile you actually drew. Your adviser will show the legal moves.</p>
        {:else}<p>{state.needsDraw ? 'Enter the discarded tile; the hidden draw is counted automatically.' : 'Enter the caller’s discard. No draw is counted.'}</p>
          <div class="flags"><label><input type="checkbox" bind:checked={discardRiichi} disabled={position.players[state.nextSeat].riichi !== 'none' || !state.needsDraw} /> Declares riichi</label><label><input type="checkbox" bind:checked={discardDrawn} disabled={!state.needsDraw} /> Discarded from the draw</label></div>
        {/if}
        {#key `${state.stage}-${state.nextSeat}-${game.log.length}`}<TileEntry label={inputPrompt} expanded onadd={tile => act(ownTurn ? { type: 'draw', tile } : { type: 'discard', tile, riichi: discardRiichi, drawn: discardDrawn })} />{/key}
        {#if !ownTurn && state.needsDraw}<details><summary>{WINDS[state.nextSeat]} declared a kan instead</summary>
          <label>Kan type<select bind:value={kanKind}><option value="concealed-kan">Concealed kan</option><option value="extended-kan">Added kan · extend a pon</option></select></label>
          <TileEntry label={`${WINDS[state.nextSeat]} kan tile`} onadd={tile => act({ type: 'kan', kind: kanKind, tile })} />
        </details>{/if}
      {:else if state.stage === 'decision'}
        <p class="eyebrow">Agent advice</p><h3>{position.phase === 'call' ? `${WINDS[position.turn]} ${position.pending_kind === 'discard' ? 'discarded' : 'declared a kan of'} ${tileWords(position.pending)}` : 'What should you play?'}</h3>
        {#if busy}<p role="status">Your adviser is thinking…</p>{/if}
        {#if analysis}<AgentWeights {analysis} {dora} onchoose={choose} disabled={busy || blocked} />
          <button class="primary record-best" onclick={() => choose(analysis.choice, false)}>Record suggested move</button>
        {:else if !busy}<button onclick={() => analyze()}>Retry advice</button>{/if}
        <p>Record the move you play at the table. The hand, calls and discards update together.</p>
      {:else if state.stage === 'responses' || state.stage === 'claim-response'}
        <p class="eyebrow">Other players’ responses</p><h3>{state.claim ? `${WINDS[state.claim.seat]} called ${state.claim.kind}. Did anyone call ron?` : `Did anyone call ${tileWords(position.pending)}?`}</h3>
        {#if state.claim}<p>Ron takes precedence. Record a win below, or confirm that the set stands.{state.claim.kind === 'chii' ? ' A simultaneous pon or open kan also takes precedence over chii.' : ''}</p>{/if}
        <button class="primary no-calls" onclick={() => act({ type: 'continue' })}>{state.claim ? 'No higher-priority calls · confirm set' : 'No other calls · continue'}</button>
        {#if !state.claim || state.claim.kind === 'chii'}<details><summary>{state.claim ? 'Record a simultaneous opponent pon or kan' : 'Record an opponent’s chii, pon or kan'}</summary><div class="fields">
          <label>Who called?<select aria-label="Who called?" bind:value={caller}><option value={-1}>Choose opponent</option>{#each candidates as candidate (candidate.seat)}<option value={candidate.seat}>{candidate.name}</option>{/each}</select></label>
          <label>Call<select aria-label="Opponent call" bind:value={callKind}><option value="pon">Pon</option><option value="chii" disabled={Boolean(state.claim)}>Chii</option><option value="kan">Open kan</option></select></label>
          {#if callKind === 'chii'}<label>Lowest tile in sequence<select aria-label="Lowest tile in sequence" bind:value={callTile}><option value="">Choose tile</option>{#each TILES.filter(t => /^[1-7][mps]$/.test(t)) as tile (tile)}<option value={tile}>{tileWords(tile)}</option>{/each}</select></label>{/if}
        </div><button disabled={caller === -1 || (callKind === 'chii' && (!callTile || Boolean(state.claim)))} onclick={() => act({ type: 'call', seat: caller, kind: callKind, tile: callTile })}>Record opponent call</button></details>{/if}
      {:else if state.stage === 'kan-response'}
        <p class="eyebrow">Kan response</p><h3>Did anyone win by robbing the kan?</h3>
        <button class="primary" onclick={() => act({ type: 'continue' })}>No ron · enter new indicator</button>
        <p>If someone won, record the hand result below.</p>
      {:else if state.stage === 'over'}
        <p class="eyebrow">Hand complete</p><h3>{state.result || 'Hand finished'}</h3>
        {#if state.ending && state.ending.kind !== 'manual'}
          {#key JSON.stringify(state.ending)}<GuidedResult {state} onsettle={input => act({ type: 'settle', input })} onnext={repeat => act({ type: 'next-hand', repeat })} />{/key}
        {:else}
          <p>This is a legacy or manually adjudicated hand end. Automatic scoring has not been applied.</p>
        <p>Settle points at the physical table, then enter the resulting scores and remaining riichi sticks.</p>
        <div class="fields">{#each position.players as player, i (i)}<label>{WINDS[i]} points<input aria-label={`${WINDS[i]} settled points`} type="number" step="100" value={player.score} oninput={e => edit(s => { s.position.players[i].score = e.currentTarget.value === '' ? null : Number(e.currentTarget.value); })} /></label>{/each}
          <label>Riichi sticks remaining<input aria-label="Riichi sticks remaining" type="number" min="0" max="100" value={position.riichi_sticks} oninput={e => edit(s => { s.position.riichi_sticks = e.currentTarget.value === '' ? null : Number(e.currentTarget.value); })} /></label>
        </div><div class="buttons"><button class="primary" onclick={() => act({ type: 'next-hand', repeat: false })}>Next hand · dealer moves</button><button onclick={() => act({ type: 'next-hand', repeat: true })}>Next hand · dealer repeats</button></div>
        {/if}
      {/if}
      {#if failure}<p class="failure" role="alert">{failure}</p>{/if}
    </div>
    {#if !['setup', 'hand', 'dora', 'over'].includes(state.stage)}
      <details class="end-hand"><summary>Someone won / hand ended</summary><div class="fields"><label>Result<select bind:value={resultKind}><option>Tsumo</option><option>Ron</option><option>Exhaustive draw</option><option>Other hand end</option></select></label><label>Winner<select bind:value={resultSeat}>{#each WINDS as wind, index (wind)}<option value={index}>{wind}</option>{/each}</select></label></div><button onclick={finish}>Record hand result</button></details>
    {/if}
    {#if !['setup', 'hand', 'dora'].includes(state.stage)}
      <details class="remembered" open><summary>Remembered table · {position.counters} honba · {position.riichi_sticks} riichi sticks</summary>
        <div class="indicators"><span>Dora indicators</span>{#each position.indicators as tile, i (i)}<Tile {tile} size="tiny" />{/each}</div>
        <div class="guided-seats">{#each position.players as player, i (i)}<section class:followed={i === position.seat} aria-label={`${WINDS[i]} remembered table`}><h4>{WINDS[i]}{i === position.seat ? ' · You' : ''} <span>{player.score?.toLocaleString()}{player.riichi !== 'none' ? ' · Riichi' : ''}</span></h4><Discards discards={player.discards} {dora} /><Melds melds={viewMelds(player)} {dora} size="tiny" /></section>{/each}</div>
      </details>
    {/if}
    {#if game.log.length}<details class="guided-log"><summary>Game history · {game.log.length} steps</summary><ol>{#each game.log as entry, i (i)}<li>{entry}</li>{/each}</ol></details>{/if}
  </fieldset>
</section>

<style>
  .guided-play, .guided-controls { display: grid; gap: 14px; min-width: 0; grid-column: 1 / -1; }
  .guided-controls { border: 0; margin: 0; padding: 0; }
  .guided-heading, .buttons, .guide-meta, .hand-caption { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; justify-content: space-between; }
  h2 { margin: 0; font-size: 1.15rem; } h3 { margin: 0; font-size: 1.25rem; line-height: 1.35; }
  h4 { margin: 0 0 10px; font-size: .88rem; } h4 span { font-weight: 400; font-size: .78rem; }
  p { margin: 0; font-size: .84rem; line-height: 1.5; }
  .guide-meta { font-size: .82rem; color: var(--gold); }
  label { display: grid; gap: 6px; font-size: .8rem; }
  .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 12px; }
  .guide-prompt { display: grid; gap: 16px; padding: 20px; border-radius: 14px; border: 1px solid #d8a12a77; background: #0004; }
  .eyebrow { color: var(--gold); font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; }
  select, input:not([type=checkbox]), button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 10px; font: inherit; box-sizing: border-box; min-width: 0; max-width: 100%; }
  select option { background: #17241f; color: var(--ivory); }
  button { cursor: pointer; font-size: .82rem; } button:disabled { opacity: .5; cursor: default; }
  .primary { background: var(--button-accent); color: var(--button-text); font-weight: 600; min-height: 48px; }
  .guide-prompt > .primary { justify-self: start; }
  summary { min-height: 36px; cursor: pointer; font-size: .85rem; line-height: 1.5; }
  details > :not(summary) { margin-top: 10px; }
  .your-hand { display: grid; gap: 12px; padding: 14px; border: 1px solid #ffffff25; border-radius: 12px; background: #0002; }
  .hand-caption { font-size: .8rem; } .held-tiles, .indicators { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .held-tiles :global(.hand-tile[data-hand-drawn=true]) { margin-left: 10px; }
  .flags { display: flex; gap: 12px; flex-wrap: wrap; } .flags label { display: flex; align-items: center; gap: 8px; }
  input[type=checkbox] { width: 20px; height: 20px; accent-color: var(--gold); }
  .end-hand, .remembered, .guided-log { border: 1px solid #ffffff22; border-radius: 12px; padding: 12px; }
  .indicators { font-size: .8rem; }
  .guided-seats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  .guided-seats section { min-width: 0; padding: 12px; background: #0003; border-radius: 10px; }
  .guided-seats .followed { outline: 1px solid #d8a12a77; }
  .guided-seats :global(.melds:not(:empty)) { margin-top: 12px; }
  .guided-log ol { max-height: 320px; overflow: auto; font-size: .8rem; line-height: 1.7; padding-left: 28px; }
  .failure { color: var(--warning-text); background: #5c201c55; border-radius: 8px; padding: 12px; }
  @media (max-width: 560px) { .guide-prompt { padding: 14px; } .guided-seats { grid-template-columns: 1fr; } .guide-meta > span { width: 100%; } .guided-heading .buttons { width: 100%; } }
</style>
