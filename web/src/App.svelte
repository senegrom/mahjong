<script>
  import { onMount, setContext, tick, untrack } from 'svelte';
  import init, { Game } from './wasm/riichi.js';
  import Tile, { preloadTiles } from './lib/Tile.svelte';
  import HandTile from './lib/HandTile.svelte';
  import { startOffline, watchOffline, prepareOfflineAi, refreshOffline } from './lib/offline.js';
  import Seat from './lib/Seat.svelte';
  import Discards from './lib/Discards.svelte';
  import Melds from './lib/Melds.svelte';
  import ScoreScreen from './lib/ScoreScreen.svelte';
  import Standings from './lib/Standings.svelte';
  import Review from './lib/Review.svelte';
  import AgentWatch from './lib/AgentWatch.svelte';
  import PhysicalPlay from './lib/PhysicalPlay.svelte';
  import GuidedPlay from './lib/GuidedPlay.svelte';
  import { chooseAction, modelIsAvailable, reportProgress, resetPolicy } from './lib/policy.js';
  import { MatchSession, SETTINGS_KEY, readSettings } from './lib/session.js';
  import { acceptsHandKey, heldSafeCount, callLabel, callTiles, moveHandFocus, analyzeDiscards, unseenTileCounts } from './lib/ui.js';
  import { MatchStore } from './lib/save-store.js';
  import { tileWords } from './lib/tiles.js';
  import { TILE_FACE_CONTEXT, TILE_FACE_OPTIONS } from './lib/tile-faces.js';
  import { normalizeOpponents, OPPONENT_LABELS, OPPONENT_TYPES, OPPONENT_POSITIONS } from './lib/opponents.js';

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };
  const storage = (() => { try { return window.localStorage; } catch { return null; } })();
  const touch = matchMedia('(pointer: coarse)').matches;
  const preferences = readSettings(storage, touch);
  const requested = new URLSearchParams(location.search).get('opponents');
  const requestedMode = new URLSearchParams(location.search).get('mode');
  let mode = $state(['watch', 'physical', 'guided'].includes(requestedMode) ? requestedMode : 'play');
  let difficulty = $state(['beginner', 'club', 'neural'].includes(requested) ? requested : preferences.difficulty);
  let opponents = $state(normalizeOpponents(OPPONENT_TYPES.includes(requested) ? requested : preferences.opponents ?? preferences.difficulty));
  let draftOpponents = $state(['club', 'club', 'club']);
  let customDialog = $state(null);
  let pendingOpponent = $state(null);
  let hints = $state(preferences.hints);
  let confirmDiscards = $state(preferences.confirmDiscards);
  let shortcuts = $state(preferences.shortcuts);
  let tileFace = $state(preferences.tileFace);
  let reviewAdviser = $state(preferences.reviewAdviser);
  setContext(TILE_FACE_CONTEXT, () => tileFace);
  let ready = $state(false);
  let startupNote = $state('Preparing the game for offline play…');
  // One trained network ships with the game, so its download is the only
  // optional one: readiness and availability are read straight from offline.
  let offline = $state({ coreReady: false, aiReady: false, hasModel: false, phase: 'checking', progress: 0, warning: '', coreWarning: '', coreLoading: false, persistent: false, updateReady: false });
  let failure = $state('');
  let storageWarning = $state('');
  let saveConflict = $state('');
  let notice = $state('');
  let loadNote = $state('');
  let session = $state.raw(null);
  let view = $state(null);
  let choices = $state([]);
  let log = $state([]);
  let busy = $state(false);
  let thinking = $state(false);
  let recovery = $state(false);
  let trainedAvailable = $state(false);
  let standings = $state(null);
  let notes = $state(null);
  let picked = $state(null);
  let selected = $state(null);
  let handElement = $state(null);
  let callElement = $state(null);
  let tableDialog = $state(null);
  let guideOpen = $state(false);
  let settingsOpen = $state(false);

  let me = $derived(view?.seats[0]);
  let right = $derived(view?.seats[1]);
  let across = $derived(view?.seats[2]);
  let left = $derived(view?.seats[3]);
  let handTiles = $derived(me ? [...me.hand, ...(me.drawn ? [me.drawn] : [])] : []);
  let discardChoices = $derived(choices.filter((choice) => choice.kind === 'discard'));
  let callChoices = $derived(choices.filter((choice) => choice.kind !== 'discard'));
  let myTurn = $derived(view?.phase === 'act' && me?.turn);
  let shownDora = $derived(hints ? (view?.dora_types ?? []) : []);
  let safeCount = $derived(heldSafeCount(view));
  let selectedTile = $derived(selected === null ? null : handTiles[selected]);
  let previewTile = $derived(handTiles[selected ?? picked] ?? null);
  // Recompute once when the engine offers a new decision, not on every hover,
  // selection or animation frame. The selected tile and all readiness rings
  // use the same hypothetical-discard results, including open hands.
  let discardHints = $derived(hints && myTurn && !busy && !failure && !saveConflict && !session?.closed
    ? analyzeDiscards(session?.engine, discardChoices) : new Map());
  let discardHint = $derived(previewTile ? discardHints.get(previewTile) ?? null : null);
  let displayWaits = $derived(discardHint?.waits ?? view?.waits ?? []);
  let displayLeft = $derived(discardHint?.waits_left ?? view?.waits_left ?? []);
  let remainingByTile = $derived(hints && view ? unseenTileCounts(view) : new Map());
  let uraIndicators = $derived(view?.outcome?.wins?.find(win => win.ura_indicators?.length)?.ura_indicators ?? []);

  $effect(() => {
    const value = { version: 1, difficulty, opponents: [...opponents], hints, confirmDiscards, shortcuts, tileFace, reviewAdviser };
    try { storage?.setItem(SETTINGS_KEY, JSON.stringify(value)); } catch { /* Gameplay still works. */ }
  });

  // A call is offered under a table that fills the window, so the choices
  // can sit below the fold on an ordinary screen. Being asked a question
  // whose answers are off screen is no question at all, so the section is
  // brought into view when it appears, by the shortest scroll that does
  // it and none at all when it is already there.
  $effect(() => {
    if (mode !== 'play' || !callChoices.length || !callElement) return;
    const buttons = callElement.querySelector('.call-options') ?? callElement;
    const box = buttons.getBoundingClientRect();
    if (box.bottom <= window.innerHeight && box.top >= 0) return;
    const gentle = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    buttons.scrollIntoView({ block: 'end', behavior: gentle ? 'auto' : 'smooth' });
  });

  // When the turn comes to you and nothing else has focus, the hand takes
  // it, so the arrows and Enter work at once: the shortcuts only run with
  // focus inside the hand, which keeps them off the controls.
  $effect(() => {
    if (mode !== 'play' || busy || failure || saveConflict || !myTurn || !handElement || !shortcuts || touch) return;
    const active = document.activeElement;
    if (!active || active === document.body || handElement.contains(active)) {
      handElement.focus({ preventScroll: true });
    }
  });

  function update(owner) {
    if (session !== owner) return;
    view = owner.view;
    choices = owner.choices;
    log = owner.events;
    busy = owner.busy;
    thinking = owner.thinking;
    failure = owner.failure;
    recovery = owner.needsRecovery;
    difficulty = owner.difficulty;
    opponents = [...owner.opponents];
    pendingOpponent = owner.pendingOpponent;
    standings = owner.over ? owner.engine.standings() : null;
    if (!thinking) loadNote = '';
  }

  function stopForConflict(message) {
    saveConflict = message;
    session?.dispose();
    session = null;
    choices = [];
    busy = false;
    thinking = false;
    recovery = false;
    pendingOpponent = null;
    customDialog?.close();
    selected = null;
    picked = null;
  }

  const matchStore = new MatchStore(storage, navigator.locks, {
    onConflict: stopForConflict,
    onWarning: (message) => { storageWarning = message; },
  });

  const callbacks = {
    // Greedy: the opponent plays its best move, not a sample of them.
    ai: (planes, mask, signal) => chooseAction(planes, mask, 0, 20000, signal),
    onChange: update,
    onSave: (snapshot) => matchStore.save(snapshot),
    guard: (task) => matchStore.run(task),
  };

  // Direct links to Watch or Physical play must not resume, create or save a
  // regular match. Restore it once, when the user first enters Play.
  let playStarted = false;
  $effect(() => {
    if (!ready || mode !== 'play' || playStarted) return;
    playStarted = true;
    // Only readiness and the mode decide this; what the restore reads on the
    // way must not make it run again on every change of opponent or network.
    untrack(() => {
      const saved = matchStore.read();
      if (saved) {
        try {
          session = MatchSession.restore(Game, saved, callbacks);
          update(session);
          if (session.opponents.includes('neural')) downloadAi();
          void session.run();
        } catch (error) {
          // Preserve the only saved copy until the user chooses New game.
          failure = `The saved match could not be restored: ${error.message}. Choose New game to start again.`;
        }
      } else start();
    });
  });

  onMount(() => {
    let mounted = true;
    const unwatchOffline = watchOffline(value => { if (mounted) offline = value; });
    modelIsAvailable().then((available) => { if (mounted) trainedAvailable = available; });
    reportProgress((note) => { if (mounted && thinking) loadNote = note; });
    (async () => {
      await startOffline();
      if (!mounted) return;
      await Promise.all([init(), preloadTiles((done, total) => {
        if (mounted) startupNote = `Loading all tile graphics… ${done}/${total}`;
      })]);
    })().then(() => {
      if (!mounted) return;
      ready = true;
    }).catch((error) => {
      failure = `The game could not finish loading: ${error.message ?? error}. Reconnect and reload to retry.`;
    });
    // Completed actions are already saved inside their writer transaction.
    // Never write an old snapshot during pagehide. A cached page must restore
    // the current record rather than resume a stale engine on Back navigation.
    const leave = () => { session?.dispose(); resetPolicy(); matchStore.close(); };
    const returnToPage = (event) => { if (event.persisted) location.reload(); };
    const storageChanged = (event) => { if (playStarted) matchStore.changed(event); };
    const checkOffline = () => { if (document.visibilityState === 'visible') void refreshOffline().catch(() => {}); };
    document.addEventListener('visibilitychange', checkOffline);
    window.addEventListener('online', checkOffline);
    window.addEventListener('pagehide', leave);
    window.addEventListener('pageshow', returnToPage);
    window.addEventListener('storage', storageChanged);
    return () => {
      mounted = false;
      unwatchOffline();
      document.removeEventListener('visibilitychange', checkOffline);
      window.removeEventListener('online', checkOffline);
      window.removeEventListener('pagehide', leave);
      window.removeEventListener('pageshow', returnToPage);
      window.removeEventListener('storage', storageChanged);
      matchStore.close();
      session?.dispose();
      reportProgress(null);
      resetPolicy();
    };
  });

  function downloadAi() { void prepareOfflineAi().catch(() => {}); }

  function start(strength = opponents) {
    if (saveConflict) return;
    session?.dispose();
    matchStore.newMatch();
    picked = null;
    selected = null;
    notes = null;
    notice = '';
    loadNote = '';
    session = new MatchSession(Game, Date.now() % 2 ** 31, strength, callbacks);
    if (session.opponents.includes('neural')) downloadAi();
    void session.run();
  }

  function startFresh(strength = opponents) {
    if (!ready || saveConflict) return false;
    if (session?.progressed && !confirm('Leave this unfinished match and deal a new one?')) return false;
    start(strength);
    return true;
  }

  function configureTable() {
    settingsOpen = false;
    draftOpponents = [...opponents];
    customDialog?.showModal();
  }

  function startCustomTable() {
    if (startFresh([...draftOpponents])) customDialog?.close();
  }

  function changeOpponents(event) {
    const strength = event.currentTarget.value;
    // The control displays the active mode until the change is confirmed.
    event.currentTarget.value = difficulty;
    if (strength === 'custom') configureTable();
    else if (strength !== difficulty) startFresh(strength);
  }

  function canDiscard(tile) { return !busy && !failure && !saveConflict && discardChoices.some((choice) => choice.tile === tile); }

  async function choose(choice) {
    if (!session || busy) return;
    const owner = session;
    const fromHand = handElement?.contains(document.activeElement);
    picked = null;
    selected = null;
    await owner.choose(choice);
    await tick();
    if (session !== owner || owner.closed || !fromHand || busy || failure || saveConflict || !myTurn || !shortcuts || touch) return;
    const active = document.activeElement;
    if (active === document.body || handElement?.contains(active)) handElement?.focus({ preventScroll: true });
  }

  function discard(tile) {
    const choice = discardChoices.find((entry) => entry.tile === tile);
    if (choice && canDiscard(tile)) void choose(choice);
  }

  function selectTile(tile, index) {
    if (!canDiscard(tile)) return;
    picked = null;
    if (confirmDiscards && selected !== index) selected = index;
    else discard(tile);
  }

  function syncHandFocus(event) {
    const tile = event.target.closest('[data-hand-index]');
    const index = tile ? Number(tile.dataset.handIndex) : null;
    if (picked !== null) picked = index;
    // Native Tab navigation must not leave a confirmation for a different tile.
    if (selected !== null && selected !== index) selected = null;
  }

  async function onKey(event) {
    if (mode !== 'play' || !shortcuts || !myTurn || busy || failure || !acceptsHandKey(event, handElement)) return;
    if (event.key === 'Escape') { picked = null; selected = null; return; }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const owner = session;
      const step = event.key === 'ArrowLeft' ? -1 : 1;
      selected = null;
      await tick();
      if (session !== owner || busy || failure || saveConflict || !myTurn || !handElement?.contains(document.activeElement)) return;
      // Only enabled, actually focused tiles can acquire a keyboard marker.
      picked = moveHandFocus(handElement, step);
      return;
    }
    if (event.repeat) { event.preventDefault(); return; }
    if (event.key === 'Enter' || event.key === ' ') {
      const focused = event.target.closest('[data-hand-index]');
      const index = focused ? Number(focused.dataset.handIndex) : picked;
      if (index !== null && canDiscard(handTiles[index])) {
        event.preventDefault();
        discard(handTiles[index]);
      }
      return;
    }
    if (/^[1-9]$/.test(event.key)) {
      const tile = me.hand[Number(event.key) - 1];
      if (tile) { event.preventDefault(); discard(tile); }
    } else if (event.key === '0' && me.drawn) {
      event.preventDefault();
      discard(me.drawn);
    } else if (event.key.toLowerCase() === 'r') {
      const first = callElement?.querySelector('[data-choice="riichi"]');
      if (first) { event.preventDefault(); first.focus(); }
    } else if (event.key.toLowerCase() === 't') {
      const tsumo = choices.find((choice) => choice.kind === 'tsumo');
      if (tsumo) { event.preventDefault(); void choose(tsumo); }
    }
  }

  function retryAi() { resetPolicy(); void session?.retry(); }
  function continueClub() { resetPolicy(); void session?.continueWithClub(); }
  function continueThisOpponent() { resetPolicy(); void session?.continueOpponentWithClub(); }
  async function nextHand() {
    if (busy || !session) return;
    const owner = session;
    picked = null;
    selected = null;
    const advanced = await owner.nextHand();
    if (session === owner && advanced && !owner.over) notes = null;
  }

  function saveLog() {
    if (!session) return;
    try {
      const text = session.engine.log();
      const url = URL.createObjectURL(new Blob([text + '\n'], { type: 'application/jsonl' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `riichi-${view.round}-${view.kyoku}-${Date.now()}.mjai.jsonl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { notice = 'The hand could not be saved to a file. Your current match is unchanged.'; }
  }

  function showReview() {
    try { notes = session?.engine.review() ?? []; }
    catch { notice = 'The review could not be built. Your match is unchanged.'; }
  }

  function inspectTable() { tableDialog?.showModal(); }
</script>

<svelte:window onkeydown={onKey} />

<main>
  <header class="bar">
    <h1>Riichi</h1>
    {#if mode === 'play'}<label class="opponents">
      <span>Opponents</span>
      <select value={difficulty} onchange={changeOpponents} disabled={!ready || Boolean(saveConflict)} aria-label="opponent strength">
        <option value="beginner">Beginner</option>
        <option value="club">Club</option>
        {#if trainedAvailable || opponents.includes('neural')}<option value="neural">Trained</option>{/if}
        <option value="custom">Custom table</option>
      </select>
      <svg class="select-chevron" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
    </label>{/if}
    <button class="settings-trigger" aria-label="Game settings" aria-expanded={settingsOpen} aria-controls="game-preferences"
      onclick={() => settingsOpen = !settingsOpen}>
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7h3m4 0h9M4 17h9m4 0h3" /><circle cx="9" cy="7" r="2" /><circle cx="15" cy="17" r="2" /></svg>
    </button>
    {#if mode === 'play'}<button class="restart" onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>{/if}
  </header>

  <nav class="game-modes" aria-label="Game mode">
    {#each [['play', 'Play'], ['watch', 'Agent watch'], ['physical', 'Physical agent play'], ['guided', 'Guided physical game']] as [key, label] (key)}
      <button aria-pressed={mode === key} disabled={!ready || (mode === 'play' && busy)} onclick={() => { mode = key; settingsOpen = false; }}>{label}</button>
    {/each}
  </nav>

  {#if settingsOpen}<button class="settings-backdrop" aria-label="Close game settings" onclick={() => settingsOpen = false}></button>{/if}
  <div id="game-preferences" class="preferences" class:mobile-open={settingsOpen}>
  <div class="mobile-preferences-head"><strong>Game settings</strong><button onclick={() => settingsOpen = false}>Done</button></div>
  <label class="compact-mode-selector">Game mode
    <select value={mode} aria-label="Game mode" disabled={!ready || (mode === 'play' && busy)}
      onchange={event => { mode = event.currentTarget.value; settingsOpen = false; }}>
      <option value="play">Play</option><option value="watch">Agent watch</option><option value="physical">Physical agent play</option><option value="guided">Guided physical game</option>
    </select>
  </label>
  {#if mode === 'play'}<button class="mobile-new-game" onclick={() => { if (startFresh()) settingsOpen = false; }} disabled={!ready || Boolean(saveConflict)}>New game</button>{/if}
  {#if mode === 'play' && difficulty === 'custom'}
    <button class="edit-table" onclick={configureTable} disabled={!ready || Boolean(saveConflict)}>Edit opponents</button>
  {/if}
  <details class="options">
    <summary>Options</summary>
    <div class="option-fields">
      <label>Tile face
        <select bind:value={tileFace} aria-label="Tile face">
          {#each TILE_FACE_OPTIONS as face}
            <option value={face.value}>{face.label}</option>
          {/each}
        </select>
      </label>
      <label><input type="checkbox" bind:checked={hints} /> Hints and markings</label>
      <label><input type="checkbox" bind:checked={confirmDiscards} onchange={() => selected = null} /> Select before discarding</label>
      <label><input type="checkbox" bind:checked={shortcuts} onchange={() => picked = null} /> Keyboard shortcuts in your hand</label>
      <p>With confirmation on, tap a tile to select it, then tap it again or press Discard. Your match and preferences are saved on this device.</p>
    </div>
  </details>
  <details class="guide" bind:open={guideOpen}>
    <summary>Tile markings and rules</summary>
    <div class="guide-body">
      <dl>
        <dt>3 tiles away from a wait</dt>
        <dd>
          How many tiles you must still exchange before one more tile would
          win. At a wait you are one tile from a complete hand; players call
          this the shanten count.
        </dd>

        <dt>Waiting on</dt>
        <dd>
          The tiles that would complete your hand, each with how many of the
          four nobody has seen yet. A wait with none left is marked in red.
          Select a tile for discard to see the waits of the hand you would keep;
          the preview does not play the move. Enable Select before discarding
          in Options to preview with a mouse or touch, or use the arrow keys.
        </dd>

        <dt>The number below a tile</dt>
        <dd>How many copies of that same tile nobody can see yet. Zero is red and one is gold, so thin tiles stand out without covering the artwork.</dd>

        <dt><span class="swatch dora"></span> a red ring, and a shine</dt>
        <dd>The tile is dora and adds a han to whatever your hand scores.</dd>

        <dt><span class="swatch safe"></span> a green ring</dt>
        <dd>
          The tile is safe against the opponents who have declared riichi, because
          they threw it themselves or it has already passed them. It can still deal into an undeclared hand. The safe count includes copies in your concealed hand only.
        </dd>

        <dt><span class="swatch one-away"></span> a silver ring</dt>
        <dd>Discarding this tile leaves your hand one tile from ready (one shanten).</dd>

        <dt><span class="swatch ready"></span> a gold ring</dt>
        <dd>
          Discarding this tile leaves a ready hand (tenpai), whether closed or open.
          This marks the tile shape, not a guaranteed win: you still need a yaku,
          and riichi is available only when its other conditions are met.
        </dd>

        <dt>A tile held apart</dt>
        <dd>The tile you just drew is separated by a gap, not given its own border.</dd>

        <dt><span class="swatch marker"></span> a blue ring</dt>
        <dd>
          The tile under the keyboard marker. It appears when you press an
          arrow key and goes when you throw.
        </dd>

        <dt><span class="swatch striped"></span> stripes</dt>
        <dd>More than one of those at once: each colour takes its turn around the tile.</dd>

      </dl>

      <dl>
        <dt>furiten</dt>
        <dd>
          You may not win on a discard: a wait is in your own discards, or you have passed a winning discard. You may still win on your own draw.
        </dd>

        <dt>Keys</dt>
        <dd>
          Focus your hand first. An arrow key brings the marker up on the tile you drew, the arrows
          move it along your hand, and Enter throws the marked tile. The
          numbers 1 to 9 throw a tile directly and 0 throws the one you just
          drew. Press r to focus the riichi choices and t to win on your own draw. Shortcuts never run while you are using another control and can be turned off in Options.
        </dd>

        <dt>Rules</dt>
        <dd>
          The European Mahjong Association's Riichi Competition Rules, 2025
          edition, in force since 1 January 2026.
        </dd>
      </dl>
    </div>
  </details>
  <details class="offline-settings">
    <summary data-offline-status>{offline.aiReady && offline.coreReady ? 'Offline: game + AI ready' : offline.phase === 'ai' ? `Saving AI… ${offline.progress}%` : offline.coreReady ? 'Offline: game ready' : 'Offline: not ready'}</summary>
    <div class="option-fields">
      <p data-core-status data-core-ready={offline.coreReady} role="status"><strong>Game and all tile graphics — automatic.</strong>
        {offline.coreWarning || (offline.coreReady
          ? 'Fully saved on this device. Beginner and Club already work offline; no download button is needed.'
          : offline.supported === false ? 'Offline storage is unavailable here, but all tile graphics still load before play.'
          : 'Downloading the complete game and every tile graphic automatically. Stay connected until ready.')}</p>
      <p data-ai-status role="status"><strong>Trained AI — optional.</strong>
        {(offline.phase === 'incomplete' && offline.warning) || (offline.aiReady
          ? 'The network and its runtime are saved too.'
          : offline.phase === 'ai' ? `Saving the trained network and runtime… ${offline.progress}%`
          : 'Only the trained network and its runtime need this extra download. Selecting a Trained opponent also starts it automatically.')}</p>
      {#if offline.hasModel && !offline.aiReady}
        <button data-download-ai onclick={downloadAi} disabled={!offline.coreReady || offline.phase === 'ai'}>{offline.phase === 'incomplete' ? 'Retry trained AI download' : 'Download trained AI for offline play'}</button>
      {/if}
      <p class="offline-detail">{offline.persistent ? 'Persistent storage granted.' : 'Your browser can remove website downloads when storage is low.'} Clearing website data removes downloads. On iPhone, check this status inside the Home Screen app before flying.</p>
      {#if offline.updateReady}<p>A new version is downloaded. Close all Mahjong windows and reopen to use it; this match is saved.</p>{/if}
    </div>
  </details>
  </div>
  {#if notice}<p class="notice" role="status">{notice}</p>{/if}

  <dialog class="custom-dialog" bind:this={customDialog} aria-labelledby="custom-table-title">
    <h2 id="custom-table-title">Custom table</h2>
    <p>Choose each opponent for a new match. They keep their type as the seat winds change.</p>
    <div class="opponent-fields">
      {#each [2, 1, 0] as position (position)}
        <label>
          <span>{OPPONENT_POSITIONS[position]}</span>
          <select bind:value={draftOpponents[position]} aria-label={`${OPPONENT_POSITIONS[position]} opponent`}>
            {#each OPPONENT_TYPES as type (type)}
              <option value={type} disabled={type === 'neural' && !trainedAvailable && !opponents.includes('neural')}>{OPPONENT_LABELS[type]}</option>
            {/each}
          </select>
        </label>
      {/each}
    </div>
    <p class="custom-help">Beginner plays simply. Club uses tile efficiency and defence. Trained uses the published network; all Trained players share one loaded model.</p>
    {#if !trainedAvailable && !opponents.includes('neural')}<p class="custom-help">The trained model is not currently available. Beginner and Club can still be mixed.</p>{/if}
    <div class="custom-actions">
      <button onclick={() => customDialog.close()}>Cancel</button>
      <button class="primary" onclick={startCustomTable} disabled={!ready || Boolean(saveConflict)}>Start custom game</button>
    </div>
  </dialog>

  {#if saveConflict}
    <section class="save-conflict" role="alert">
      <p>{saveConflict}</p>
      <button onclick={() => location.reload()}>Reload latest match</button>
    </section>
  {/if}

  {#if failure && (mode === 'play' || !ready)}
    <section class="failure" aria-label="game recovery">
      <p role="alert">{failure}</p>
      {#if !ready}<button onclick={() => location.reload()}>Reload to retry</button>{/if}
      {#if recovery && pendingOpponent}<p class="recovery-note">Waiting for the {pendingOpponent.position.toLowerCase()} opponent. Retry the network or explicitly switch that player to Club.</p>{/if}
      {#if recovery}
        <div class="recovery-actions">
          <button onclick={retryAi} disabled={busy}>Retry trained opponent</button>
          {#if pendingOpponent}
            <button onclick={continueThisOpponent} disabled={busy}>Use Club for {pendingOpponent.position.toLowerCase()} opponent only</button>
          {/if}
          <button onclick={continueClub} disabled={busy}>{difficulty === 'custom' ? 'Use Club for all Trained opponents' : 'Continue with Club opponents'}</button>
          <button onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>
        </div>
      {/if}
    </section>
  {/if}
  {#if storageWarning}<p class="notice" role="status">{storageWarning}</p>{/if}

  {#if !ready && !failure}
    <p class="loading" role="status">{startupNote}</p>
  {:else if ready && mode === 'watch'}
    <AgentWatch {ready} {trainedAvailable} {opponents} {hints} />
  {:else if ready && mode === 'guided'}
    <GuidedPlay {ready} {trainedAvailable} {storage} {hints} />
  {:else if ready && mode === 'physical'}
    <PhysicalPlay {ready} {trainedAvailable} {storage} />
  {:else if mode === 'play' && view}
    <div class="board">
      <div class="place across"><Seat seat={across} side="across" dealer={across.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === across.player} /></div>
      <div class="place left"><Seat seat={left} side="left" dealer={left.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === left.player} /></div>
      <div class="centre" aria-label="the table">
        <div class="round-details">
          <div class="round"><strong>{NAMES[view.round]} {view.kyoku}</strong><span>Round</span></div>
          <div class="wall"><strong>{view.wall}</strong><span>tiles left</span></div>
        </div>
        <div class="table-indicators">
          <div class="indicator-line"><span>Dora</span><div class="indicators" aria-label="dora indicators">
            {#each view.dora_indicators as indicator, slot (slot)}<Tile tile={indicator} size="small" />{/each}
          </div></div>
          {#if uraIndicators.length}
            <div class="indicator-line"><span>Ura-dora</span><div class="ura-indicators" aria-label="ura-dora indicators">
              {#each uraIndicators as indicator, slot (slot)}<Tile tile={indicator} size="small" />{/each}
            </div></div>
          {/if}
        </div>
        {#if view.counters || view.riichi_sticks}
          <div class="table-extras">
            {#if view.counters}<span>{view.counters} honba (+{view.counters * 300})</span>{/if}
            {#if view.riichi_sticks}<span>{view.riichi_sticks} riichi bet{view.riichi_sticks === 1 ? '' : 's'}</span>{/if}
          </div>
        {/if}
        <button class="inspect" onclick={inspectTable} aria-label="Inspect all discards and called sets">
          <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="3" y="3" width="5" height="6" rx="1" /><rect x="12" y="3" width="5" height="6" rx="1" /><rect x="3" y="12" width="5" height="5" rx="1" /><rect x="12" y="12" width="5" height="5" rx="1" /></svg>
          Discards
        </button>
      </div>
      <div class="place right"><Seat seat={right} side="right" dealer={right.seat === 'east'} dora={shownDora} thinking={thinking && pendingOpponent?.player === right.player} /></div>
    </div>

    <div class="play-area" class:ended={view.phase === 'over' || Boolean(standings)}>
      <section class="mine" class:turn={myTurn} aria-label="your seat">
        <header>
          <strong>You are {NAMES[me.seat]}</strong>
          <span class="score">{me.score.toLocaleString()}</span>
          {#if me.seat === 'east'}<span class="my-dealer">Dealer</span>{/if}
          {#if me.riichi}<span class="riichi">Riichi</span>{/if}
          {#if hints}
            <span class="hint">
              {#if !discardHint && view.shanten < 0}Complete tile shape
              {:else if displayWaits.length}
                {discardHint ? `After discarding ${tileWords(previewTile)}, waiting on` : me.riichi || !me.drawn ? 'Waiting on' : 'Wait before this draw:'}
                {#each displayWaits as wait, index (index)}
                  <span class="wait"><Tile tile={wait} size="tiny" dora={shownDora.includes(wait)} /><span class="remaining" class:none={displayLeft[index] === 0} aria-label="{displayLeft[index]} unseen">{displayLeft[index]}</span></span>
                {/each}
              {:else if (discardHint?.shanten ?? view.shanten) === 0}
                Select a discard to see its waits
              {:else}
                {#if discardHint}After discarding {tileWords(previewTile)}: {/if}
                {discardHint?.shanten ?? view.shanten} tile{(discardHint?.shanten ?? view.shanten) === 1 ? '' : 's'} from a wait
              {/if}
            </span>
          {/if}
        </header>
        {#if hints && (safeCount || view.dora.length || view.furiten)}
          <div class="hand-facts">
            {#if safeCount}<span class="safe-note">{safeCount} held tile{safeCount === 1 ? '' : 's'} safe against declared riichi</span>{/if}
            {#if view.dora.length}<span class="dora-note">{view.dora.length} dora han in hand</span>{/if}
            {#if view.furiten}<span class="furiten">Furiten — self-draw wins only</span>{/if}
          </div>
        {/if}
        <div class="hand" class:has-draw={Boolean(me.drawn)} role="group" aria-label="your tiles" aria-describedby={view.phase === 'over' ? undefined : 'hand-help'} tabindex="-1" bind:this={handElement} onfocusin={syncHandFocus}>
          {#each handTiles as tile, index (index)}
            <HandTile {tile} handIndex={index} onclick={() => selectTile(tile, index)}
              disabled={!canDiscard(tile)} muted={view.phase === 'over'} selected={myTurn && (picked === index || selected === index)}
              drawn={Boolean(me.drawn) && index === me.hand.length}
              discardShanten={discardHints.get(tile)?.shanten ?? null}
              remaining={remainingByTile.get(tile) ?? null} showRemaining={hints}
              safe={hints && view.phase !== 'over' && view.safe.includes(tile)} dora={shownDora.includes(tile)} />
          {/each}
        </div>
        {#if me.melds.length}<div class="my-melds"><Melds melds={me.melds} size="small" dora={shownDora} /></div>{/if}
        <div class="own-discards">
          <span class="caption">Your discards ({me.discards.length})</span>
          <Discards discards={me.discards} compact={false} dora={shownDora} />
        </div>
      </section>

      <section class="controls" aria-label="your choices" bind:this={callElement}>
        {#if standings}
          <Standings {standings} onagain={() => start()} />
        {/if}
        {#if view.phase === 'over' && view.outcome}
          <ScoreScreen outcome={view.outcome} seats={view.seats} dora={shownDora} {hints} {busy}
            bets={view.riichi_sticks ?? 0} gameOver={session?.over ?? false} onnext={nextHand}
            ongame={() => start()} onreview={showReview} reviewed={notes !== null} onlog={saveLog} finalHand={Boolean(standings)} />
          {#if notes !== null}<Review {notes} {hints} engine={session?.engine} {trainedAvailable} bind:adviser={reviewAdviser} />{/if}
        {:else}
          <p id="hand-help" class="prompt" role="status">
            {#if saveConflict}Reload the latest match to continue here.
            {:else if failure}Choose a recovery option above to continue.
            {:else if busy}{thinking ? (loadNote || (pendingOpponent ? `${pendingOpponent.position} Trained opponent is thinking…` : 'Trained opponents are thinking…')) : 'Playing the turn…'}
            {:else if myTurn}
              {confirmDiscards ? 'Your turn. Select a tile, then confirm the discard.' : 'Your turn. Choose a tile to discard.'}
              {#if !touch && shortcuts}<span class="key-help">Focus your hand to use arrows and Enter.</span>{/if}
            {:else if !callChoices.length}Waiting for the others…
            {:else}Choose a call, or pass.{/if}
          </p>
          {#if view.phase === 'call' && view.pending_discard && callChoices.length}
            <div class="call-stage" aria-live="polite">
              <span class="call-kicker">Discard</span>
              <Tile tile={view.pending_discard} dora={shownDora.includes(view.pending_discard)} />
              <span class="call-message"><strong>{NAMES[view.pending_from] ?? 'An opponent'}</strong><span>discarded the {tileWords(view.pending_discard)}</span></span>
            </div>
          {/if}
          {#if selectedTile && myTurn && !busy}
            <div class="confirm-discard" aria-live="polite">
              <span>Selected: {tileWords(selectedTile)}</span>
              <button class="primary" onclick={() => discard(selectedTile)}>Discard {tileWords(selectedTile)}</button>
              <button onclick={() => selected = null}>Cancel</button>
            </div>
          {/if}
          {#if callChoices.length}
            <div class="call-options">
              {#each callChoices as choice, index (index)}
                <button class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} class:win-call={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}
                  aria-label={callLabel(choice)} disabled={busy || Boolean(failure)} onclick={() => choose(choice)}>
                  <span class="call-label">{callLabel(choice)}</span>
                  {#if callTiles(choice, view.pending_discard).length}
                    <span class="call-preview" aria-hidden="true">
                      {#each callTiles(choice, view.pending_discard) as tile, slot (slot)}<Tile {tile} size="tiny" />{/each}
                    </span>
                  {/if}
                </button>
              {/each}
            </div>
          {/if}
        {/if}
      </section>
    </div>

    <details class="history">
      <summary>Hand history · {log.length} event{log.length === 1 ? '' : 's'}</summary>
      <section class="log" aria-label="what happened, oldest first">
        {#each log as line, index (index)}<p>{line}</p>{/each}
      </section>
    </details>

    <dialog class="table-dialog" bind:this={tableDialog} aria-labelledby="table-dialog-title">
      <header><h2 id="table-dialog-title">All discards and called sets</h2><button onclick={() => tableDialog.close()}>Close</button></header>
      <div class="inspection-grid">
        {#each view.seats as seat, index (index)}
          <section><h3>{index === 0 ? 'You' : NAMES[seat.seat]} · {NAMES[seat.seat]} · {seat.score.toLocaleString()}</h3>
            <Discards discards={seat.discards} dora={shownDora} />
            {#if seat.melds.length}<Melds melds={seat.melds} dora={shownDora} />{/if}
            {#if seat.discards.length === 0}<p>No discards yet.</p>{/if}
          </section>
        {/each}
      </div>
    </dialog>
  {/if}
</main>

<style>
  .game-modes { display: flex; gap: 4px; grid-column: 1 / -1; padding: 3px; border-radius: 12px; background: #0002; }
  .game-modes button { flex: 1; min-width: 0; padding: 8px 6px; border-color: transparent; border-radius: 9px; background: transparent; color: color-mix(in srgb, var(--ivory) 72%, transparent); font-size: .8rem; }
  .game-modes button[aria-pressed=true] { color: var(--ivory); background: #ffffff16; box-shadow: 0 1px 3px #0002; }
  .compact-mode-selector { display: none; }
  @media (max-width: 360px) and (max-height: 640px) {
    .game-modes { display: none; }
    .compact-mode-selector { display: flex; flex-basis: 100%; align-items: center; justify-content: space-between; gap: 8px; margin: 4px 0 8px; font-size: .8rem; }
    .compact-mode-selector select { min-width: 0; max-width: 70%; }
  }
  .custom-dialog { width: min(460px, calc(100vw - 24px)); max-height: calc(100dvh - 24px); box-sizing: border-box; padding: 20px; border: 1px solid var(--gold); border-radius: 12px; background: var(--felt-deep); color: var(--ivory); }
  .custom-dialog::backdrop { background: #000a; }
  .custom-dialog h2 { margin: 0; font-size: 1.15rem; }
  .custom-dialog p { font-size: .85rem; line-height: 1.45; }
  .opponent-fields { display: grid; gap: 10px; }
  .opponent-fields label { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .opponent-fields select { width: 60%; min-width: 0; }
  .custom-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; }
  .custom-help, .recovery-note { opacity: .85; }
  .edit-table { font-size: .8rem; padding: 4px 8px; }
  .save-conflict { padding: 12px; border: 1px solid var(--gold); border-radius: 8px; background: var(--felt-deep); }
  .save-conflict p { margin: 0 0 8px; }
  .save-conflict button { min-height: 44px; padding: 8px 14px; }
  main { width: 100%; max-width: 1100px; box-sizing: border-box; margin: 0 auto; grid-template-columns: minmax(0, 1fr); padding: max(10px, env(safe-area-inset-top)) max(10px, env(safe-area-inset-right)) max(24px, env(safe-area-inset-bottom)) max(10px, env(safe-area-inset-left)); display: grid; gap: 10px; }
  .bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; border-bottom: 1px solid #ffffff28; padding-bottom: 8px; }
  h1 { margin: 0; font-size: 1.1rem; letter-spacing: .16em; text-transform: uppercase; }
  .opponents { position: relative; margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: .85rem; }
  select, button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 12px; font: inherit; }
  button { cursor: pointer; touch-action: manipulation; }
  button:hover:not(:disabled) { background: #0007; }
  button:disabled { opacity: .55; cursor: default; }
  select option { color: #17241f; background: #f7f2e4; }
  .opponents select { appearance: none; padding-right: 34px; border-color: #ffffff24; border-radius: 12px; background: #ffffff09; }
  .select-chevron { position: absolute; right: 12px; width: 14px; height: 14px; pointer-events: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; opacity: .7; }
  .restart { font-size: .85rem; }
  .preferences { display: flex; flex-wrap: wrap; align-items: center; gap: 0 20px; min-width: 0; }
  .preferences details[open] { flex-basis: 100%; order: 1; }
  .options, .guide, .history, .offline-settings { font-size: .85rem; min-width: 0; }
  summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .option-fields { display: flex; gap: 4px 20px; flex-wrap: wrap; background: #0003; padding: 10px; border-radius: 8px; }
  .option-fields label { min-height: 44px; display: flex; gap: 8px; align-items: center; }
  .option-fields p { flex-basis: 100%; margin: 4px 0; max-width: 70ch; }
  input[type=checkbox] { width: 20px; height: 20px; accent-color: var(--gold); }
  .guide-body { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 8px 28px; padding: 10px 14px; border-radius: 10px; background: #0003; }
  .guide dl { margin: 0; min-width: 0; }
  .guide dt { font-weight: 600; margin-top: 8px; display: flex; align-items: center; gap: 6px; }
  .guide dd { margin: 2px 0 0; max-width: 62ch; }
  .swatch { display: inline-block; width: 12px; height: 16px; margin-right: 4px; border-radius: 3px; background: var(--ivory); flex: none; }
  .swatch.dora { box-shadow: 0 0 0 2px #e2453d; }
  .swatch.safe { box-shadow: 0 0 0 2px #7fd1a0; }
  .swatch.one-away { box-shadow: 0 0 0 2px #c5cbd3; }
  .swatch.ready { box-shadow: 0 0 0 2px var(--gold); }
  .swatch.marker { box-shadow: 0 0 0 2px #4ea3ff; }
  .swatch.striped { border: 3px solid transparent; background: linear-gradient(var(--ivory),var(--ivory)) padding-box, repeating-linear-gradient(45deg,#e2453d 0 4px,var(--gold) 4px 8px,#7fd1a0 8px 12px) border-box; }
  .board { display: grid; grid-template-columns: minmax(0,1fr) minmax(190px,auto) minmax(0,1fr); gap: 10px; align-items: start; }
  .across { grid-area: 1 / 2; }
  .left { grid-area: 2 / 1; justify-self: start; }
  .right { grid-area: 2 / 3; justify-self: end; }
  .centre { grid-area: 2 / 2; display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px 18px; max-width: 350px; border-radius: 12px; padding: 12px; background: #0004; }
  .round-details { display: contents; }
  .round, .wall { display: grid; text-align: center; }
  .round strong, .wall strong { font-size: 1.3rem; }
  .round span, .wall span { font-size: .7rem; }
  .table-indicators { display: grid; gap: 6px; min-width: 0; }
  .indicator-line { display: grid; gap: 4px; font-size: .65rem; }
  .ura-indicators { display: flex; gap: 4px; flex-wrap: wrap; align-items: flex-end; }
  .indicators { display: flex; gap: 4px; flex-wrap: wrap; align-items: flex-end; }
  .table-extras { display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: .8rem; color: var(--gold); }
  .inspect { display: inline-flex; align-items: center; justify-content: center; gap: 6px; font-size: .8rem; }
  .inspect svg { width: 17px; height: 17px; flex: none; stroke: currentColor; stroke-width: 1.25; }
  .play-area { display: grid; gap: 10px; min-width: 0; }
  .mine { display: grid; gap: 10px; padding: 10px 12px; border-radius: 12px; background: #0004; border: 1px solid #d8a12a60; min-width: 0; }
  .mine header { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; font-size: .9rem; }
  .score { font-variant-numeric: tabular-nums; }
  .riichi, .furiten { color: var(--warning-text); font-weight: 600; }
  .hint { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-left: auto; font-size: .85rem; }
  .wait { position: relative; display: inline-flex; margin-right: 4px; }
  .remaining { position: absolute; right: -3px; bottom: -2px; min-width: 12px; padding: 0 2px; border-radius: 6px; background: #231d12; color: #fff; font-size: .65rem; line-height: 1.2; text-align: center; }
  .remaining.none { background: #932812; }
  .hand-facts { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: .8rem; }
  .safe-note { color: #9cddb5; }
  .dora-note { color: var(--gold); }
  .hand { --draw-gap: 12px; display: flex; gap: 5px; align-items: end; flex-wrap: wrap; padding: 6px 4px; min-width: 0; }
  .hand :global(button.tile[data-drawn=true]) { margin-inline-start: var(--draw-gap); }
  /* The hand takes focus on every turn, so its ring is a hint, not a
     frame: softer than a control's, and set out from the tiles. */
  .hand:focus-visible { border-radius: 8px; outline: 2px solid rgba(216, 161, 42, 0.45); outline-offset: 6px; }
  .my-melds { padding: 4px; }
  /* Always in view: the row is your furiten record and what the table
     sees of you, so it is not folded away on any screen. */
  .own-discards { display: grid; gap: 4px; }
  .own-discards .caption { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6; }
  .controls { display: grid; gap: 8px; min-width: 0; }
  .prompt { margin: 0; font-size: .9rem; }
  .key-help { display: block; font-size: .78rem; }
  .call-options { display: flex; flex-wrap: wrap; gap: 8px; }
  .call-options button { display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; text-align: left; }
  .call-label { min-width: 0; overflow-wrap: anywhere; }
  .call-preview { display: inline-flex; align-items: center; gap: 4px; }
  .confirm-discard { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .confirm-discard > span { font-size: .85rem; }
  button.primary { background: var(--button-accent); color: var(--button-text); border-color: var(--button-accent); font-weight: 600; }
  button.primary:hover { background: var(--button-accent-hover); }
  .failure { background: var(--accent-soft); color: var(--accent); padding: 12px; border-radius: 8px; }
  .failure p { margin: 0; }
  .recovery-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .notice { margin: 0; font-size: .8rem; }
  .log { font-size: .82rem; display: grid; gap: 4px; max-height: 40vh; overflow-y: auto; padding: 8px 12px; background: #0003; border-radius: 8px; }
  .log p { margin: 0; }
  .table-dialog { max-width: min(900px, calc(100vw - 20px)); width: 100%; max-height: 90dvh; background: var(--felt-deep); color: var(--ivory); border: 1px solid #d8a12a99; border-radius: 12px; padding: 14px; }
  .table-dialog::backdrop { background: #0009; }
  .table-dialog header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .table-dialog h2 { margin: 0; font-size: 1rem; }
  .inspection-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,280px),1fr)); gap: 16px; --tile-width: 60px; }
  .inspection-grid section { min-width: 0; display: grid; gap: 8px; align-content: start; }
  .inspection-grid h3 { font-size: .85rem; }
  @media (max-width: 760px) {
    main { gap: 6px; }
    .bar { gap: 8px; flex-wrap: nowrap; }
    h1 { font-size: 1rem; letter-spacing: .1em; }
    .bar select, .restart { padding: 8px; }
    .preferences { gap: 0 16px; }
    .opponents { gap: 5px; }
    .opponents > span { display: none; }
    .options, .guide, .offline-settings { font-size: .8rem; }
    .board { grid-template-columns: repeat(3,minmax(0,1fr)); gap: 6px; }
    .centre { grid-area: 1 / 1 / 2 / -1; max-width: none; justify-content: space-between; gap: 6px 10px; padding: 8px; }
    .round strong, .wall strong { font-size: 1rem; }
    .round span, .wall span { font-size: .65rem; }
    .indicators, .ura-indicators { --tile-width: 36px; }
    .left { grid-area: 2 / 1; justify-self: stretch; }
    .across { grid-area: 2 / 2; }
    .right { grid-area: 2 / 3; justify-self: stretch; }
    .place { --tile-width: 28px; }
    .play-area:not(.ended) { position: sticky; top: 6px; bottom: auto; z-index: 2; background: var(--felt-deep); padding: 6px 0 max(6px, env(safe-area-inset-bottom)); border-radius: 10px; box-shadow: 0 -6px 16px #0003; }
    .mine { padding: 8px; gap: 4px; }
    .mine header { font-size: .8rem; }
    .hint { margin-left: 0; font-size: .75rem; }
    .hand { display: grid; grid-template-columns: repeat(7,minmax(0,1fr)); gap: 6px; padding: 6px 3px; }
    .hand :global(button.tile) { width: 100%; min-height: 44px; }
    /* Reserve the extra gap within the grid, keeping all tile faces equal
       sized and the drawn tile inside the viewport even on a 320px phone. */
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
    .controls { padding: 0 8px; }
    .prompt { font-size: .82rem; }
    .call-options { gap: 6px; }
    .call-options button { font-size: .8rem; padding: 6px 10px; }
  }
  @media (max-width: 359px) {
    .place { --tile-width: 23px; }
    .preferences { column-gap: 8px; }
    .opponents { min-width: 0; }
    .bar select { min-width: 0; max-width: 100%; }
    .restart { white-space: nowrap; flex-shrink: 0; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    main { max-width: none; grid-template-columns: minmax(270px, .85fr) minmax(340px, 1.15fr); align-items: start; }
    .bar, .preferences, .notice, .failure, .save-conflict, .loading, .history { grid-column: 1 / -1; }
    .board { grid-column: 1; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 4px; }
    .centre { grid-area: 1 / 1 / 2 / -1; max-width: none; gap: 4px 8px; padding: 6px; }
    .left { grid-area: 2 / 1; justify-self: stretch; }
    .across { grid-area: 2 / 2; }
    .right { grid-area: 2 / 3; justify-self: stretch; }
    .place { --tile-width: 23px; }
    .play-area { grid-column: 2; grid-row: auto; position: sticky; top: 8px; bottom: auto; }
    .mine { padding: 6px 8px; gap: 4px; }
    .mine header { font-size: .8rem; }
    .hand { display: grid; grid-template-columns: repeat(7,minmax(0,44px)); gap: 5px; padding: 6px 3px; }
    .controls { font-size: .8rem; }
    .prompt { font-size: .8rem; }
    .own-discards .caption { font-size: .62rem; }
    .hand :global(button.tile) { width: 100%; min-height: 44px; }
    /* Reserve the extra gap within the grid, keeping all tile faces equal
       sized and the drawn tile inside the viewport even on a 320px phone. */
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
    .hint { margin-left: 0; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }


  /* --- 2026 game-feel pass ------------------------------------------------ */
  .settings-trigger, .settings-backdrop,
  .mobile-preferences-head, .mobile-new-game { display: none; }

  .my-dealer {
    padding: 1px 7px;
    border: 1px solid rgba(216, 161, 42, .55);
    border-radius: 999px;
    color: var(--gold);
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  .mine.turn {
    border-color: rgba(216, 161, 42, .9);
    box-shadow: 0 0 0 1px rgba(216, 161, 42, .16), 0 0 24px rgba(216, 161, 42, .10);
  }

  /* Counts sit below both hand tiles and wait tiles; never cover artwork. */
  .wait {
    position: static;
    display: inline-grid;
    justify-items: center;
    align-items: end;
    gap: 3px;
    margin-right: 5px;
  }
  .remaining {
    position: static;
    min-width: 18px;
    padding: 2px 3px 0;
    border-top: 1px solid rgba(247, 242, 228, .26);
    border-radius: 0;
    background: none;
    color: rgba(247, 242, 228, .75);
    font-size: .62rem;
    font-weight: 650;
    line-height: 1;
  }
  .remaining.none {
    border-color: rgba(255, 184, 164, .55);
    background: none;
    color: var(--warning-text);
  }

  .hand { --draw-gap: 18px; align-items: flex-end; }
  .hand :global(button.tile[data-drawn=true]) { margin-inline-start: 0; }
  .hand :global(.hand-tile[data-hand-drawn=true]) { margin-inline-start: var(--draw-gap); }

  .call-stage {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    align-items: center;
    gap: 9px 12px;
    padding: 10px 12px;
    border: 1px solid rgba(216, 161, 42, .45);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(216,161,42,.11), rgba(0,0,0,.23));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    animation: offer-in .24s ease-out both;
  }
  .call-kicker {
    grid-column: 1 / -1;
    margin-bottom: -5px;
    color: var(--gold);
    font-size: .64rem;
    font-weight: 750;
    letter-spacing: .14em;
    text-transform: uppercase;
  }
  .call-message { display: grid; min-width: 0; line-height: 1.25; }
  .call-message span { opacity: .78; font-size: .82rem; }
  .call-options { animation: choices-in .22s .06s ease-out both; }
  .call-options button.win-call {
    border-color: #efc45e;
    background: linear-gradient(180deg, #b93b24, #8e2414);
    box-shadow: 0 5px 16px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.16);
  }
  @keyframes offer-in { from { opacity: 0; transform: translateY(5px) scale(.985); } to { opacity: 1; transform: none; } }
  @keyframes choices-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  @media (min-width: 761px) and (min-height: 501px) {
    main { max-width: 1280px; padding-inline: 18px; gap: 14px; }
    .board {
      grid-template-columns: minmax(260px, 1fr) minmax(330px, .92fr) minmax(260px, 1fr);
      min-height: 390px;
      padding: 18px 20px;
      gap: 14px 20px;
      border: 1px solid rgba(247,242,228,.10);
      border-radius: 28px;
      background:
        radial-gradient(ellipse at center, rgba(78,150,109,.18) 0 26%, transparent 57%),
        linear-gradient(145deg, rgba(255,255,255,.025), rgba(0,0,0,.13));
      box-shadow: inset 0 0 50px rgba(0,0,0,.13), 0 16px 34px rgba(0,0,0,.10);
    }
    .place { --tile-width: 54px; align-self: stretch; }
    .across { width: min(450px, 100%); justify-self: center; }
    .left, .right { width: min(340px, 100%); align-self: center; }
    .centre {
      width: min(340px, 100%);
      min-height: 190px;
      align-self: center;
      justify-self: center;
      padding: 18px;
      border: 1px solid rgba(216,161,42,.28);
      border-radius: 20px;
      background: radial-gradient(circle at 50% 35%, rgba(47,111,80,.54), rgba(3,25,16,.48));
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 12px 28px rgba(0,0,0,.14);
    }
    .mine {
      --tile-width: 52px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(4, 30, 20, .43);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
    }
    .inspect { border-radius: 999px; }
  }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    main { padding-top: max(5px, env(safe-area-inset-top)); gap: 5px; }
    .bar {
      position: sticky;
      top: 0;
      z-index: 18;
      min-height: 56px;
      padding: 4px 2px;
      gap: 8px;
      border-bottom: 0;
      background: color-mix(in srgb, var(--felt) 92%, transparent);
      backdrop-filter: blur(12px);
    }
    h1 { margin-right: auto; font-size: 1rem; letter-spacing: .18em; white-space: nowrap; }
    .opponents { margin-left: 0; }
    .opponents select { min-height: 44px; width: 128px; max-width: 128px; padding: 5px 30px 5px 12px; font-size: .82rem; }
    .settings-trigger {
      display: inline-flex;
      width: 44px;
      min-height: 44px;
      align-items: center;
      justify-content: center;
      padding: 0;
      flex: none;
      border-color: #ffffff24;
      border-radius: 12px;
      background: #ffffff09;
    }
    .settings-trigger svg { width: 22px; height: 22px; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; }
    .restart { display: none; }

    .settings-backdrop {
      display: block;
      position: fixed;
      inset: 0;
      z-index: 38;
      width: 100%;
      height: 100%;
      min-height: 0;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: rgba(0,0,0,.58);
    }
    .preferences { display: none; }
    .preferences.mobile-open {
      display: flex;
      position: fixed;
      left: 8px;
      right: 8px;
      bottom: max(8px, env(safe-area-inset-bottom));
      z-index: 39;
      max-height: min(82dvh, 680px);
      overflow: auto;
      align-content: flex-start;
      align-items: stretch;
      gap: 2px 12px;
      padding: 14px;
      border: 1px solid rgba(216,161,42,.45);
      border-radius: 20px;
      background: color-mix(in srgb, var(--felt-deep) 96%, black 4%);
      box-shadow: 0 22px 70px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.05);
    }
    .preferences.mobile-open details, .preferences.mobile-open .edit-table { flex-basis: 100%; }
    .mobile-preferences-head {
      display: flex;
      flex-basis: 100%;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 4px;
      font-size: 1rem;
    }
    .mobile-preferences-head button { min-height: 44px; padding: 5px 11px; }
    .mobile-new-game { display: block; flex-basis: 100%; margin-bottom: 5px; }
    .preferences .option-fields { background: rgba(0,0,0,.24); }
    .notice {
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,.10);
      border-radius: 10px;
      background: #0002;
      font-size: .78rem;
    }

    .board { margin-top: 3px; }
    .centre {
      display: grid;
      grid-template-columns: minmax(78px, 1fr) minmax(0, auto) auto;
      gap: 8px;
      padding: 11px 12px;
      border: 0;
      border-radius: 14px;
      background: #0002;
      box-shadow: none;
    }
    .round-details { display: grid; grid-area: 1 / 1; gap: 1px; }
    .round { text-align: left; }
    .round strong { font-size: 1.18rem; font-weight: 650; line-height: 1.35; letter-spacing: -.02em; }
    .round span { display: none; }
    .wall { display: flex; gap: 4px; align-items: baseline; text-align: left; color: color-mix(in srgb, var(--ivory) 68%, transparent); }
    .wall strong, .wall span { font-size: .73rem; font-weight: 400; line-height: 1.5; }
    .wall strong { font-variant-numeric: tabular-nums; }
    .table-indicators { grid-area: 1 / 2; max-width: 140px; }
    .indicator-line { gap: 3px; font-size: .6rem; }
    .indicator-line > span { color: color-mix(in srgb, var(--ivory) 68%, transparent); }
    .indicators, .ura-indicators { --tile-width: 42px; gap: 3px; }
    .table-extras { grid-column: 1 / -1; gap: 4px 12px; padding-top: 5px; border-top: 1px solid #ffffff14; font-size: .7rem; }
    .inspect { grid-area: 1 / 3; padding: 8px 10px; border-color: transparent; border-radius: 10px; background: #ffffff0d; font-size: .76rem; }
    .mine { border-radius: 15px; background: rgba(3,29,19,.40); }
    .hand { --draw-gap: 14px; }
    .hand :global(.hand-tile) { width: 100%; }
    .hand :global(.hand-tile[data-hand-drawn=true]) { margin-inline-start: var(--draw-gap); }
    .hand.has-draw { padding-inline-end: calc(3px + var(--draw-gap)); }
    .call-stage { grid-template-columns: auto auto minmax(0,1fr); padding: 9px 10px; }

    .custom-dialog {
      position: fixed;
      inset: auto 0 0 0;
      width: 100%;
      max-width: none;
      max-height: 86dvh;
      margin: 0;
      padding: 18px 16px max(18px, env(safe-area-inset-bottom));
      overflow: auto;
      border-radius: 22px 22px 0 0;
      box-shadow: 0 -24px 80px rgba(0,0,0,.42);
    }
  }
</style>
