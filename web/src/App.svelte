<script>
  import './lib/app/controls.css';
  import AppSettings from './lib/app/AppSettings.svelte';
  import OpponentDialog from './lib/app/OpponentDialog.svelte';
  import MatchTable from './lib/app/MatchTable.svelte';
  import PlayerHand from './lib/app/PlayerHand.svelte';
  import TurnChoices from './lib/app/TurnChoices.svelte';
  import { onMount, setContext, tick, untrack } from 'svelte';
  import init, { Game } from './wasm/riichi.js';
  import { preloadTiles } from './lib/tile-preload.js';
  import { INITIAL_OFFLINE, startOffline, watchOffline, prepareOfflineAi, refreshOffline } from './lib/offline.js';
  import ScoreScreen from './lib/ScoreScreen.svelte';
  import Standings from './lib/Standings.svelte';
  import Review from './lib/Review.svelte';
  import GameReview from './lib/GameReview.svelte';
  import { MatchReview } from './lib/game-review.js';
  import { readTrend, recordTrend } from './lib/review-trend.js';
  import AgentWatch from './lib/AgentWatch.svelte';
  import PhysicalPlay from './lib/PhysicalPlay.svelte';
  import GuidedPlay from './lib/GuidedPlay.svelte';
  import { analyzeBatch, chooseAction, modelIsAvailable, reportProgress, resetPolicy } from './lib/policy.js';
  import { watchModelAvailability } from './lib/model-availability.js';
  import { MatchSession, readSettings, writeSettings } from './lib/session.js';
  import { acceptsHandKey, moveHandFocus, pressedByKeyboard, waitingNote } from './lib/ui.js';
  import { MatchStore } from './lib/save-store.js';
  import { gameFileName, matchLabels, playerNames, saveLogFile } from './lib/game-log.js';
  import { TILE_FACE_CONTEXT, normalizeTileFace } from './lib/tile-faces.js';
  import { DEFAULT_OPPONENT, normalizeOpponents, opponentPreset, OPPONENT_TYPES, sameOpponents } from './lib/opponents.js';
  import { MANIFEST, playNetwork } from './lib/trained-network.js';

  const storage = (() => { try { return window.localStorage; } catch { return null; } })();
  const touch = matchMedia('(pointer: coarse)').matches;
  const preferences = readSettings(storage, touch);
  // The version of the trained network the player picked, if any, and the
  // network the page plays for it, the default without one. Picking another
  // reloads the page (changeNetwork), so every download, digest check and
  // offline report this page makes is about this network.
  let trainedNetwork = preferences.trainedNetwork;
  const network = playNetwork(trainedNetwork);
  const requested = new URLSearchParams(location.search).get('opponents');
  const requestedMode = new URLSearchParams(location.search).get('mode');
  let mode = $state(['watch', 'physical', 'guided'].includes(requestedMode) ? requestedMode : 'play');
  let opponents = $state(normalizeOpponents(OPPONENT_TYPES.includes(requested) ? requested : preferences.opponents ?? preferences.difficulty));
  // The table as the settings name it: the one type when all three opponents
  // match, otherwise 'custom'. It follows the opponents, so the two agree.
  let difficulty = $derived(opponentPreset(opponents));
  // Replaced by the current table each time the dialog opens.
  let draftOpponents = $state(normalizeOpponents(DEFAULT_OPPONENT));
  let customDialog = $state(null);
  let pendingOpponent = $state(null);
  let hints = $state(preferences.hints);
  let confirmDiscards = $state(preferences.confirmDiscards);
  let shortcuts = $state(preferences.shortcuts);
  let tileFace = $state(preferences.tileFace);
  let pendingTileFace = $state(null);
  let faceWarning = $state('');
  let faceLoad = null;
  let reviewAdviser = $state(preferences.reviewAdviser);
  setContext(TILE_FACE_CONTEXT, () => tileFace);
  let ready = $state(false);
  let startupNote = $state('Loading the game and selected tile graphics…');
  // A page plays one trained network, so its download is the only optional
  // one: readiness and availability are read straight from offline.
  let offline = $state(INITIAL_OFFLINE);
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
  // How many hands the game's log holds: the finished ones only.
  let loggedHands = $state(0);
  let notes = $state(null);
  // The hand review's heading, which takes focus as the review opens.
  let reviewHeading = $state(null);
  // The final standings' heading, which takes focus as the game ends, and
  // whether the keyboard sent it there, for the result to step aside for it.
  let standingsHeading = $state(null);
  let standingsFocused = $state(false);
  let picked = $state(null);
  let selected = $state(null);
  let handElement = $state(null);
  let callElement = $state(null);
  let settingsOpen = $state(false);
  // The whole-game review of the finished match, kept while that match is,
  // and whether its screen is open. It reviews on while the screen is shut.
  let matchReview = $state.raw(null);
  let reviewingGame = $state(false);
  // The reviewed games this browser keeps, which a finished review joins.
  let reviewTrend = $state.raw(readTrend(storage));

  let me = $derived(view?.seats[0]);
  let handTiles = $derived(me ? [...me.hand, ...(me.drawn ? [me.drawn] : [])] : []);
  let discardChoices = $derived(choices.filter((choice) => choice.kind === 'discard'));
  let callChoices = $derived(choices.filter((choice) => choice.kind !== 'discard'));
  let myTurn = $derived(view?.phase === 'act' && me?.turn);
  let shownDora = $derived(hints ? (view?.dora_types ?? []) : []);
  let selectedTile = $derived(selected === null ? null : handTiles[selected]);

  // Stored when a preference changes, not whenever this effect reruns. A
  // player who has not picked a network has no trainedNetwork in the record.
  let storedSettings = null;
  const currentSettings = () =>
    ({ difficulty, opponents: [...opponents], hints, confirmDiscards, shortcuts, tileFace, reviewAdviser, trainedNetwork });
  $effect(() => {
    storedSettings = writeSettings(storage, currentSettings(), storedSettings);
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
    // Every move updates the view. A fresh copy of an unchanged table would
    // rerun the preferences effect each time, over other tabs' changes.
    if (!sameOpponents(opponents, owner.opponents)) opponents = [...owner.opponents];
    pendingOpponent = owner.pendingOpponent;
    standings = owner.over ? owner.engine.standings() : null;
    loggedHands = owner.engine.game_log_hands();
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
    loggedHands = 0;
    customDialog?.close();
    selected = null;
    picked = null;
    dropGameReview();
  }

  const matchStore = new MatchStore(storage, navigator.locks, {
    onConflict: stopForConflict,
    onWarning: (message) => { storageWarning = message; },
  });

  const callbacks = {
    // Greedy: the opponent plays its best move, not a sample of them.
    ai: (planes, mask, signal) => chooseAction(planes, mask, signal),
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
    const unwatchModel = watchModelAvailability({
      probe: modelIsAvailable, watchOffline, refreshOffline,
      onChange: available => { if (mounted) trainedAvailable = available; },
    });
    reportProgress((note) => { if (mounted && thinking) loadNote = note; });
    const startup = new AbortController();
    // Full-package caching has its own honest status. An unavailable unused
    // face set must not prevent a game with the selected graphics opening.
    void startOffline();
    Promise.all([init(), preloadTiles((done, total) => {
      if (mounted) startupNote = `Loading selected tile graphics… ${done}/${total}`;
    }, { face: tileFace, signal: startup.signal })]).then(() => {
      if (mounted && !startup.signal.aborted) ready = true;
    }).catch((error) => {
      if (mounted && !startup.signal.aborted) failure = `The game could not finish loading: ${error.message ?? error}. Reconnect and reload to retry.`;
    });
    // Completed actions are already saved inside their writer transaction.
    // Never write an old snapshot during pagehide. A cached page must restore
    // the current record rather than resume a stale engine on Back navigation.
    const leave = () => { startup.abort(); faceLoad?.abort(); matchReview?.dispose(); session?.dispose(); resetPolicy(); matchStore.close(); };
    const returnToPage = (event) => { if (event.persisted) location.reload(); };
    const storageChanged = (event) => { if (playStarted) matchStore.changed(event); };
    window.addEventListener('pagehide', leave);
    window.addEventListener('pageshow', returnToPage);
    window.addEventListener('storage', storageChanged);
    return () => {
      mounted = false;
      startup.abort();
      faceLoad?.abort();
      unwatchOffline();
      unwatchModel();
      window.removeEventListener('pagehide', leave);
      window.removeEventListener('pageshow', returnToPage);
      window.removeEventListener('storage', storageChanged);
      matchStore.close();
      matchReview?.dispose();
      session?.dispose();
      reportProgress(null);
      resetPolicy();
    };
  });

  async function changeTileFace(value) {
    if (!ready) return;
    const next = normalizeTileFace(value);
    faceLoad?.abort();
    faceWarning = '';
    const owner = new AbortController();
    faceLoad = owner;
    pendingTileFace = next === tileFace ? null : next;
    if (next === tileFace) { faceLoad = null; return; }
    try {
      await preloadTiles(() => {}, { face: next, signal: owner.signal });
      if (faceLoad === owner && !owner.signal.aborted) tileFace = next;
    } catch (error) {
      if (faceLoad === owner && !owner.signal.aborted) faceWarning = `The selected tile graphics could not load: ${error.message ?? error}. Your previous tiles and match are unchanged. Select the face again to retry.`;
    } finally {
      if (faceLoad === owner) { pendingTileFace = null; faceLoad = null; }
    }
  }

  function downloadAi() { void prepareOfflineAi().catch(() => {}); }

  // Another network is loaded with the page, not swapped into this one: the
  // opponents' worker, every adviser and the offline status are about the
  // network a page plays. So the pick is saved and the page loads again, in
  // the mode it was in. Matches, guided games and physical tables are saved
  // and come back as they were; a watched game is not saved, and ends, and
  // the physical table's undo history lives only in the page.
  function changeNetwork(version) {
    if (version === network.version) return true;
    const lost = { watch: ', and the watched game ends', physical: ', and undo history starts afresh' }[mode] ?? '';
    // Offline, a network that was never saved here cannot be loaded, and
    // Trained then waits for a connection. The one playing now stays saved
    // if it was, so picking it again brings it back.
    const offlineNote = navigator.onLine ? '' : ' Offline, it plays only if it was saved on this device.';
    if (!confirm(`Switch to Trained ${version}? The page reloads to load it${lost}.${offlineNote}`)) return false;
    // Picking the default is no pick, so the page moves on with the default
    // when a newer network is published; another version stays picked.
    trainedNetwork = version === MANIFEST.version ? undefined : version;
    storedSettings = writeSettings(storage, currentSettings(), storedSettings);
    // A pick that is not kept would be lost as the page reloads.
    if (readSettings(storage, touch).trainedNetwork !== trainedNetwork) {
      trainedNetwork = preferences.trainedNetwork;
      storedSettings = writeSettings(storage, currentSettings(), storedSettings);
      notice = 'This browser is not keeping your settings, so another trained network cannot be loaded here.';
      return false;
    }
    const address = new URL(location.href);
    if (mode === 'play') address.searchParams.delete('mode');
    else address.searchParams.set('mode', mode);
    // The same address with a fragment would only scroll, not load again.
    if (address.href === location.href) location.reload();
    else location.replace(address.href);
    return true;
  }

  function start(strength = opponents) {
    if (saveConflict) return;
    session?.dispose();
    dropGameReview();
    matchStore.newMatch();
    picked = null;
    selected = null;
    notes = null;
    standingsFocused = false;
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
  // Next hand on the last hand brings the standings in place of another
  // deal, and its button goes for the Final hand caption, so focus goes on
  // to the standings' heading, which says where the player finished, rather
  // than to the page (land). On a phone the result is a sheet over the
  // standings. When the keyboard pressed the button, the sheet steps aside
  // for them, so that focus is not hidden under it; a tap leaves the result
  // up, as it was.
  async function nextHand(event) {
    if (busy || !session) return;
    const owner = session, byKeyboard = pressedByKeyboard(event);
    picked = null;
    selected = null;
    const advanced = await owner.nextHand();
    if (session !== owner || !advanced) return;
    if (!owner.over) { notes = null; return; }
    standingsFocused = byKeyboard;
    await tick();
    land(standingsHeading, byKeyboard);
  }

  function saveLog() {
    if (!session) return;
    try {
      saveLogFile(`riichi-${view.round}-${view.kyoku}-${Date.now()}.mjai.jsonl`, session.engine.log());
    } catch { notice = 'The hand could not be saved to a file. Your current match is unchanged.'; }
  }

  // The whole game from East 1. The engine leaves out a hand still being
  // played, whose deal would show the opponents' tiles.
  function saveGame() {
    if (!session || session.closed || !view) return;
    try {
      const names = playerNames(view, matchLabels(session.initialOpponents, session.opponents));
      saveLogFile(gameFileName(), session.engine.game_log(names));
    } catch { notice = 'The game could not be saved to a file. Your current match is unchanged.'; }
  }

  // Focus goes on to a heading that says where the player now is. When the
  // keyboard pressed the button, the heading is brought into view as well,
  // by as little movement as will show it, since that player follows focus
  // and is shown a ring there. A mouse's or a finger's press leaves the page
  // where it was: that player is shown no ring to follow, and on a phone the
  // scroll would take the top of the table off the screen (pressedByKeyboard).
  function land(heading, byKeyboard) {
    heading?.focus({ preventScroll: true });
    if (byKeyboard) heading?.scrollIntoView({ block: 'nearest' });
  }

  // The result's button goes as it opens the hand's review, and on a phone
  // the result steps aside with it, so focus goes on to the review, to its
  // heading, rather than to the page (land). The way back to the result is
  // the control just before the review: on a phone the result's chip, which
  // hands focus back to it, and where the result is not a sheet, its own
  // buttons.
  async function showReview(event) {
    const byKeyboard = pressedByKeyboard(event);
    try { notes = session?.engine.review() ?? []; }
    catch { notice = 'The review could not be built. Your match is unchanged.'; return; }
    await tick();
    land(reviewHeading, byKeyboard);
  }

  // The finished match is replayed from its saved commands on an engine of
  // the review's own, so the match and its save are left as they are. Says
  // whether the review is open. The network this page plays judges it, and
  // the trend keeps which version that was.
  function openGameReview() {
    if (!session?.over || session.closed) return false;
    if (!matchReview) {
      try {
        matchReview = new MatchReview(Game, session.snapshot(), { analyzeBatch,
          onComplete: review => {
            reviewTrend = recordTrend(storage, { ...review.summary, network: network.version, at: Date.now() });
          } });
      } catch (error) {
        notice = `The game could not be replayed for review: ${error?.message ?? error}. Your saved match is unchanged.`;
        return false;
      }
    }
    reviewingGame = true;
    return true;
  }

  // Back to the standings' own button, wherever the review was opened from:
  // on a phone the result sheet that offered it has stepped aside for them.
  // It is brought to the middle of the screen, clear of the bar at the top
  // and the result's chip at the bottom.
  async function closeGameReview() {
    reviewingGame = false;
    await tick();
    const button = document.querySelector('.standings [data-review-game]');
    button?.focus({ preventScroll: true });
    button?.scrollIntoView({ block: 'center' });
  }

  function dropGameReview() {
    matchReview?.dispose();
    matchReview = null;
    reviewingGame = false;
  }

  // Watch, Guided and Physical play ask the same network, and go first.
  $effect(() => { if (mode !== 'play') untrack(() => matchReview?.stop()); });

</script>

<svelte:window onkeydown={onKey} />

<main>
  <AppSettings bind:mode bind:settingsOpen bind:hints bind:confirmDiscards bind:shortcuts {tileFace} {pendingTileFace} onfacechange={changeTileFace}
    network={network.version} onnetworkchange={changeNetwork}
    {difficulty} {opponents} {ready} {busy} {saveConflict} {trainedAvailable} {offline}
    {changeOpponents} {startFresh} {configureTable} {downloadAi} {loggedHands} onsavegame={saveGame}
    onconfirmationchange={() => selected = null} onshortcutschange={() => picked = null} />
  {#if faceWarning}<p class="notice" data-face-error role="alert">{faceWarning}</p>{/if}
  <!-- A screen reader hears a status line only once it knows it, so this one
       waits in the page, empty and out of sight, before its words arrive. -->
  <p class={notice ? 'notice' : 'visually-hidden'} data-notice role="status">{notice}</p>

  <OpponentDialog bind:customDialog bind:draftOpponents {opponents} {trainedAvailable}
    {ready} {saveConflict} {startCustomTable} />

  {#if saveConflict}
    <section class="save-conflict" role="alert">
      <p>{saveConflict}</p>
      <button class="app-control" onclick={() => location.reload()}>Reload latest match</button>
    </section>
  {/if}

  {#if failure && (mode === 'play' || !ready)}
    <section class="failure" aria-label="game recovery">
      <p role="alert">{failure}</p>
      {#if !ready}<button class="app-control" onclick={() => location.reload()}>Reload to retry</button>{/if}
      {#if recovery && pendingOpponent}<p class="recovery-note">Waiting for the {pendingOpponent.position.toLowerCase()} opponent. Retry the network or explicitly switch that player to Club.</p>{/if}
      {#if recovery}
        <div class="recovery-actions">
          <button class="app-control" onclick={retryAi} disabled={busy}>Retry trained opponent</button>
          {#if pendingOpponent}
            <button class="app-control" onclick={continueThisOpponent} disabled={busy}>Use Club for {pendingOpponent.position.toLowerCase()} opponent only</button>
          {/if}
          <button class="app-control" onclick={continueClub} disabled={busy}>{difficulty === 'custom' ? 'Use Club for all Trained opponents' : 'Continue with Club opponents'}</button>
          <button class="app-control" onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>
        </div>
      {/if}
    </section>
  {/if}
  {#if storageWarning}<p class="notice" role="status">{storageWarning}</p>{/if}

  {#if !ready && !failure}
    <p class="loading" role="status">{startupNote}</p>
  {:else if ready && mode === 'watch'}
    <AgentWatch {trainedAvailable} {hints} {offline} />
  {:else if ready && mode === 'guided'}
    <GuidedPlay {trainedAvailable} {storage} {hints} {offline} />
  {:else if ready && mode === 'physical'}
    <PhysicalPlay {trainedAvailable} {storage} />
  {:else if mode === 'play' && view}
    {#if reviewingGame && matchReview}
      <GameReview review={matchReview} trend={reviewTrend} {hints} {trainedAvailable} {offline} onclose={closeGameReview} />
    {/if}
    <!-- Hidden rather than closed while the review is open, so the result
         and the standings come back as they were left. -->
    <div class="match-view" class:away={reviewingGame && matchReview}>
      <MatchTable {view} {hints} {thinking} {pendingOpponent} />

      <div class="play-area" class:ended={view.phase === 'over' || Boolean(standings)}>
        <PlayerHand {view} engine={session?.engine} closed={session?.closed ?? false}
          {hints} {busy} blocked={Boolean(failure || saveConflict)} {discardChoices} {confirmDiscards} {picked} {selected}
          {canDiscard} {selectTile} {syncHandFocus} bind:handElement />

        <section class="controls" aria-label="your choices" bind:this={callElement}>
          {#if standings}
            <Standings {standings} onagain={() => start()} onsave={saveGame} onreview={openGameReview} bind:heading={standingsHeading} />
          {/if}
          {#if view.phase === 'over' && view.outcome}
            <ScoreScreen outcome={view.outcome} seats={view.seats} dora={shownDora} {hints} {busy}
              bets={view.riichi_sticks ?? 0} onnext={nextHand}
              onreview={showReview} reviewed={notes !== null} onlog={saveLog} ongame={saveGame} ongamereview={openGameReview}
              gameReviewOpen={reviewingGame} {standingsFocused} finalHand={Boolean(standings)} />
            {#if notes !== null}<Review {notes} {hints} engine={session?.engine} {trainedAvailable} {offline} bind:adviser={reviewAdviser} bind:heading={reviewHeading} />{/if}
          {:else}
            <TurnChoices {view} {shownDora} {busy} {thinking} {failure} {saveConflict}
              loadNote={waitingNote(loadNote, offline)} {pendingOpponent} {myTurn} {confirmDiscards} {shortcuts} {touch}
              {selectedTile} {callChoices} {choose} {discard} oncancel={() => selected = null} />
          {/if}
        </section>
      </div>

      <details class="history">
        <summary>Hand history · {log.length} event{log.length === 1 ? '' : 's'}</summary>
        <section class="log" aria-label="what happened, oldest first">
          {#each log as line, index (index)}<p>{line}</p>{/each}
        </section>
      </details>
    </div>

  {/if}
</main>

<style>
  .recovery-note { opacity: .85; }
  .match-view { display: contents; }
  .match-view.away { display: none; }
  .save-conflict { padding: 12px; border: 1px solid var(--gold); border-radius: 8px; background: var(--felt-deep); }
  .save-conflict p { margin: 0 0 8px; }
  .save-conflict button { min-height: 44px; padding: 8px 14px; }
  main { width: 100%; max-width: 1100px; box-sizing: border-box; margin: 0 auto; grid-template-columns: minmax(0, 1fr); padding: max(10px, env(safe-area-inset-top)) max(10px, env(safe-area-inset-right)) max(24px, env(safe-area-inset-bottom)) max(10px, env(safe-area-inset-left)); display: grid; gap: 10px; }
  .history { font-size: .85rem; min-width: 0; }
  summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .play-area { display: grid; gap: 10px; min-width: 0; }
  .controls { display: grid; gap: 8px; min-width: 0; }
  .failure { background: var(--accent-soft); color: var(--accent); padding: 12px; border-radius: 8px; }
  .failure p { margin: 0; }
  .recovery-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .notice { margin: 0; font-size: .8rem; }
  .log { font-size: .82rem; display: grid; gap: 4px; max-height: 40vh; overflow-y: auto; padding: 8px 12px; background: #0003; border-radius: 8px; }
  .log p { margin: 0; }
  @media (max-width: 760px) {
    main { gap: 6px; }
    .play-area:not(.ended) { position: sticky; top: 6px; bottom: auto; z-index: 2; background: var(--felt-deep); padding: 6px 0 max(6px, env(safe-area-inset-bottom)); border-radius: 10px; box-shadow: 0 -6px 16px #0003; }
    .controls { padding: 0 8px; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    main { max-width: none; grid-template-columns: minmax(270px, .85fr) minmax(340px, 1.15fr); align-items: start; }
    .notice, .failure, .save-conflict, .loading, .history, main > :global(.game-review) { grid-column: 1 / -1; }
    .play-area { grid-column: 2; grid-row: auto; position: sticky; top: 8px; bottom: auto; }
    .controls { font-size: .8rem; }
  }

  @media (min-width: 761px) and (min-height: 501px) {
    main { max-width: 1280px; padding-inline: 18px; gap: 14px; }
  }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    main { padding-top: max(5px, env(safe-area-inset-top)); gap: 5px; }
    .notice {
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,.10);
      border-radius: 10px;
      background: #0002;
      font-size: .78rem;
    }
  }
</style>
