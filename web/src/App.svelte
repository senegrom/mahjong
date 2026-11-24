<script>
  import './lib/app/controls.css';
  import AppSettings from './lib/app/AppSettings.svelte';
  import OpponentDialog from './lib/app/OpponentDialog.svelte';
  import MatchTable from './lib/app/MatchTable.svelte';
  import PlayerHand from './lib/app/PlayerHand.svelte';
  import TurnChoices from './lib/app/TurnChoices.svelte';
  import { onMount, setContext, tick, untrack } from 'svelte';
  import init, { Game, PhysicalAnalysis, review_facts } from './wasm/riichi.js';
  import { preloadTiles } from './lib/tile-preload.js';
  import { INITIAL_OFFLINE, startOffline, watchOffline, prepareOfflineAi, refreshOffline } from './lib/offline.js';
  import ScoreScreen from './lib/ScoreScreen.svelte';
  import Standings from './lib/Standings.svelte';
  import Review from './lib/Review.svelte';
  import GameReview from './lib/GameReview.svelte';
  import { Asking, MatchReview, batchMostFor } from './lib/game-review.js';
  import { MatchInsight, anyGate, countedBatches, reviewInsight } from './lib/review-insight-policy.js';
  import { readTrend, recordTrend } from './lib/review-trend.js';
  import AgentWatch from './lib/AgentWatch.svelte';
  import PhysicalPlay from './lib/PhysicalPlay.svelte';
  import GuidedPlay from './lib/GuidedPlay.svelte';
  import { analyzeBatch, analyzePolicy, modelIsAvailable, networkInWorker, reportLoading, reportProgress, resetPolicy } from './lib/policy.js';
  import { policyRuntime, reportRuntime } from './lib/gpu-batches.js';
  import { watchModelAvailability } from './lib/model-availability.js';
  import { MatchSession, readSettings, writeSettings } from './lib/session.js';
  import { RELOAD_TO_CONTINUE, RESULT_SHEET, acceptsHandKey, handKeyActs, heldMove, land, moveHandFocus, pressedByKeyboard, repeatedClick, startupWords,
    waitIsDownload, waitingNote } from './lib/ui.js';
  import { MatchStore } from './lib/save-store.js';
  import { gameFileName, matchLabels, playerNames, saveLogFile } from './lib/game-log.js';
  import { TILE_FACE_CONTEXT, normalizeTileFace } from './lib/tile-faces.js';
  import { DEFAULT_OPPONENT, normalizeOpponents, opponentPreset, OPPONENT_TYPES, sameOpponents, standInFor, tableAfterStandIn,
    trainedAtHand, trainedUnderWay } from './lib/opponents.js';
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
  // The player's own table as their record holds it, seat by seat, or null
  // where it holds none: a record from before tables were stored so, one a
  // reset read as Trained, or a page whose address asks for a table, which
  // is no stored choice. A stand-in is known by it after a reload (below).
  const storedTable = !OPPONENT_TYPES.includes(requested) && Array.isArray(preferences.opponents) ? preferences.opponents : null;
  // The table as the settings name it: the one type when all three opponents
  // match, otherwise 'custom'. It follows the opponents, so the two agree.
  let difficulty = $derived(opponentPreset(opponents));
  // The table this match's Club opponents stand in for, or null while it
  // plays no stand-in (opponents.js). A player who would not wait for the
  // trained network to download took Club meanwhile: the match is Club's and
  // shows it, and this, the table they chose, stays their stored preference,
  // for the next game to go back to once the network is at hand. It ends as
  // the player picks opponents themself (start).
  let returnTable = $state.raw(null);
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
  // Whether Trained AI's reviews keep its move back until the player asks to
  // see it, and the other players' tiles too. Both are stored with the other
  // preferences, so a quiz the player set up is still one after a reload.
  let reviewHideMove = $state(preferences.reviewHideMove);
  let reviewHideTiles = $state(preferences.reviewHideTiles);
  // Which decisions the review of the game steps through, stored with the
  // other preferences, so that it is as the player left it in the next game
  // and after a reload.
  let reviewFilter = $state(preferences.reviewFilter);
  setContext(TILE_FACE_CONTEXT, () => tileFace);
  let ready = $state(false);
  let startupNote = $state(startupWords(null, false, false));
  // A page plays one trained network, so its download is the only optional
  // one: readiness and availability are read straight from offline.
  let offline = $state(INITIAL_OFFLINE);
  let failure = $state('');
  let storageWarning = $state('');
  let saveConflict = $state('');
  let notice = $state('');
  let loadNote = $state('');
  // What the opponent's worker last said of the network while it loads it,
  // for the modes other than Play, which show it beside their own thinking.
  let workerNote = $state('');
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
  // Why a press on a set of buttons could not be done, which is said beside
  // them (say): the result's (ScoreScreen), the final standings' (Standings)
  // and the settings' Export game's (AppSettings).
  let failures = $state({ result: '', standings: '', export: '' });
  // How many presses on the standings' own buttons have failed, for the
  // result to step aside on a phone, where it is a sheet over them.
  let standingsFailures = $state(0);
  // The final standings' heading, which takes focus as the game ends, and
  // whether the keyboard sent it there, for the result to step aside for it.
  let standingsHeading = $state(null);
  let standingsFocused = $state(false);
  // Whether the result's chip shows (ScoreScreen says): the page then ends
  // lower, so that its last line can be scrolled clear of the chip.
  let resultChip = $state(false);
  let picked = $state(null);
  let selected = $state(null);
  let handElement = $state(null);
  let callElement = $state(null);
  let settingsOpen = $state(false);
  // Trained AI's review of the match, kept while that match is: of the game
  // so far from any hand's result, taken on to each later one, and of the
  // whole game once it is over. And whether its screen is open. It reviews
  // on while the screen is shut, until the next deal (nextHand), and waits
  // while the hand's own notes ask Trained AI (notesAsking), and while its
  // reasoning about a decision does (insightAsking).
  let matchReview = $state.raw(null);
  let reviewingGame = $state(false);
  // The review's reasoning about the match's decisions (MatchInsight): each
  // decision's Why line and what would change Trained AI's mind, kept beside
  // the review for the match, and stopped wherever the review stops for the
  // next deal.
  let matchInsight = $state.raw(null);
  // Where the player is in that review, `{ hand, step }`, kept for the match
  // so that closing the review and opening it again comes back to the same
  // decision. It is let go when the review is taken on to hands played since,
  // so that the review opens on the newest of them, and with the review.
  let reviewPlace = $state(null);
  // What the player has revealed in that review, decision by decision
  // ("hand:step"), where the settings keep Trained AI's move or the other
  // players' tiles back, with any move they picked there: kept for the match
  // beside the place, so that no decision asks again, through steps, closing
  // and opening and the hands added later. Nothing of it is saved: a new
  // review of another match, a new game or a reload asks afresh.
  let reviewRevealed = $state({});
  // The hand's own notes, opened by View table / my hand or Review final
  // hand, ask Trained AI a decision at a time (Review), of the page's one
  // worker, which answers its requests in turn. The review asks the same
  // worker up to eight decisions a request, so each note would wait behind
  // one of its batches: a hand's notes would take twenty seconds rather than
  // two or three. So the notes say when they begin and stop asking, and
  // before each batch the review waits for them at its gate, then goes on
  // exactly where it was: the same hand, the same hands queued and the same
  // word from the player. Nothing need start it again, so the review of the
  // whole game behind the standings still reaches the trend. Each time the
  // notes begin to ask, only their first question can still wait behind a
  // batch, the one already sent, whichever of the two was opened first, the
  // review's screen open or shut. Its deadline runs from when the worker
  // begins it (requestPolicy), so that wait slows it but cannot make it run
  // out of time. Once the review has timed the device (batchFor), the wait is
  // about ten seconds at most on one that takes up to ten seconds a position:
  // the decisions Trained AI hesitates over are asked again in a batch of
  // their own (suits.js), which waits at the gate as the first did.
  const notesAsking = new Asking();
  // The review's reasoning asks the same worker, in passes that take about as
  // long as the review's batch, the tables it hesitates over asked again
  // (review-insight-policy.js), each request sent only while the notes
  // ask nothing and no batch is with the worker: the review's, counted as it
  // is sent (workerBatches), or another pass. Its spells (insightAsking)
  // shut the review's gate as the notes' do, so the review waits there for
  // the whole of a Why line or a search, until the worker has answered its
  // last pass, and sends no batch to wait behind one. Neither ever waits in
  // the worker's queue for the other, only at its gate in the page. The
  // notes' first question waits behind whichever of the two is with the
  // worker, one at most, as it waited behind a batch before.
  const insightAsking = new Asking(), workerBatches = new Asking();
  const reviewBatch = countedBatches(analyzeBatch, workerBatches);
  // A review and its reasoning ask bigger batches once their batches run on
  // the graphics card (batchMostFor, gpu-batches.js), which the settings
  // name too. Play never runs there.
  const batchMost = () => batchMostFor(policyRuntime());
  let trainedRuntime = $state(policyRuntime());
  const reasoning = reviewInsight({ PhysicalAnalysis, review_facts, analyzeBatch, asking: insightAsking, notes: notesAsking,
    batches: workerBatches, positionMs: () => matchReview?.positionMs ?? null, batchMost });
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
  const currentSettings = () => {
    // While a stand-in holds, the preference is the table it stands in for,
    // not the Club this one match plays.
    const table = returnTable ?? opponents;
    return { difficulty: opponentPreset(table), opponents: [...table], hints, confirmDiscards, shortcuts, tileFace, reviewAdviser, reviewHideMove,
      reviewHideTiles, reviewFilter, trainedNetwork };
  };
  $effect(() => {
    storedSettings = writeSettings(storage, currentSettings(), storedSettings);
  });

  // Held sideways, a phone's screen has room under the game's bar for the
  // hand and the player's choices only with nothing between them, so while
  // a hand is played the choices are drawn in the player's seat, right under
  // the hand and over the called sets and discards (PlayerHand), rather
  // than after the seat. They are drawn in one place or the other rather
  // than moved by CSS, so that a screen reader and the keyboard go through
  // them in the order they are seen. As the screen turns, a choice that had
  // focus would go with the place it was in, dropping focus to the page, so
  // the same choice where they are now drawn has it instead.
  const SIDEWAYS = '(max-height: 500.98px) and (orientation: landscape)';
  let sideways = $state(matchMedia(SIDEWAYS).matches);
  let choicesInSeat = $derived(sideways && view?.phase !== 'over' && !standings);
  onMount(() => {
    const media = matchMedia(SIDEWAYS);
    const turned = async () => {
      const kept = [...(callElement?.querySelectorAll('button') ?? [])].indexOf(document.activeElement);
      sideways = media.matches;
      if (kept < 0) return;
      await tick();
      callElement?.querySelectorAll('button')[kept]?.focus();
    };
    media.addEventListener('change', turned);
    return () => media.removeEventListener('change', turned);
  });

  // A call is offered under a table that fills the window, so the choices
  // can sit below the fold on an ordinary screen. Being asked a question
  // whose answers are off screen is no question at all, so the buttons are
  // brought into view when they appear, by the shortest scroll that does
  // it and none at all when they are already there. They are measured
  // themselves: held sideways they share their rows with the staged
  // discard, and their own row has no box (TurnChoices). Held sideways, the
  // hand the player answers from comes into view too, below the game's bar,
  // which stays at the top of the screen (AppSettings): with the buttons in
  // view and the hand's top above that, as at the foot of the page, the
  // page scrolls back up to the hand, as far as keeps the buttons in view.
  // Upright and on a desktop the page stays where it is once the buttons
  // show. All is measured once the play area has finished easing and the
  // choices have slid in (TurnChoices): with reduced motion every change of
  // style eases over a hundredth of a millisecond (app.css), which takes a
  // frame or two, and until then a hand that has just thrown its drawn tile
  // keeps that tile's gap, its tiles a pixel narrower and the buttons that
  // much higher; and the slide's delay is not shortened, so until it ends
  // the buttons sit 4 pixels low.
  $effect(() => {
    if (mode !== 'play' || !callChoices.length || !callElement) return;
    const section = callElement, hand = handElement;
    let live = true;
    const bringIntoView = () => {
      if (!live) return;
      const buttons = [...section.querySelectorAll('.call-options button')];
      if (!buttons.length) return;
      const boxes = buttons.map(button => button.getBoundingClientRect());
      const top = Math.min(...boxes.map(box => box.top)), bottom = Math.max(...boxes.map(box => box.bottom));
      const gentle = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      const behavior = gentle ? 'auto' : 'smooth';
      if (bottom > window.innerHeight || top < 0) {
        buttons[boxes.findIndex(box => box.bottom === bottom)].scrollIntoView({ block: 'end', behavior });
        return;
      }
      if (!sideways) return;
      const bar = document.querySelector('.bar');
      const roof = bar && getComputedStyle(bar).position === 'sticky' ? bar.getBoundingClientRect().bottom : 0;
      const lift = Math.min(roof - (hand?.getBoundingClientRect().top ?? roof), window.innerHeight - bottom);
      if (lift > 0) window.scrollBy({ top: -lift, behavior });
    };
    const easing = [
      ...section.closest('.play-area')?.getAnimations({ subtree: true }).filter(animation => animation instanceof CSSTransition) ?? [],
      ...section.getAnimations({ subtree: true }).filter(animation => animation instanceof CSSAnimation
        && Number.isFinite(animation.effect?.getComputedTiming().endTime)),
    ];
    void Promise.allSettled(easing.map(motion => motion.finished)).then(bringIntoView);
    return () => { live = false; };
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
    if (!sameOpponents(opponents, owner.opponents)) {
      // Given to Club while a Trained opponent was awaited, the match plays
      // a stand-in, and the table it had stays the player's preference.
      // Club taken after a failure is the player's new table, as before.
      if (owner.stoodIn && !returnTable) returnTable = standInFor(opponents, owner.opponents);
      opponents = [...owner.opponents];
    }
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
    // Greedy: the opponent plays its best move, by the last hand's rules
    // where they apply, not a sample of them. Its weights say whether it
    // hesitates, and the renamed game is then asked too
    // (MatchSession.trainedMove).
    ai: (planes, mask, signal) => analyzePolicy(planes, mask, signal),
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
          // Nothing saved marks a stand-in. A match that plays Club exactly
          // where the player's own stored table says Trained is one, taken
          // before the page was loaded again: the table stays their
          // preference rather than being written over with the match's, and
          // the network they wait for is asked for again where it can be
          // saved (where it cannot, nothing fetches it until a Trained move
          // asks, and the next new game is their table: opponents.js). A record that
          // holds no table of its own is written over, as it always was.
          returnTable = standInFor(storedTable, session.opponents);
          update(session);
          if (session.opponents.includes('neural') || returnTable) downloadAi();
          void session.run();
          // A finished match opens on its standings where the result is a
          // sheet over them, on a phone: the sheet starts stepped aside there,
          // and the standings take focus and the top of the screen, as when
          // the game ended under a finger (ScoreScreen's onstandings).
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
    reportLoading((note) => { if (mounted) workerNote = note; });
    // Where a review's batches run decides its time a position, which is none
    // once they run elsewhere than where it was measured (MatchReview): its
    // screen keeps what it last read, so the review tells it to read again.
    reportRuntime((value) => {
      if (!mounted) return;
      trainedRuntime = value;
      matchReview?.notify();
    });
    const startup = new AbortController();
    // Full-package caching has its own honest status. An unavailable unused
    // face set must not prevent a game with the selected graphics opening.
    void startOffline();
    // The page waits for the engine and the selected tile graphics at once,
    // and its line says which it still waits for (startupWords).
    let tiles = null, drawn = false, engine = false;
    const show = () => { if (mounted) startupNote = startupWords(tiles, drawn, engine); };
    Promise.all([init().then(() => { engine = true; show(); }), preloadTiles((done, total) => {
      tiles = { done, total };
      show();
    }, { face: tileFace, signal: startup.signal }).then(() => { drawn = true; show(); })]).then(() => {
      if (mounted && !startup.signal.aborted) ready = true;
    }).catch((error) => {
      if (mounted && !startup.signal.aborted) failure = `The game could not finish loading: ${error.message ?? error}. Reconnect and reload to retry.`;
    });
    // Completed actions are already saved inside their writer transaction.
    // Never write an old snapshot during pagehide. A cached page must restore
    // the current record rather than resume a stale engine on Back navigation.
    const leave = () => { startup.abort(); faceLoad?.abort(); matchInsight?.stop(); matchReview?.dispose(); session?.dispose(); resetPolicy(); matchStore.close(); };
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
      matchInsight?.stop();
      matchReview?.dispose();
      session?.dispose();
      reportProgress(null);
      reportLoading(null);
      reportRuntime(null);
      resetPolicy();
    };
  });

  // Whether the tile face's failure has been said in the settings' dialog,
  // under its select (AppSettings). The page's line then shows it, once the
  // dialog is closed, without saying it again: inserted as an alert as the
  // dialog closed, it was heard a second time.
  let faceWarningSaid = $state(false);
  $effect(() => { if (faceWarning && settingsOpen) faceWarningSaid = true; });
  async function changeTileFace(value) {
    if (!ready) return;
    const next = normalizeTileFace(value);
    faceLoad?.abort();
    faceWarning = '';
    faceWarningSaid = false;
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

  // Deals a new match: at the table named, which is the player's own pick
  // and ends a stand-in, or else at the table as it is. While a stand-in
  // holds, that is the table it stands in for, and the stand-in ends, unless
  // the network is not at hand and its download is under way: then the
  // stand-in again, which goes on holding (opponents.js).
  function start(strength = null) {
    if (saveConflict) return;
    let table = strength ?? opponents;
    if (strength) returnTable = null;
    else if (returnTable) {
      table = tableAfterStandIn(returnTable, {
        atHand: trainedAtHand({ network: offline.network, workerLoaded: networkInWorker() }),
        underWay: trainedUnderWay({ network: offline.network, workerNote }),
      });
      if (!standInFor(returnTable, table)) returnTable = null;
    }
    session?.dispose();
    dropGameReview();
    matchStore.newMatch();
    picked = null;
    selected = null;
    notes = null;
    failures = { result: '', standings: '', export: '' };
    standingsFailures = 0;
    standingsFocused = false;
    notice = '';
    loadNote = '';
    session = new MatchSession(Game, Date.now() % 2 ** 31, table, callbacks);
    if (session.opponents.includes('neural') || returnTable) downloadAi();
    void session.run();
  }

  function startFresh(strength = null) {
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

  // A move made from the hand hands focus back to the hand as the turn comes
  // back: with the hand's keys on, for its arrows and Enter; and with them
  // off where the tile was pressed by keyboard, whose player would otherwise
  // tab back from the page every turn (`byKeyboard`). The pressed tile was
  // disabled while the opponents played, which dropped focus to the page.
  async function choose(choice, byKeyboard = false) {
    if (!session || busy) return;
    const owner = session;
    const fromHand = handElement?.contains(document.activeElement);
    picked = null;
    selected = null;
    await owner.choose(choice);
    await tick();
    if (session !== owner || owner.closed || !fromHand || busy || failure || saveConflict || !myTurn) return;
    if ((!shortcuts || touch) && !byKeyboard) return;
    const active = document.activeElement;
    if (active === document.body || handElement?.contains(active)) handElement?.focus({ preventScroll: true });
  }

  function discard(tile, byKeyboard = false) {
    const choice = discardChoices.find((entry) => entry.tile === tile);
    if (choice && canDiscard(tile)) void choose(choice, byKeyboard);
  }

  // Without confirmation a press throws its tile, and against opponents that
  // answer at once the next turn's tile is in its place before the second
  // press of a double click lands, which is let go (repeatedClick). With
  // confirmation the first press only selects, and a second on the same tile
  // is how it is thrown.
  function selectTile(tile, index, event) {
    if (!canDiscard(tile) || (!confirmDiscards && repeatedClick(event))) return;
    picked = null;
    if (confirmDiscards && selected !== index) selected = index;
    else discard(tile, pressedByKeyboard(event));
  }

  // The confirmation's two buttons go as either is pressed. Where one of
  // them had focus, the hand it answered for takes it, rather than the page,
  // with the keys on or off and on a touch screen too: nothing else is left
  // to hold it.
  async function confirmation(act) {
    const held = Boolean(document.activeElement?.closest?.('.confirm-discard'));
    act();
    if (!held) return;
    await tick();
    const active = document.activeElement;
    if (!active || active === document.body) handElement?.focus({ preventScroll: true });
  }

  function syncHandFocus(event) {
    const tile = event.target.closest('[data-hand-index]');
    const index = tile ? Number(tile.dataset.handIndex) : null;
    if (picked !== null) picked = index;
    // Native Tab navigation must not leave a confirmation for a different tile.
    if (selected !== null && selected !== index) selected = null;
  }

  async function onKey(event) {
    if (mode !== 'play') return;
    // Enter held down on a tile or on one of the player's choices presses it
    // once, with the keys on or off: its repeats would answer the decisions
    // that follow, each drawn where the last one was (heldMove).
    if (heldMove(event, handElement)) { event.preventDefault(); return; }
    if (!shortcuts || !myTurn || busy || failure || !acceptsHandKey(event, handElement)) return;
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
    // A key held down throws no second tile and makes no second choice. Only
    // the keys that act here are taken from the browser: a held arrow down,
    // Page Down or Tab still scrolls the page or moves focus on.
    if (event.repeat) {
      if (handKeyActs(event.key)) event.preventDefault();
      return;
    }
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

  // A Trained opponent's first move waits for the network, which on a first
  // visit is a download of minutes on a slow link. While the wait is that
  // download, and not the network starting or thinking, the turn's controls
  // offer Club meanwhile (TurnChoices): this match's Trained seats are
  // Club's from the press on, as after recovery's Continue with Club
  // opponents, while the download carries on. Nothing is reset here, so the
  // worker and whatever is fetching the network go on; the table the player
  // chose stays their preference (update, returnTable).
  let downloadWait = $derived(thinking && !failure && !saveConflict && waitIsDownload(loadNote, offline));
  let standInPressed = false;
  function playClubMeanwhile() {
    session?.clubMeanwhile();
    standInPressed = true;
  }
  // Its button goes as the wait ends. Pressed, it had focus, which the hand
  // takes rather than the page, with the keys on or off: also after a press
  // that came too late to be taken (the last answer already in, the run
  // still holding the save lock), whose button goes a moment later all the
  // same.
  $effect(() => {
    if (downloadWait || !standInPressed) return;
    standInPressed = false;
    const active = document.activeElement;
    if (!active || active === document.body) handElement?.focus({ preventScroll: true });
  });

  function retryAi() { resetPolicy(); void session?.retry(); }
  function continueClub() { resetPolicy(); void session?.continueWithClub(); }
  function continueThisOpponent() { resetPolicy(); void session?.continueOpponentWithClub(); }
  // Next hand on the last hand brings the standings in place of another
  // deal, and its button goes for the Final hand caption, so focus goes on
  // to the standings' heading, which says where the player finished, rather
  // than to the page (land). Where the result is not a sheet, only a
  // keyboard's press moves the page to show the heading: that player follows
  // focus, and is shown a ring there. A mouse's or a finger's press is shown
  // no ring, and the page is left as the browser leaves it when the
  // standings come in above the result, as it was before focus moved: where
  // the player had scrolled down to the result's buttons, at the foot of the
  // page, a scroll to the heading would take the Final hand caption, where
  // they pressed, off a short screen (pressedByKeyboard). On a phone the
  // result is a sheet over the standings, and left up it showed nothing of
  // them: the game had ended with no place on the screen and no Play again.
  // So there the sheet steps aside for them however the button was pressed,
  // leaving its chip, and the page moves to them: for a tap to the heading
  // at the top of the screen, the places and the buttons under it, and for a
  // keyboard by as little as shows the heading and its ring, as before.
  async function nextHand(event) {
    if (busy || !session) return;
    const owner = session, byKeyboard = pressedByKeyboard(event);
    picked = null;
    selected = null;
    // The next deal's trained opponents ask the same network as the review
    // of the game so far, which answers one question at a time, and they go
    // first: the review stops, keeping every hand it has reviewed, and goes
    // on when it is next opened. Its reasoning stops too, keeping whatever
    // it had.
    matchReview?.stop();
    matchInsight?.stop();
    const recorded = owner.commands.length;
    const advanced = await owner.nextHand();
    if (session !== owner || owner.closed) return;
    // The deal is recorded before the new hand's opponents are asked, so a
    // Trained opponent that fails on its first move leaves the hand dealt and
    // the press unanswered. The notes and the failure line of the hand before
    // go with that hand all the same, by what the engine did: kept, they
    // would open as the next result's review, of a hand no longer there.
    if (owner.commands.length > recorded && !owner.over) { notes = null; failures.result = ''; }
    if (!advanced || !owner.over) return;
    // A game whose every hand was reviewed as it was played joins the trend
    // as it ends, rather than whenever its review is next opened, which may
    // be never. Should the review not be taken on to the finished match, the
    // standings' Review whole game tries again, and says why.
    try { takeReviewOn(owner.snapshot()); } catch { /* Said by the standings' button when pressed. */ }
    const sheet = matchMedia(RESULT_SHEET).matches;
    standingsFocused = byKeyboard || sheet;
    await tick();
    land(standingsHeading, byKeyboard || sheet, sheet && !byKeyboard ? 'start' : 'nearest');
  }

  // What is said beside a set of buttons when a press on one could not be
  // done, where the player pressed it (say). The page's notice line would
  // not do: it sits at the top of the page, above the screen at desktop
  // width while the result's and the standings' buttons are pressed, and on
  // a phone under the result's sheet, or behind the settings' dialog, which
  // leaves all outside it unheard by a screen reader while it is open.
  const REVIEW_NOT_BUILT = 'The review could not be built. Your match is unchanged.';
  const HAND_NOT_SAVED = 'The hand could not be saved to a file. Your current match is unchanged.';
  const GAME_NOT_SAVED = 'The game could not be saved to a file. Your current match is unchanged.';
  // How a replay that failed begins; what follows names its error.
  const NOT_REPLAYED = 'The game could not be replayed for review:';

  // Says beside the result's, the standings' or Export game's buttons
  // (`line`) why a press on one failed, or, once the same thing is done from
  // any of them, takes back what was said of it wherever it shows, since it
  // is no longer true; a different failure's words stay. What was said of it
  // is the same words (`same`), or for a replay, which names its error, any
  // reason it could not be replayed. Answers whether it was done, which
  // tells the buttons whether to bring the line into view (ScoreScreen,
  // Standings, AppSettings). On a phone the result is a sheet over the
  // standings, whose buttons a keyboard or a screen reader still reaches
  // beneath it, so a press there that fails has the sheet step aside, for
  // its words and the button with focus to be seen (ScoreScreen).
  function say(line, words, done, same = text => text === words) {
    if (!done) {
      failures[line] = words;
      if (line === 'standings') standingsFailures += 1;
    } else {
      for (const other of Object.keys(failures)) if (failures[other] && same(failures[other])) failures[other] = '';
    }
    return done;
  }

  // The hand as one mjai log, from the result's button.
  function saveLog() {
    if (!session) return;
    try {
      saveLogFile(`riichi-${view.round}-${view.kyoku}-${Date.now()}.mjai.jsonl`, session.engine.log());
    } catch { return say('result', HAND_NOT_SAVED, false); }
    return say('result', HAND_NOT_SAVED, true);
  }

  // The whole game from East 1, from the result, the standings or Export
  // game (`line`). The engine leaves out a hand still being played, whose
  // deal would show the opponents' tiles. Says whether the file was made.
  function saveGame(line) {
    if (!session || session.closed || !view) return;
    try {
      const names = playerNames(view, matchLabels(session.initialOpponents, session.opponents));
      saveLogFile(gameFileName(), session.engine.game_log(names));
    } catch { return say(line, GAME_NOT_SAVED, false); }
    return say(line, GAME_NOT_SAVED, true);
  }

  // The result's button goes as it opens the hand's review, and on a phone
  // the result steps aside with it, so focus goes on to the review, to its
  // heading, rather than to the page (land). Where the result is not a
  // sheet, the heading is brought into view however the button was pressed:
  // nothing was covered there, and the review opens below the result, often
  // below the screen, where a player who clicked would see the button go and
  // nothing else. Where the result is a sheet, on a phone, only a keyboard's
  // press moves the page, since that player follows focus and is shown a
  // ring there. A mouse's or a finger's press is shown no ring and leaves the
  // page where it was: it asked for what the sheet covers, which the sheet
  // uncovers by stepping aside, and the scroll would take the top of the
  // table off a small screen (pressedByKeyboard). The way back to the result
  // is the control just before the review: on a phone the result's chip,
  // which hands focus back to it, and where the result is not a sheet, its
  // own buttons. A review the engine cannot build opens nothing, and false
  // tells the result so: it stays as it was, its button too, with focus, and
  // says why beside its buttons (say).
  function showReview(event) {
    const byKeyboard = pressedByKeyboard(event);
    try { notes = session?.engine.review() ?? []; }
    catch { return say('result', REVIEW_NOT_BUILT, false); }
    say('result', REVIEW_NOT_BUILT, true);
    void tick().then(() => land(reviewHeading, byKeyboard || !matchMedia(RESULT_SHEET).matches));
    return true;
  }

  // Why the match could not be replayed for review, the last time it could
  // not, for the same press to take back once a review opens.
  let replayFailure = '';
  // Takes the review on to the match as it now stands (MatchReview.extend).
  // Where that adds hands, the player's place is let go, so that the review
  // opens next on the newest of them, the hand just played or the final one,
  // as it did when first opened. Throws as extend does, changing nothing.
  function takeReviewOn(snapshot) {
    const known = matchReview?.hands.length ?? 0;
    matchReview?.extend(snapshot);
    if ((matchReview?.hands.length ?? 0) > known) reviewPlace = null;
  }
  // The match is replayed from its saved commands on an engine of the
  // review's own, so the match and its save are left as they are. The first
  // press builds the review; each press after takes it on to the match as it
  // now stands, so a hand reviewed at an earlier result is not asked about
  // again. Says whether the review is open. The network this page plays
  // judges it, averaging over the suits, and the trend keeps which version
  // that was and that it averaged (review-trend.js). Its batches are
  // counted as they go to the worker, and it waits at the gate of the notes'
  // spells and of its reasoning's, which is built with it for the match, and
  // of the worker's batches: a batch the review let go at Stop runs on in the
  // worker, and Continue waits for it here rather than in the worker's queue.
  // Running, the review sends a batch only once its last is answered, so
  // that gate is open then.
  function openGameReview() {
    try {
      if (matchReview) takeReviewOn(session.snapshot());
      else {
        matchReview = new MatchReview(Game, session.snapshot(), { analyzeBatch: reviewBatch, gate: anyGate(notesAsking, insightAsking, workerBatches), batchMost,
          onComplete: review => {
            reviewTrend = recordTrend(storage, { ...review.summary, network: network.version, at: Date.now() });
          } });
        matchInsight = new MatchInsight(matchReview, reasoning);
        reviewPlace = null;
        reviewRevealed = {};
      }
    } catch (error) {
      replayFailure = `${NOT_REPLAYED} ${error?.message ?? error}. Your saved match is unchanged.`;
      return false;
    }
    reviewingGame = true;
    return true;
  }
  // Offered wherever a hand's result is shown (`line`): any hand's result,
  // and once the match is over the standings, beside which the last hand
  // stays on the table, and on a phone the final result over them. Not while
  // the next hand is being dealt, which would leave the review open over a
  // table that has moved on. A match that could not be replayed is said
  // beside the button pressed (say).
  function reviewGame(line) {
    if (!session || session.closed || busy || view?.phase !== 'over') return;
    const opened = openGameReview();
    return say(line, replayFailure, opened, text => text.startsWith(NOT_REPLAYED));
  }

  // Back to the button that opened the review. While the match is under way
  // that is the result's own. Where the result is a sheet, on a phone, the
  // sheet is up again as it was left, with the button in its row at its
  // foot, so nothing need move; elsewhere the button is brought to the
  // middle of the screen. Once the match is over it is the standings' own,
  // wherever the review was opened from: on a phone the result sheet that
  // offered it has stepped aside for them. It is brought to the middle of
  // the screen, clear of the bar at the top and the result's chip at the
  // bottom.
  async function closeGameReview() {
    const over = Boolean(session?.over);
    reviewingGame = false;
    await tick();
    const button = document.querySelector(over ? '.standings [data-review-game]' : '.screen [data-review-game]');
    button?.focus({ preventScroll: true });
    if (over || !matchMedia(RESULT_SHEET).matches) button?.scrollIntoView({ block: 'center' });
  }

  function dropGameReview() {
    matchInsight?.stop();
    matchInsight = null;
    matchReview?.dispose();
    matchReview = null;
    reviewingGame = false;
    reviewPlace = null;
    reviewRevealed = {};
  }

  // Watch, Guided and Physical play ask the same network, and go first.
  $effect(() => {
    if (mode !== 'play') untrack(() => { matchReview?.stop(); matchInsight?.stop(); });
  });

</script>

<svelte:window onkeydown={onKey} />

<main class:chip={resultChip}>
  <AppSettings bind:mode bind:settingsOpen bind:hints bind:confirmDiscards bind:shortcuts bind:reviewHideMove bind:reviewHideTiles
    {tileFace} {pendingTileFace} {faceWarning} onfacechange={changeTileFace}
    network={network.version} runtime={trainedRuntime} onnetworkchange={changeNetwork}
    {difficulty} {opponents} {ready} {saveConflict} {trainedAvailable} {offline}
    {changeOpponents} {startFresh} {configureTable} {downloadAi} {loggedHands} onsavegame={() => saveGame('export')} exportFailure={failures.export}
    onconfirmationchange={() => selected = null} onshortcutschange={() => picked = null} />
  <!-- On a phone the Tile face select is in the settings' dialog, and while
       that is open this alert would lie behind it, unheard and dimmed: the
       dialog says it under the select instead (AppSettings), and the page
       shows it once the dialog is closed, as an alert only if the dialog has
       not said it already (faceWarningSaid). -->
  {#if faceWarning && !settingsOpen}<p class="notice" data-face-error role={faceWarningSaid ? null : 'alert'}>{faceWarning}</p>{/if}
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
  <!-- Waits in the page, empty and out of sight, for its words, as the
       notice does. -->
  <p class={storageWarning ? 'notice' : 'visually-hidden'} data-storage-warning role="status">{storageWarning}</p>

  {#if !ready && !failure}
    <p class="loading" role="status">{startupNote}</p>
  {:else if ready && mode === 'watch'}
    <AgentWatch {trainedAvailable} {hints} {offline} {workerNote} onchip={shown => { resultChip = shown; }} />
  {:else if ready && mode === 'guided'}
    <GuidedPlay {trainedAvailable} {storage} {hints} {offline} {workerNote} />
  {:else if ready && mode === 'physical'}
    <PhysicalPlay {trainedAvailable} {storage} {offline} {workerNote} />
  {:else if mode === 'play' && view}
    {#if reviewingGame && matchReview}
      <GameReview review={matchReview} insight={matchInsight} trend={reviewTrend} {hints} {trainedAvailable} {offline} {workerNote} onclose={closeGameReview} {shortcuts} {touch}
        bind:filter={reviewFilter} bind:place={reviewPlace} bind:revealed={reviewRevealed} bind:hideMove={reviewHideMove}
        bind:hideTiles={reviewHideTiles} />
    {/if}
    <!-- Hidden rather than closed while the review is open, so the result
         and the standings come back as they were left. -->
    <div class="match-view" class:away={reviewingGame && matchReview}>
      <MatchTable {view} {hints} {thinking} {pendingOpponent} />

      <div class="play-area" class:ended={view.phase === 'over' || Boolean(standings)}>
        <PlayerHand {view} engine={session?.engine} closed={session?.closed ?? false}
          {hints} {busy} blocked={Boolean(failure || saveConflict)} {discardChoices} {confirmDiscards} {picked} {selected}
          {canDiscard} {selectTile} {syncHandFocus} bind:handElement controls={choicesInSeat ? controls : null} />

        {#if !choicesInSeat}{@render controls()}{/if}
      </div>

      <!-- The player's choices, or the hand's result and the standings once
           it is over: after the player's seat, or in it, under the hand,
           while a hand is played on a phone held sideways (choicesInSeat). -->
      {#snippet controls()}
        <section class="controls" aria-label="your choices" bind:this={callElement}>
          <!-- Once another window has taken the match on, no button of the
               standings or of the result can do anything here: they cannot
               be pressed, and each set's own line says to reload, as the
               turn's prompt does while a hand is played. -->
          {#if standings}
            <Standings {standings} onagain={() => start()} onsave={() => saveGame('standings')} onreview={() => reviewGame('standings')}
              failure={saveConflict ? RELOAD_TO_CONTINUE : failures.standings} blocked={Boolean(saveConflict)} bind:heading={standingsHeading} />
          {/if}
          {#if view.phase === 'over' && view.outcome}
            <ScoreScreen outcome={view.outcome} seats={view.seats} dora={shownDora} {hints} {busy}
              bets={view.riichi_sticks ?? 0} onnext={nextHand} blocked={Boolean(saveConflict)}
              onreview={showReview} reviewed={notes !== null} failure={saveConflict ? RELOAD_TO_CONTINUE : failures.result} onlog={saveLog} ongame={() => saveGame('result')}
              ongamereview={() => reviewGame('result')} gameReviewOpen={reviewingGame} {standingsFocused} {standingsFailures} finalHand={Boolean(standings)}
              onstandings={() => land(standingsHeading, true, 'start')}
              onchip={shown => { resultChip = shown; }} />
            {#if notes !== null}<Review {notes} {hints} engine={session?.engine} {trainedAvailable} {offline} {workerNote} bind:adviser={reviewAdviser} bind:heading={reviewHeading}
              onask={() => notesAsking.begin()} hideAnswers={reviewHideMove} />{/if}
          {:else}
            <TurnChoices {view} {shownDora} {busy} {thinking} {failure} {saveConflict}
              loadNote={waitingNote(loadNote, offline)} onclub={downloadWait ? playClubMeanwhile : null}
              {pendingOpponent} {myTurn} {confirmDiscards} {shortcuts} {touch}
              {selectedTile} {callChoices} {choose} discard={tile => confirmation(() => discard(tile))}
              oncancel={() => confirmation(() => { selected = null; })} />
          {/if}
        </section>
      {/snippet}

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
  /* What went wrong, and the ways on, on a pale panel in the light theme and
     a dark one in the dark: its words in a red that reads on either (5.2 and
     4.8 to 1), and its buttons filled, as the page's other buttons that act
     are. As plain controls they were the panel's red on a grey wash of it,
     2.3 to 1, and 1.3 under the pointer. */
  .failure { background: var(--accent-soft); color: var(--accent-text); padding: 12px; border-radius: 8px; }
  .failure p { margin: 0; }
  .failure :global(button.app-control) { background: var(--button-accent); color: var(--button-text); border-color: var(--button-accent); }
  .failure :global(button.app-control:hover:not(:disabled)) { background: var(--button-accent-hover); }
  .recovery-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .notice { margin: 0; font-size: .8rem; }
  .log { font-size: .82rem; display: grid; gap: 4px; max-height: 40vh; overflow-y: auto; padding: 8px 12px; background: #0003; border-radius: 8px; }
  .log p { margin: 0; }
  /* A finger needs more of a line to press than a pointer. */
  @media (pointer: coarse) {
    summary { min-height: 44px; padding: 10px 0; }
  }
  /* On a phone the play area scrolls with the page. It used to stay at the
     top of the screen as the page moved, six pixels down, where the game's
     bar, which stays there too (AppSettings), covered the head of the
     player's seat; and a play area taller than the screen, as with a call
     late in a hand, held its own foot, the call's buttons, under the
     screen's. */
  @media (max-width: 760.98px) {
    main { gap: 6px; }
    .play-area:not(.ended) { background: var(--felt-deep); padding: 6px 0 max(6px, env(safe-area-inset-bottom)); border-radius: 10px; box-shadow: 0 -6px 16px #0003; }
    .controls { padding: 0 8px; }
    /* The result's chip lies over the foot of the screen (ScoreScreen), so
       while it shows the page ends that much lower, and its last line, the
       hand's history, can be scrolled clear of it. The page is told that
       the chip shows (resultChip), so that the room is kept in a browser
       without :has too, by which a style could look for the chip itself. */
    main.chip { padding-bottom: calc(max(10px, env(safe-area-inset-bottom)) + 56px); }
  }
  @media (min-width: 640px) and (max-height: 500.98px) and (orientation: landscape) {
    main { max-width: none; grid-template-columns: minmax(270px, .85fr) minmax(340px, 1.15fr); align-items: start; }
    .notice, .failure, .save-conflict, .loading, .history, main > :global(.game-review) { grid-column: 1 / -1; }
    .play-area { grid-column: 2; grid-row: auto; position: sticky; top: 8px; bottom: auto; }
    .controls { font-size: .8rem; }
  }
  /* Held sideways, a phone's screen leaves the play area less room than it
     takes under the game's bar, which stays at the top (AppSettings): the
     hand's two rows alone are 157 pixels of the 264 to 374 left. Stuck to
     the top of the screen, the play area hid its own foot, and the bar hid
     the hand, so while a hand is played it scrolls with the page instead,
     its choices drawn right under the hand, in the seat (choicesInSeat),
     and a call brings the hand and its buttons into view together (above).
     A finished hand's area, with its result, stays as it was. */
  @media (max-height: 500.98px) and (orientation: landscape) {
    .play-area:not(.ended) { position: static; }
    .play-area:not(.ended) .controls { padding: 0; }
  }

  @media (min-width: 761px) and (min-height: 501px) {
    main { max-width: 1280px; padding-inline: 18px; gap: 14px; }
  }
  /* A window under 700 pixels high, as a 1280 by 720 laptop's browser is,
     keeps the table and the player's hand on one screen only with less
     between the page's rows (MatchTable has the table's own part). */
  @media (min-width: 761px) and (min-height: 501px) and (max-height: 699.98px) {
    main { gap: 8px; }
  }

  @media (max-width: 760.98px), (min-width: 640px) and (max-height: 500.98px) and (orientation: landscape) {
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
