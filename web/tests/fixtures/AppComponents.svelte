<script lang="ts">
  import { setContext } from 'svelte';
  import AppSettings from '../../src/lib/app/AppSettings.svelte';
  import OpponentDialog from '../../src/lib/app/OpponentDialog.svelte';
  import PlayerHand from '../../src/lib/app/PlayerHand.svelte';
  import TurnChoices from '../../src/lib/app/TurnChoices.svelte';
  import { TILE_FACE_CONTEXT } from '../../src/lib/tile-faces.js';
  import type { GameMode, TileFace, Opponent, GameView, LegalChoice, OfflineState, DiscardAnalyser, SeatName } from '../../src/lib/app/types';

  let mode = $state<GameMode>('play'), settingsOpen = $state(false);
  let hints = $state(true), confirmDiscards = $state(true), shortcuts = $state(true);
  let tileFace = $state<TileFace>('classic');
  let opponents = $state<Opponent[]>(['club', 'club', 'club']);
  let draftOpponents = $state<Opponent[]>(['club', 'club', 'club']);
  let customDialog = $state<HTMLDialogElement | null>(null);
  let handElement = $state<HTMLDivElement | null>(null);
  let selected = $state<number | null>(0);
  let callbacks = $state<string[]>([]);
  let offline = $state<OfflineState>({ coreReady: true, aiReady: false, hasModel: true, phase: 'incomplete', progress: 0, warning: 'Interrupted fixture download' });
  const seats: SeatName[] = ['east', 'south', 'west', 'north'];
  const view: GameView = {
    phase: 'act', round: 'east', kyoku: 1, wall: 60, counters: 0, riichi_sticks: 0,
    seats: seats.map((seat, player) => ({ seat, player, score: 30000, turn: player === 0, riichi: false,
      hand: player === 0 ? ['1m', '2m'] : [], drawn: player === 0 ? '3m' : null, discards: [], melds: [] })),
    dora_indicators: ['4z'], dora_types: [], dora: [], safe: [], shanten: 0,
    waits: ['1m'], waits_left: [2], furiten: false, pending_discard: null, pending_from: null,
  };
  const choices: LegalChoice[] = ['1m', '2m', '3m'].map(tile => ({ kind: 'discard', tile }));
  const engine: DiscardAnalyser = { discard_hint: () => ({ shanten: 0, waits: ['1m'], waits_left: [2] }) };
  setContext(TILE_FACE_CONTEXT, () => tileFace);
  function configureTable() { draftOpponents = [...opponents]; customDialog?.showModal(); }
</script>

<main>
  <AppSettings bind:mode bind:settingsOpen bind:hints bind:confirmDiscards bind:shortcuts {tileFace}
    difficulty="custom" {opponents} ready={true} busy={false} saveConflict="" trainedAvailable={false} {offline}
    changeOpponents={() => configureTable()} startFresh={() => { callbacks.push('new'); return true; }} {configureTable}
    downloadAi={() => { callbacks.push('download'); offline = { ...offline, phase: 'ready', aiReady: true }; }}
    onfacechange={face => { tileFace = face; callbacks.push(`face:${face}`); }}
    onconfirmationchange={() => { selected = null; callbacks.push('confirmation'); }}
    onshortcutschange={() => callbacks.push('shortcuts')} />
  <OpponentDialog bind:customDialog bind:draftOpponents {opponents} trainedAvailable={false} ready={true} saveConflict=""
    startCustomTable={() => { opponents = [...draftOpponents]; callbacks.push('custom'); customDialog?.close(); }} />
  <PlayerHand {view} {engine} closed={false} {hints} busy={false} blocked={false} discardChoices={choices}
    picked={null} {selected} canDiscard={() => true} selectTile={(_tile, index) => selected = index}
    syncHandFocus={() => {}} bind:handElement />
  <TurnChoices {view} shownDora={[]} busy={false} thinking={false} failure="" saveConflict="" loadNote=""
    pendingOpponent={null} myTurn={true} {confirmDiscards} {shortcuts} touch={false}
    selectedTile={selected === null ? null : ['1m', '2m', '3m'][selected]} callChoices={[{ kind: 'riichi', tile: '1m' }]}
    choose={choice => { callbacks.push(choice.kind); }} discard={tile => callbacks.push(`discard:${tile}`)} oncancel={() => selected = null} />
  <output data-parent-state>{JSON.stringify({ mode, settingsOpen, hints, confirmDiscards, shortcuts, tileFace, opponents, selected, callbacks, handBound: Boolean(handElement) })}</output>
</main>

<style>
  main { display: grid; gap: 12px; padding: 10px; }
  output { overflow-wrap: anywhere; }
</style>
