<script lang="ts">
  import type { Opponent, OpponentDialogProps } from './types';
  import { opponentName, OPPONENT_TYPES, OPPONENT_POSITIONS } from '../opponents.js';

  // Only the draft is editable here. App confirms replacement of the current
  // match before promoting these assignments to the active opponents.
  let {
    customDialog = $bindable(null), draftOpponents = $bindable(),
    opponents, trainedAvailable, ready, saveConflict, startCustomTable,
  }: OpponentDialogProps = $props();
  // How each type plays: its option's title, there when it is asked for
  // rather than read in the dialog.
  const PLAYS: Record<Opponent, string> = {
    beginner: 'Plays simply',
    club: 'Uses tile efficiency and defence',
    neural: 'Plays the network picked under Trained network in the settings; all Trained players share one loaded model',
  };
</script>

<dialog class="custom-dialog" bind:this={customDialog} aria-labelledby="custom-table-title">
  <h2 id="custom-table-title">Custom table</h2>
  <p>Opponents keep their type as the seat winds change.</p>
  <div class="opponent-fields">
    {#each [2, 1, 0] as position (position)}
      <label>
        <span>{OPPONENT_POSITIONS[position]}</span>
        <select class="app-control" bind:value={draftOpponents[position]} aria-label={`${OPPONENT_POSITIONS[position]} opponent`}>
          {#each OPPONENT_TYPES as type (type)}
            <option value={type} title={PLAYS[type]} disabled={type === 'neural' && !trainedAvailable && !opponents.includes('neural')}>{opponentName(type)}</option>
          {/each}
        </select>
      </label>
    {/each}
  </div>
  {#if !trainedAvailable && !opponents.includes('neural')}<p class="custom-help">The trained network is not available right now.</p>{/if}
  <div class="custom-actions">
    <button class="app-control" onclick={() => customDialog?.close()}>Cancel</button>
    <button class="app-control primary" onclick={startCustomTable} disabled={!ready || Boolean(saveConflict)}>Start custom game</button>
  </div>
</dialog>

<style>
  .custom-dialog { width: min(460px, calc(100vw - 24px)); max-height: calc(100dvh - 24px); box-sizing: border-box; padding: 20px; border: 1px solid var(--gold); border-radius: 12px; background: var(--felt-deep); color: var(--ivory); }
  .custom-dialog::backdrop { background: #000a; }
  .custom-dialog h2 { margin: 0; font-size: 1.15rem; }
  .custom-dialog p { font-size: .85rem; line-height: 1.45; }
  .opponent-fields { display: grid; gap: 10px; }
  .opponent-fields label { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .opponent-fields select { width: 60%; min-width: 0; }
  /* Kept apart from the selects, with or without the note between them. */
  .custom-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .custom-help { opacity: .85; }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {

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
