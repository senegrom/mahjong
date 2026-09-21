<script>
  import { OPPONENT_LABELS, OPPONENT_TYPES, OPPONENT_POSITIONS } from '../opponents.js';

  // Only the draft is editable here. App confirms replacement of the current
  // match before promoting these assignments to the active opponents.
  let {
    customDialog = $bindable(null), draftOpponents = $bindable(),
    opponents, trainedAvailable, ready, saveConflict, startCustomTable,
  } = $props();
</script>

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

<style>
  .custom-dialog { width: min(460px, calc(100vw - 24px)); max-height: calc(100dvh - 24px); box-sizing: border-box; padding: 20px; border: 1px solid var(--gold); border-radius: 12px; background: var(--felt-deep); color: var(--ivory); }
  .custom-dialog::backdrop { background: #000a; }
  .custom-dialog h2 { margin: 0; font-size: 1.15rem; }
  .custom-dialog p { font-size: .85rem; line-height: 1.45; }
  .opponent-fields { display: grid; gap: 10px; }
  .opponent-fields label { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .opponent-fields select { width: 60%; min-width: 0; }
  .custom-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; }
  .custom-help { opacity: .85; }
  select, button { min-height: 44px; color: inherit; background: #0004; border: 1px solid #ffffff55; border-radius: 8px; padding: 8px 12px; font: inherit; }
  button { cursor: pointer; touch-action: manipulation; }
  button:hover:not(:disabled) { background: #0007; }
  button:disabled { opacity: .55; cursor: default; }
  select option { color: #17241f; background: #f7f2e4; }
  button.primary { background: var(--button-accent); color: var(--button-text); border-color: var(--button-accent); font-weight: 600; }
  button.primary:hover { background: var(--button-accent-hover); }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }

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
