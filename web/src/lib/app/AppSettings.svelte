<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { cycleDialogFocus } from '../dialog-focus.js';
  import { TILE_FACE_OPTIONS } from '../tile-faces.js';
  import OfflineStatus from './OfflineStatus.svelte';
  import type { SettingsProps, GameMode, TileFace } from './types';

  // App owns persistence and match lifecycle; only panel/focus state is local.
  let {
    mode = $bindable('play'), settingsOpen = $bindable(false),
    hints = $bindable(true), confirmDiscards = $bindable(false),
    shortcuts = $bindable(true), tileFace, pendingTileFace = null,
    difficulty, opponents, ready, busy, saveConflict, trainedAvailable, offline,
    changeOpponents, startFresh, configureTable, downloadAi, onfacechange,
    onconfirmationchange, onshortcutschange,
  }: SettingsProps = $props();
  const MODES: [GameMode, string][] = [['play', 'Play'], ['watch', 'Agent watch'], ['physical', 'Physical agent play'], ['guided', 'Guided physical game']];
  const COMPACT = '(max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape)';
  let compact = $state(false);
  let guideOpen = $state(false), optionsOpen = $state(false), offlineOpen = $state(false);
  let settingsDialog = $state<HTMLDialogElement | null>(null);
  let settingsTrigger = $state<HTMLButtonElement | null>(null);
  let preferencesElement = $state<HTMLDivElement | null>(null);

  function closeSettings() {
    settingsOpen = false;
    settingsDialog?.close(); // Restores the opener before another dialog opens.
  }
  function openTable() { closeSettings(); configureTable(); }
  function changeMode(next: GameMode) {
    if (!ready || (mode === 'play' && busy)) return;
    mode = next;
    closeSettings();
  }
  onMount(() => {
    let alive = true;
    const media = matchMedia(COMPACT);
    compact = media.matches;
    const changed = () => {
      const restore = settingsOpen || preferencesElement?.contains(document.activeElement);
      closeSettings();
      compact = media.matches;
      if (restore) void tick().then(() => {
        if (!alive) return;
        (compact ? settingsTrigger : preferencesElement?.querySelector<HTMLElement>('summary'))?.focus({ preventScroll: true });
      });
    };
    media.addEventListener('change', changed);
    return () => { alive = false; media.removeEventListener('change', changed); settingsDialog?.close(); };
  });
  $effect(() => {
    if (compact && settingsOpen && settingsDialog && !settingsDialog.open) {
      settingsDialog.showModal();
      settingsDialog.querySelector<HTMLElement>('[data-close-settings]')?.focus({ preventScroll: true });
    } else if ((!compact || !settingsOpen) && settingsDialog?.open) settingsDialog.close();
  });
</script>

<header class="bar">
  <h1>Riichi</h1>
  {#if mode === 'play'}<label class="opponents">
    <span>Opponents</span>
    <select class="app-control" value={difficulty} onchange={changeOpponents} disabled={!ready || Boolean(saveConflict)} aria-label="opponent strength">
      <option value="beginner">Beginner</option>
      <option value="club">Club</option>
      {#if trainedAvailable || opponents.includes('neural')}<option value="neural">Trained</option>{/if}
      <option value="custom">Custom table</option>
    </select>
    <svg class="select-chevron" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
  </label>{/if}
  <button class="app-control settings-trigger" aria-label="Game settings" aria-expanded={settingsOpen} aria-controls="game-preferences"
    bind:this={settingsTrigger} onclick={() => settingsOpen = !settingsOpen}>
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7h3m4 0h9M4 17h9m4 0h3" /><circle cx="9" cy="7" r="2" /><circle cx="15" cy="17" r="2" /></svg>
  </button>
  {#if mode === 'play'}<button class="app-control restart" onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>{/if}
</header>

<nav class="game-modes" aria-label="Game mode">
  {#each MODES as [key, label] (key)}
    <button class="app-control" aria-pressed={mode === key} disabled={!ready || (mode === 'play' && busy)} onclick={() => changeMode(key)}>{label}</button>
  {/each}
</nav>

{#snippet preferences()}
<div id="game-preferences" class="preferences" class:mobile-open={compact && settingsOpen} bind:this={preferencesElement}>
<div class="mobile-preferences-head"><strong id="game-settings-title">Game settings</strong><button class="app-control" data-close-settings onclick={closeSettings}>Done</button></div>
<label class="compact-mode-selector">Game mode
  <select class="app-control" value={mode} aria-label="Game mode" disabled={!ready || (mode === 'play' && busy)}
    onchange={event => changeMode(event.currentTarget.value as GameMode)}>
    <option value="play">Play</option><option value="watch">Agent watch</option><option value="physical">Physical agent play</option><option value="guided">Guided physical game</option>
  </select>
</label>
{#if mode === 'play'}<button class="app-control mobile-new-game" onclick={() => { if (startFresh()) closeSettings(); }} disabled={!ready || Boolean(saveConflict)}>New game</button>{/if}
{#if mode === 'play' && difficulty === 'custom'}
  <button class="app-control edit-table" onclick={openTable} disabled={!ready || Boolean(saveConflict)}>Edit opponents</button>
{/if}
<details class="options" bind:open={optionsOpen}>
  <summary>Options</summary>
  <div class="option-fields">
    <label>Tile face
      <select class="app-control" value={pendingTileFace ?? tileFace} aria-label="Tile face" disabled={!ready} aria-busy={pendingTileFace !== null}
        onchange={event => onfacechange(event.currentTarget.value as TileFace)}>
        {#each TILE_FACE_OPTIONS as face}
          <option value={face.value}>{face.label}</option>
        {/each}
      </select>
    </label>
    {#if pendingTileFace}<p role="status" data-face-progress>Loading selected tile graphics… Your current tiles remain available.</p>{/if}
    <label><input type="checkbox" bind:checked={hints} /> Hints and markings</label>
    <label><input type="checkbox" bind:checked={confirmDiscards} onchange={onconfirmationchange} /> Select before discarding</label>
    <label><input type="checkbox" bind:checked={shortcuts} onchange={onshortcutschange} /> Keyboard shortcuts in your hand</label>
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
<OfflineStatus {offline} {downloadAi} bind:open={offlineOpen} />
</div>
{/snippet}

<dialog class="settings-dialog" bind:this={settingsDialog} aria-labelledby="game-settings-title"
  onkeydown={cycleDialogFocus} oncancel={closeSettings} onclose={() => { if (!settingsDialog?.open) settingsOpen = false; }}>
  {#if compact}{@render preferences()}{/if}
</dialog>
{#if !compact}{@render preferences()}{/if}

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
  .edit-table { font-size: .8rem; padding: 4px 8px; }
  .bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; border-bottom: 1px solid #ffffff28; padding-bottom: 8px; }
  h1 { margin: 0; font-size: 1.1rem; letter-spacing: .16em; text-transform: uppercase; }
  .opponents { position: relative; margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: .85rem; }
  .opponents select { appearance: none; padding-right: 34px; border-color: #ffffff24; border-radius: 12px; background: #ffffff09; }
  .select-chevron { position: absolute; right: 12px; width: 14px; height: 14px; pointer-events: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; opacity: .7; }
  .restart { font-size: .85rem; }
  .preferences { display: flex; flex-wrap: wrap; align-items: center; gap: 0 20px; min-width: 0; }
  .preferences :global(details[open]) { flex-basis: 100%; order: 1; }
  .options, .guide { font-size: .85rem; min-width: 0; }
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
  @media (max-width: 760px) {
    .bar { gap: 8px; flex-wrap: nowrap; }
    h1 { font-size: 1rem; letter-spacing: .1em; }
    .bar select, .restart { padding: 8px; }
    .preferences { gap: 0 16px; }
    .opponents { gap: 5px; }
    .opponents > span { display: none; }
    .options, .guide { font-size: .8rem; }
  }
  @media (max-width: 359px) {
    .preferences { column-gap: 8px; }
    .opponents { min-width: 0; }
    .bar select { min-width: 0; max-width: 100%; }
    .restart { white-space: nowrap; flex-shrink: 0; }
  }
  @media (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
    .bar, .preferences { grid-column: 1 / -1; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
  .settings-trigger, .mobile-preferences-head, .mobile-new-game { display: none; }

  @media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {
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

    .preferences.mobile-open :global(details), .preferences.mobile-open .edit-table { flex-basis: 100%; }
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
    .preferences :global(.option-fields) { background: rgba(0,0,0,.24); }
  }

  .settings-dialog { width: calc(100% - 16px); max-width: 680px; max-height: min(82dvh, 680px); margin: auto auto max(8px, env(safe-area-inset-bottom)); padding: 14px; border: 1px solid rgba(216,161,42,.45); border-radius: 20px; color: var(--ivory); background: color-mix(in srgb, var(--felt-deep) 96%, black 4%); box-shadow: 0 22px 70px rgba(0,0,0,.48); overflow: auto; }
  .settings-dialog::backdrop { background: rgba(0,0,0,.58); }
  .preferences.mobile-open { align-items: stretch; align-content: flex-start; gap: 2px 12px; }
</style>
