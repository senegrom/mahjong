<script>
  import { TILE_FACE_OPTIONS } from '../tile-faces.js';

  // Preferences flow back to App, where they are persisted once. This component
  // never creates a match or writes storage when a settings panel is opened.
  let {
    mode = $bindable('play'), settingsOpen = $bindable(false),
    hints = $bindable(true), confirmDiscards = $bindable(false),
    shortcuts = $bindable(true), tileFace = $bindable('classic'),
    difficulty, opponents, ready, busy, saveConflict, trainedAvailable, offline,
    changeOpponents, startFresh, configureTable, downloadAi,
    onconfirmationchange, onshortcutschange,
  } = $props();
  let guideOpen = $state(false);
</script>

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
  .options, .guide, .offline-settings { font-size: .85rem; min-width: 0; }
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
    .options, .guide, .offline-settings { font-size: .8rem; }
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
  .settings-trigger, .settings-backdrop, .mobile-preferences-head, .mobile-new-game { display: none; }

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
  }
</style>
