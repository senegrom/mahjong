from pathlib import Path


def swap(path, old, new, expected=1):
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != expected:
        raise SystemExit(f'{path}: expected {expected} occurrences, found {count}: {old[:100]!r}')
    file.write_text(text.replace(old, new))


def append_before(path, marker, addition):
    file = Path(path)
    text = file.read_text()
    if addition.strip() in text:
        raise SystemExit(f'{path}: polish block already present')
    if marker not in text:
        raise SystemExit(f'{path}: marker not found')
    file.write_text(text.replace(marker, addition + marker, 1))


# A hand tile owns its optional count below the face. Keeping the count outside
# Tile means result/discard/meld tiles retain their exact layout and semantics.
Path('web/src/lib/HandTile.svelte').write_text(r'''<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  let {
    tile,
    remaining = null,
    showRemaining = false,
    handIndex = null,
    onclick = null,
    disabled = false,
    muted = disabled,
    selected = false,
    drawn = false,
    discardShanten = null,
    safe = false,
    dora = false,
  } = $props();
</script>

<span class="hand-tile" data-tile={tile} data-drawn={drawn ? 'true' : undefined}>
  <Tile {tile} {handIndex} {onclick} {disabled} {muted} {selected} {drawn}
    {discardShanten} {safe} {dora} />
  {#if showRemaining && remaining !== null}
    <span class="copy-count" class:dead={remaining === 0} class:thin={remaining === 1}
      title={`${remaining} unseen ${tileWords(tile)} ${remaining === 1 ? 'remains' : 'remain'}`}
      aria-label={`${remaining} unseen ${tileWords(tile)} ${remaining === 1 ? 'remains' : 'remain'}`}>
      {remaining}
    </span>
  {/if}
</span>

<style>
  .hand-tile {
    display: inline-flex;
    width: var(--tile-width);
    min-width: 0;
    flex: none;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
  }

  .copy-count {
    width: 72%;
    min-height: 12px;
    margin-top: 4px;
    padding-top: 2px;
    border-top: 1px solid rgba(247, 242, 228, .25);
    color: rgba(247, 242, 228, .72);
    font-size: .64rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    text-align: center;
  }

  .copy-count.thin {
    border-color: rgba(216, 161, 42, .62);
    color: #efc86c;
  }

  .copy-count.dead {
    border-color: rgba(255, 184, 164, .55);
    color: var(--warning-text);
  }
</style>
''')

# Exact unseen-copy counts, matching the rules engine's visibility semantics:
# claimed discards are represented by their meld, so never count both.
ui = Path('web/src/lib/ui.js')
ui_text = ui.read_text().rstrip()
ui_add = r'''

const ALL_TILES = [
  ...['m', 'p', 's'].flatMap(suit => Array.from({ length: 9 }, (_, i) => `${i + 1}${suit}`)),
  ...Array.from({ length: 7 }, (_, i) => `${i + 1}z`),
];

/** Copies of every tile kind nobody can currently see from the human seat.
 * This is the same public information used for wait width: own concealed
 * tiles, unclaimed discards, all called sets and the exposed dora indicators.
 */
export function unseenTileCounts(view) {
  const left = new Map(ALL_TILES.map(tile => [tile, 4]));
  const see = tile => {
    if (!left.has(tile)) return;
    left.set(tile, Math.max(0, left.get(tile) - 1));
  };
  for (const [index, seat] of (view?.seats ?? []).entries()) {
    if (index === 0) {
      for (const tile of seat.hand ?? []) see(tile);
      if (seat.drawn) see(seat.drawn);
    }
    for (const discard of seat.discards ?? []) if (!discard.claimed) see(discard.tile);
    for (const meld of seat.melds ?? []) for (const tile of meld.tiles ?? []) see(tile);
  }
  for (const tile of view?.dora_indicators ?? []) see(tile);
  return left;
}
'''
if 'export function unseenTileCounts' in ui_text:
    raise SystemExit('web/src/lib/ui.js already has unseenTileCounts')
ui.write_text(ui_text + ui_add)

# App wiring: compact mobile chrome, hand counts, turn/think state, stronger call
# presentation. Keep the just-landed Matisse tile-face setting untouched.
swap('web/src/App.svelte',
     "  import Tile, { preloadTiles } from './lib/Tile.svelte';\n",
     "  import Tile, { preloadTiles } from './lib/Tile.svelte';\n  import HandTile from './lib/HandTile.svelte';\n")
swap('web/src/App.svelte',
     "  import { acceptsHandKey, heldSafeCount, callLabel, callTiles, moveHandFocus, analyzeDiscards } from './lib/ui.js';",
     "  import { acceptsHandKey, heldSafeCount, callLabel, callTiles, moveHandFocus, analyzeDiscards, unseenTileCounts } from './lib/ui.js';")
swap('web/src/App.svelte',
     "  let guideOpen = $state(false);\n",
     "  let guideOpen = $state(false);\n  let settingsOpen = $state(false);\n")
swap('web/src/App.svelte',
     "  let displayLeft = $derived(discardHint?.waits_left ?? view?.waits_left ?? []);\n  let uraIndicators = $derived(view?.outcome?.wins?.find(win => win.ura_indicators?.length)?.ura_indicators ?? []);",
     "  let displayLeft = $derived(discardHint?.waits_left ?? view?.waits_left ?? []);\n  let remainingByTile = $derived(hints && view ? unseenTileCounts(view) : new Map());\n  let uraIndicators = $derived(view?.outcome?.wins?.find(win => win.ura_indicators?.length)?.ura_indicators ?? []);")
swap('web/src/App.svelte',
     "  function configureTable() {\n    draftOpponents = [...opponents];\n    customDialog?.showModal();\n  }",
     "  function configureTable() {\n    settingsOpen = false;\n    draftOpponents = [...opponents];\n    customDialog?.showModal();\n  }")

old_header = '''  <header class="bar">
    <h1>Riichi</h1>
    <label class="opponents">
      <span>Opponents</span>
      <select value={difficulty} onchange={changeOpponents} disabled={!ready || Boolean(saveConflict)} aria-label="opponent strength">
        <option value="beginner">Beginner</option>
        <option value="club">Club</option>
        {#if trainedAvailable || opponents.includes('neural')}<option value="neural">Trained</option>{/if}
        <option value="custom">Custom table</option>
      </select>
    </label>
    <button class="restart" onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>
  </header>

  <div class="preferences">'''
new_header = '''  <header class="bar">
    <h1><span>Riichi</span>{#if view}<span class="header-round"> · {NAMES[view.round]} {view.kyoku}</span>{/if}</h1>
    <div class="compact-status" aria-label="app status">
      <span class="status-dot" class:ready={offline.coreReady}
        title={offline.coreReady ? 'Game available offline' : 'Offline preparation incomplete'}
        aria-label={offline.coreReady ? 'Game available offline' : 'Offline preparation incomplete'}>●</span>
      <span class="save-mark" class:ready={Boolean(storage) && !storageWarning && !saveConflict}
        title={Boolean(storage) && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}
        aria-label={Boolean(storage) && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}>✓</span>
    </div>
    <label class="opponents">
      <span>Opponents</span>
      <select value={difficulty} onchange={changeOpponents} disabled={!ready || Boolean(saveConflict)} aria-label="opponent strength">
        <option value="beginner">Beginner</option>
        <option value="club">Club</option>
        {#if trainedAvailable || opponents.includes('neural')}<option value="neural">Trained</option>{/if}
        <option value="custom">Custom table</option>
      </select>
    </label>
    <button class="settings-trigger" aria-label="Game settings" aria-expanded={settingsOpen}
      onclick={() => settingsOpen = !settingsOpen}>⚙</button>
    <button class="restart" onclick={() => startFresh()} disabled={!ready || Boolean(saveConflict)}>New game</button>
  </header>

  {#if settingsOpen}<button class="settings-backdrop" aria-label="Close game settings" onclick={() => settingsOpen = false}></button>{/if}
  <div class="preferences" class:mobile-open={settingsOpen}>
  <div class="mobile-preferences-head"><strong>Game settings</strong><button onclick={() => settingsOpen = false}>Done</button></div>
  <button class="mobile-new-game" onclick={() => { if (startFresh()) settingsOpen = false; }} disabled={!ready || Boolean(saveConflict)}>New game</button>'''
swap('web/src/App.svelte', old_header, new_header)

# Explain the new count directly where the other learning aids live.
swap('web/src/App.svelte',
     '''        </dd>\n\n        <dt><span class="swatch dora"></span> a red ring, and a shine</dt>'''.replace('\\n','\n'),
     '''        </dd>\n\n        <dt>The number below a tile</dt>\n        <dd>How many copies of that same tile nobody can see yet. Zero is red and one is gold, so thin tiles stand out without covering the artwork.</dd>\n\n        <dt><span class="swatch dora"></span> a red ring, and a shine</dt>'''.replace('\\n','\n'))

# Saved/restored messages are independent of the settings sheet on phones.
swap('web/src/App.svelte',
     "  {#if notice}<p class=\"notice\" role=\"status\">{notice}</p>{/if}\n  </div>\n\n  <dialog class=\"custom-dialog\"",
     "  </div>\n  {#if notice}<p class=\"notice\" role=\"status\">{notice}</p>{/if}\n\n  <dialog class=\"custom-dialog\"")

# Opponent turn/thinking state is visible in the actual seat card.
swap('web/src/App.svelte',
     '<div class="place across"><Seat seat={across} side="across" dealer={across.seat === \'east\'} dora={shownDora} /></div>',
     '<div class="place across"><Seat seat={across} side="across" dealer={across.seat === \'east\'} dora={shownDora} thinking={thinking && pendingOpponent?.player === across.player} /></div>')
swap('web/src/App.svelte',
     '<div class="place left"><Seat seat={left} side="left" dealer={left.seat === \'east\'} dora={shownDora} /></div>',
     '<div class="place left"><Seat seat={left} side="left" dealer={left.seat === \'east\'} dora={shownDora} thinking={thinking && pendingOpponent?.player === left.player} /></div>')
swap('web/src/App.svelte',
     '<div class="place right"><Seat seat={right} side="right" dealer={right.seat === \'east\'} dora={shownDora} /></div>',
     '<div class="place right"><Seat seat={right} side="right" dealer={right.seat === \'east\'} dora={shownDora} thinking={thinking && pendingOpponent?.player === right.player} /></div>')
swap('web/src/App.svelte',
     '<button class="inspect" onclick={inspectTable} aria-label="Inspect all discards and called sets">All discards</button>',
     '<button class="inspect" onclick={inspectTable} aria-label="Inspect all discards and called sets"><span aria-hidden="true">▦</span> Discards</button>')
swap('web/src/App.svelte',
     '<section class="mine" aria-label="your seat">',
     '<section class="mine" class:turn={myTurn} aria-label="your seat">')
swap('web/src/App.svelte',
     '          <span class="score">{me.score.toLocaleString()}</span>\n          {#if me.riichi}',
     '          <span class="score">{me.score.toLocaleString()}</span>\n          {#if me.seat === \'east\'}<span class="my-dealer">Dealer</span>{/if}\n          {#if me.riichi}')

old_hand = '''          {#each handTiles as tile, index (index)}
            <Tile {tile} handIndex={index} onclick={() => selectTile(tile, index)}
              disabled={!canDiscard(tile)} muted={view.phase === 'over'} selected={myTurn && (picked === index || selected === index)}
              drawn={Boolean(me.drawn) && index === me.hand.length}
              discardShanten={discardHints.get(tile)?.shanten ?? null}
              safe={hints && view.phase !== 'over' && view.safe.includes(tile)} dora={shownDora.includes(tile)} />
          {/each}'''
new_hand = '''          {#each handTiles as tile, index (index)}
            <HandTile {tile} handIndex={index} onclick={() => selectTile(tile, index)}
              disabled={!canDiscard(tile)} muted={view.phase === 'over'} selected={myTurn && (picked === index || selected === index)}
              drawn={Boolean(me.drawn) && index === me.hand.length}
              discardShanten={discardHints.get(tile)?.shanten ?? null}
              remaining={remainingByTile.get(tile) ?? null} showRemaining={hints}
              safe={hints && view.phase !== 'over' && view.safe.includes(tile)} dora={shownDora.includes(tile)} />
          {/each}'''
swap('web/src/App.svelte', old_hand, new_hand)

old_offer = '''          {#if view.phase === 'call' && view.pending_discard && callChoices.length}
            <div class="offered-tile"><Tile tile={view.pending_discard} size="small" dora={shownDora.includes(view.pending_discard)} />
              <span><strong>{NAMES[view.pending_from] ?? 'An opponent'}</strong> offers the {tileWords(view.pending_discard)}</span>
            </div>
          {/if}'''
new_offer = '''          {#if view.phase === 'call' && view.pending_discard && callChoices.length}
            <div class="call-stage" aria-live="polite">
              <span class="call-kicker">Discard</span>
              <Tile tile={view.pending_discard} dora={shownDora.includes(view.pending_discard)} />
              <span class="call-message"><strong>{NAMES[view.pending_from] ?? 'An opponent'}</strong><span>discarded the {tileWords(view.pending_discard)}</span></span>
            </div>
          {/if}'''
swap('web/src/App.svelte', old_offer, new_offer)
swap('web/src/App.svelte',
     "                <button class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}",
     "                <button class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'} class:win-call={choice.kind === 'ron' || choice.kind === 'tsumo'} data-choice={choice.kind}")

# Layer the visual overhaul after the existing mature responsive rules, so it
# does not disturb the tested geometry unless deliberately overridden.
polish_css = r'''

  /* --- 2026 game-feel pass ------------------------------------------------ */
  .header-round, .compact-status, .settings-trigger, .settings-backdrop,
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
  .hand :global(.hand-tile[data-drawn=true]) { margin-inline-start: var(--draw-gap); }

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

  @media (min-width: 761px) {
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

  @media (max-width: 760px) {
    main { padding-top: max(5px, env(safe-area-inset-top)); gap: 5px; }
    .bar {
      position: sticky;
      top: 0;
      z-index: 18;
      min-height: 48px;
      padding: 5px 0 6px;
      gap: 7px;
      border-bottom-color: rgba(255,255,255,.12);
      background: linear-gradient(180deg, color-mix(in srgb, var(--felt-deep) 96%, transparent), color-mix(in srgb, var(--felt-deep) 88%, transparent));
      backdrop-filter: blur(12px);
    }
    h1 { margin-right: auto; font-size: .91rem; letter-spacing: .10em; white-space: nowrap; }
    .header-round { display: inline; font-size: .76rem; font-weight: 500; letter-spacing: 0; opacity: .68; text-transform: none; }
    .compact-status { display: inline-flex; align-items: center; gap: 4px; }
    .status-dot, .save-mark { color: #b07161; font-size: .7rem; line-height: 1; opacity: .8; }
    .status-dot.ready, .save-mark.ready { color: #8bd6a7; opacity: 1; }
    .save-mark { font-size: .8rem; font-weight: 800; }
    .opponents { margin-left: 0; }
    .opponents select { min-height: 38px; max-width: 116px; padding: 5px 8px; border-radius: 10px; font-size: .8rem; }
    .settings-trigger {
      display: inline-flex;
      width: 38px;
      min-height: 38px;
      align-items: center;
      justify-content: center;
      padding: 0;
      border-radius: 50%;
      font-size: 1rem;
    }
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
    .mobile-preferences-head button { min-height: 38px; padding: 5px 11px; }
    .mobile-new-game { display: block; flex-basis: 100%; margin-bottom: 5px; }
    .preferences .option-fields { background: rgba(0,0,0,.24); }
    .notice {
      position: fixed;
      top: max(54px, calc(env(safe-area-inset-top) + 48px));
      right: 10px;
      z-index: 17;
      max-width: min(70vw, 260px);
      padding: 5px 8px;
      border: 1px solid rgba(255,255,255,.10);
      border-radius: 999px;
      background: rgba(8,35,24,.83);
      box-shadow: 0 4px 14px rgba(0,0,0,.18);
      font-size: .68rem;
      pointer-events: none;
    }

    .board { margin-top: 1px; }
    .centre { border: 1px solid rgba(216,161,42,.18); box-shadow: inset 0 1px 0 rgba(255,255,255,.035); }
    .mine { border-radius: 15px; background: rgba(3,29,19,.40); }
    .hand { --draw-gap: 14px; }
    .hand :global(.hand-tile) { width: 100%; }
    .hand :global(.hand-tile[data-drawn=true]) { margin-inline-start: var(--draw-gap); }
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
'''
append_before('web/src/App.svelte', '</style>', polish_css)

# Stronger opponent cards and explicit dealer/controller/turn status.
swap('web/src/lib/Seat.svelte',
     "  let { seat, side = 'across', dealer = false, dora = [] } = $props();",
     "  let { seat, side = 'across', dealer = false, dora = [], thinking = false } = $props();")
swap('web/src/lib/Seat.svelte',
     "  let announcement = $derived(seat.riichi ? 'Riichi' : '');\n",
     "  let announcement = $derived(seat.riichi ? 'Riichi' : '');\n  const shortScore = score => `${(score / 1000).toFixed(1)}k`;\n")
swap('web/src/lib/Seat.svelte',
     '<section class="seat {side}" class:turn={seat.turn} aria-label="{NAMES[seat.seat]} seat">',
     '<section class="seat {side}" class:turn={seat.turn} class:thinking aria-label="{NAMES[seat.seat]} seat">')
swap('web/src/lib/Seat.svelte',
     '''    <span class="wind" class:dealer>{NAMES[seat.seat]}</span>
    <span class="score">{seat.score.toLocaleString()}</span>''',
     '''    <span class="wind" class:dealer>{NAMES[seat.seat]}</span>
    {#if dealer}<span class="dealer-badge" aria-label="dealer">Dealer</span>{/if}
    <span class="score score-full">{seat.score.toLocaleString()}</span>
    <span class="score score-short" title={seat.score.toLocaleString()}>{shortScore(seat.score)}</span>''')
seat_css = r'''

  .dealer::after { content: none; }
  .dealer-badge {
    padding: 1px 6px;
    border: 1px solid rgba(216,161,42,.48);
    border-radius: 999px;
    color: var(--gold);
    font-size: .56rem;
    font-weight: 750;
    letter-spacing: .07em;
    text-transform: uppercase;
  }
  .score-short { display: none; }
  .opponent-type {
    padding: 1px 6px;
    border-radius: 999px;
    background: rgba(255,255,255,.06);
  }
  .seat { position: relative; background: rgba(2,25,16,.32); box-shadow: inset 0 1px 0 rgba(255,255,255,.025); }
  .turn {
    border-color: rgba(216,161,42,.88);
    background: rgba(3,31,20,.48);
    box-shadow: 0 0 0 1px rgba(216,161,42,.12), 0 0 22px rgba(216,161,42,.13), inset 0 1px 0 rgba(255,255,255,.04);
  }
  .thinking::before {
    content: '';
    position: absolute;
    inset: -3px;
    pointer-events: none;
    border: 1px solid rgba(239,196,94,.72);
    border-radius: 13px;
    animation: thinking-pulse 1.35s ease-in-out infinite;
  }
  @keyframes thinking-pulse { 0%,100% { opacity: .22; transform: scale(.995); } 50% { opacity: .9; transform: scale(1.008); } }

  @media (min-width: 761px) {
    .seat { min-height: 150px; padding: 12px 14px; border-radius: 14px; }
    .back { width: 8px; height: 17px; }
  }
  @media (max-width: 760px), (max-height: 500px) and (orientation: landscape) {
    .score-full { display: none; }
    .score-short { display: inline; }
    .dealer-badge { padding-inline: 4px; font-size: .5rem; }
  }
  @media (prefers-reduced-motion: reduce) { .thinking::before { animation: none; opacity: .7; } }
'''
append_before('web/src/lib/Seat.svelte', '</style>', seat_css)

# Results become a proper result card on desktop and a dismissible bottom sheet
# on phones, with the value of the hand visually dominant.
Path('web/src/lib/ScoreScreen.svelte').write_text(r'''<script>
  import Tile from './Tile.svelte';
  import Melds from './Melds.svelte';

  let {
    outcome,
    seats = [],
    onnext,
    ongame,
    onreview,
    reviewed = false,
    onlog,
    gameOver = false,
    finalHand = false,
    dora = [],
    bets = 0,
    busy = false,
    hints = true,
  } = $props();

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };
  let minimized = $state(false);
  let primaryWin = $derived(outcome?.wins?.[0] ?? null);

  function reveal(node) {
    const id = requestAnimationFrame(() => {
      const phone = matchMedia('(max-width: 760px)').matches;
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!phone) node.scrollIntoView({ block: 'nearest', behavior: reduced ? 'instant' : 'smooth' });
      node.focus({ preventScroll: true });
    });
    return { destroy: () => cancelAnimationFrame(id) };
  }

  function signed(value) {
    if (value === 0) return '0';
    return value > 0 ? `+${value.toLocaleString()}` : value.toLocaleString();
  }

  function heroPayment(payment = '') {
    return payment.split(',')[0]
      .replace(/\b\d{4,6}\b/g, value => Number(value).toLocaleString())
      .replace(/\s+from.*$/, '');
  }

  function reviewTable() {
    onreview?.();
    minimized = true;
  }
</script>

{#if minimized}
  <button class="result-chip" onclick={() => minimized = false}>Show result · {outcome.line}</button>
{/if}

<section class="screen" class:minimized aria-label="how the hand ended" tabindex="-1" use:reveal>
  <header class="result-header">
    <div class="result-title">
      <span class="eyebrow">Hand result</span>
      <h2>{outcome.line}</h2>
    </div>
    {#if primaryWin}
      <div class="hero-score" aria-label={`${NAMES[primaryWin.seat]} ${primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}, ${primaryWin.limit ?? `${primaryWin.han} han`}, ${heroPayment(primaryWin.payment)}`}>
        <span>{NAMES[primaryWin.seat]} · {primaryWin.by === 'self-draw' ? 'Tsumo' : 'Ron'}</span>
        <strong>{primaryWin.limit ? primaryWin.limit.toUpperCase() : `${primaryWin.han} HAN`}</strong>
        <b>{heroPayment(primaryWin.payment)}</b>
      </div>
    {/if}
    {#if outcome.kind === 'draw' && outcome.tenpai.length}
      <p class="waiting">Waiting: {outcome.tenpai.map((seat) => NAMES[seat]).join(', ')}</p>
    {/if}
    {#if outcome.kind === 'draw' && bets > 0}
      <p class="waiting">{bets} riichi bet{bets > 1 ? 's' : ''} stay{bets > 1 ? '' : 's'} on the table for the next winner.</p>
    {/if}
  </header>

  {#each outcome.wins as win (win.seat)}
    <article class="win">
      <div class="win-heading">
        <h3>{NAMES[win.seat]} — {win.by === 'self-draw' ? 'Tsumo' : 'Ron'}</h3>
        <span>{win.han} han{#if win.fu} · {win.fu} minipoints{/if}</span>
      </div>
      <div class="tiles">
        {#each win.hand as tile, index (tile + index)}
          <Tile {tile} size="small" dora={hints && (win.dora_types ?? dora).includes(tile)} />
        {/each}
        <span class="gap"></span>
        <span class="winning"><Tile tile={win.winning_tile} size="small" dora={hints && (win.dora_types ?? dora).includes(win.winning_tile)} /></span>
        {#if win.melds.length}
          <span class="gap"></span>
          <Melds melds={win.melds} size="small" dora={hints ? (win.dora_types ?? dora) : []} />
        {/if}
      </div>

      <div class="bonus-indicators">
        <div class="indicator-row" aria-label="winning hand dora indicators">
          <span>Dora indicators</span>
          <div class="indicator-tiles">{#each win.dora_indicators ?? [] as tile, slot (slot)}<Tile {tile} size="small" />{/each}</div>
        </div>
        {#if win.ura_indicators?.length}
          <div class="indicator-row ura" aria-label="winning hand ura-dora indicators">
            <span>Ura-dora indicators</span>
            <div class="indicator-tiles">{#each win.ura_indicators as tile, slot (slot)}<Tile {tile} size="small" />{/each}</div>
          </div>
          <details class="indicator-help">
            <summary>ⓘ Ura-dora</summary>
            <p>Revealed after a riichi win. Each indicator points to the next tile and repeated indicators count again.{win.limit === 'yakuman' ? ' Dora do not add han to yakuman.' : ''}</p>
          </details>
        {/if}
      </div>

      <div class="working">
        <ul class="yaku">
          {#each win.yaku as yaku (yaku.name)}<li><span>{yaku.name}</span><b>{yaku.han}</b></li>{/each}
          {#if win.dora || win.ura_indicators?.length}<li class="dora-count"><span>Dora</span><b>{win.dora - (win.ura_dora ?? 0)}</b></li>{/if}
          {#if win.ura_indicators?.length}<li class="ura-count"><span>Ura-dora</span><b>{win.ura_dora ?? 0}</b></li>{/if}
        </ul>
        <p class="total"><b>{win.han}</b> han{#if win.fu}<span class="fu">, {win.fu} minipoints</span>{/if}{#if win.limit}<span class="limit">{win.limit}</span>{/if}</p>
        <p class="payment">{win.payment}{#if win.bets > 0}, and {win.bets.toLocaleString()} in bets from the table{/if}</p>
      </div>
    </article>
  {/each}

  <table class="changes">
    <thead><tr><th scope="col">Player</th><th scope="col">Change</th><th scope="col">Points</th></tr></thead>
    <tbody>
      {#each seats as seat, index (seat.seat)}
        <tr><th scope="row">{index === 0 ? 'You' : NAMES[seat.seat]}</th>
          <td class:up={outcome.changes[index] > 0} class:down={outcome.changes[index] < 0}>{signed(outcome.changes[index] ?? 0)}</td>
          <td class="after">{seat.score.toLocaleString()}</td></tr>
      {/each}
    </tbody>
  </table>

  <div class="buttons">
    {#if finalHand}<span class="final-caption">Final hand</span>
    {:else if gameOver}<button disabled={busy} class="primary" onclick={ongame}>Play again</button>
    {:else}<button disabled={busy} class="primary" onclick={onnext}>Next hand</button>{/if}
    {#if onreview && !reviewed}<button class="quiet" onclick={reviewTable}>{finalHand ? 'Review final hand' : 'View table / my hand'}</button>{/if}
    {#if onlog}<button class="quiet" onclick={onlog} title="The hand as an mjai event log, which replayers and other riichi programs read">{finalHand ? 'Save final hand' : 'Save this hand'}</button>{/if}
  </div>
</section>

<style>
  .result-chip { display: none; }
  .screen {
    display: grid;
    gap: 14px;
    padding: 18px 20px;
    border: 1px solid rgba(216,161,42,.42);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(2,31,20,.62), rgba(0,0,0,.31));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 14px 34px rgba(0,0,0,.12);
  }
  .result-header { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 10px 20px; }
  .result-title { min-width: min(100%, 260px); }
  .eyebrow { display: block; margin-bottom: 3px; color: var(--gold); font-size: .64rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  h2 { margin: 0; font-size: 1.12rem; font-weight: 650; }
  h3 { margin: 0; font-size: .9rem; }
  .hero-score { display: grid; justify-items: end; line-height: 1.08; }
  .hero-score span { font-size: .72rem; opacity: .7; }
  .hero-score strong { margin-top: 3px; color: var(--gold); font-size: 1.02rem; letter-spacing: .08em; }
  .hero-score b { margin-top: 3px; font-size: 1.75rem; font-variant-numeric: tabular-nums; }
  .waiting { flex-basis: 100%; margin: 0; font-size: .83rem; opacity: .78; }
  .win { display: grid; gap: 10px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.12); }
  .win-heading { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 12px; }
  .win-heading span { font-size: .72rem; opacity: .7; }
  .tiles { display: flex; align-items: flex-end; gap: 2px; flex-wrap: wrap; }
  .gap { width: 12px; }
  .winning { display: inline-flex; border-radius: 6px; box-shadow: 0 0 0 2px var(--gold), 0 0 14px rgba(216,161,42,.18); }
  .bonus-indicators { display: flex; flex-wrap: wrap; gap: 10px 22px; min-width: 0; }
  .indicator-row { display: grid; gap: 6px; min-width: 0; font-size: .78rem; }
  .indicator-tiles { display: flex; flex-wrap: wrap; gap: 5px; }
  .indicator-help { flex-basis: 100%; font-size: .76rem; opacity: .82; }
  .indicator-help summary { min-height: 28px; padding: 2px 0; color: var(--gold); }
  .indicator-help p { margin: 2px 0 0; max-width: 62ch; }
  .final-caption { align-self: center; font-size: .85rem; }
  .working { display: grid; gap: 4px; }
  .yaku { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,210px),1fr)); gap: 1px 18px; font-size: .88rem; }
  .yaku li { display: flex; justify-content: space-between; gap: 12px; padding: 1px 0; border-bottom: 1px dotted rgba(255,255,255,.16); }
  .total { margin: 4px 0 0; font-size: 1rem; }
  .fu { opacity: .85; }
  .limit { margin-left: 8px; color: var(--gold); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
  .payment { margin: 0; font-size: .9rem; opacity: .9; }
  .changes { border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; }
  .changes th { padding: 2px 16px 2px 0; text-align: left; font-weight: 500; opacity: .84; }
  .changes td { padding: 2px 16px 2px 0; }
  .up { color: #7fd1a0; }
  .down { color: var(--warning-text); }
  .after { opacity: .7; }
  .buttons { display: flex; flex-wrap: wrap; gap: 8px; }
  .buttons button { min-height: 44px; padding: 8px 18px; border: 1px solid var(--button-accent); border-radius: 999px; background: var(--button-accent); color: var(--button-text); font-weight: 650; cursor: pointer; }
  .buttons button.quiet { border-color: rgba(255,255,255,.25); background: transparent; color: inherit; font-weight: 500; }

  @media (max-width: 760px) {
    .screen {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 50;
      max-height: 88dvh;
      overflow: auto;
      padding: 18px 18px max(14px, env(safe-area-inset-bottom));
      border-width: 1px 0 0;
      border-radius: 22px 22px 0 0;
      background: color-mix(in srgb, var(--felt-deep) 97%, black 3%);
      box-shadow: 0 -100vh 0 100vh rgba(0,0,0,.60), 0 -18px 55px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.05);
      animation: result-up .28s cubic-bezier(.2,.8,.2,1) both;
      overscroll-behavior: contain;
    }
    .screen.minimized { display: none; }
    .result-chip {
      display: block;
      position: fixed;
      right: 10px;
      bottom: max(10px, env(safe-area-inset-bottom));
      z-index: 42;
      max-width: calc(100vw - 20px);
      min-height: 42px;
      padding: 7px 13px;
      border: 1px solid rgba(216,161,42,.65);
      border-radius: 999px;
      background: rgba(8,42,28,.94);
      box-shadow: 0 8px 25px rgba(0,0,0,.35);
      font-size: .78rem;
    }
    .result-header { position: relative; }
    .hero-score { flex-basis: 100%; justify-items: start; padding: 9px 0 3px; border-top: 1px solid rgba(255,255,255,.09); }
    .hero-score b { font-size: 2rem; }
    .buttons {
      position: sticky;
      bottom: calc(-1 * max(14px, env(safe-area-inset-bottom)));
      z-index: 2;
      margin: 2px -18px calc(-1 * max(14px, env(safe-area-inset-bottom)));
      padding: 10px 18px max(14px, env(safe-area-inset-bottom));
      background: linear-gradient(180deg, transparent, color-mix(in srgb,var(--felt-deep) 97%,black 3%) 20%);
    }
    .buttons .primary { flex: 1 1 145px; }
    @keyframes result-up { from { opacity: 0; transform: translateY(28px); } to { opacity: 1; transform: none; } }
  }

  @media (prefers-reduced-motion: reduce) { .screen { animation: none; } }
</style>
''')

# Slightly richer felt/card separation globally.
swap('web/src/app.css',
     "  background: radial-gradient(circle at 50% 30%, var(--felt) 0%, var(--felt-deep) 78%);",
     "  background: radial-gradient(circle at 50% 18%, var(--felt) 0%, var(--felt-dark) 56%, var(--felt-deep) 100%);\n  background-attachment: fixed;")

# Unit regression for public/unseen tile counting, including a claimed discard.
swap('web/tests/session.test.js',
     "import { heldSafeCount, callTiles, callLabel } from '../src/lib/ui.js';",
     "import { heldSafeCount, callTiles, callLabel, unseenTileCounts } from '../src/lib/ui.js';")
with Path('web/tests/session.test.js').open('a') as file:
    file.write(r'''

test('remaining-copy hints count public information once, including claimed tiles via melds', () => {
  const left = unseenTileCounts({
    dora_indicators: ['1m'],
    seats: [
      { hand:['1m','2p'], drawn:'3s', discards:[], melds:[] },
      { hand:[], drawn:null, discards:[{tile:'4m',claimed:true},{tile:'5p',claimed:false}], melds:[{tiles:['4m','4m','4m']}] },
      { hand:[], drawn:null, discards:[{tile:'6z',claimed:false}], melds:[] },
      { hand:[], drawn:null, discards:[], melds:[] },
    ],
  });
  assert.equal(left.get('1m'), 2); // one held, one indicator
  assert.equal(left.get('2p'), 3);
  assert.equal(left.get('3s'), 3);
  assert.equal(left.get('4m'), 1); // claimed pond tile is not counted twice
  assert.equal(left.get('5p'), 3);
  assert.equal(left.get('6z'), 3);
  assert.equal(left.get('9s'), 4);
});
''')

# Production-browser evidence for the new visual hierarchy and non-overlay counts.
Path('web/scripts/ui-polish-check.mjs').write_text(r'''import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createFixtureHandler } from './static-fixture-server.mjs';
import init, { Game } from '../src/wasm/riichi.js';
import { MatchSession, SAVE_KEY, SETTINGS_KEY } from '../src/lib/session.js';
import { unseenTileCounts } from '../src/lib/ui.js';

await init({module_or_path:readFileSync(new URL('../src/wasm/riichi_bg.wasm',import.meta.url))});
const web=fileURLToPath(new URL('../',import.meta.url)),output=resolve(web,'test-results');
const handler=createFixtureHandler({root:resolve(web,'dist'),publicRoot:resolve(web,'dist')});
const server=createServer(handler);let browser;const contexts=[],results=[];
function opening(){const m=new MatchSession(Game,81,'club');try{m.advance(false);return {snapshot:m.snapshot(),view:m.view};}finally{m.dispose();}}
function calling(){const m=new MatchSession(Game,1,'club');try{m.advance(false);for(const [kind,tile] of [['discard','9s'],['discard','3m'],['discard','8s'],['discard','7p']]){m.apply({type:'choose',kind,tile});m.advance(false);}return m.snapshot();}finally{m.dispose();}}
function won(){const m=new MatchSession(Game,16,'club');try{m.advance(false);for(let n=0;n<240&&m.view.phase!=='over';n++){const c=m.choices;const q=c.find(x=>['ron','tsumo'].includes(x.kind))??c.find(x=>x.kind==='riichi')??c.find(x=>x.kind==='pass')??c.find(x=>x.kind==='discard'&&x.tile===m.view.seats[0].drawn)??c.find(x=>x.kind==='discard')??c[0];assert.ok(q);m.apply({type:'choose',kind:q.kind,tile:q.tile??null});m.advance(false);}assert.equal(m.view.phase,'over');return m.snapshot();}finally{m.dispose();}}
const start=opening(),call=calling(),result=won();
async function open(snapshot=start.snapshot,width=390,height=844){const context=await browser.createBrowserContext();contexts.push(context);const p=await context.newPage();p.errors=[];p.on('pageerror',e=>p.errors.push(e.message));await p.setViewport({width,height,isMobile:width<600,hasTouch:width<600});await p.emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}]);await p.evaluateOnNewDocument((key,settings,s)=>{Object.defineProperty(navigator,'serviceWorker',{value:undefined});localStorage.setItem(key,JSON.stringify(s));localStorage.setItem(settings,JSON.stringify({version:1,difficulty:'club',opponents:['club','club','club'],hints:true,confirmDiscards:true,shortcuts:true,tileFace:'classic'}));},SAVE_KEY,SETTINGS_KEY,snapshot);await p.goto(`http://127.0.0.1:${server.address().port}/mahjong/`,{waitUntil:'networkidle0'});await p.waitForSelector('.hand');return p;}
async function check(name,fn){try{await fn();results.push({name,passed:true});console.log('PASS '+name);}catch(e){results.push({name,passed:false,error:e.stack});console.error('FAIL '+name+'\n'+e.stack);}finally{while(contexts.length)await contexts.pop().close();}}
const shot=(p,name)=>p.screenshot({path:resolve(output,name+'.png'),fullPage:true});
try{await mkdir(output,{recursive:true});await new Promise(done=>server.listen(0,'127.0.0.1',done));const chrome=process.env.CHROME_BIN||['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(existsSync);assert.ok(chrome);browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 await check('phone chrome collapses settings while keeping round/opponent/status visible',async()=>{const p=await open();assert.match(await p.$eval('.bar h1',el=>el.textContent),/Riichi.*East 1/i);assert.equal(await p.$eval('.preferences',el=>getComputedStyle(el).display),'none');assert.notEqual(await p.$eval('.settings-trigger',el=>getComputedStyle(el).display),'none');assert.equal(await p.$eval('.restart',el=>getComputedStyle(el).display),'none');await p.click('.settings-trigger');assert.notEqual(await p.$eval('.preferences',el=>getComputedStyle(el).display),'none');assert.ok(await p.$('.mobile-new-game'));await shot(p,'polish-phone-settings');assert.deepEqual(p.errors,[]);});
 await check('hint counts sit below every hand tile and exactly match public unseen copies',async()=>{const p=await open(),expected=unseenTileCounts(start.view);const rows=await p.$$eval('.hand .hand-tile',els=>els.map(el=>{const tile=el.dataset.tile,button=el.querySelector('button.tile'),count=el.querySelector('.copy-count'),a=button.getBoundingClientRect(),b=count.getBoundingClientRect();return{tile,count:Number(count.textContent.trim()),below:b.top>=a.bottom-0.5};}));assert.equal(rows.length,14);for(const row of rows){assert.equal(row.count,expected.get(row.tile));assert.equal(row.below,true,row.tile);}await p.click('.settings-trigger');await p.click('.option-fields input[type=checkbox]');assert.equal(await p.$('.copy-count'),null);await shot(p,'polish-phone-game');assert.deepEqual(p.errors,[]);});
 await check('desktop table uses the expanded table surface and larger opponent zones',async()=>{const p=await open(start.snapshot,1440,1000);const box=await p.$eval('.board',el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return{w:r.width,h:r.height,bg:s.backgroundImage,radius:s.borderRadius};});assert.ok(box.w>1000);assert.ok(box.h>=380);assert.notEqual(box.bg,'none');assert.ok(parseFloat(box.radius)>=20);await shot(p,'polish-desktop-table');assert.deepEqual(p.errors,[]);});
 await check('call window presents the discard as a staged event before choices',async()=>{const p=await open(call);await p.waitForSelector('.call-stage');assert.match(await p.$eval('.call-stage',el=>el.textContent),/discarded/i);const tile=await p.$eval('.call-stage .tile',el=>el.getBoundingClientRect().width);assert.ok(tile>=40);assert.ok(await p.$('.call-options'));await shot(p,'polish-phone-call');assert.deepEqual(p.errors,[]);});
 await check('phone result is a fixed bottom sheet with hero value and can reveal the table',async()=>{const p=await open(result);await p.waitForSelector('.screen');const state=await p.$eval('.screen',el=>({position:getComputedStyle(el).position,maxHeight:getComputedStyle(el).maxHeight,bottom:el.getBoundingClientRect().bottom,hero:el.querySelector('.hero-score')?.textContent,buttons:getComputedStyle(el.querySelector('.buttons')).position}));assert.equal(state.position,'fixed');assert.ok(state.bottom<=845&&state.bottom>=842);assert.match(state.hero,/Ron|Tsumo/);assert.equal(state.buttons,'sticky');const review=await p.$('.screen .quiet');if(review){await review.click();await p.waitForSelector('.result-chip');assert.equal(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');await p.click('.result-chip');assert.notEqual(await p.$eval('.screen',el=>getComputedStyle(el).display),'none');}await shot(p,'polish-phone-result');assert.deepEqual(p.errors,[]);});
}finally{await writeFile(resolve(output,'ui-polish-report.json'),JSON.stringify(results,null,2));for(const c of contexts)await c.close();await browser?.close();if(server.listening)await new Promise(done=>server.close(done));}
console.log(`${results.filter(r=>r.passed).length}/${results.length} UI-polish browser checks passed`);if(results.some(r=>!r.passed))process.exitCode=1;
''')

# Run the new production check in normal CI forever.
swap('web/package.json',
     '"test:browser": "node scripts/ui-regression.mjs && node scripts/tile-effects-check.mjs && node scripts/full-review-check.mjs && node scripts/discard-readiness-check.mjs && node scripts/mixed-opponents-check.mjs && npm run test:toolchain && npm run test:offline"',
     '"test:browser": "node scripts/ui-regression.mjs && node scripts/tile-effects-check.mjs && node scripts/full-review-check.mjs && node scripts/discard-readiness-check.mjs && node scripts/mixed-opponents-check.mjs && node scripts/ui-polish-check.mjs && npm run test:toolchain && npm run test:offline"')

# Keep the learning-aids documentation current without changing rules semantics.
readme = Path('README.md')
text = readme.read_text()
old = '- **Learning aids**: how far the hand is from a wait, what it is waiting on\n  and how many of each are still unseen, the dora in hand, which tiles'
new = '- **Learning aids**: how far the hand is from a wait, what it is waiting on,\n  the unseen-copy count directly below every hand tile, the dora in hand, which tiles'
if old in text:
    readme.write_text(text.replace(old,new,1))
elif 'the unseen-copy count directly below every hand tile' not in text:
    raise SystemExit('README learning-aids paragraph changed concurrently')
