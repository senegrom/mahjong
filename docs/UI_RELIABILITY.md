# UI reliability and saved matches

The browser owns one `MatchSession` at a time. Disposing it aborts outstanding
inference and frees its WASM game. Every asynchronous continuation checks that
owner before applying an answer, reporting an error or changing the interface.
A failed neural opponent remains explicitly Trained until the player retries,
continues the same position with Club opponents, or starts a new match.

## Saved matches

After each successful player/AI action and each hand transition, the application
stores `riichi.match.v2` in local storage. The versioned record contains the
initial seed, original opponent mode, a validated sequence of legal commands,
and the final visible state. Neural action choices are recorded too. Reloading
replays these exact choices on the rules engine; it does not ask the network to
choose the past moves again. A pending opponent turn resumes after restoration.

Replay is deliberately fail-closed: malformed records, illegal actions and an
engine version producing a different state are rejected without overwriting the
saved record. The user can deliberately choose New game to replace it. Storage
failure leaves play available and displays a warning. This is local restoration,
not cloud sync or an offline-installation feature. Clearing website data removes
the saved match. A future change to engine rules may require a save migration.

Preferences (opponent strength, hints, discard confirmation and shortcuts) are
stored separately. Cancelling an opponent change leaves both the displayed
selection and active game unchanged. New game requests confirmation whenever
an unfinished match has progress, including between hands.

## Input and presentation

- Hand shortcuts run only with focus inside the hand. Browser modifiers, other
  controls and repeated destructive key events are excluded. The marker is
  cleared explicitly after a choice. Shortcuts can be disabled in Options.
- Select-before-discard is optional and defaults on for coarse pointers. A
  second tap or the labelled Discard button confirms the selected tile.
- Mobile play keeps the hand and actions together. All opponents' discard rows
  remain visible in a compact table, with a larger native-dialog inspection view.
- Dora types are global (including repeated indicators); held-dora han is a
  separate value. Results include applicable ura-dora, and review notes use the
  indicators from that historical position. Indicator keys identify slots,
  not tile names.
- The safe count counts held copies and is explicitly qualified as safety
  against declared riichi, not a guarantee against undeclared hands. Tile names
  expose selected, drawn, safe and dora states to assistive technology.

## Verification

From `web/`, after `npm ci`:

```sh
npm run wasm
npm run build
npm run test:unit
npm run test:browser
node scripts/check-icons.mjs
node scripts/check-icons.mjs --built
```

`test:unit` includes pure session tests and tests against the rebuilt WASM,
including a complete-match save/restore and the seed-369 duplicate-indicator
regression. Browser tests use the production build under `/mahjong/`, without
application test hooks; deterministic saved positions and test-only worker
responses exercise recovery, inputs, hints, and small-screen layouts.

Set `CHROME_BIN` when Chrome/Chromium is not in a standard system location.
Screenshots and the structured browser report are saved to `web/test-results/`.
CI requires both the rules-engine job and the web regression job before Pages
publication. Browser viewport emulation is not a physical iPhone/Safari test.

## Second-review corrections

Saved-match commands run inside a same-origin Web Lock. Each write compares the
exact revision originally read, so a stale window cannot take over merely by
acquiring the lock later. Other windows stop accepting moves when storage changes
and offer **Reload latest match**. Nothing is saved by `pagehide`; successful
commands were already saved in their transaction. Returning from the back/forward
cache reloads the current match. Browsers without usable storage/Web Locks may
play without shared persistence and show a warning rather than use a racy fallback.

Compatible old records are read from `riichi.match.v1` and migrated to the new key
without deleting the old copy. Old-version windows can no longer overwrite the
new save. Additive claimed-tile metadata and historical implicit neural passes are migrated; other rules/replay divergence is
still rejected without overwriting the original record.

A submitted human call remains pending until every other eligible claimant has
answered, including competing ron claims. Pending calls replay exactly and can
recover to Club. The retained final hand and exported log preserve their original
player-to-seat mapping. Ties have a joint place and a distinct player identifier.
Meld presentation obtains the actual claimed tile from its authoritative event,
not from its position in the sorted sequence. Native Tab focus and arrow-key
markers stay synchronised, and Enter prioritises the focused tile. Tiles disabled
while deciding a call stay readable.

The regression suite includes real shared browser tabs, simultaneous writers,
shift-Tab/mouse input, a completed match, and late-hand, multiple-meld, long-call
and tied-standings layouts at narrow portrait and landscape viewport sizes.
