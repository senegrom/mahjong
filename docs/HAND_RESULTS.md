# Hand results, waits and exported balances

## Ura-dora

After a win by a player who declared riichi, the table shows a labelled
**Ura-dora** indicator row alongside the ordinary dora indicators. Each winning
hand also shows its own ordinary and ura-dora indicators and separate bonus
counts. These are result information, so the indicators remain visible when
Hints and markings is switched off. Repeated indicators retain separate slots.

Ura-dora are never exposed during play, on an exhaustive draw, or for a winner
who did not declare riichi. In a multiple-winner result the bonus belongs only
to eligible winners. A zero ura-dora bonus still shows the revealed indicators;
a yakuman result explains that dora do not add to its value. The existing total
han, payouts and white-dragon foil effect are unchanged.

## Waits and keyboard play

A drawn tile is not included when deriving the standing hand's waits. In riichi,
this keeps the locked wait stable across unrelated draws. Before riichi the
interface distinguishes the wait before the draw from the wait after a selected
discard. Discard previews are read-only queries checked against legal actions.

Arrow keys skip disabled hand tiles, and the marker follows successful focus.
After a keyboard discard, the hand regains focus for the next decision unless
the player has deliberately focused another control. Browser shortcuts and
other focused controls remain unaffected.

## Final-hand review and saves

Opening final standings does not start another hand, so it no longer clears
the final hand's event history or an already open review. Review final hand and
Save final hand remain available below the standings, including after reload.

Save format 3 validates the corrected view. Earlier saves are replayed using
their recorded legal commands and checked against all authoritative state;
only the repaired derived waits and newly added result metadata are migrated.
An incompatible or corrupt save still fails closed rather than being silently
replaced.

## MJAI accounting

Settlement deltas describe only that individual event. A reader first processes
reach_accepted's 1,000-point payment, then each hora or ryukyoku delta once.
Multiple winners have separate incremental settlements and intermediate score
balances; their changes are not duplicated. Hand::deltas() intentionally remains
the cumulative change for the score screen.

The native log replay test reconciles every settlement against the preceding
balance and sums all events, including riichi payments, against the final hand
total. The browser/WASM regressions include seeds 3, 17 and 248 for these cases.

## Verification

Run the existing checks after building the WASM engine:

```sh
cargo test --workspace
cd web
npm ci
npm run wasm
npm run build
npm run test:unit
npm run test:browser
```

The full-review suite adds tests for seed-81 riichi waits, seed-31 disabled-tile
navigation, consecutive keyboard turns, saved-state migration, actual ura-dora
indicators with hints off, and final-hand review/history/export access.
The production browser command also retains the existing UI and dragon-effect
checks. The results and screenshots are written to web/test-results.

Browser checks run in Chromium, including narrow portrait and landscape sizes;
they do not constitute physical-iPhone or Safari testing.
