# Regular-play UI boundaries

`../../App.svelte` owns startup, mode selection, preferences, the active
`MatchSession`, and the `MatchStore` writer transactions. It also owns selection,
keyboard handling, AI recovery, and match review. Switching between modes does
not recreate that owner or replay a regular match.

| Component | Responsibility |
| --- | --- |
| `AppSettings.svelte` | Header, mode navigation, preferences, rules guide, and offline status. |
| `OpponentDialog.svelte` | Editable draft opponent assignments; starting a match is a callback to App. |
| `MatchTable.svelte` | Opponent seats, round and wall information, indicators, and the inspection dialog. |
| `PlayerHand.svelte` | Hand, waits, discard previews, melds, and personal discards. |
| `TurnChoices.svelte` | Turn prompts, offered calls, and discard confirmation or cancellation. |

The components render the existing HTML sections directly, without extra layout
wrappers. Their styles remain scoped and live alongside the markup they style.
Shared wind names live in `../tiles.js` with the other display names.

## State and actions

Preferences and the opponent draft use explicit bindable props. A change returns
to App, where the existing persistence and new-game confirmation run. Opening or
closing a panel must not create a match, write a save, or change active opponents.

The hand element is bound back to App so the existing keyboard and focus guards
still check the actual DOM element. Selection is not duplicated in a child.
Hypothetical discard analysis is a read-only derived value in `PlayerHand`; moving
the marker must not rerun the analysis for an unchanged set of discard choices.

Calls, discards, and recovery go through App's guarded session callbacks. Children
must not construct sessions, write shared saves, or start their own AI workers.

## Checks

From `web`, run `npm run verify`. `tests/app-components.test.js` compiles the actual
components for client and server targets with no warnings. The production browser
checks in `scripts/cleanup-check.mjs` exercise settings propagation, selection
cancellation, mode round trips, both dialogs, and responsive layout. The remaining
browser and session suites cover keyboard navigation, scoring, review, save
conflicts, recovery, and cancellation of stale AI work.
