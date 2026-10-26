# App presentation boundaries

`App.svelte` owns the live match, storage transactions, keyboard guards, AI
cancellation/recovery, mode lifetime, selected tiles, scoring and review. The
review's reasoning about its decisions (`MatchInsight`) is App's too, kept for
the match beside the review: App gives it, the review and the hand's notes
their spells of asking at the one worker, so that none queues behind another,
and stops it at the next deal and on leaving Play.
Presentation components cannot write a match or choose a fallback opponent.

| Component | Boundary |
| --- | --- |
| `AppSettings` | Bind preference scalars to App, the review's two hiding settings among them, which the game's review binds in its header too; own the responsive settings dialog and focus; ask App to load another trained network, which App does by reloading the page. |
| `OfflineStatus` | Render actual core/AI persistence status and invoke an explicit download callback. |
| `OpponentDialog` | Edit a separate draft; ask App to confirm replacement of the match. |
| `MatchTable` | Render opponents and table inspection; own all regular-table breakpoints, and those of a review's table, which shows every hand face up as it stood at a past decision, or the opponents' turned face down in their racks (`concealed`), on a dense board from 761 pixels wide (the centre a bar across the top, the three opponents in one row), and, where the player answers a discard, outlines the tile on offer in the row of the seat that let it go. |
| `PlayerHand` | Render the hand and read-only hypothetical discard analysis; bind its DOM element to App. In play on a phone held sideways, draw App's controls (`controls`) right under the hand, over the called sets and discards. In a review, name the tiles for the seat, as one of the table's four hands (`listed`), and put words over the tiles the review's moves concern (`tags`). |
| `TurnChoices` | Render choices and forward actions/cancellation to App's guards. On a phone held sideways, stage a call's discard in a small card in the rows of its buttons, each the call's word beside its tiles. |
| `ReviewInsight` (`src/lib`, in the review's `DecisionReplay`) | Render what App's `MatchInsight` has for the decision on show, the Why line and "What would change its mind?" with its progress, Cancel and what it found, only once Trained AI's answer is shown; ask it to explain, search, cancel or leave a decision, never the worker itself. |

`types.ts` declares the view, choice, offline, preference and callback contracts.
The components use type-checked Svelte scripts; this is not a second state store.
`controls.css` is imported once by App and opts in via `.app-control`. Tile buttons
never receive it. Component-specific rules remain scoped.

## Startup and artwork

Interactive startup waits for the engine and the selected face set (including its
back, fallback and foil images). Full artwork/core caching continues independently
through the service worker. A playable page does **not** imply the complete offline
package is ready. `OfflineStatus` remains the authority for that separate claim.

Face changes decode the replacement before updating or persisting `tileFace`.
Failure keeps the current face and match; selecting it again retries. A newer
selection or unmount aborts stale work. `image-preloader.js` limits concurrency,
drains aborted workers, reuses verified decode successes and bounds retained Image
objects to the displayed set plus the current partial attempt. Its timeout covers
both load and decode. `tile-preload.js` owns the face-specific URL inventory.

## Settings focus

Compact layouts use native `showModal()`: underlying content is inert, focus stays
inside, Escape closes, and the opener regains focus. Desktop preferences remain
inline. Changing breakpoints closes the modal, preserves open sections/preferences
and moves focus to a visible corresponding control. Opening opponent configuration
closes settings synchronously before opening the next modal.

## Verification

Run `npm run verify` from `web`, the checks of CI's npm audit and web app jobs.
It includes svelte-check, which compiles every component and fails on any
compiler warning, TypeScript contract checks, the real offline-status component
and bounded-preloader unit tests. `component-check.mjs` mounts the real
components under a reactive parent and checks bindings, context, callbacks,
modal focus, resizing, unmount and effective styles. `cleanup-check.mjs`
exercises the production app, including genuine preference reloads,
saved-match preservation and unavailable unused artwork. Test fixtures are not
imported by the production entry point.
