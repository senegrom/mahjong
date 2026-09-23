# App presentation boundaries

`App.svelte` owns the live match, storage transactions, keyboard guards, AI
cancellation/recovery, mode lifetime, selected tiles, scoring and review.
Presentation components cannot write a match or choose a fallback opponent.

| Component | Boundary |
| --- | --- |
| `AppSettings` | Bind preference scalars to App; own the responsive settings dialog and focus. |
| `OfflineStatus` | Render actual core/AI persistence status and invoke an explicit download callback. |
| `OpponentDialog` | Edit a separate draft; ask App to confirm replacement of the match. |
| `MatchTable` | Render opponents and table inspection; own all regular-table breakpoints. |
| `PlayerHand` | Render the hand and read-only hypothetical discard analysis; bind its DOM element to App. |
| `TurnChoices` | Render choices and forward actions/cancellation to App's guards. |

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

Run `npm run verify` from `web`, the same command as CI. It includes actual component
compilation, TypeScript contract checks, the real offline-status component and
bounded-preloader unit tests. `component-check.mjs` mounts the real components under
a reactive parent and checks bindings, context, callbacks, modal focus, resizing,
unmount and effective styles. `cleanup-check.mjs` exercises the production app,
including genuine preference reloads, saved-match preservation and unavailable
unused artwork. Test fixtures are not imported by the production entry point.
