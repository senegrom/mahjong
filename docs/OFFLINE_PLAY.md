# Offline play and first-download preparation

## Before travelling

Open the updated game while connected. On iPhone, open the actual Home Screen
app you will use on the flight. The complete game and all tile graphics download
and save automatically at startup. **No click is required for the game, graphics,
Beginner or Club opponents.** Play becomes available once the engine and selected
face set are ready; this does not mean the full offline package has finished.

Only the trained network and its runtime are optional. Select a Trained opponent,
or choose **Download trained AI for offline play** in the status panel without
changing your current match. Wait for **Offline: game + AI ready** when you need
Trained opponents; **Offline: game ready** is enough for Beginner and Club.
Closing and reopening the app, restoring a match and starting another game use
saved files. There is no need to clear website data or reinstall; either can
remove downloads.

## Artwork loading and readiness

The production build generates a scoped service worker from an inventory of the
actual output, including hashed JavaScript chunks, the rules engine, all runtime
tile graphics, the white-dragon artwork, icons and installation manifest.

Interactive startup loads and decodes only the selected face set and its back,
fallback and foil images. Other sets continue saving through the service worker.
An unavailable unused set therefore does not prevent play, but the offline panel
correctly remains incomplete until every required core file has been verified.
A failed selected-set load stops startup rather than showing missing artwork.

Changing tile faces decodes the new set before displaying or saving that choice.
Failure keeps the previous tiles and match; select the desired face again to
retry. Loading has bounded concurrency and a timeout covering both transport
and decode. Replacement attempts cancel and drain old work, ignore stale progress,
reuse successful decodes and release images no longer needed by the displayed set.

The status panel separates **Game and all tile graphics — automatic** from
**Trained AI — optional**. Missing core files are repaired automatically at
startup, on reconnect and when returning to the app. This recovery never opts
into AI. The optional download/retry button requests only the trained AI package,
not an already complete game or its graphics.

## Trained opponents and storage

Selecting any Trained opponent, including in a custom table, downloads and saves
the complete AI package: model weights, worker code, runtime module and WASM.
This starts even when the human has the first turn. The inference timer starts
after the download; abandoning a match detaches its request without cancelling
shared asset preparation or applying the old response to a replacement match.

Cached availability does not need a successful online HEAD request. Reopening,
reloading, retrying an AI worker or starting another match reads the downloaded
bytes, not the server.

Every file is checked against its build-time byte length and SHA-256 before it
is marked as saved. Captive portals, partial responses and failed writes cannot
become successful downloads. Retrying requests only missing files. Storage is
scoped to this app; other projects on the same GitHub Pages origin are untouched.
The service worker responds cache-first and caches neither unknown URLs nor
error responses. Equal content is stored once, including duplicate runtime
outputs and unchanged files across application updates.

Updates wait until existing Mahjong windows close. When AI was previously
requested, an update must also finish its matching AI package before installing;
an interrupted upgrade leaves the working version and its downloads intact.
No downloaded model is replaced merely because a newer UI was published.

The app requests persistent storage where supported, and reports whether the
browser granted it. Clearing website data removes downloads. Browsers may also
evict non-persistent storage under storage pressure; no web app can guarantee
retention after the operating system deletes its data. Readiness is checked
against CacheStorage at startup and when the app returns to the foreground,
not inferred from a flag in the saved match. Unsupported storage is explicitly
labelled online-only. Saved matches remain a separate, unchanged mechanism.

## Verification

Run `npm run verify` in `web` for the same checks as CI. The component tests mount
real components under a reactive parent and exercise bindings, context, download
callbacks, modal focus, breakpoint changes and unmount. Production checks cover
saved preferences across reload, unchanged matches, unavailable unused artwork
and failed face switches. Unit tests cover preload cancellation and retry.

`npm run test:offline` exercises the real production service worker and shipped
network, including a browser-process restart with HTTP cache cleared, disabled
network access, all graphics reloaded offline, continued mixed-opponent play,
interrupted downloads and version updates. Unit tests independently cover
hash/length validation, quotas, eviction, cache isolation and failed upgrades.
The cold-restart checks also refuse game assets at the HTTP server, so ordinary
HTTP caching cannot conceal a missing offline file. Existing small-phone layout
assertions remain part of verification.

Automated browser checks use Chromium. They do not replace physical iPhone,
Safari installation or VoiceOver testing. Inspect the PR's CI results for the
exact tested revision rather than treating an old test count as current evidence.
