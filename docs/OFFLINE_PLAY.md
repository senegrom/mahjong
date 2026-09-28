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
This starts even when the human has the first turn. Abandoning a match detaches
its request without cancelling shared asset preparation or applying the old
response to a replacement match.

Loading the network into the opponent's worker is its own phase. Where the page
could not save the network (no service worker, an installation still running, a
hard reload, a storage failure), the worker downloads it itself, showing its
progress. That phase has no fixed deadline: it ends when the worker has reported
nothing for two minutes. The 20-second deadline for a move starts only once the
network is loaded. A Retry after a failed start tries offline saving again.

The service worker saves the runtime (a few megabytes); the page saves the
network itself (over 100 MB). A browser abandons a service-worker event that is
still running after a few minutes (five in Chrome), and a half-received body is
not kept, so on a slow link a worker could never finish the network. A page has
no such deadline: a transfer that keeps receiving bytes may take up to three
hours, and one that stalls for a minute fails and can be retried.

Cached availability does not need a successful online HEAD request. Reopening,
reloading, retrying an AI worker or starting another match reads the downloaded
bytes, not the server.

Every file is checked against its build-time byte length and SHA-256 before it
is marked as saved. Captive portals, partial responses and failed writes cannot
become successful downloads. Retrying requests only missing files. Storage is
scoped to this app; other projects on the same GitHub Pages origin are untouched.
The service worker responds cache-first and caches neither unknown URLs nor
error responses. Equal content is stored once, including duplicate runtime
outputs and unchanged files across application updates. The runtime is served
from a folder named by a hash of its files (`ort/<hash>/`), so no address ever
holds two versions' bytes in a browser, CDN or service-worker cache. A version
whose runtime was never saved cannot fetch it once a later deploy has removed
its folder; Trained opponents then wait for that version's windows to close so
the update takes over.

Updates wait until existing Mahjong windows close. An update installs with the
game and graphics alone, so a slow connection still completes it. When Trained
AI was ever requested, the running version's page then saves the waiting
update's runtime and network while you play; if that does not finish, the
update's own page saves them after it starts, once per visit while connected,
with the status panel showing progress. A new version never plays an older
network: until its own is saved, **Offline: game ready** is shown and Trained
opponents need a connection. Check for **Offline: game + AI ready** after an
update before flying. The previous network stays stored until the new one is
saved. No downloaded model is replaced merely because a newer UI was published.

The app requests persistent storage where supported, and reports whether the
browser granted it. Clearing website data removes downloads. Browsers may also
evict non-persistent storage under storage pressure; no web app can guarantee
retention after the operating system deletes its data. Readiness is checked
against CacheStorage at startup and when the app returns to the foreground,
not inferred from a flag in the saved match. Unsupported storage is explicitly
labelled online-only. Saved matches remain a separate, unchanged mechanism.
When the browser reports too little free storage for the network, the page does
not download it only to have it refused; the opponent's worker fetches it once
for the session. When storage refuses a network the page has already verified,
those bytes go to the worker instead of being downloaded again.
These checks read the size and digest recorded when the network was verified
and stored; they never read or hash the 116 MB again. The bytes handed to the
runtime are hashed in full each time it loads them.

## Verification

Run `npm run verify` in `web` for the same checks as CI. The component tests mount
real components under a reactive parent and exercise bindings, context, download
callbacks, modal focus, breakpoint changes and unmount. Production checks cover
saved preferences across reload, unchanged matches, unavailable unused artwork
and failed face switches. Unit tests cover preload cancellation and retry.

`npm run test:offline` exercises the real production service worker and shipped
network, including a browser-process restart with HTTP cache cleared, disabled
network access, all graphics reloaded offline, continued mixed-opponent play,
interrupted downloads and version updates. It also checks that the optional
button downloads only trained-AI bytes without changing the match, and that
missing tile and icon entries are repaired on reconnect without downloading
AI. Unit tests independently cover hash/length validation, quotas, eviction,
cache isolation and failed upgrades.
The cold-restart checks also refuse game assets at the HTTP server, so ordinary
HTTP caching cannot conceal a missing offline file. Existing small-phone layout
assertions remain part of verification.

Automated browser checks use Chromium. They do not replace physical iPhone,
Safari installation or VoiceOver testing. Inspect the PR's CI results for the
exact tested revision rather than treating an old test count as current evidence.
