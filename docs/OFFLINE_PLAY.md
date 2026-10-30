# Offline play and first-download preparation

## Before travelling

Open the updated game while connected. On iPhone, open the actual Home Screen
app you will use on the flight. The complete game and all tile graphics download
and save automatically at startup. **No click is required for the game, graphics,
Beginner or Club opponents.** Play becomes available once the engine and selected
face set are ready; this does not mean the full offline package has finished.

Only the trained network and its runtime are optional. The network is the one
picked under **Game settings → Trained network**, 0.1.2610 unless another was
picked; the offline panel's Trained AI line names it, and after picking
another, wait for **Offline: game + AI ready** again before flying. Trained
opponents are the default for a new player, so a first match starts their
download by itself, and
the table shows its progress while a Trained opponent waits for it; the hand
review, Agent watch and a guided game show it too when their first Trained
advice is what starts the download. On a first visit the network's download
begins only once the game and every tile graphic are saved, which reports no
percentage, so until then the table says that the game is being saved first.
A player whose saved table is Club or
Beginner can select a Trained opponent, or choose **Download trained AI for
offline play** in the status panel without changing the current match. Wait
for **Offline: game + AI ready** when you need Trained opponents; **Offline:
game ready** is enough for Beginner and Club. When the network cannot be
downloaded, the table offers **Continue with Club opponents**.
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
into AI. A first install that outlasts the page's two-minute wait, one that
failed, and a page that a hard reload loaded past its worker are each named in
the game's line, never reported as storage the browser lacks; the panel checks
again by itself when the finished worker takes control of the page. The
optional download/retry button requests only the trained AI package, not an
already complete game or its graphics.

## Trained opponents and storage

Selecting any Trained opponent, including in a custom table, downloads and saves
the complete AI package: model weights, worker code, runtime module and WASM.
This starts even when the human has the first turn. Abandoning a match detaches
its request without cancelling shared asset preparation or applying the old
response to a replacement match.

A build lists every network a player may pick, and its service worker knows
them all. A page plays one, the one picked (or the default), for as long as
it is open: picking another reloads the page. That network alone is what
the page asks for, checks against its own size and SHA-256, saves and runs,
and every message to the worker names the pick, so the worker reports
whether that network is saved, never another. The worker deletes no network
its build lists: a player who goes back to a network saved earlier finds it
still saved, at the cost of keeping each one picked (about 117 MB apiece).
A network the site stops listing goes the way an older version's does: it
is kept only while it is the one saved before the current one.

Loading the network into the opponent's worker is its own phase. Where the page
could not save the network (no service worker, an installation still running, a
hard reload, a storage failure), the worker downloads it itself, showing its
progress. That phase has no fixed deadline: it ends when the worker has reported
nothing for two minutes. The 20-second deadline for a move starts only once the
network is loaded and the worker begins that move. A Retry after a failed start
tries offline saving again.

The service worker saves the runtime (a few megabytes); the page saves the
network itself (over 100 MB), because a browser cuts a service worker's work
short after a few minutes, too soon for the network on a slow link (see
[cache lifetime and upgrades](ONNX_RUNTIME.md#cache-lifetime-and-upgrades)).
A page has no such deadline: a transfer that keeps receiving bytes may take up
to three hours, and one that stalls for a minute fails and can be retried.

Cached availability does not need a successful online HEAD request. Reopening,
reloading, retrying an AI worker or starting another match reads the downloaded
bytes, not the server. Offline, Trained is offered only when the service worker
reports the runtime saved as well as the network: the opponent's worker can
store the network by itself, and a network without its runtime cannot play.

Every file is checked against its build-time byte length and SHA-256 before it
is marked as saved. Captive portals, partial responses and failed writes cannot
become successful downloads. Retrying requests only missing files. No file has
a fixed time limit of its own: one that keeps arriving goes on, and one that
stops for a minute is tried again later. A file that fails does not stop the
others, and every file an attempt finishes stays saved, so an install the
browser abandons after its few minutes (five in Chrome) resumes on the next
visit or reconnect with only what is still missing. Only a link too slow to
bring the largest tile graphics within those minutes, below about 300 kbit/s,
never completes it; play still works online. Storage is scoped to this app;
other projects on the same GitHub Pages origin are untouched.
The service worker responds cache-first and caches neither unknown URLs nor
error responses. Equal content is stored once, including unchanged files
across application updates. The runtime is served from a folder named by a
hash of its files (`ort/<hash>/`), so no address ever holds two versions'
bytes in a browser, CDN or service-worker cache. A version
whose runtime was never saved cannot fetch it once a later deploy has removed
its folder; Trained opponents then wait for that version's windows to close so
the update takes over.

Updates wait until existing Mahjong windows close. An update installs with the
game and graphics alone, never the network, and on a slow link it resumes over
several attempts like a first install. When Trained AI was ever requested, the
update's runtime and network are saved while you play; if that does not
finish, the update saves them itself once it starts, with the status panel
showing progress (see [cache lifetime and
upgrades](ONNX_RUNTIME.md#cache-lifetime-and-upgrades)). The update plays the
network you picked while it still lists it, and its own default otherwise;
until that network is saved, **Offline: game ready** is shown and Trained
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
These checks never read the 116 MB again: they read the size and digest
recorded when the network was verified and stored. The network is checked in
full again each time it is loaded to play (see [verification and
memory](ONNX_RUNTIME.md#verification-and-memory)).

## Verification

Run `npm run verify` in `web` for the checks of CI's npm audit and web app jobs.
The component tests mount real components under a reactive parent and exercise
bindings, context, download callbacks, modal focus, breakpoint changes and
unmount. Production checks cover saved preferences across reload, unchanged
matches, unavailable unused artwork and failed face switches. Unit tests cover
preload cancellation and retry.

`scripts/offline-check.mjs`, part of `npm run test:browser`, exercises the real
production service worker and the default network, including a browser-process restart with HTTP cache cleared, disabled
network access, all graphics reloaded offline, continued mixed-opponent play,
interrupted downloads and version updates. It also checks that the optional
button downloads only trained-AI bytes without changing the match, and that
missing tile and icon entries are repaired on reconnect without downloading
AI. Unit tests independently cover hash/length validation, quotas, eviction,
cache isolation and failed upgrades.
The cold-restart checks also refuse game assets at the HTTP server, so ordinary
HTTP caching cannot conceal a missing offline file. Existing small-phone layout
assertions remain part of verification.
`scripts/network-picker-check.mjs` picks each listed network but the default
in the production build and then the default again, and checks that each page
asks for its own network's address alone, names it, and that the real service
worker reports on it, and that settings saved before 2610 (record version 4,
3 from before 1410, or 2 from before 1110) play the default whatever they
picked, keep every other choice and resume their match;
unit tests run the worker with several listed networks, picked, saved,
updated and pruned.

Automated browser checks use Chromium. They do not replace physical iPhone,
Safari installation or VoiceOver testing. Inspect the PR's CI results for the
exact tested revision rather than treating an old test count as current evidence.
