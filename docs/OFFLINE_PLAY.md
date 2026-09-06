# Offline play and first-download preparation

## Before travelling

Open the updated game while connected. On iPhone, open the actual Home Screen
app you will use on the flight. The complete game and all tile graphics download
and save automatically at startup. **No click is required for the game, graphics,
Beginner or Club opponents.** Every tile graphic is decoded before the first hand.

Only the trained network and its runtime are optional. Select a Trained opponent,
or choose **Download trained AI for offline play** in the status panel without
changing your current match. Wait for **Offline: game + AI ready** when you need
Trained opponents; **Offline: game ready** is enough for Beginner and Club.
Closing and reopening the app, restoring a match and starting another game use
saved files. There is no need to clear website data or reinstall; either can
remove downloads.

## What is saved

The production build generates a scoped service worker from an inventory of the
actual output, including hashed JavaScript chunks, the rules engine, all tile
SVGs, the white-dragon artwork, icons and installation manifest.

Before the first hand is shown, all 36 tile SVGs and the dragon image are loaded
and decoded. The previous delayed, one-image-at-a-time preloader is removed.
A failed tile download stops startup with a retry message, not missing artwork
that only becomes apparent when that tile is drawn.

The status panel separates **Game and all tile graphics — automatic** from
**Trained AI — optional**. Missing core files are repaired automatically at
startup, on reconnect and when returning to the app. This recovery never opts
into AI. The optional download/retry button requests only the trained AI package,
not an already complete game or its graphics.

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

`npm run test:offline` exercises the real production service worker and shipped
network, including a browser-process restart with HTTP cache cleared, disabled
network access, all graphics reloaded offline, continued mixed-opponent play,
interrupted downloads and version updates. Unit tests independently cover
hash/length validation, quotas, eviction, cache isolation and failed upgrades.

The automatic-download release candidate passed 89 unit/session/cache tests and
102 browser checks, including all nine offline checks. Separate cold-restart
checks confirm both Beginner and Club play offline without pressing a download
button or fetching any AI files. The manual AI-only test verifies that the button
neither refetches a core asset nor changes the saved match. Another check deletes
a cached tile and icon, then confirms automatic repair on reconnect with no AI
download. Coordinator tests exercise startup and recovery separately from AI.

The cold-restart checks refuse every game asset at the HTTP server as well as
disabling browser networking, so ordinary HTTP caching cannot conceal a missing
offline file. The Trained test plays the actual shipped network after restart,
checks every tile graphic and starts another match. Existing small-phone layout
assertions are retained unchanged.

These automated browser results are from Chromium. Physical iPhone and Safari
installation testing is not claimed.
