# Offline play and first-download preparation

The production build generates a scoped service worker from an inventory of the
actual output, including hashed JavaScript chunks, the rules engine, all tile
SVGs, the white-dragon artwork, icons and installation manifest.

Before the first hand is shown, all 36 tile SVGs and the dragon image are loaded
and decoded. The previous delayed, one-image-at-a-time preloader is removed.
A failed tile download stops startup with a retry message, not missing artwork
that only becomes apparent when that tile is drawn.

Selecting any Trained opponent, including in a custom table, downloads and saves
the complete AI package: model weights, worker code, runtime module and WASM.
This starts even when the human has the first turn. A manual **Download AI for
offline play** action is also available. The inference timer starts after the
download; abandoning a match detaches its request without cancelling shared
asset preparation or applying the old response to a replacement match.

Wait for **Offline: game + AI ready** before disconnecting. On iPhone check this
inside the Home Screen app you intend to use. Cached availability does not need
a successful online HEAD request. Reopening, reloading, retrying an AI worker
or starting another match reads the downloaded bytes, not the server.

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

`npm run test:offline` exercises the real production service worker and shipped
network, including a browser-process restart with HTTP cache cleared, disabled
network access, all graphics reloaded offline, continued mixed-opponent play,
interrupted downloads and version updates. Unit tests independently cover
hash/length validation, quotas, eviction, cache isolation and failed upgrades.
