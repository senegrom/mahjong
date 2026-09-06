# Automatic game downloads, optional trained AI

Opening the game automatically saves the complete core package: the application,
rules engine, all 36 tile faces/backs, white-dragon artwork, icons and manifest.
Every tile graphic is loaded and decoded before the first hand. No offline
button, opponent selection or visit to the status panel is needed for this.
Beginner and Club therefore work offline once startup preparation finishes.

The optional **Download trained AI for offline play** button adds only the model
weights and inference runtime. Selecting a Trained opponent starts the same AI
preparation automatically. The button does not enable basic offline play, and
AI download/retry does not download an already complete core package again.

The status panel labels the game and graphics as **automatic** and trained AI
as **optional**, with separate status messages. Missing core cache entries are
repaired automatically at startup, on reconnect and on returning to the app.
Core recovery never opts into the trained model.

Offline storage availability and retention remain browser-dependent; the panel
does not claim that failed or evicted downloads are ready. The first download
still requires a connection. Saved-match storage and opponent assignments are
unchanged.

Regression tests cover both built-in opponent types after a full browser restart
with the ordinary HTTP cache cleared and all game URLs refused by the server,
automatic tile/icon repair, and a manual AI-only download without changing the
match or refetching any core resource.
