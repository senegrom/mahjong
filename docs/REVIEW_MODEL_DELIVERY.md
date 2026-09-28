# Model delivery and scoring explanations

Repairs the eight findings in the review of `1cb9d84`. These changes do not
publish new model weights, change the scoring rules, certify a reader, or
claim a playing-strength gain.

## Verified network bytes versus durable offline readiness

`network-transfer.js` is used by ordinary inference and inlined into the
self-contained service worker at build time. Both cache hits and fresh
responses must match the build's exact byte count and SHA-256. Bad/unreadable
cached entries are discarded best-effort; they are never sent to ONNX Runtime.
Online inference may use a verified download even when cache access or a write
fails. Offline preparation explicitly requires durable storage instead, and
its status rechecks the actual cached bytes. The UI keeps offline readiness
false and displays a storage warning while permitting online inference.

Each transfer owns an abort controller, a 60-second idle deadline, a total
deadline (30 minutes then; three hours since the page, not the service worker,
downloads the network), and the manifest's decoded-byte ceiling (up to 512 MiB). These
limits cover headers and streamed bodies even if a transport ignores abort.
The compressed Content-Length is not used to size the decoded payload. There
is no unbounded chunk accumulation. Failure cancels the reader and clears the
shared preparation job; retry creates a new attempt. A departing match releases
its own subscription without cancelling a healthy shared download.

The production offline manifest now includes the exact remote model identity.
A user who requested trained offline play can install a model-changing update
only after its runtime and replacement model are verified and stored. Failed
installation leaves the old shell/runtime/model usable. Activation rechecks
readiness before pruning old local assets. UI-only updates reuse unchanged
weights. The remote cache is not globally cleared, and other applications'
caches are untouched. Storage eviction after a completed update remains a
browser capability constraint, not a persistence guarantee.

Superseded: downloading the replacement model inside installation meant a
link below about 3 Mbit/s could never install such an update, because a
browser abandons an install still running after five minutes and keeps
nothing of the body. Updates now install with the game alone and the page
saves the network; see `OFFLINE_PLAY.md` and `ONNX_RUNTIME.md`.

## Publisher

The publisher validates model path, source, nonnegative safe-integer generation,
HTTPS origin, and bucket before upload. Each attempt owns a unique staged gzip
file; interleaving attempts cannot replace each other's upload bodies. The
manifest is staged and synced beside its destination, then atomically renamed.
Failure before rename leaves the previous manifest; a failure after rename may
leave the complete new manifest visible. Temporary files are removed on success
or failure. Concurrent manifests are last-completed-writer-wins, not a model
promotion ordering policy. The publisher still expects an already checked ONNX
export; hashing does not replace export parity or playing-strength evaluation.

## Scoring explanations

A score now retains its selected decomposition and completed block. Both normal
and physical settlement views serialize these exact groups with displayed tile
locations, including the separately shown winning tile. A ron-completed triplet
is not marked concealed; a tsumo-completed triplet is. Specific dragon and wind
yaku carry distinct identities and the actual value tile. Buttons use those
identities rather than repeated display names. Four equal sequences count as
two pairs of sequences. Legacy results lacking attribution do not substitute
the first wind triplet when wind context is unavailable, and missing attribution
is not described as a circumstance yaku. Physical results show the winning tile
once rather than repeating it inside their standing hand.

Regressions cover native-to-JavaScript attribution, ambiguous highest-scoring
readings, both wind triplets, ron/tsumo, four identical sequences, multiple dragon
triplets, poisoned caches, interrupted upgrades, bounded stalled/overlong streams,
quota failures and publication interleaving.

## Saved-result compatibility

Regular match format 6 carries the scorer's added presentation fields. Formats
0/2/3/4/5 still replay their exact commands, tiles, yaku and payments before only
the newly added attribution is regenerated. Modern-format state remains strict.
Guided saved settlements and undo snapshots similarly migrate their old full-hand
presentation only after recomputation matches every prior ledger and score field;
no settlement is applied again and no stored balance changes. Partially modern or
altered score metadata is rejected, not silently repaired.

Browser model-download assertions observe the page, its workers and the
service worker, since the page now owns durable preparation of the network.
They still require a real single download and trained inference; no network or
offline checks are skipped.
