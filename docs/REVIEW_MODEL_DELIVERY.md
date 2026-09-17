# Placement artifacts, model delivery, and scoring explanations

Repairs the eight findings in the review of `1cb9d84`. These changes do not
publish new model weights, change the scoring rules, certify a reader, or
claim a playing-strength gain.

## Placement heads and replay

`neural.placement_contract` is the shared representation validator. Feature
versions 1, 2, and 3 require a frozen-feature fingerprint. Version 4 reads a
known observation directly and uses its observation identity (`planes-1012`
for Mortal-v4, `planes-97` for engine planes). All teacher replay additionally
requires the SHA-256 of the exact head-file snapshot. Mortal teacher replay
refuses a version-4 engine-plane head, missing identities, and unknown versions.
Saving, loading, ordinary serving, replay collection, reopening, fitting, and
resume use these same meanings. Existing supported v1/v2 replay is unchanged;
v3/v4 provenance is now accepted instead of failing before collection.

Tile-attention heads and independent towers preserve their actual `looks`
setting. Early v4 files wrote zero for the depth: loading reconstructs a
contiguous depth from the saved tower's parameter names, not an assumed default.
New files write the actual depth. Save/load prediction parity is tested after
optimization at 1, 2, 4, and 6 blocks/looks, together with load-and-resave.
Matching schema fields returned by loading may be resaved, but metadata cannot
change head-owned architecture. Atomic staged publication remains in place.
Search API remains 5 and training API remains 2. No old data is relabelled and
no reader proposal declaration is changed.

## Verified network bytes versus durable offline readiness

`network-transfer.js` is used by ordinary inference and inlined into the
self-contained service worker at build time. Both cache hits and fresh
responses must match the build's exact byte count and SHA-256. Bad/unreadable
cached entries are discarded best-effort; they are never sent to ONNX Runtime.
Online inference may use a verified download even when cache access or a write
fails. Offline preparation explicitly requires durable storage instead, and
its status rechecks the actual cached bytes. The UI keeps offline readiness
false and displays a storage warning while permitting online inference.

Each transfer owns an abort controller, a 60-second idle deadline, a 30-minute
total deadline, and the manifest's decoded-byte ceiling (up to 512 MiB). These
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
quota failures, publication interleaving, and placement replay consumer paths.

## Saved-result compatibility

Regular match format 6 carries the scorer's added presentation fields. Formats
0/2/3/4/5 still replay their exact commands, tiles, yaku and payments before only
the newly added attribution is regenerated. Modern-format state remains strict.
Guided saved settlements and undo snapshots similarly migrate their old full-hand
presentation only after recomputation matches every prior ledger and score field;
no settlement is applied again and no stored balance changes. Partially modern or
altered score metadata is rejected, not silently repaired.

Browser model-download assertions observe the service-worker network target,
which now owns durable preparation. They still require a real single download
and trained inference; no network or offline checks are skipped.
