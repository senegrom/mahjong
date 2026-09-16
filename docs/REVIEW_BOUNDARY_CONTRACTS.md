# Physical matching, fresh continuation chance and model identity

Follow-up to the full review of `659fb3c`. The current dependency-pin update is
preserved. No new trained model, calibration claim or strength promotion is
introduced by these source changes.

## Physical-table matching

Claimed discards are matched injectively to compatible called melds using
augmenting paths. An ambiguous chii assignment may be reassigned when a later
discard needs it. Validation does not reorder the recorded meld history, reuse
one meld for two claims, or relax copy-count/source-seat checks. The regression
covers 123p/123m/345m claiming 1p/3m/5m through the actual position builder,
including all 36 meld/discard permutations and invalid extra/source claims.
The WASM physical-advice test exercises the serialized public entry point too.

## Search chance and compatibility

**Rebuild `riichi-py` for SEARCH_API_VERSION=5. Training stays API 2.**
Every imagined proposal receives a fresh continuation seed from that game's
search RNG before resampling. Coalescing repeated proposal IDs keeps the seed
with the original proposal; sorting or reindexing retained slots never changes
its chance identity. Incumbent/challenger siblings share it within one evidence
stage. Discovery and confirmation receive fresh seeds, including for later
hands, instead of repeating a sequence derived from the slot ordinal.
The environment RNG and actual game's future deals are untouched.

The Rust `leaves_from`, `leaves_from_for_objective`, `Lookahead::begin` and
`Lookahead::begin_moves` APIs now require one explicit `chance_seeds` entry per
world. The higher-level `leaves` and Python Arena generate these automatically.
Callers must pair siblings within a stage, not reuse discovery's seeds in a
confirmation stage. This is reproducible pseudorandom sampling, not a guarantee
of statistical coverage or a proof of playing strength.

While updating this path, the club adapter's dropped `placement_only` argument
was also repaired. A zero placement head now produces zero utility at a first-hand
boundary even when the root hand moved points. Hybrid search retains those points.

New diagnostic records, cloud experiment identities and supervised teacher
metadata all declare API 5. API-4 teacher replay is not relabelled or accepted as
new teacher evidence; recollect it for the corrected chance/utility semantics.
Original version-1 supervised replay retains its existing separate contract.
Existing policy checkpoints and their action/reward meanings are unchanged.

Tests drive actual discovery-plus-confirmation through hand boundaries with both
network and club rollouts. A scripted critic forces the branch under test; it is
not a trained model measurement. The tests check sibling pairing, refreshed
future deals, repeatability of the whole experiment, and unchanged real play.

## Reader declarations survive checkpoints

A producer's explicit positive-u32 `reader_proposal_version` is now serialized
by `PolicyValueNet.payload_fields()` and restored by `from_payload()` and the
normal `zoo.load_player()` path. Engine and Mortal observation readers both
round-trip. The contained standalone model's marker also survives combined
checkpoint loading; it does not invent a reader on the combined policy.

The proposal layout/distribution version is still **4**, distinct from search
API 5. Saving/loading never automatically calibrates a reader or stamps a new
model. Only the producer, after fitting and checking against that proposal,
should set the marker. Missing, stale and unknown versions remain explicitly
uniform at search time. Malformed markers (including booleans/floats) fail.
A declaration is producer metadata, not independent evidence of calibration.

## Sibling-head supporting features

Create trainable heads with `sibling_head.new_head(net)`. The trainer does this
itself when no head is supplied. A head retains the exact identity of its
supporting stem/tower/tail and observation/action contract: SHA-256 over named,
shaped, typed tensor bytes plus the feature-layout version. For combined models,
these are the actual `ours` features consumed by this head, not unused towers.
The feature identity does not depend on a mutable checkpoint pathname.

Saving validates and preserves this contract; user metadata cannot replace it.
Loaded and explicitly supplied heads must match before ranked inference or
continued head training. This check is shared by the CLI and its cloud caller.
Serving assumes frozen supporting weights for the lifetime of a RankedPlayer.
Legacy unbound head files remain inspectable, but cannot be attached or silently
published as verified heads. Refit them against the intended frozen actor;
pointing an old pathname at new weights cannot establish their provenance.

Consumer regressions save/reload the original actor and head, accept them, reject
another same-shaped actor, and reject a replaced actor path before the real ranked
CLI begins a duel. They also cover missing/malformed contracts and action-layout
changes. Atomic auxiliary-file failure protections are retained.
