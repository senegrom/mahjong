# Physical matching and model identity

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

## Reader declarations survive checkpoints

A producer's explicit positive-u32 `reader_proposal_version` is now serialized
by `PolicyValueNet.payload_fields()` and restored by `from_payload()` and the
normal `zoo.load_player()` path. Engine and Mortal observation readers both
round-trip. The contained standalone model's marker also survives combined
checkpoint loading; it does not invent a reader on the combined policy.

Saving/loading never automatically calibrates a reader or stamps a new model.
Only the producer, after fitting and checking against that proposal, should
set the marker. Missing, stale and unknown versions remain loadable; the search
that weighed worlds by them has been removed. Malformed markers (including
booleans/floats) fail.
A declaration is producer metadata, not independent evidence of calibration.
