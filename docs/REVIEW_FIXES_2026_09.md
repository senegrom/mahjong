# September 2026 review repairs

## Physical-table wall accounting

`recordDiscard` counts an implicit draw only for an unknown hand whose seat
is not already at its `act` decision. That decision means a draw is already
accounted for or pon/chii has just supplied the extra tile. This also avoids
counting the initial dealer draw and an explicitly entered replacement draw
twice. A still-pending kan replacement consumes one live-wall tile. An
implicit draw from an empty or incomplete wall is rejected without changing
the draft, discards, or riichi payments.

Regression coverage: `web/tests/physical-wall.test.js` exercises known and
unknown hands after pon/chii, subsequent ordinary draws, all three kan kinds,
an explicitly recorded draw followed by hiding the hand, and the final draw.
Guided mode retains its own `needsDraw` accounting and post-call wall override.

## Resuming training with a different round size

All three trainers already use `training_state.round_seed`. Its version-2
schedule reserves an immutable block of `2**32` consecutive u64 deal seeds
for each absolute generation:

```
first = base_seed + (generation + 1) * 2**32
```

Only the requested `games` prefix of that block is played. The first seed no
longer depends on `games`, so reducing or increasing `--games` on resume cannot
move the round into a previous generation's allocation. Inputs are checked,
rounds must contain between 1 and `2**32` games, and u64 overflow is rejected.
The unused ranges cost no memory and require no checkpoint cursor.

This deliberately changes future deals even for small rounds; it is not a
bit-for-bit continuation of the old deal schedule. At legacy resume generation
G, each earlier old-schedule round with at most `2**32` games ends below
`base_seed + G * 2**32`. The new range therefore starts beyond that history.
Keep the same base seed and the absolute checkpoint generation when resuming.
Do not downgrade to the old allocator. Resuming an older checkpoint deliberately
replays its future seed ranges, as before; this is not rollback detection.

`neural/tests/test_round_seeds.py` covers both batch-size directions, mixed
round sizes, deterministic restarts, migration from both older formulas,
adjacent full blocks, and invalid/overflowing inputs. Existing Mortal stop-
generation tests remain in place.

## Unsaved-play identity fallback

`MatchStore.newMatch` marks a new identity without creating a UUID. Identity
allocation happens only inside the first real save transaction. Missing or
denied storage/Locks therefore never requires `crypto.randomUUID`. If identity
creation itself fails, the store warns and falls back to unsaved play without
replacing the existing saved record. Persistent games still receive one fresh
identity per game and keep it across moves.

Regression coverage: `web/tests/save-identity.test.js` runs the actual store
source with unavailable crypto/UUID, denied Locks, and working persistence.

## Required real-network memory coverage in CI

The web CI job prepares the exact network named in `model-manifest.js` using
the existing bounded, byte-count- and SHA-256-verified network transfer, then
sets `MAHJONG_NETWORK` for the standard `npm run verify` command. A missing
model path is a failure in CI rather than a skipped memory test. Local unit
runs without `CI` may still skip it unless that path is provided.

The runtime regression retains its original reservation and headroom checks
and runs 36 inferences across three model loads; its diagnostic now reports
those actual counts. Network preparation has a five-minute total deadline
and a one-minute idle deadline. It does not change the published network,
the browser's delivery behavior, or any tile artwork.
