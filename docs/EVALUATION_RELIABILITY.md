# Independent opponents and interrupted replay publication

The native arena settles all configured heuristic-bot actions and claims
before exposing another decision. This includes every responder in a
multi-player claim window, and tracks player identity as winds rotate.
`evaluate_games` additionally rejects a native extension that exposes a bot's
decision. Rebuild `riichi_py` when updating this code; old wheels are not a
valid benchmark backend.

The arena CLI loads checkpoints through `zoo.load_player`, including the
46-action Mortal-space adapter. Duplicate evaluation uses the same score-only
loop as the fixed-chair benchmark, without retaining training batches or
inventing oracle labels for adapters. Legacy 78-action networks retain the
benchmark's imagined-world RNG consumption. `--max-steps` controls the safety
budget and incomplete simulations still fail instead of reporting a result.

Historical benchmark measurements made with the bot-ownership defect are
not directly comparable to corrected ones. Remeasure checkpoint candidates
on the corrected engine; do not treat a change in benchmark semantics as a
change in policy strength.

Native terminal search and Python training share the placement-reward
fixtures in `engine/riichi-core/tests/fixtures/placement-rewards.csv`.
Tests exercise each fixture under all 24 player permutations. Ties pool the
rewards of the occupied places; a four-way tie has zero placement reward.

Replay cleanup never deletes the destination generation from a failed
publication attempt: an atomic manifest rename may already have committed
it before a catchable interrupt was delivered. A dirty live writer reloads
the committed manifest before its next read or write. Reopening synchronizes
the manifest's directory before reclaiming unreferenced generations. Both
normal write failures and actual SIGINT at the publication boundary are
covered. The cache remains single-writer; concurrently opening a second
writer is unsupported. Existing wrong rewards and legacy mixed batches are
not repaired retrospectively.
