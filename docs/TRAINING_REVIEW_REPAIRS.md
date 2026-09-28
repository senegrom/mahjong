# Training review repairs — 12 September 2026

This follow-up is reconciled against `393d9ba2393f5b765883b8bc03f1b197483498d8`.
It preserves PR #42's caller-owned precision, bounded baseline batches,
cancellable prefetching, policy-drift control, native action validation,
checkpoint migration, and the current engine and browser changes.

## Cloud and auxiliary learning

All three cloud PPO entry points reject nonpositive literal generation counts
before creating a workspace or child process. The CLI's zero-round sentinel is
not used to reinterpret a request for zero work. Opponents are resolved as a
complete population and copied through the validated checkpoint writer. Missing
or corrupt inputs abort the invocation. Indexed subdirectories prevent collisions
(including punctuation collisions), while the filenames retain the roster's
reference names and weights. `opponents.json` records requested names, copied-byte
SHA-256 hashes and generations; cloud publication retains the manifest.

Imitation and re-heading validate their requested
resume inputs and update budgets before creating models or output. Singleton
auxiliary batches and empty update runs are rejected. Actual update counts are
logged, and both total loss and gradient finiteness are checked. Teacher and
re-heading masks use core legality for both reach decisions; mapped red/plain
aliases share, rather than duplicate, the source action's probability.

Dense replay has exact shape/dtype checks and bounded-memory value validation:
returns must be `(N,)` float32, hand targets must be probability distributions
or empty hands, byte planes must be binary, and each row needs a legal action.
The existing publication and signal-recovery tests remain; their artificial
fixtures now use the actual production schema.

## Observation and action schemas

Observation and action schemas are independent. The Mortal observation adapter
supports both 46-action and 78-action models, and refuses unknown dimensions.
The search contracts this review also repaired (search distillation targets,
native continuation metadata, imagined hand boundaries and world resampling)
went with the search, which has been removed from the trainer and the engine.

## Exploration and diagnostics

The concurrent main-branch design is retained: recorded likelihoods are the
raw policy's, and forced rows do not contribute to PPO, its drift guard,
entropy or leash. They remain available for value/belief supervision. The
competing mixture-gradient implementation prepared earlier in this review is
not installed. Per-stage exploration coefficients are retained as diagnostic
provenance; the reach tile decision is never itself forced. Excluded rows are
masked before exponential ratio arithmetic, so an extreme ignored likelihood
cannot poison a finite policy loss. Checkpoint controls name this policy.

`after_exploration` records subsequent same-player information states within the
hand, rather than relabelling the pre-action observation with its current coin
flip.

## Validation

Run the complete discovered Python suite with both native engines installed,
alongside the existing Rust workspace tests, Clippy and formatting checks. New
regressions exercise actual small-network optimization, narrow legal masks,
frozen targets, critic gradients, caller-selected value heads, modern action
schemas, native exploration bookkeeping and cloud request staging.
CPU tests and controlled interfaces do not establish CUDA/AMP/compiled execution
parity or playing strength. Preserve and regenerate historically mislabelled
replay rather than silently relabelling it.
