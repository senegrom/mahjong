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

Imitation, re-heading and native search distillation validate their requested
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

## Search and learning contracts

Distillation stores the frozen actor probabilities with each searched decision.
Its improvement targets are constructed once per collection, not reconstructed
from the moving student. Illegal log-probability terms are removed before their
zero targets are multiplied. The outcome loss trains the evaluator chosen with
`--valued-by`; that same choice reaches searched evaluation. This remains a
shallow-search improvement distribution, not invented visit counts.

Observation and action schemas are independent. The Mortal observation adapter
supports both 46-action and 78-action models, and refuses unknown dimensions.
Native continuation metadata includes a wanted-value mask; terminal/broken
slots may omit history, but a **nonterminal** slot without reconstructible
history raises `UnsupportedSearchLayout`. It is never evaluated at all-zero
Mortal input or silently treated as terminal.

The concurrent main-branch fix reconstructs new-hand and seating events after
imagined hand boundaries and is preserved, along with its native regressions.
Missing nonterminal events still fail closed. This repair does not claim full
modern tree search, stronger play, or completion of AlphaZero-style training.

World resampling is invariant under a common finite log-weight shift. Its
all-invalid fallback returns the requested number of samples, with replacement
when needed. Invalid dimensionality and counts are rejected.

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
flip. Counterfactual reporting uses these groups, the selected value head, and
an oracle head when the checkpoint has one. It no longer infers that coverage
is the cause from an MSE difference. Matched branch continuations and controlled
opponents are still needed for a causal coverage or action-ranking experiment.

## Promotion

The `neural.promote` CLI uses `neural.gate.compare`, requires an explicit fresh
`--seed`, and publishes only an accepted immutable input snapshot. It never
reloads a moving training path after the match. Publication verifies the copied
snapshot's hash, writes a hash-bound verdict under `promotion-reports/`, and
atomically replaces `champion.pt` with the exact evaluated bytes. The old
object-level paired-error gate remains diagnostic for compatibility. The legacy
in-memory publication helper now uses validated atomic checkpoint saving.

Example (parameters and seed range must be chosen independently of results):

```sh
python -m neural.promote candidate.pt champion.pt --seed 950000000 \
  --games 512 --attempt 1 --confidence .95 --out runs/champion
```

A conservative gate may need more games to establish small edges. The caller
still owns fresh seed allocation and the increasing attempt counter. Individual
files are atomic, not a distributed multi-file transaction; verdicts are bound
to checkpoint hashes, not a mutable sidecar name. No production checkpoint,
replay cache, or cloud training run is modified by these code changes.

## Validation

Run the complete discovered Python suite with both native engines installed,
alongside the existing Rust workspace tests, Clippy and formatting checks. New
regressions exercise actual small-network optimization, narrow legal masks,
frozen targets, critic gradients, caller-selected value heads, nonterminal
history rejection, modern action schemas, native exploration bookkeeping,
resampling invariance, cloud request staging and promotion failure injection.
CPU tests and controlled interfaces do not establish CUDA/AMP/compiled execution
parity or playing strength. Preserve and regenerate historically mislabelled
replay rather than silently relabelling it.
