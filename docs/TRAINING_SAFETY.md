# Training safety

Checkpoints, run identity, rewards, replay and the controls on each policy
update: what the trainers guarantee and what they do not.

## Completed checkpoint publication

All five neural checkpoint-producing commands (`train`, `train_mortal`,
`train_combined`, `imitate`, `rehead`) use `checkpoints.atomic_save`.
Each writes a unique temporary file alongside the destination, flushes and
fsyncs it, and safely deserializes its tensor/primitive payload on CPU before
atomic replacement. A serialization error, invalid payload or pre-publication
interruption leaves the prior destination untouched. An interrupt after the
rename leaves the complete new file; cleanup never removes the destination.
POSIX directory syncing requests rename durability. Windows has no portable
matching directory-fsync operation, so process-interruption safety does not
imply protection against every power failure or filesystem fault.

This adds CPU validation and disk IO per save; it does not construct a second
GPU model. New checkpoint payloads must be readable with `weights_only=True`.
An abrupt process exit can leave an unused uniquely named `.partial` file;
it is never used as a resume source. Existing corrupt files are not repaired.

`atomic_save` also keeps the destination's previous bytes as
`<name>.previous`, moved aside by its own atomic rename, unless the caller
passes `keep_previous=False`. The backup is local and single-writer, not a
transaction, and no proof that the older file was sound. Restoring from it
is an operator's choice, never automatic; cloud publication neither
restores from it nor uploads it.

The cloud publisher validates staged copies of **all** checkpoint files before
changing any live checkpoint. `generation.txt` and tenth-generation archives
use the generation in the copied `latest.pt`, not a stale stdout counter.
Checkpoint renames are individually atomic, not one transaction across logs,
latest, best, reference and metadata. A failed controller stops its child;
a failed training process does not trigger another final publication. Earlier
completed generations already published remain available.

## Container reuse and resume

Each cloud training, distillation and reheading invocation receives its own
fresh temporary directory and explicit `run.json` identity. An existing local
`latest.pt` is never a reason to resume. Only a checkpoint resolved from the
requested volume source enables resume. Same-run resume carries the saved
reference, best checkpoint and history explicitly; cross-lineage starts do
not inherit the target run's old reference or history. Completed output is
validated before it is copied to the volume.

The invocation owns and cleans up only its own temporary directory. Existing
scratch directories, volume checkpoints, and compiler caches are not deleted.
Replay is local to the invocation and starts empty on the next cloud call;
it remains available across generations within that call. Compiler caches
remain separately shared. Use a new run identity to start an independent
lineage. Simultaneous publishers for the same volume run are not supported;
this change does not add a distributed run lock or a transactional volume.

Opponents named for a cloud run are resolved as a whole population before
anything is staged: one missing checkpoint aborts the invocation rather than
shrinking the population. Each is copied through the validated checkpoint
writer into its own numbered directory, and `opponents.json` records the
requested name, the staged copy's SHA-256 and its generation; cloud
publication keeps that manifest beside the checkpoints.

Deal seeds do not depend on the round size, so `--games` may change on
resume. `training_state.round_seed` gives every absolute generation its own
block of 2**32 consecutive seeds, starting at
`base_seed + (generation + 1) * 2**32`, and a round plays the first `games`
of them. Keep the same base seed and the checkpoint's absolute generation
when resuming; the blocks then never overlap an earlier round's.

## Fusion belief compatibility

The fusion correction projects Mortal features into independent logits for
all three opponents and 34 tiles. It no longer adds a softmax-invariant scalar
per opponent. New output weights start at zero, preserving the initial base
belief distribution while receiving tile-specific gradients immediately.

Loading the old Conv1d head replicates each of its constant rows into the new
Linear head; probabilities are preserved before further training. Combined
training expands the corresponding Adam moments with the same mapping,
retaining step counts and all unrelated optimizer state. Subsequent updates
can now learn different corrections for different tiles.

## Native actions and acting precision

`riichi_py.Arena.step` checks every live row's action before it changes any
table, so an illegal later row cannot leave earlier games advanced without
their training records. An illegal action raises `ValueError` rather than
becoming a pass or the first legal move; finished rows are ignored.
`strict=False` opts into the old lenient behaviour, and no trainer or
evaluator uses it.

`neural.policy_inference` fixes the precision every player that chooses moves
at a table acts at: a 46-action policy on CUDA runs under bfloat16 autocast,
and CPU policies and legacy 78-action policies at float32. It sets that
context explicitly rather than inheriting whatever autocast surrounds it;
training forwards keep the precision their callers choose.
`neural/tests/test_policy_precision.py` checks both. This fixes the
precision, not bitwise agreement across batch sizes or hardware.

## Rewards, ties and measurements

Self-play pays each hand's score change to the decisions made in that hand
after every arena step, including steps on which only other players moved,
and refuses a round in which a hand's ending went unnoticed. Placement
rewards reach every decision only once every game has finished.

`neural.outcomes` is shared by self-play, the heuristic benchmark, the
duplicate arena and duels. Tied players take the average of the ranks they
share and split the rewards of those places, so a four-way tie places
everyone at 2.5 with no placement reward under the default reward vector. A
tie for first splits one win: `wins` counts shares of first place, not
outright wins. The rewards are checked against
`engine/riichi-core/tests/fixtures/placement-rewards.csv` under every
ordering of the players. Measurements from before ties were pooled broke
them by player index and are not comparable with current ones.

Self-play, the score-only measurement and duels refuse an unfinished batch
with `IncompleteGamesError` rather than read provisional scores as final
results. The default budget is 4,000 steps (`--max-steps` in the arena and
duels); a game that finishes on the last permitted step counts. The
heuristic benchmark seats the network as one fixed player against three
native bots whose actions and claims the arena settles itself; if a
`riichi_py` build ever hands one of their decisions to Python,
`evaluate_games` stops and asks for a rebuild.

## Replay validation and completed learning rounds

Sparse planes require native int64 offsets, uint16 columns, native floating
values, consistent monotone offsets and finite values. Every column must be
less than 34,408. Validation runs before replay publication, at loading, and
before dense or resident conversion; scans use bounded temporary allocations.
Malformed loaded slots are reported and excluded, never silently densified
into the next observation. Validation cannot recover previously wrong rewards
or structurally valid legacy mixed batches.

The replay ring is a single-writer cache: never open one directory from two
training processes. Each pushed batch goes to a fresh generation directory;
its arrays are flushed before `ring.json` is atomically replaced to reference
them, and retired batches are removed only after that. Reopening reclaims
generations nothing references, and cleanup after a failed publication never
deletes the attempt's own destination, since the manifest rename may already
have committed it. None of this repairs wrong rewards in replay written
before these checks: rebuild such a cache from new self-play, and keep an old
one only for diagnosis.

The three self-play trainers reject nonpositive batch/epoch settings and
rollouts too small to produce one full minibatch before publishing replay or
performing auxiliary work. They also require a positive actual update count
before reporting success. `optimizer_updates` and (for the main trainer)
`replay_optimizer_updates` are recorded. Remainders of otherwise trainable
rollouts remain dropped to preserve compiled fixed-shape minibatches.
`checkpoint_generation` explicitly reports the completed checkpoint counter;
the existing `generation` field remains the zero-based iteration for log
compatibility.

Mortal-only checkpoints now snapshot PyTorch CPU and CUDA random streams
following measurement. Resume restores them after constructing all models,
optimizers, compiler wrappers and opponents. Legacy missing-state resumes warn
that exact historical sampling cannot be reconstructed. The deterministic CPU
regression exercises the actual trainer entry point with controlled rollouts;
GPU/compiler determinism and deployed Modal behaviour need environment-specific
verification and are not implied by that test.

## Policy update controls

All three self-play trainers accept `--target-kl` and `--baseline-batch`,
and the cloud training functions and the main cloud launcher forward them.
The KL guard is opt-in (`0`, the default, keeps the full update budget). For
example, as a CPU smoke run:

```sh
python -m neural.train --rounds 1 --games 1 --channels 8 --blocks 1 \
  --batch 32 --epochs 2 --replay-steps 1 --measure-games 1 \
  --baseline-batch 16 --target-kl 0.02 --out runs/smoke
```

A smoke configuration checks correctness, not playing strength; choose
production settings by held-out comparisons. Nonfinite learning rates and
weights and invalid clipping or precision settings are refused before a
learner is built.

`--baseline-batch` bounds the rows of each value-baseline forward. Without it
the forward takes `--batch` rows, capped at the rollout's size; the final
chunk is still padded to that size for compiled shape stability.

For a sampled log-probability change `d = new_log_prob - old_log_prob`, the
guard's diagnostic is `mean(expm1(d) - d)`. It is nonnegative, and under
old-policy samples with equal legal support its expectation is
`KL(old || new)`. The signed `approx_kl` field remains for log
compatibility; the guard's own fields are `sampled_kl_last`,
`sampled_kl_max`, `target_kl` and `kl_early_stop`.

When the diagnostic exceeds the threshold, the pending optimizer step is not
applied and the round's remaining PPO epochs stop. Updates already made are
not rolled back: this is a sampled minibatch safeguard, not a hard bound on
the whole policy's KL. The main trainer may still run its detached auxiliary
replay pass. A round stopped before any update fails rather than publish a
learned generation. Checkpoints record the settings under
`training_controls` for provenance, but the command line is authoritative
and must supply them again on resume. With the guard on, every minibatch
reads a scalar back from the GPU, which costs time; with it off, nothing
waits.

PPO clipping alone does not guarantee a small policy change; the original
Spinning Up implementation stops early too:
https://spinningup.openai.com/en/latest/algorithms/ppo.html
