# Checkpoints, run identity and supported training contracts

## Completed checkpoint publication

All six neural checkpoint-producing commands (`train`, `train_mortal`,
`train_combined`, `imitate`, `distil`, `rehead`) use `checkpoints.atomic_save`.
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

## Search input contracts

The network-only `searched.play(..., searcher=None)` baseline accepts both
current 1,012-plane action layouts through the ordinary player adapters.
Actual **native lookahead remains limited to 97-plane, 78-action engine-space
networks**. That API supplies hypothetical snapshots without the complete mjai
history a Mortal encoder requires. All three native search entry points now
reject unsupported layouts before touching the arena, with an actionable
error. They never pad, relabel or silently approximate the observation.
Use `neural.arena` to benchmark current checkpoints. Enabling native lookahead
for them requires an additional historical-state API, not just a reshape.

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

## Replay validation and completed learning rounds

Sparse planes require native int64 offsets, uint16 columns, native floating
values, consistent monotone offsets and finite values. Every column must be
less than 34,408. Validation runs before replay publication, at loading, and
before dense or resident conversion; scans use bounded temporary allocations.
Malformed loaded slots are reported and excluded, never silently densified
into the next observation. Validation cannot recover previously wrong rewards
or structurally valid legacy mixed batches.

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
