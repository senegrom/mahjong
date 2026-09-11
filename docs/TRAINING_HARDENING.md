# Training hardening: publication, gradients and environment contracts

This change builds on PR #40 at `087cf7d915274dc17f5e390e37aed1b46a4d22c8`.
It fixes additional training correctness failures; it does **not** claim that
search now improves the policy or that an AlphaZero loop has been completed.

## Checkpoints

`neural.checkpoint.atomic_save` is used by the standalone, combined and Mortal
PPO trainers and the imitation/search-distillation trainers. It serializes to
a unique temporary file in the destination directory, flushes it, preserves
the old bytes at `<filename>.previous`, and atomically replaces the live path.
POSIX parent directories are synced. No checkpoint is unpickled as part of
publication. There must be one writer per destination.

Failure before replacement leaves the live checkpoint unchanged. Failure or
interruption just after replacement can expose the complete new checkpoint;
cleanup never deletes the destination or the previous snapshot. Uncatchable
process exits may leave harmless temporary files. Windows guarantees here are
about process interruption, not every possible power loss or filesystem error.
Existing checkpoints are not retroactively validated or repaired.

Recovery is explicit: investigate the error and resume from the chosen valid
snapshot. There is no silent fallback to an older generation. The redundant
end-of-run overwrite was removed from PPO trainers so `latest.pt.previous`
remains the preceding saved generation rather than a second copy of the last
one. Old output directories and user checkpoints are not deleted by this PR.

## Combined belief projection

The old `hands_fix -> broadcast -> shared 1x1 hands_out` added one constant per
opponent to every tile logit. Softmax cancels that constant, so this additional
Mortal-conditioned head could not change its tile distribution.

`hands_out` now projects the hidden vector to three independent 34-tile rows.
It starts at zero, preserving fresh-model behaviour. Loading an older 1x1
projection repeats its weights and biases over the tile positions. This
preserves its predictions while allowing tile-specific gradients thereafter.
Only the two resized projection parameters' old optimizer moments are reset,
with a warning; other model, optimizer and random state is retained. New-format
checkpoints round-trip normally. Unexpected optimizer shapes remain errors.

The combined value baseline now also reads detached policy-tower features,
consistent with the standalone trainer's separation. Auxiliary losses train
their heads, not either policy backbone. This is an explicit training change,
not evidence by itself of increased playing strength.

## Training steps and resume

Eager PPO learning now includes the final incomplete minibatch, including a
round smaller than `--batch`. The optional compiled path keeps fixed-size
batches and drops only a shuffled remainder, as before. A compiled round
smaller than one batch fails explicitly: reduce `--batch` or omit `--compile`.
It cannot masquerade as a successful training generation with zero updates.

Each PPO log records `optimizer_steps`. Advantage normalization uses population
standard deviation so a singleton batch does not create NaNs. Nonfinite
optimizer gradients are rejected before the update. Batch, epoch, game and
measurement counts are validated; nonexistent resume/opponent checkpoints do
not silently change the experiment. Mutually exclusive freeze flags are
rejected. The standalone trainer explicitly rejects a 46-action learner rather
than feeding it the 78-action collection mask.

The Mortal trainer now saves and restores Torch/CUDA random streams using the
same helper as the combined trainer. Restoration happens after all network
constructors. Legacy checkpoints cannot recreate random state they did not
save; the existing warning remains intentional.

## Native API version 2

Rebuild and reinstall **riichi_py** before running the updated collectors or
benchmarks. They check `TRAINING_API_VERSION == 2` and reject older extensions
with a rebuild diagnostic rather than silently using the old semantics.

`Arena.step` validates every live row before advancing any game. Illegal actions
raise a Python ValueError with the game/action identifier, leaving the whole
batch unadvanced. Finished rows remain ignored. Explicit `strict=False` is a
legacy lenient escape hatch; training and evaluation never request it.

Imagined-world sampling has its own per-table random generator. It no longer
consumes the generator that deals subsequent real hands. The benchmark no
longer generates unused proposal hands merely to reproduce the old side
effect. Same-seed historical benchmark numbers therefore are not interchangeable
with this environment version. Remeasure competing checkpoints under the same
version. The PPO checkpoints/logs record that version, and resumed runs discard
an older version's smoothed/best benchmark history while retaining learned
weights and optimizer state.

This does not make policy-dependent game trajectories identical. Choices still
affect real play; the isolation prevents *simulation effort alone* from changing
the deals. Separate policy sampling, opponent selection and freeze RNG state
remain part of reproducibility.

## Search remains an experiment

`searched.py` and `distil.py` now reject incompatible modern checkpoints rather
than silently treating a combined player as its standalone subnetwork. Supported
legacy search still uses the 97-plane/78-action layout. `distil.collect` rejects
unfinished games and exposes `--max-steps`; CPU measurement gets the correct
device argument.

The remaining work from the handoff is still required:

- Build modern observation/action adapters for simulated continuations while
  respecting each player's information.
- Correct top-k world truncation and validate action-ranking quality.
- Demonstrate search improvement over the same frozen policy before using it
  as a teacher.
- Collect search distributions and completed-game outcomes, and train the
  evaluator as well as the policy on those data.
- Replace heuristic-only `best.pt` selection with candidate/champion evaluation
  and an opponent population. An improved heuristic score is not a promotion
  certificate.

Keep old replay data for diagnosis when needed, but regenerate reward data
known to have been affected by earlier collector errors. Do not relabel old
labels or benchmark measurements as corrected by this code change.

## Regression coverage

The added tests exercise interrupted checkpoint writes/publication (including
hard subprocess exit), legacy projection prediction parity and useful gradients,
auxiliary gradient isolation, per-parameter optimizer migration, small eager
rounds, explicit compiled no-op rejection, actual combined/Mortal optimizer
updates, serialized combined resume parity, native batch-atomic illegal-action
rejection, claim-window preservation, and complete native games with extra
imaginary-world calls but unchanged real deals/outcomes.

Run `python -m unittest discover -s neural/tests -v` with both freshly built
native engines, plus unfiltered workspace tests and Clippy. Validation results
belong to the specific commit/run, not merely this coverage description.
