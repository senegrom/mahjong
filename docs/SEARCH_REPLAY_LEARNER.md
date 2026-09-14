# Fitting completed search replay

`neural.train_search` adds the optimizer/checkpoint stage to
[the current-lineage collector](SEARCH_TRAINING_REPLAY.md). It accepts the
versioned, completed-game `SearchReplay` format, not `neural.searched.Recording`.
The separate `neural.teach` rollout-value teaching tool on main is unchanged.

## A pilot run

Build and install the current native engines first. From a checkout containing
this work, collect enough complete deals to provide both fitting and validation
examples. Start small and measure actual throughput before scaling search:

```sh
python -m neural.collect_search combined.pt \
  --source-revision "$(git rev-parse HEAD)" \
  --out runs/search-data/round-0001 --games 10 --seed 20260920 \
  --worlds 8 --candidates 4 --pool 1 --played-by network

python -m neural.train_search \
  --checkpoint combined.pt --replay runs/search-data/round-0001 \
  --out runs/search-fit/block-1 --epochs 2 --batch 256 --lr 1e-5
```

This trains the current standalone or combined 1012-plane, 46-action policy/value
network. It uses the existing frozen action-improvement target and completed-game
hybrid reward; it does not invent MCTS visits or a PPO behavior probability.
The same selected value evaluator used by the collection is trained. Belief
supervision is not added by this fitting step.

All input checkpoints are first copied and safely validated privately. A new
run starts a fresh AdamW optimizer; it does not reinterpret a PPO optimizer as
search-training state. The input model is never overwritten, and neither
`champion.pt` nor `best.pt` is written.

## Validation follows deals, not decision rows

By default, a deal whose environment seed is divisible by five is held out.
Every player, observation, and both stages of riichi in that deal stay together.
Repeated deals across shards or actors remain in the same split. Shard paths,
input ordering, and minibatch shuffling cannot move a game across this boundary.

The learner rejects a split with fewer than two training rows, or no validation
rows. `--validation-every 0` explicitly disables validation for a smoke test;
its result reports `validation_loss: null`, not a fictitious zero error.

Losses are weighted by decision row, not by shard size. All training rows are
used once in each epoch, including the final partial minibatch. Observations
are gathered only for the current minibatch; the replay is not made dense on
the accelerator. Current supported models allow singleton shard groups/tails;
Mortal's batch-normalization statistics retain their model-defined frozen mode.

Validation measures prediction error on the collected distribution. It does
**not** measure playing strength, counterfactual search quality, or independence
from all earlier experiments. Use fresh held-out games and the existing duel/gate
procedures before drawing conclusions about improvement or publishing a player.

## Exact continuation versus a new experiment

Resume into a **new output directory**:

```sh
python -m neural.train_search \
  --resume runs/search-fit/block-1/latest.pt \
  --replay runs/search-data/round-0001 \
  --out runs/search-fit/block-2 --epochs 2
```

`--epochs` is the additional number of complete *fitting epochs*. Reusing a fixed
corpus does not count as another generation of self-play. The generic checkpoint
`generation` advances from the starting checkpoint by completed fitting epochs;
`search_training.epochs` and the source checkpoint identity make this explicit.

Continuation restores Adam moments, step counts, a private NumPy shuffle stream,
Torch CPU randomness, and the selected CUDA device's generator state. It checks
the tensor parameter order/shapes, runtime signature, exact data fingerprints,
semantic contracts, source revision, value-head choice, and training settings.
An explicit active-parameter manifest also detects partial deletion of Adam
history; saved slot ordering is checked before PyTorch binds moments to weights.
The configured holdout must match the actual dataset split, including through
the low-level Python API. Outer generation/native metadata must agree with the
nested resume state.
It refuses missing/reset optimizer history, malformed/nonfinite moments, changed
hyperparameters or modified replay bytes rather than silently starting over.
Settings omitted on the command line inherit their saved values. Reordering the
same replay paths is harmless; listing identical data twice is rejected.

For a deliberately changed dataset or training configuration, start a new run
with `--checkpoint previous/latest.pt`, not `--resume`. It starts fresh optimizer
history and records that checkpoint as its initial parent. To build successive
self-learning rounds, collect new data with that checkpoint before fitting it.
This makes a collect/fit/recollect workflow possible; it is not an autonomous
population manager or a champion-promotion controller.

## Safety and review findings

The review identified boundaries that the earlier loss-helper example did not
manage: compatible multi-shard input, correlated validation rows, repeatable
optimizer continuation, weighting across unequal shards, and failed epochs.
These checks now belong to the executable learner instead of its caller:

- Only current native API and explicitly supported action/observation/reward
  semantics enter optimization. Even numerically equal boolean/float schema
  versions are rejected. Mixed collecting revisions/evaluators are refused.
- Checkpoint identity hashes the actual validated arrays in bounded chunks,
  including their shapes, dtypes and metadata, rather than trusting a filename.
  Shards remain an immutable-input contract; do not mutate their files during
  a run. These checks are not a hostile-concurrent-writer security guarantee.
- Nonfinite losses and gradients abort before the affected optimizer step;
  nonfinite parameters/optimizer state are refused before checkpoint publication.
  A failed epoch cannot be continued or checkpointed through the learner object;
  reload the previous completed checkpoint.
- Each completed epoch is atomically saved using `checkpoints.atomic_save`,
  retaining the prior file. Metrics and RNG/optimizer state live inside that
  same checkpoint. Partial epochs are never labelled complete.

The new output-directory requirement avoids competing writers and protects
previous runs, including a restart from an older recovery point. An interruption
before the first completed epoch leaves no `latest.pt`; after one, it leaves the
last durable completed checkpoint. There is no multi-file/distributed transaction.

## Tests and limits

```sh
python -m unittest discover -s neural/tests -p 'test_train_search.py' -v
```

The local controlled-network tests exercise real AdamW updates, dropout RNG,
serialization/restart equivalence, multi-shard weighting, whole-deal splits,
input mismatch rejection, singleton tails, numerical failures, and interrupted
checkpoint publication. Additional CI tests use real standalone and combined
networks, both native engines, and a completed multi-candidate searched game.

Exact CPU continuation is regression-tested with the same runtime. CUDA,
AMP/compiled parity and playing-strength improvement are not established here.
The learner intentionally has no AMP/compile or automatic deployment flags.
The hybrid reward, teacher quality, exploration, belief calibration, population
refresh, and promotion-test power still require separate work and experiments.
