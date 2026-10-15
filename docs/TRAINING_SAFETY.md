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
restores from it nor uploads it. The self-play trainers skip it when
`MAHJONG_KEEP_PREVIOUS` is `0`, which the cloud launcher sets for its
children: a cloud workspace is deleted when its call ends, so there the
backup only cost a copy of the checkpoint every generation. On the desktop
it is kept.

The cloud publisher validates staged copies of **all** checkpoint files before
changing any live checkpoint. `generation.txt` and tenth-generation archives
use the generation in the copied `latest.pt`, not a stale stdout counter.
Checkpoint renames are individually atomic, not one transaction across logs,
latest, best, reference and metadata. A failed controller stops its child;
a failed training process does not trigger another final publication. Earlier
completed generations already published remain available. Within one call,
`best.pt` and `reference.pt` are copied again only when they have changed
since the call last published them (by size, modification time and inode);
the first publication of every call copies both.

## Container reuse and resume

Each cloud training, distillation and reheading invocation receives its own
fresh, empty temporary directory, its name prefixed with the run's. An
existing local `latest.pt` is never a reason to resume. Only a checkpoint
resolved from the requested volume source enables resume. Same-run resume
carries the saved reference, best checkpoint and history explicitly;
cross-lineage starts do not inherit the target run's old reference or
history. A resume name that resolves to nothing starts a new lineage (a
fresh network, the published Mortal or a join) only in a run that holds no
`latest.pt` yet; in one that does, the call fails before anything is copied,
staged or started, because the new lineage's first publication would
overwrite the run's checkpoints and log. Completed output is validated
before it is copied to the volume.

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
publication keeps that manifest beside the checkpoints. A block seated
without others publishes no manifest and removes the one an earlier block
left, so the published manifest always describes the block that published
last.

Deal seeds do not depend on the round size, so `--games` may change on
resume. `training_state.round_seed` gives every absolute generation its own
block of 2**32 consecutive seeds, starting at
`base_seed + (generation + 1) * 2**32`, and a round plays the first `games`
of them. Keep the same base seed and the checkpoint's absolute generation
when resuming; the blocks then never overlap an earlier round's.

A resumed optimiser takes Adam's moments and step counts from its
checkpoint, and with them the settings no trainer gives, such as the betas
and epsilon. Every trainer sets each group's learning rate from the command
line, and its weight decay and fused kernel as its own code builds them
(`ppo_loop.restore_optimiser`), so a change to either in the code reaches a
resumed run as it reaches a fresh one.

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

## The placement reward, credited hand by hand

By default every decision is paid its hand's points over 4,000 plus the
game's placement bonus (1.5, 0.5, -0.5 or -1.5), credited to every decision
of the game. `train_combined --reward placement` pays instead

    U(the next hand's starting scoreboard) + alpha x (the hand's points) / 4,000

and, for a decision in a game's last hand, the bonus itself plus the same
points term. U is the expected final bonus read from the scoreboard alone
(`neural/place_model.py`): a small network over the stage (East 1 to South
4), the counters, the riichi sticks on the table, where the dealer sits, the
four raw scores and the gaps to the other three, in seat order from the
player's own, and which kind of player holds each seat (the learner or a
named member of the run's roster). It answers the four places' chances,
fitted to where players finished with tied places as shared labels, so its
expected bonus is the one `outcomes.placement_rewards` pays; its answers
for the four views of one scoreboard are centred, since a game's four
bonuses always add up to nought (`place_model.centred`). EMA carries
nothing from one hand to the next but the scores, the counters, the sticks
and the dealer, so cutting the return at a hand's start loses nothing about
the future; summed over a game, the hands' changes in U add up to the bonus
less U at East 1 (`place_model.hand_credits`, tested). At `--alpha 0` the
reward is placement alone, what the ruler counts; at 0.05 points only break
ties, since even a 60,000-point swing is then worth less than one place.

The scoreboard comes from the start_kyoku events the arena already hands
the follower (`scoreboard.HandStarts`, read in `observe.Views.advance`),
with raw scores, so a score below zero survives where Mortal's planes clamp
it. A round played with `selfplay.play(..., ledger=True)` carries a
`scoreboard.Ledger`: every decision's game, hand and player and its own
hand's points, every hand's start, who sat where and the final scores. A
ledger whose hands do not match the engine's own count of finished hands
is refused (`Ledger.check`), because a start read late or missed would pay
a decision another hand's boundary. Two things keep every start read: a
table nobody follows, which otherwise stops writing its log
(`Arena.forget_events`), keeps writing it for a ledger, which reads it
every step; and once the round is over the ledger reads what the last step
wrote, since native bots can deal and play out a game's last hand on it
without asking anyone, and a relaunch would replay that round and be
refused again. A round settles by placement in `selfplay.settle_placement`,
which replaces the returns and nothing else.

**The default does not change.** A round asked for no ledger plays and pays
exactly as before: `neural/tests/test_placement_reward.py` pins today's
returns, moves and final scores at three fixed seeds to digests taken at
7a0c405, before the change, over the learner alone, a learner seated by
player beside an other and the Club, and a table Mortal's engine follows;
the engine deals from its seed and the returns are sums of integers, so
the digests hold on every machine. Keeping a ledger changes no field of a
round. With none of its options, `train_combined` keeps no ledger, builds
the same optimiser and writes the same checkpoint keys and log records.

Options (all refused, before anything is built, without `--reward
placement`; the cloud's `train_combined` passes none of them on unless
`reward="placement"`, see `cloud_requests.placement_arguments`):

- `--alpha A`: the points' weight, finite and nonnegative.
- `--place-model PATH`: the U a run starting the reward settles by
  (`python -m neural.place_model fit ledger.npz ... --out u.pt`, or the
  pre-test's `place-model.pt`). A resumed run settles by the U its
  checkpoint carries and says that it does not read the file, so a
  relaunch with the same command line cannot undo a refit.
- `--value-warm-up N` and `--value-warm-up-lr R` (1e-4): the reward's
  first N generations teach the critic alone, our value head and the
  head's correction from Mortal's vector, in an AdamW of their own at R,
  with everything else held still and the PPO epochs skipped. Both parts
  read features held back from them, so no policy can move: every other
  weight, the main optimiser's moments and every move's probability are
  bit for bit the same after such a generation
  (`neural/tests/test_placement_training.py`). The count is kept in the
  checkpoint, so a relaunch does not warm up again.
- `--place-refit-every K` and `--place-refit-rounds M` (4): every K
  generations of the reward, U is fitted again on the run's last M rounds,
  an eighth of their games held out, and kept only if it reads those games
  at least as well as the U it would replace; the log's
  `place_model_refit` says how it read and whether it was kept. Rounds
  that seated none of some kind of player U reads are not fitted on, and
  U is kept: a U fitted on them would read that kind through weights no
  game taught. The Club sits out about one round in four, so two rounds
  both lack it one time in sixteen and four rounds one time in 256.
  Likewise a run starting the reward refuses a U fitted on rounds that
  seated none of a member it seats (`fitted["unseen"]`).

A checkpoint of a placement run carries `reward` (kind, alpha, the
generation the reward began, the warm-up generations had, the last refits)
and `place_model`. Resuming one without `--reward placement`, or at another
alpha, is refused: start a new run to change either. U must know every
member the run seats (`--opponents`), or the run is refused before it
plays. A placement run's records add `reward`, `alpha`, `value_warm_up`,
and the return's variance and the critic's error over the East round's
decisions and the South's apart (`return_variance_east`,
`value_error_south`, ...).

### The pre-test and its kill rules

`python -m neural.placement_pretest` (on Modal: `placement_pretest`) plays
one round of a checkpoint at its run's tables, keeping the ledger, and
writes `report.json`, `ledger.npz` and `place-model.pt` (U fitted on the
whole round, which a fork starts from). It reads:

1. **U's calibration**, every game held out once (`--folds`): the mean U
   against the mean bonus by stage (with U's explained share beside the
   boundary judge's 0.04 to 0.77), by tenth of U's answers, and by tenth
   within each stage. **Kill** when some tenth at some stage misses by
   more than 0.03 beyond two of that cell's standard errors, counted over
   games. The rule as first written, any miss over 0.03, cannot be applied
   to a few thousand games: the error of a tenth at East 1 is about as
   large as the threshold, and noise alone would fail it. The literal
   count is reported beside it (`cells_missing_literally`).
2. **The split-half agreement of one update**, for today's return and the
   placement reward at each `--alphas` (0 and 0.05), each game paid by a U
   that never saw it, each decision's baseline from a copy of the critic
   taught the other half of the games' returns as the warm-up would teach
   it. The games are split into halves three ways; each half's PPO
   gradient over the fusion head at the generation's first step is
   compared with the other's (the cosine), and each half's gradient, taken
   as a generation's step (`--lr` times the generation's optimiser steps)
   preconditioned by the checkpoint's AdamW second moments, is applied to
   the head and read on about 20,000 positions where the checkpoint's top
   move is under 0.8, from greedy rounds against three published Mortals
   (the flips: the overlap of the two halves' changed best moves). Both
   are read again with the advantages shuffled among the decisions, which
   is chance; agreement is the reading above chance. **Kill** when alpha
   0's agreement is under 1.5 times today's on both measures; go when it
   is at least that on both; when the two disagree the verdict says so and
   the owner decides. A measure with fewer than 20 flips in a half is
   unread, and the verdict then rests on the other.

### The fork's stop rules

A fork from fusion-long's newest checkpoint, at alpha 0 and alpha 0.05,
trains with fusion-long's recipe at its rates before generation 873 (2e-6,
8e-7, 6e-7), after one warm-up generation, refitting U every 50:

- **At +50:** stop when the fork's placement against published Mortal in
  the log's `matchups` is more than 0.006 worse than fusion-long's plateau
  (2.463 to 2.467), that is above 2.473. Run
  `E:/tmp-claude/mahjong/compare-810/probe3.py` and
  `feature-probes/probe_alllast.py` on it: at level scores the turn-one 7m
  should drift towards 1m, far behind it should keep 7m, and South 4
  riichi should stop being 100% where dama already moves the place.
- **At +100:** a ruler on seed 9,100,000 (20 x 1,000), paired with 810, 910
  and 1010 on the same deals. At or below the plateau's mean: stop.
  +0.006 or more above it: replicate with two more training seeds. In
  between: continue to +200 and rule again.

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
weights, an entropy floor whose options are nonfinite, negative or leave it
nothing to fall back to or no room to rise, and invalid clipping or
precision settings are refused before a learner is built.

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
