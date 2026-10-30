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
   count is reported beside it (`cells_missing_literally`, also on the
   last line's VERDICT), and the owner decides if the two disagree. Also
   **kill** when a member of the roster never sat (`unseen`), since U
   then reads its seats through weights no game taught. A seed whose round
   leaves a member out of its pool (the Club sits out about one round in
   four) is refused before anything is loaded; 9,700,000, the default,
   seats everyone.
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
   are read again with each half's advantages shuffled among its own
   decisions, which is chance: a half keeps the advantages it had and
   loses only which move each went with. Agreement is the reading above
   chance. A measure fails when alpha 0's agreement is under 1.5 times
   today's, or when neither return agrees better than chance. **Kill**
   when both measures fail; go when both pass; when the two disagree the
   verdict says so and the owner decides. A measure with fewer than 20
   flips in a half is unread, and the verdict then rests on the other.

### The fork's stop rules

A fork from fusion-long's newest checkpoint trains with fusion-long's
recipe at its rates before generation 873 (2e-6, 8e-7, 6e-7), after one
warm-up generation, refitting U every 50. Since one place is worth 1 in the
reward and a hand's points count alpha / 4,000 each, a place is worth
4,000 / alpha points:

| alpha | a place is worth | what it is |
|---|---|---|
| 0 | any number of points | placement alone, what the ruler counts |
| 0.05 | 80,000 points | lexicographic: points only break ties |
| 0.2 | 20,000 points | trained only if 0.4 passes its ruler at +100 |
| 0.4 | 10,000 points | EMA's own rate, the uma's 15,000 / 5,000 / -5,000 / -15,000 |
| about 1 | 4,000 points | today's return, for comparison |

Alpha 0, 0.05 and 0.4 started together from 1010 on 16 October 2026. The
trainers run one container at a time, and Modal pools a function's
containers by the options a call is spawned with, so each fork is spawned
with its run's name in its environment: a pool of its own, side by side
with the others, while one run still never trains in two containers.

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
  between: continue to +200 and rule again. A registered series of
  `docs/RULER.md` plays the published rulers' seeds, so it pairs with
  them deal by deal; register its rule before its first seed, as that
  document says, for an effect of 0.006 or less. The forks' series are
  `place-<arm>-g1110`, asked by seating without graphs (the same games as
  the old duel, 2.15 times as fast) and read `--against g1010`, fusion-long's
  ruler of 1010 kept as a series, by the usual rule of up to 20 seeds.

## The noise pre-test: an oracle critic and twin deals

Registered on 30 October 2026, before any spawn, for one call on a005's
generation 2820: `python -m neural.noise_pretest`, on Modal the function
`noise_pretest` of an app of its own, mahjong-noise, which writes only
`noise-pretest/<name>` on the volume. It plays one round of 4,096 games at
a005's own tables (published Mortal, mortal-run's generation 30 and the
Club, seated by player at a half; seed 9,700,000, whose round seats every
member; the reach's tile previewed, as a005 plays), paid the reward the
checkpoint carries: the placement reward at its alpha, settled by its own
U. A checkpoint without a placement reward and a U is refused, and so is
one whose U does not know a member the round seats. The round keeps, for
every decision the learner records, the oracle's 32 planes: the
opponents' concealed tiles, the next sixteen draws and the ura indicators
(`selfplay.play(..., keep_oracle=True)`; a riichi's two records share their
row's). No move and no reward reads them, and without `keep_oracle` a round
is played and recorded exactly as before.

**Baselines**, each cross-fitted by deal: the games, each a deal of its own,
are split into halves, and whatever is taught one half is asked only about
the other.

- `own`: the checkpoint's judge, the run's own baseline, which never saw
  the round.
- `warm`: the judge taught one half's returns as the reward's warm-up
  teaches it (two epochs at 1e-4).
- `oracle`: own's judge held still, plus a term that starts at nought: our
  pooled features beside a tower shaped as our network's untrained oracle
  critic (`ours.oracle_*`: a convolution of the 1,012 planes and the oracle's
  32 to 128 channels, four residual blocks, pooled over the line), read by
  two layers whose last starts at nought. Taught on three quarters of a
  half's deals (AdamW at 1e-3 falling on a cosine to 1e-5, batch 2,048, the
  warm-up's loss and clip, at most ten epochs), it is kept at the epoch that
  read the half's fourth quarter best, the start counted as epoch nought
  (the term at nought: own), and stops after two epochs without a better
  reading.
- `control`: the same term from the same seed, taught on the same
  minibatches, fed noughts for the oracle's planes: the same capacity,
  nothing hidden.

**Readings**, every decision against baselines that never saw its deal: V =
Var(G − b), plain and weighted by (1 − π(a))² from the probabilities play
recorded, over every decision, by stage and by tiles left in tens, each
with a delete-a-group jackknife error over 20 random groups of deals. The
critics are not taught again without the group, so the error is a lower
bound. The **information cut** is 1 − V_w(oracle) / min(V_w(control),
V_w(own)), what the oracle's planes add over the better of the two
baselines that see nothing hidden, the minimum taken again with each group
left out; the **capacity cut** 1 − V_w(control) / V_w(own), what the term's
capacity adds. The oracle's cuts against the control alone and against own
alone are reported beside them.

**The gradient's noise**: one pass of the head's gradient
(`placement_pretest.head_gradients`) summed over each group's decisions for
own, the control and the oracle, 60 vectors, each on the return less the
baseline, centred over the round and not standardised. Only their Gram
matrices are kept: K, and K′ with every sum divided by the checkpoint's
AdamW √v̂ + ε. N(arm) = (tr K − 1ᵀK1/20) / 19 over the arm's 20 sums, the
spread of the sums: every baseline here leaves the gradient's expectation
where it was, so it can only spread them less. N(oracle) / N(own) and
N(oracle) / N(control) are read from K and from K′, each with a jackknife
error over the groups.

**The twin ceiling**, from this plain round: with q = Σπ² over the legal
moves at each learner decision, the chance two independent draws agree,
twin copies of a deal stay one game only until a draw of any of its
learner seats differs. Over the game's learner decisions, every seat's in
one sequence, in the order made, they part at decision i with chance
Π_{j<i} q_j · (1 − q_i). Only that parting pairs two draws from one state:
after it no seat of the game has two draws from one state. The ceiling is
the expected partings over Σ (1 − q), the round's gradient-carrying draws
(1 − q is also what a draw carries of the gradient on average,
Σ_a π_a |e_a − π|²), read game by game. Even that is a ceiling: a differing
answer to a call that a ron or a pon overrides does not part the copies.
The same read seat by seat, each seat's decisions a sequence of their own,
is reported beside it and not read: where each seat's own draws would all
but surely part the copies, it counts a game's parting about once for each
of its learner seats (2.06 a game at a seat share of a half), and so reads
near twice the share a twin round could pair.

Registered:

- **Gate.** The fit worked when V_w(control) ≤ 1.02 × V_w(own) over every
  decision. The slack is there because the control is kept at the epoch
  that read its own half's check deals best and is then read on the other
  half, where it can come out a little above own by chance though it fit;
  the information cut, read against the better of the two, gains nothing
  by it. When the gate fails, or when any of the four fits (the oracle's
  term and the control, each half) ran all its epochs with its best its
  last, still more than 0.2% below the epoch before, the oracle's reading is
  **unread**, and the owner decides whether to run again.
- **Oracle: go** only if both hold: the weighted information cut over every
  decision, 1 − V_w(oracle) / min(V_w(control), V_w(own)), is at least 0.30,
  and N(oracle) / N(own) from K is at most 0.8, with its value plus two of
  its errors below 0.9. Otherwise **kill**. The readings from K′, the cuts
  against the control alone and against own alone, and N against the
  control are reported, not registered.
- **Duplicates: build a twin round next** only if the twin ceiling, read
  game by game, is at least 20%; otherwise **stop**. The reading seat by
  seat is reported, not registered.

The output's last line is `VERDICT {"oracle": ..., "duplicates": ...}`. The
network is loaded first and the gradient pass tried on one chunk of
`--gradient-chunk` made-up rows with all 60 vectors before anything is
played. A chunk the card has no room for, or whose peak
(`torch.cuda.max_memory_allocated`, the network and the seated others
included) is above 80% of the card, is halved for the whole pass, down to
32 rows. No room is PyTorch's out-of-memory error, or a RuntimeError that
names out of memory, CUDNN_STATUS_ALLOC_FAILED,
CUDNN_STATUS_EXECUTION_FAILED or CUBLAS_STATUS_ALLOC_FAILED: cuDNN's and
cuBLAS's allocations fail that way. Products that cannot be taken batched
for any other reason end the run there. The chunk's time is projected over
the round and added to the minutes spent and to what the other phases are
known to cost at 4,096 games on the L40S, 27 minutes (play 14, the critics'
inputs 3, the warm-up 1, the oracle's term and its control 8, the readings
1); a run whose total is over `--max-minutes`, 75 for this call, stops
before it plays. In the pass, a chunk without room is taken again at half
its rows and the rest of the pass read at that size. `--out` must be empty
or absent, and the cloud function refuses a `noise-pretest/<name>` that
already holds anything before its child starts. The round's ledger, every
decision's baselines and a report are written before the gradient pass,
each file under a hidden name renamed into place once whole, and the cloud
function copies whatever the child wrote to the volume whether or not it
succeeded.

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

The fusion trainer's record also says what bounded each generation, reading
clocks and counters only: `card`, the card's name (null on the processor);
`host`, the processors' model, how many the system shows and this process may
use, the container's quota of them and the encoder's threads, as far as the
platform shows them; `cpu_seconds`, the process's processor time over the
generation; `throttling`, what the cgroup's throttled periods and time, the
steal time and the CPU pressure grew by meanwhile, where they are shown;
`seated_encode_seconds`, the seated others' encoding, most of it hidden
behind the learner's deciding and so not a part of `play_split`; and, with
`--play-ahead`, `learning_beside_round_seconds`, `learning_alone_seconds`,
`updates_beside_round` and `updates_alone`, how much of the learning went on
beside the next round and how much after that round was done.

Combined checkpoints include the freeze-mode generator, CPU Torch state and
available CUDA states. Restore happens after model, optimizer and opponent
construction. Legacy checkpoints advance the freeze schedule to their absolute
generation using the supplied seed and mode list, but warn that their missing
Torch stream cannot be reproduced. CPU tests compare uninterrupted training
with serialized, resumed training, including AdamW state and resulting weights.

Mortal-only checkpoints snapshot PyTorch CPU and CUDA random streams
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

The joined player's trainer (`train_combined`) on a card reads neither the
guard's figure nor the gradient clip's finite check on the host
(`ppo_control.StepGuard`). Its fused AdamW refuses on the card, through
`found_inf`, the first update either refuses and every update after it, and
the host reads each update's verdict from page-locked memory two updates
later. The epochs stop, or the same error is raised, those few updates late
with nothing applied in between: the weights, AdamW's moments and step
counts, the generators and the record are bit for bit those of deciding at
once, which on an L40S cost 1.7% of an update (286.1 ms against 281.2 in
mode none). With an entropy floor, which reads every update's entropy on the
host anyway, and on the processor, it decides at once as before.

PPO clipping alone does not guarantee a small policy change; the original
Spinning Up implementation stops early too:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

## The last hand's rules in self-play

`train_combined --last-hand-rules all-last-dama decline-last` plays the
joined player's own self-play by the two rules of the last hand the project
kept on 31 October (`neural.all_last`, `neural.late_rules`; their series are
in docs/RULER.md), in every seat the learner holds at every table
(`neural.selfplay_rules`). The seated others are opponents and stay plain,
as each series measured its rule against a plain incumbent, and so does the
in-training measurement, which plays the network's own argmax against the
Club bots and only picks `best.pt`. It is off unless asked for: without the
flag self-play hands the learner nothing new, the record and the checkpoint
gain no key, and the cloud launches pass no flag
(`cloud_requests.rule_arguments`). The rules need `--preview-reach`, and are
refused without it: with the preview the reach's tile comes from the forward
that chose the reach and nothing is told to the follower before the rules
decide.

**What the rules read.** Self-play draws its moves, so each rule reads the
move drawn, not the network's best, inside the learner's deciding
(`mortal_learner.decide_in_mortal_space`). Decline-last is asked about every
learner row whose drawn move is a win: where the engine says the win would
leave the player alone in last, a ron becomes the pass, and a self-draw the
best other move of the same forward's logits, the win struck, as the ruler
plays them. Then the all-last dama rule is asked about every row whose move
is still a riichi, at the tile the reach drew from its preview: where the
engine says that tile keeps the player first, its discard is played instead.
Decline comes first because a declined self-draw can leave a riichi, which
dama then judges. Neither rule reads anything hidden: dama reads the
player's own hand, the scores and what the player has seen, and decline
reads a riichi win at the most its ura dora could add. CI's worker job holds
the learner's greedy play by both rules to the ruler's wrappers composed in
this order, the average over the suits off, on the cases it holds the site's
Trained AI to (`neural/tests/export_rules_fixture.py`).

**Why the best other move and not a draw.** It is the ruler's own choice
(`late_rules.decline_last`), and a move drawn from the moves that remain
could not be recorded so that the learner reproduces it: the mask is an
input to the head (Mortal's duelling head subtracts the mean over the legal
moves, the fusion's correction reads the mask), so a mask without the win
would not give the move the probability it was drawn with. The move is
recorded alone, as every rule's move is.

**How a rule's move is recorded.** Alone: its row's mask holds only the move
played, its log-probability is exactly nought, it keeps its own planes,
value, hand and return, and it is flagged (`Records.ruled`, `Batch.ruled`).
Where dama fires, the reach's tile record is dropped from every field: the
position after the reach never happens at the table. A reach left standing
after a declined self-draw keeps its tile record, an ordinary decision of
the policy, drawn from the preview of a position that does happen.

**Why that keeps it out of the policy's learning, exactly.** The learning
forward masks every move outside the recorded mask to minus infinity, so a
one-move row has one finite logit whatever the weights: its log-probability
is x - logsumexp(x) = 0 in float32, its log-softmax's gradient is g - 1 x g =
0 at the move and nought elsewhere, its ratio is exactly one and never
clipped, its entropy and its KL term are nought, and the drift guard's term
is expm1(0) - 0 = 0. Beyond the means' denominators, it touches the policy's
update only through its return in the round's standardisation of the
advantages, which runs over every row, and through the critic's share of
the single clip on the gradient's norm; never through its own probability.
Both are slight: a few hundred such rows in a round of a million and a half,
and a gradient's norm the clip seldom reaches. Such rows were common already
(a declared riichi's draw has one move), and `ppo_epochs` is unchanged. The
critic and the reading of the hands never read the mask, so they still learn
the position, and returns come per hand and player from the ledger: a rule's
consequence, a game that goes on after a declined win or a win without the
reach's thousand, reaches every decision of that hand. The policy is not
taught the rules. They stay play-time rules, which must be played wherever
the network plays to deliver their gain, as a ruler and the site's Trained AI
play them (`web/src/lib/trained-move.js`); what training gets is games,
values and credit consistent with rule-following play.

**Checks.** Each raises `selfplay_rules.RuleCheckFailed`:

- at a decision: a judged row's code must be one a legal move can have, a
  fired dama discard must be legal, and a fired decline must leave a move;
- each step: the tile records' rows are exactly the recorded rows that play
  a reach, each reach throws the tile its record holds, and each row a rule
  decided sends exactly its recorded move (`selfplay_rules.check_step`);
- at the round's end: the decisions flagged, the account's records and the
  firings (dama's and decline's, less the rows both fired at) agree, the
  reaches sent to the table and the tile records gathered, counted by the
  round itself as it steps the table and gathers the records, agree with
  each other and with the account, and decline's firings are its claim
  windows' and its turns' (`selfplay_rules.reconcile`);
- before learning: a round of a run played by the rules must say it was
  played by exactly those (`train_combined.check_rules_played`), and every
  flagged row must hold one move, its action, at a recorded log-probability
  of exactly nought, and the baseline pass's log-probability of it, read on
  every round that gets the pass, played ahead or not, within 1e-6 of
  nought, the largest recorded as `ruled_begun_max`
  (`train_combined.check_ruled`; nought where no pass reads it, a round
  valued from play, which a005's recipe never asks for).

A failed check stops the trainer before the round is learned or saved,
whether it failed here or on the worker that played the round ahead: the
trainer writes why to `rule-check-failed.txt` in its workspace and exits with
75 (`cloud_requests.RULE_CHECK_EXIT`). On that code alone the cloud's trainer
runner publishes the last generation the trainer completed once more, as a
clean exit does, its record perhaps not yet in the volume's log, and nothing
new: nothing of the refused round exists. It then writes
`rule-check-failed.json` beside the run's checkpoints (the generation the
run had published, the message and the time), commits the volume, and ends
the call with an answer that starts "stopped: rule check failed", rather
than fail it for Modal to retry the same round to the same end. Every other
exit is as before. Every checkpoint published passed every check, so after
a fix and a redeploy the run can resume from its `latest.pt` once the marker
is moved aside; a new run name is needed only if a round already learned is
shown to have been wrong.

**What the record says.** Each generation's record carries the round's
account, in the ruler's vocabulary, the tables and hands as counts. An
illustration, not a measurement:

```json
"last_hand_rules": {
  "rules": ["all-last-dama", "decline-last"],
  "all-last-dama": {"judged": 1700, "fired": 115, "tables": 105, "hands": 110,
                    "kept": {"no_yaku": 670, "short": 915, "dead": 0},
                    "riichi_after_firing": 3, "calls": 1050},
  "decline-last": {"judged": 2000, "fired": 150, "in_window": 120, "at_turn": 30,
                   "to_riichi": 2, "tables": 120, "hands": 130,
                   "kept": {"not_last": 1850}, "kept_after_firing": 4, "calls": 1400},
  "ruled_records": 264, "both": 1, "tile_records": 1180, "reaches_sent": 1180,
  "ruled_begun_max": 0.0, "seconds": 0.412
}
```

There `judged` is `fired` plus every `kept`, decline's `fired` is `in_window`
plus `at_turn`, `both` (a declined self-draw's reach that dama then left
undeclared) is at most `to_riichi`, which is at most `at_turn`, and
`ruled_records` is the two `fired` less `both`; `calls` are the engine's
questions and `seconds` the time the rules' own work took inside deciding.
The checkpoint's `training_controls` gains `"last_hand_rules"`, so whoever
ships the network knows it was trained on games played by them; nothing but
the tests reads `training_controls`, so such a checkpoint loads wherever a
checkpoint of the trainer does (`zoo.load_player`, a ruler, the cloud's
copies).

**What it costs.** No forward more: the reach's tile comes from the preview
the recipe already asks for. At most one question to the engine for each
rule a step, only when some learner row drew a win or a riichi, its rows
chosen by array masks; in a round that keeps a ledger, as every run paid the
placement reward does, only about rows the engine can answer, a non-dealer's
in South 4, read from the hands begun, and about nothing before any game has
begun South 4 (`selfplay_rules.last_hand`, `may_ask`), since each question
wakes the engine's threads. The logits come back from the card only for a
self-draw declined at the player's turn, a handful a round. On this desktop
(2 November), rounds of 256 tables of a learner that declares wherever it
can, so that the rules have the most to judge, spent 0.22 to 0.59 seconds in
the rules' own work, 0.36 on average, against 109 to 199 seconds of play
(the slowest round was slowed throughout by other work on the desktop), and
a round's processors' time was 357 seconds with the rules against 349
without (within 0.1% and 0.9% in two runs on 1 November). Scaled linearly by
tables to a005's 4,096 that is 5.7 seconds a round, an upper bound and not
the measure: it multiplies by 16 the work done once a step, though 16 times
the tables take only about 1.4 times the steps; the engine's questions,
though a step asks at most one of each kind; and the rows judged, of which
this learner gives about three a table to a005's one. The measure is built
by parts from one instrumented round of the same learner: a step that asks
the engine nothing cost 45 microseconds, a question 162 besides the
engine's answer, and the engine 38 a row judged (124 at its first asking,
which wakes its threads and carries the desktop's waits). a005's round of
about 2,000 steps, at most two questions a step over its 1,500 from the
first South 4 on and about 3,700 rows judged, comes to 0.7 seconds, 0.2% of
its 350 seconds of play, and to 1.2 seconds with a step's work extended to
a005's 800 rows and the engine's answers at their first asking. The
record's seconds are wall-clock: in a round played ahead they also count
waits for the learning beside it, as every part of play's time does there
(a005's `translate` read 1.1 seconds in round 2610 and 8.5 in 2611), so
there the generation's seconds against a005's are what judge the cost.

**The first lineage played by them.** `fusion-place-ruled`, from
fusion-place-a005's `history/gen-02610` at half of a005's three rates,
`--lr` 1e-6, `--lr-ours` 4e-7 and `--lr-mortal` 3e-7, trained on the app
`mahjong-train-rules`. The halved `--lr` is the optimiser's first group,
which trains the critic and the reading of the hands as well as the head
(`Combined.always_trained`): they learn at half of a005's rate too, as the
owner asked.
