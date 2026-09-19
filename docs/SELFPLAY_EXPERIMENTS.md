# Placement-aligned self-play experiments

This branch adds an experimental trainer, `neural.train_league`, without replacing
the three existing trainers or changing the browser policy. The code implements
the experiments below; it does **not** establish that any of them increases
playing strength. No long training run, promotion, or export happens automatically.

## The objective and the control

`--objective placement` trains the actor on `2.5 - final_rank`, including the
engine's shared-rank convention for ties. The utility is undiscounted and ends at
the end of the full match, not at a hand boundary. This uses `Batch.placements`;
the collector's existing hybrid `Batch.returns` remains available.

`--objective hybrid` is the control: current-hand points divided by 4,000 plus
final-placement utility. The selected actor objective has a versioned contract
in each league checkpoint. Resuming with a different objective or critic shape
requires `--reset-critics`; hybrid-trained value weights are never silently
reused as a placement baseline. Resetting the critics restarts their warm-up.

The deployed model's original value head deliberately retains **hybrid-v1** units.
A separate, detached auxiliary pass trains that value head and the opponent-hand
predictor. Search and browser callers must not mistake it for the new training
critic. This preserves existing export/search contracts.

## A first run

Build and install the two real engine extensions as described in the README.
Use an existing 46-action checkpoint; old 78-action models must first be re-headed.
The initial actor can be native, combined, or Mortal-only. Raw Mortal checkpoints
must be compatible with the repository's existing safe checkpoint loader.

```sh
python -m neural.train_league \
  --initial runs/joined/latest.pt \
  --opponents runs/mortal/latest.pt runs/published-mortal.pt \
  --champion runs/champion.pt \
  --out runs/placement \
  --objective placement --critic privileged --advantage mc \
  --games 1024 --batch 2048 --epochs 2 --rounds 20 \
  --target-kl 0.01 --table-mix 0.4 0.4 0.2 \
  --seed-ledger runs/seeds.json --device cuda --amp
```

Checkpoint paths are examples, not files bundled with this repository. On CPU,
use `--device cpu` and omit `--amp`. Compilation is optional (`--compile`), not
required for correctness. Rates, architecture, table mixture and KL settings
are experimental starting points rather than tuned optima.

The first generation trains the critics without moving the actor. Change
`--critic-warmup` explicitly to ablate this. The new public critic has its own
observation tower. With `--critic privileged`, a second independent critic also
receives the simulator's pre-action oracle planes, including hidden tiles. Both
predict the selected objective. Neither shares trainable features or an optimizer
with the actor. Only public policy methods are exposed to the acting wrapper.

The baseline is measured **before** fitting on the new round. Monte Carlo
advantages are terminal utility minus this old baseline. Critics continue fitting
even if PPO reaches its KL limit. Critic fitting uses current-rollout targets,
not replayed returns from old policies with different opponents.

## Temporal credit assignment

`--advantage gae --gae-lambda 0.95` enables generalized advantage estimation as a
separate placement-only experiment. Gamma is fixed at one. The collector saves
fixed player identity, hand identity, next-decision index and match-terminal
flags. Next-decision links follow the same player through interleaved games,
rotating seats, hand boundaries and both stages of a riichi declaration.

At lambda one, this implementation reduces to the Monte Carlo advantage. Lower
lambda trades reduced variance against critic bias. Critics are still fitted to
full terminal returns, making the advantage-estimator experiment distinct from a
change to their targets. Downsampling a round clears its trajectory links; a
boundary-only dataset cannot be used as though it were a complete trajectory.
Unfinished games remain an error, never invented terminal outcomes.

Reference: Schulman et al., *High-Dimensional Continuous Control Using Generalized
Advantage Estimation*, https://arxiv.org/abs/1506.02438.

## PPO and auxiliary isolation

The trainer reconstructs the complete frozen old action distribution and verifies
its chosen-action likelihoods against collection before updating. Legal-action
KL is computed over all 46 actions, avoiding arithmetic on masked infinities.
The default KL guard is 0.01. It stops additional actor updates, not a rollback
of steps already applied. The log also measures final KL separately for discard,
riichi, call and other decision states, including the last optimizer step.

The actor and auxiliary heads have disjoint optimizers and gradient clipping.
Large hand-prediction or value gradients therefore cannot reduce the actor step
through a shared clipping norm. `--actor-clip` and `--aux-clip` control them
separately. The combined model's Mortal batch-normalization statistics remain
frozen, as in the original trainer.

`--schedule staged` freezes Mortal initially (`--freeze-generations 20`) and then
trains jointly at separate component rates. `--schedule random` retains the
existing six-mode random-freezing experiment; `--schedule joint` trains all actor
components. These schedules concern combined actors; Mortal-only and native
controls train their actor normally.

`--leash` optionally penalizes KL to an immutable public reference. By default it
is the starting actor in `reference.pt`. Replacing it on resume requires both
`--refresh-reference` and `--reference PATH`, so an intentional champion-based
anchor change cannot occur silently at a restart.

There is no uniform forced exploration in this trainer. Policy actions are still
sampled: zero forced exploration is not greedy self-play. The PPO update rejects
intervention-containing batches rather than assuming that masking a forced row
makes the subsequent trajectory on-policy.

Reference: Schulman et al., *Proximal Policy Optimization Algorithms*,
https://arxiv.org/abs/1707.06347.

## Opponents and immutable identities

With a nonempty roster, the default table shares are:

| Share | Table |
|---|---|
| 40% | Four learner players |
| 40% | One learner and three frozen opponents |
| 20% | Three learners and one frozen opponent |

`--table-mix SELF ONE_LEARNER THREE_LEARNERS` changes these shares. Each frozen
seat is independently drawn from the weighted roster and held by the same player
for the whole match. Chairs are randomized before play, not after observing the
deal. Without opponents the explicit fallback is pure four-player self-play.
A nonzero frozen share with an empty roster is an error.

The roster weights are champion 3, reference 2, recent 1, older 0.5. Named
`--opponents` are references. `--champion`, `--recent` and `--older` supply other
roles. Every requested file is copied, hashed and loaded from an immutable
`snapshots/<sha256>.pt`. Checkpoint metadata records requested path, content hash,
generation and role. A changing `latest.pt` cannot silently change a resumed
population. Use `--refresh-opponents` deliberately to introduce new snapshots.
Missing requested players fail rather than silently changing the distribution.

Training metrics report complete table compositions separately. They are noisy
training diagnostics, not promotion evidence. The mean placement of all four
learner seats at a pure-self-play table is necessarily 2.5 and is not a strength
measurement. One-learner tables cost more simulation per learner decision; gains
must be compared against both independent matches and elapsed compute.

## Seeds, recovery and comparison runs

The new seed ledger reserves each contiguous game block **before** collection.
Training, validation, test, promotion and counterfactual collection occupy
separate unsigned-64-bit domains. Domain addresses remain below signed-64-bit
limits so game identities survive tensor serialization. A reservation records its
start, exclusive end, count, purpose and attempt index. Changed batch sizes and
restart cannot reuse a completed range. The original trainers now use the same
allocator instead of a fixed 1,000-seed generation stride.

All related new runs should share `--seed-ledger runs/seeds.json`. `--seed`
controls actor sampling and initialization; the new trainer uses ledger root
zero independently of that sampling seed. Separate ledger files intentionally
reproduce the same initial deals: do not pool them as independent evidence.
Legacy trainers use per-seed lanes; a lane holds 16,777,216 games per domain and
fails explicitly when exhausted. Do not delete a ledger to reset an evidence
budget. A leftover writer lock is fail-closed and requires checking that no
writer is active before removal.

`latest.pt` contains the actor, all optimizers, both training critics when enabled,
objective contract, random states, seed high-water marks, roster and reference
hash. Copy `reference.pt` and `snapshots/` with a moved run. Preserve `seeds.json`
too: a file may record an interrupted attempt newer than the checkpoint. A
checkpoint can restore a missing ledger, but cannot know reservations made after
it was saved. Shared durable ledgers are the authoritative allocation record.

Resume with the same experiment flags and an explicit source:

```sh
python -m neural.train_league --resume runs/placement/latest.pt \
  --out runs/placement --rounds 20 --games 1024 --batch 2048 \
  --objective placement --critic privileged --advantage mc \
  --seed-ledger runs/seeds.json --device cuda --amp
```

A whole-run writer lock prevents concurrent trainers from replacing one another's
checkpoints. Actor inputs are snapshotted before any loader reopens them.
The sampling seed must match. The objective and critic shape must match unless
reset explicitly. Rates and other controls remain user-selectable on resume;
use the run's `experiment.json` and checkpoint `run_options` when reproducing a
continuation. Opponents, mixture and reference are preserved unless explicitly
changed. Exact restart tests cover unchanged options on CPU, not cross-device
or cross-PyTorch-version bitwise reproducibility.

## A–E ablations and the Mortal-only control

`neural.selfplay_experiments` prints commands by default. Nothing trains until
those commands are run or `--execute` is supplied.

```sh
python -m neural.selfplay_experiments runs/joined/latest.pt \
  --opponents runs/mortal/latest.pt runs/champion.pt \
  --mortal runs/published-mortal.pt --out runs/ablations --seeds 11 22 33
```

A uses the hybrid objective, an independent public critic, pure self-play,
random freezing and no KL stop. B changes to placement. C introduces the
privileged baseline. D adds mixed tables. E changes freezing and enables the KL
guard. Compare E's individual controls separately when attributing its effects.
All use Monte Carlo advantages, so GAE can be tested afterward.

A is a **new-infrastructure control**, not a claim to reproduce the old trainer
exactly. The original `train`, `train_mortal` and `train_combined` remain available
for that comparison. The optional Mortal-only run establishes whether fusion is
worth its extra computation. Use multiple sampling seeds and compare quality at
fixed game and compute budgets. Do not infer a strength gain from lower value
loss, sharper entropy, or a few lucky duels.

## Evaluation, nomination and promotion

The trainer nominates at a fixed `--nominate-every` cadence. It writes
`candidate.pt` and a generation checkpoint, never `champion.pt`. A low heuristic
placement no longer determines which new-trainer generations can be compared.

```sh
python -m neural.league_evaluate runs/placement/candidate.pt \
  --opponents runs/champion.pt runs/published-mortal.pt \
  --ledger runs/seeds.json --games 512 --domain validation --out panel.json

python -m neural.league_gate runs/placement/candidate.pt runs/champion.pt \
  --ledger runs/evidence.json --games 2048
```

The panel reports each opponent separately. Every comparison runs all four seats
in both 1-vs-3 directions, paired by game seed. Standard errors use independent
deals, not correlated seatings. A panel is diagnostic; keep an untouched test
panel for final reporting.

The new promotion command defaults to an empirical Bernstein lower bound on the
symmetric per-deal edge, in [-1.5,1.5]. `--method hoeffding` preserves the original
range-only control. The ledger binds method, confidence, minimum edge and minimum
deals before evaluation. Attempts use alpha `(1-confidence)/(i*(i+1))`, spending a
summable budget across fresh candidate tests. Interrupted attempts consume their
seed allocation and attempt. Inputs are immutable copies bound to their hashes.

This is a fixed-sample bound **per attempt**, not an anytime confidence sequence.
Choose the game count and settings before reading outcomes; do not repeatedly
peek at partial games, reuse promotion seeds, or switch bounds after seeing the
answer. The confidence statement assumes independent game randomness and the
specified bounded matchup statistic. It is not a proof of strength against
arbitrary players, especially in nontransitive populations.

Only an explicit `--out RUN` publishes a passing evaluated snapshot as
`RUN/champion.pt`. Without it, the command writes evidence only. A failure or tie
leaves the champion untouched. Publication is hash-bound and atomic per file,
not a multi-file transaction.

Reference: Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance
Penalization*, Theorem 4, https://arxiv.org/abs/0907.3740.

## Placement-judge data hygiene

`neural.placement` now partitions by immutable game identity into approximately
80% training, 10% validation and 10% test, using a stable mixed hash. All seats,
positions and duplicate recordings of a deal stay together. `--keep-best`
selects only from validation performance. The test partition is measured once,
after loading the selected epoch, and reported separately. Very small datasets
may have empty partitions; a kept-best fit without validation is refused.
There is no promise of exact 80/10/10 sizes on small samples.

## Counterfactual action differences: a separate research track

```sh
python -m neural.paired_actions collect runs/placement/candidate.pt \
  --ledger runs/seeds.json --games 256 --root-step 40 --candidates 4 --out pairs-01.pt
python -m neural.paired_actions fit pairs-01.pt --out action-ranker.pt
python -m neural.paired_actions evaluate runs/placement/candidate.pt action-ranker.pt \
  --ledger runs/seeds.json --games 512 --out ranker-evaluation.json
```

Collection freezes the actor, records its public policy's top-k first-stage
moves, and runs each intervention to the **end of the full match** on paired
seeds. A riichi declaration is intervened on before the translator updates its
follower; its second-stage discard then follows the frozen actor. Root
fingerprints must match before interventions. Original legal masks remain the
authority. Data uses a distinct paired-action schema, not PPO rollout records.

The learned public-only head minimizes squared error in action-return differences
relative to the policy's first choice, with equal weight per root. A single
revealed wall gives a noisy return sample, never a hindsight-optimal action label.
Game-grouped train/validation/test partitions prevent leakage. The selected
prediction is evaluated on held-out realized returns, and reported errors are
clustered by deal. The maximum realized candidate return is not used as an
unbiased estimate of available improvement.

The optional ranked player only reranks the same top-k first-stage candidates;
second-stage reach decisions keep the original policy. It is bound to the exact
actor hash used to create its data. Its gameplay evaluator reports both duel
directions on fresh test seeds. It never promotes itself or enables search in
self-play or the browser. Data from a single root-step distribution may be poorly
calibrated elsewhere: collect several preselected root steps and test actual
playing strength before considering deployment. No superiority is claimed.

## Cloud entry point and tests

The optional launcher uses the existing Modal image, managed subprocess and
atomic volume publisher:

```sh
modal run neural/league_modal.py::train_league \
  --initial joined-run/latest --run placement-run --rounds 20
```

It defaults to a fresh experimental run directory, never a production champion.
The shared volume's `league-seeds.json` allocates related cloud runs. Its function
has one container at a time; do not concurrently write the same ledger from an
uncoordinated external job. Modal volume durability still depends on commits:
a hard container loss before publication can lose reservations newer than the
last committed volume state. Checkpoint high-water marks protect completed
published generations, not unseen work after a lost volume commit.

Run regression tests with both real engines installed:

```sh
OMP_NUM_THREADS=1 python -m unittest neural.tests.test_selfplay_experiments -v
OMP_NUM_THREADS=1 python -m unittest neural.tests.test_paired_action_research -v
python -m unittest discover -s neural/tests -v
```

The new tests cover objective contracts, same-player GAE, immutable populations,
variable-size seed reservations, exact unchanged-option trainer restart,
independent actor/critic gradients, full masked KL, combined/Mortal compatibility,
real mixed-table collection, real paired continuations, validation/test separation
and failed promotion attempts. GPU precision/compilation, cloud execution and
long-run playing strength require their respective environments and budgets.
