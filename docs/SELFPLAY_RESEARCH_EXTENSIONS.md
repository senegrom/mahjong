# Self-play research extensions: danger, targeted practice, collection and evaluation

These are opt-in extensions to `neural.train_league`. They do not change the
browser model, enable search, promote a checkpoint, or launch cloud training.
Rebuild **both** Python engines before use. `riichi_py.RESEARCH_API_VERSION == 1`
and `libriichi.follow.Follower.fork()` are required by the new research paths.
The ordinary engine API and existing league checkpoints remain supported.

## 4. Public danger features in the actor

`neural.danger` has three commands: `collect`, `fit`, and `attach`.
The predictor reads the same 1,012 public-observation planes as the actor.
For each of the three opponents and each tile it predicts:

- the probability that the opponent **can legally ron** that discard;
- the payment by the discarder, conditional on that opponent alone claiming ron.

"Can legally ron" is not "will choose ron." Labels are obtained by cloning the
real hand, making each legal discard, and asking the **existing rules engine**
for legal calls. No separate approximation of yaku, furiten, last-tile rules,
liability or honba is implemented. Normal and riichi-declaration discards are
labelled separately. A riichi deposit refunded when the declaration loses to
ron is not subtracted from the payment label. Riichi deposits awarded to the
winner are not a payment by the discarder.

The actual hidden hands are used only to make labels. They never enter the
predictor's forward pass or the actor. A public prediction can be uncertain or
wrong; it does not prohibit dangerous moves or change the placement reward.

```bash
python -m neural.danger collect runs/champion.pt \
  --ledger runs/seeds.json --games 256 --every 8 --device cuda \
  --out runs/danger-data-01.pt
python -m neural.danger fit runs/danger-data-01.pt \
  --epochs 8 --device cuda --out runs/danger-predictor.pt
python -m neural.danger attach runs/champion.pt runs/danger-predictor.pt \
  --out runs/danger-actor.pt
python -m neural.train_league --initial runs/danger-actor.pt \
  --opponents runs/mortal.pt --out runs/danger-run \
  --seed-ledger runs/seeds.json --games 1024 --rounds 20 --device cuda --amp
```

Collection completes whole matches. Fitting partitions by original deal seed,
keeping all players, decisions and riichi stages of a game together. Validation
selects the epoch; the test partition is measured once after selection. Reports
include legal-target count, positive ron count, Brier score, probability
calibration bins, and payment error conditional on a legal ron. Check positive
coverage and calibration: a fitted artifact is not proof of a useful predictor.
Duplicate shards and overlapping declared seed blocks are rejected.

Attachment creates a small residual over the base logits, legality mask,
predicted ron probabilities and predicted expected payments. Its final layer
starts at zero, so the initial policy is unchanged. The fitted predictor is
frozen, excluded from both actor and auxiliary optimizers, and fingerprinted.
Checkpoint publication refuses a predictor whose tensor content changed during
actor training. To test a new predictor, explicitly attach it to a new actor
and begin a new run; updating the predictor underneath PPO would change the
policy even with detached gradients.

Both native and combined/Mortal actors can be wrapped, provided they use the
46-action layout. Normal zoo loading, league training, paired research, duels
and the new evaluation panel retain the wrapper. The checkpoint contains the
base policy, frozen predictor and residual; it does not need the original
predictor path when resumed. Use `--initial` to start from an attached actor,
then `--resume runs/danger-run/latest.pt` for subsequent league rounds. Existing
browser export has not been extended to this experimental wrapper.

## 5. Targeted counterfactual practice and exact snapshots

`neural.curriculum` selects roots from public pre-action information. A root is
a decision position from which several candidate actions are played separately.
Each requested category has a bounded uniform reservoir; candidate payoffs are
not read during selection. Available categories are:

| Category | Public criterion |
|---|---|
| `uniform` | Any decision with multiple legal actions. |
| `riichi_pressure` | At least one other player has declared riichi. |
| `close_endgame` | South 3 or later, with a score gap of at most 8,000 to another player. |
| `call_pass` | A call and pass are both legal. |
| `riichi_choice` | Declaring riichi is one of multiple legal actions. |
| `close_policy` | The two highest legal logits differ by at most 0.5. |
| `riichi_discard` | The conditional discard after a hypothetical legal declaration, with multiple legal tiles. |

Keep a `uniform` component to retain ordinary positions. The reservoirs are
uniform **within categories**, not an unbiased uniform sample of all decisions
after categories are combined. A root retained by several categories is stored
once with all category tags. The report includes category eligibility counts.

```bash
python -m neural.curriculum runs/champion.pt \
  --ledger runs/seeds.json --games 128 --per-category 32 --candidates 4 \
  --categories uniform riichi_pressure close_endgame call_pass riichi_choice riichi_discard \
  --device cuda --out runs/targeted-pairs-01.pt
python -m neural.paired_actions fit runs/targeted-pairs-01.pt \
  --device cuda --out runs/targeted-ranker.pt
python -m neural.paired_actions evaluate runs/champion.pt runs/targeted-ranker.pt \
  --ledger runs/seeds.json --games 512 --device cuda --out runs/ranker-evaluation.json
```

Candidates are the frozen policy's top legal actions, and every valid candidate
is continued to **final match placement**. The regression target is the payoff
difference from the policy's first choice, not the action that happened to win
on a revealed wall. All sibling candidates and repeated roots of a deal stay
in the same fit/validation/test partition. Stage metadata prevents a first-stage
ranker from automatically altering second-stage riichi discards; a stage must
have training examples before the saved ranker may act on it.

`research_state.Snapshot.capture(views, rows)` copies the selected simulator
states and their observation follower, including RNG states, pending calls,
player identities, log cursors, bot state and the follower's pending riichi
flags. `restore()` produces independent copies. The roots' public observations
are verified before intervention. Conditional riichi snapshots keep the
follower's declaration and the engine's still-pending composite action in the
same state, so no declaration is replayed twice.

Snapshots are **private, in-memory, process-local handles**, not portable save
files, policy inputs, or objects to pickle across workers. Finish outstanding
search handles before taking one. Normal callers synchronize through
`research_state.decision()`; a conditional snapshot deliberately retains the
follower one declaration ahead of the engine. Continuations use a frozen greedy
policy, so no external Torch sampling state needs to be saved. The simulator's
future random deals are copied exactly; pairing does not eliminate every
source of outcome variance after actions diverge.

Targeted data is separate research data, **never mixed into on-policy PPO**.
No ranker is loaded automatically by the browser, PPO or search. The ranker is
bound to the exact actor hash and shortlist definition used to collect it.

## 6. Synchronous parallel collection

```bash
python -m neural.train_league --initial runs/champion.pt \
  --opponents runs/mortal.pt --out runs/parallel-run --seed-ledger runs/seeds.json \
  --games 1024 --collectors 4 --inference-batch 512 \
  --rounds 20 --device cuda --amp
```

The default remains one collector. With multiple collectors, each owns its
arena, follower, opponent seating and private sampling generator. One service
thread owns model inference and combines requests, bounded by the requested
inference batch size and a finite queue. Requests for different policies are
never mixed. Native encoder/simulator work can run concurrently; Python work
still has the interpreter's usual concurrency limitations.

The learner does **not** update while collection runs. All collectors finish
against the same frozen policy version; seed reservations are partitioned into
disjoint contiguous blocks. The completed batches are merged in partition
order, repairing game indices and same-player successor indices. The usual
old-probability verification and trajectory checks run before any PPO update.
A collector or inference failure joins the workers and refuses the entire
incomplete round. There is no asynchronous policy lag or uncorrected stale
PPO replay.

Collecting with a different number of workers intentionally changes the
private sampling-stream partition. Omitted collector/batch settings are
inherited on resume. Floating-point batching can still affect exact replay on
some devices; stored-probability checks remain active. The optional Modal
launcher accepts `collectors` and `inference_batch` without starting a run merely
by being imported or deployed.

Reports include collector count, total collection wall time, completed matches
and meaningful actor decisions per second, inference calls/rows, maximum
inference batch, queue peak, and the existing engine/encoder/network timings.
Training time is also separated into old-policy reconstruction, critic
prediction, PPO, critic fitting and auxiliary fitting. Per-worker timing sums
can exceed wall time because work overlaps. GPU inference time includes result
synchronization; these are end-to-end timings, not isolated kernel benchmarks.
Compare **learning per wall-clock hour**, not throughput alone. No speedup is
claimed without profiling the actual machine and checkpoint.

## 7. Mixed-table evaluation and experimental chance controls

`neural.table_evaluate` supports heterogeneous and two-and-two policy lineups.
Policy index zero is the candidate; the other indices follow `--opponents`.
Two identical policies at a table are still two independent Mahjong players,
not teammates with a shared reward.

```bash
python -m neural.table_evaluate runs/candidate.pt \
  --opponents runs/mortal.pt runs/older.pt runs/reference.pt \
  --lineups 0,1,2,3 0,0,1,1 --ledger runs/seeds.json \
  --games 512 --domain validation --device cuda --out runs/mixed-panel.json
```

Each lineup is repeated in all four rotations with the same deal seeds. Error
bars use per-deal averages, not the correlated rotations or the multiple
candidate copies as independent samples. The report retains each table's
placement distribution and contextual diagnostics: calls versus available call
opportunities, deal-ins per discard, deal-ins under another player's declared
riichi, and final placement for players observed leading/trailing on first
entry into South 3 or later. These counters explain behavior; they are not
additional rewards or independently tested promotion criteria. Multi-ron is
counted as one deal-in for the discarder. Tied placements receive fractional
occupancy in the distribution.

### A deliberately narrow chance-only control variate

The experimental `neural.chance_control` is motivated by the zero-mean chance
correction idea in Burch et al., *AIVAT: A New Variance Reduction Technique for
Agent Evaluation in Imperfect Information Games* (arXiv:1612.06915). It is **not
full AIVAT**: information-set aggregation and player-action corrections are not
implemented or claimed. Existing raw paired evaluation remains the authority.

The native engine exposes the four initial **13-tile** hands before the dealer's
extra draw. Under the uniform 136-tile deal, tile counts have mean `13 / 34`.
For each hand, the expected number of equal-tile pairs is exactly

```
choose(13, 2) * 34 * choose(4, 2) / choose(136, 2).
```

Subtracting these known means gives 140 zero-mean chance features. A ridge
regression fitted on a separate pilot panel learns fixed coefficients. The
subsequent result is `raw_utility - centered_features @ coefficients`. The
expected correction is zero for any fixed coefficient vector under the stated
dealing law, whether or not the regression is accurate. Variance reduction is
not guaranteed: the report retains both estimates and their variance ratio.

```bash
python -m neural.table_evaluate runs/candidate.pt \
  --opponents runs/mortal.pt runs/older.pt runs/reference.pt \
  --lineups 0,1,2,3 0,0,1,1 --ledger runs/seeds.json --games 512 \
  --domain validation --want-controls --device cuda --out runs/chance-pilot.json
python -m neural.chance_control runs/chance-pilot.json --out runs/chance-model.json
python -m neural.table_evaluate runs/candidate.pt \
  --opponents runs/mortal.pt runs/older.pt runs/reference.pt \
  --lineups 0,1,2,3 0,0,1,1 --ledger runs/seeds.json --games 1024 \
  --domain test --chance-model runs/chance-model.json --device cuda \
  --out runs/chance-test.json
```

Policy hashes, lineups, feature/native versions and rotation protocol must match
between pilot and evaluation. Seed ranges must not overlap, and evaluation
outcomes never fit coefficients. Use one shared ledger; copying or resetting
ledgers can intentionally reproduce deals and does not create independent
samples. The chance features contain hidden pre-deal information, are used only
for evaluation, and never enter the acting policy.

Adjusted outcomes can lie **outside** raw placement bounds. They are labelled
`diagnostic_only`, carry their own conservative range, and are never sent to
the promotion gate. A full AIVAT implementation, sequential evidence for the
corrected estimator, and GPU-scale strength/throughput experiments remain
separate research tasks—not completed benchmarks of this branch.

## Verification

`python -m unittest neural.tests.test_research_extensions -v` exercises native
snapshot isolation and future deals, both riichi stages, positive legal-ron
labels and exact payments, frozen-predictor PPO updates, safe predictor fitting,
parallel worker failure/cleanup, probability/trajectory parity, mixed-table
symmetry, deal grouping, and pilot/test separation. Run the complete neural
suite with the real engines; CUDA-specific checks require CUDA hardware.
