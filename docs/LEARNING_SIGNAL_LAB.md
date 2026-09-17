# Learning-signal experiments

`python -m neural.learning_lab` is a new opt-in experiment runner. It does not
change the legacy trainers, their defaults, the web app, model weights, or the
promotion gate. Its baseline is a **new controlled baseline**, not a promise to
reproduce the old trainer's learning curve. Compare each preset against this
baseline with matching checkpoints, games, opponents, training seeds and compute.

## Implemented experiments

| Area | Runnable implementation |
| --- | --- |
| Asymmetric baseline | A training-only critic receives the player's public history **and** either opponents' hands or the full existing oracle encoding. It never chooses an action. Predictions are frozen before any optimiser touches the current round. |
| Distributional placement | Scalar or four-category final-rank prediction, with soft labels over the occupied positions on tied finishes. The expected utility uses `[1.5, 0.5, -0.5, -1.5]`. Hybrid runs add a separate current-hand score-change prediction. |
| Objective alignment | An explicit `hybrid` or `placement` policy objective. Complete data retain both components; no old reward is relabelled. Rank-head and reward changes can be ablated independently. |
| Defensive supervision | Public-only, separately trained tenpai, immediate legal-ron and conditional-payment heads. Exact labels come from cloned engine positions. Missing call-window labels and unevaluated discards are masked, never counted as safe negatives. |
| Counterfactual lessons | Fresh teacher discovery, one independent confirmation, and a third independent audit for ranker labels. Confirmation adjusts only the compared pair's existing policy mass. Unsearched alternatives and engine/policy aliases retain their probability contracts. |
| Richer ranker | Frozen own features, frozen own-plus-Mortal features, or a dedicated public tower. Learns paired advantages and can be evaluated at the real table through the existing seat-balanced duel. |
| Adaptive league | Frozen, identified reference/champion/recent/older/exploiter opponents, an evolving development matchup matrix, harder-opponent weighting, and a positive uniform exposure floor. An exploiter run trains one seat against three copies of one fixed champion. |
| Selective reanalysis | Uniform-quota plus public-priority root selection, an optional public-policy disagreement selector, immutable action traces and public-state hashes, fresh versioned labels under newer teachers, Gumbel candidate sampling, and fixed-budget sequential halving. |

These are experimental algorithms, not demonstrated playing-strength gains.
The implementation is not a full AlphaZero tree search or a complete AlphaStar
league manager. League membership is fixed within an exact-resume run; adding a
new exploiter/recent checkpoint starts a new experiment with an explicit roster.

## Install and test

Use the repository's normal training dependencies and rebuild `riichi-py` for
the additive `LEARNING_LABEL_API_VERSION = 1` before requesting defence labels or independently seeded reanalysis:

```sh
maturin build --release --locked --manifest-path engine/riichi-py/Cargo.toml --out wheels
python -m pip install --force-reinstall wheels/riichi_py-*.whl
python -m unittest neural.tests.test_learning_lab neural.tests.test_learning_lab_native -v
cargo test -p riichi-py --locked learning_labels::tests
```

Both native Python engines are required, as for the existing pipeline. Training
API remains 2 and search API remains 5. No existing native method changes its
meaning. CPU is the CLI default; `--device cuda` is an explicit local device
selection, not a cloud launch. For AMP experiments use `--set amp=true` consistently
in collection and learning. GPU execution must be tested on actual GPU hardware.

## Matched on-policy experiments

The input below is an existing current-lineage 1012-plane/46-action checkpoint;
`PolicyValueNet` and fused `Combined` policies are supported. Each command writes a **new** output
directory and never promotes its result.

```sh
python -m neural.learning_lab train --checkpoint combined.pt \
  --preset baseline --opponents champion.pt reference.pt --roles champion reference \
  --rounds 5 --out runs/lab-baseline

python -m neural.learning_lab train --checkpoint combined.pt \
  --preset oracle --opponents champion.pt reference.pt --roles champion reference \
  --rounds 5 --out runs/lab-oracle
```

Other single-change presets are `categorical`, `placement`, `defence`, and
`league`. `all` combines oracle hands, categorical placement, placement-only
policy learning, defensive supervision and an adaptive league. It requires
explicit `--opponents`. It is useful for integration tests, but a comparison of
`all` against `baseline` cannot identify which individual change helped.

Defaults include two critic-only warm-up rounds, 32 games per round, bounded
minibatches, a fixed initial-policy KL penalty and a sampled-KL early-stop check.
`--rounds` counts **additional policy-update rounds**, after any remaining warm-up.
Warm-up does not update actor weights or increment actor generation. All defaults
are starting points, not tuned hyperparameters. `--set field=value` overrides a
single validated configuration field; a JSON `--config` can store an entire run.
The `fixed` backbone control applies to fused networks; standalone models have no
Mortal backbone to freeze.

Exact continuation inherits the saved configuration and frozen league:

```sh
python -m neural.learning_lab train --resume runs/lab-oracle/latest.pt \
  --rounds 5 --out runs/lab-oracle-continued
```

The checkpoint retains both critics, Adam moments and active-slot manifests,
all relevant RNG streams, the fixed policy anchor, frozen opponents, league
history, source/runtime identity and completed data identities. A changed
objective, source, runtime or missing state is **not** silently accepted as an
exact resume. Use `--checkpoint` to start a deliberately new experiment instead.
Do not run simultaneous writers into an output directory. Failed rounds cannot
publish a checkpoint; completed round data may remain useful for diagnosis.

PPO uses only newly sampled on-policy actions. Actor inference keeps its
collection-time normalisation/dropout behaviour during differentiation. Privileged
critic inputs and defensive losses never enter the actor's computation graph.
The inherited embedded auxiliary heads in the actor are **not** refitted by this
runner. The new critics are explicit training-sidecar state. Use the lab's
reanalysis route for its placement teacher, rather than assuming that legacy
`--valued-by critic` automatically reads these new critics.

## Fit and inspect a critic before using it

Collect multiple complete games with one frozen actor, retaining the requested
private training labels. The `hands` mode stores only opponent-hand planes, not
future wall planes. `full` uses the existing complete oracle encoding.

```sh
python -m neural.learning_lab collect --checkpoint combined.pt --preset oracle \
  --set rank_head=categorical --set defence_weight=0.1 --set games=128 \
  --out runs/lab-critic-data

python -m neural.learning_lab fit-critics --round runs/lab-critic-data \
  --preset oracle --set rank_head=categorical --set defence_weight=0.1 \
  --epochs 3 --out runs/lab-critics.pt
```

The split keeps whole underlying game seeds together and rejects overlapping
or repeated games across shards. Oracle prefit refuses mixed collecting-policy
identities. Reports expose held-out value MSE, rank cross-entropy and Brier score
as applicable. They are not promotion results. An absent holdout is reported as
missing, not zero loss; `--validation-every 0` explicitly disables it for smoke
checks only. Training rounds with changed actors are not a substitute for this
fixed-policy prefit dataset.

Pass `--critic-init runs/lab-critics.pt` to a new `train` run with the **same**
starting actor, critic architecture, oracle inputs and objective. The fitted
critic identity is retained; incompatible or stale initialisation is refused.
A prefit critic can be followed by the normal warm-up or an explicitly chosen
`--set warmup_rounds=0`. Fit error alone does not establish variance reduction or
better policy updates.

## Defensive targets: precise scope

`Arena.learning_defence()` emits `(games, 3, 104)` float32 values in relative
opponent order: validity, tenpai, 34 evaluated-discard flags, 34 legal-ron flags,
and 34 point charges. Only **currently legal ordinary discards** are evaluated.
A cloned hand executes each discard; the engine's call/scoring rules determine
legal ron, including yaku and furiten. A payment is the discarder charge for that
opponent winning alone, including honba/liability, not a prediction of whether
they choose to win, a joint multiple-winner settlement, or danger several turns
in the future. Call windows are deliberately unlabelled.

The public defensive heads share a separate critic encoder, not actor parameters.
Thus defensive losses can improve public value/search features without directly
overwriting the actor's policy representation. When PPO uses the oracle baseline,
the public defensive heads do not by themselves change that oracle baseline;
their immediate use is the public critic/teacher and diagnostics. No direct
risk-penalty rule or clairvoyant acting input has been introduced.

## Reanalyse retained roots with a new teacher

Every collection retains a bounded number of exact public roots and the complete
engine-action trace needed to reconstruct them. Root priorities use public policy
entropy, calls/riichi and late-round context; standalone `collect --peer other.pt`
can additionally use disagreement between two public-only policies. A positive
uniform quota avoids selecting only the deliberately prioritised positions.
The selected data are not an IID sample or an unbiased importance-sampled loss.

```sh
python -m neural.learning_lab reanalyse \
  --trace runs/lab-oracle/data/round-000002 \
  --teacher runs/lab-oracle/latest.pt --out runs/lab-lessons-v1 \
  --worlds 8 --candidates 4 --confirm-worlds 16 --audit-worlds 8 \
  --candidate-method gumbel --race-budget 96
```

A standard actor checkpoint instead needs an explicitly fitted, compatible
`--placement-head`: either a legacy placement head or a `fit-critics` sidecar for
exactly that actor. A lab teacher uses its **public** placement expectation,
never its oracle. This route compares network-played, hand-boundary,
placement-only continuations, including representable call alternatives.
Reconstruction must match each saved public-state hash and the final game scores.
Changed engine/follower source semantics require fresh traces, not relabelling.

The three simulation stages are separated:

1. Discovery selects one challenger, optionally through sequential halving. The
   incumbent is protected. `race-budget` counts candidate-world slots in discovery
   only; confirmation and the independent audit have separately declared budgets.
2. A new world batch tests that single challenger against the incumbent. There
   is no repeated retry until a desired move passes.
3. A third new world batch labels candidate advantages for the ranker. Its result
   does not feed back into the policy-target acceptance decision.

A positive confirmation margin produces a bounded exponential tilt **within the
compared pair's old probability mass**. Everything unsearched remains unchanged.
The conversion between joint engine moves and declaration/conditional policy rows
preserves alias proportions. Unconfirmed roots retain the original distribution;
zero-probability support is not invented. Counterfactual riichi lessons are marked
as policy-only examples, not actual on-policy rewards or trajectories.

A standard-error margin is not a formal confidence guarantee, and an independent
confirmation does not repair a biased hidden-world model or evaluator. Test
teachers at the real table before using their labels as trusted supervision.
Gumbel sampling/sequential halving here does not inherit AlphaZero guarantees.
The reanalysis seed controls a separate native search RNG, without changing the
recorded dealing stream; different seeds do not silently reuse the same native
world sequence. Repeating identical seeds/configuration is deliberately reproducible.
Old data are never overwritten: a later teacher writes `runs/lab-lessons-v2`.

## Fit a richer ranker or distil policy lessons

```sh
python -m neural.learning_lab fit-ranker \
  --checkpoint runs/lab-oracle/latest.pt --lessons runs/lab-lessons-v1 \
  --features fused --out runs/lab-ranker.pt

python -m neural.learning_lab evaluate-ranker \
  --checkpoint runs/lab-oracle/latest.pt --ranker runs/lab-ranker.pt \
  --opponent champion.pt --games 128 --seed 9100000 --out runs/lab-ranker-duel.json

python -m neural.learning_lab distill \
  --checkpoint runs/lab-oracle/latest.pt --lessons runs/lab-lessons-v1 \
  --out runs/lab-distilled
```

`--features ours`, `fused` and `own-tower` are separate ranker ablations. A fused
ranker uses Mortal's frozen global features beside its own per-tile features.
Its offline audit-label report is game-held-out, not a whole-game strength claim.
The explicit duel command runs both seat-role directions using the existing
engine and does not promote anything. Choose fresh seeds for each final test.

Distillation is masked supervised cross-entropy with an optional fixed-policy
anchor. Its actor is loadable through the existing model interfaces, but it
explicitly drops stale lab critic state and requires refitting. A new on-policy
run can instead add `--lessons PATH --set lesson_weight=0.1`; those examples enter
only a separate supervised loss, never the PPO likelihood ratio. The default
lesson weight is zero, because having a working teacher pipeline is not evidence
that the teacher is stronger.

## Exploiters and league updates

```sh
python -m neural.learning_lab train --checkpoint combined.pt --preset exploiter \
  --opponents champion.pt --roles champion --rounds 5 --out runs/lab-exploiter
```

The champion's copied weights never change. After independent evaluation, the
result can be explicitly added as an `exploiter` opponent to a **new** main run.
The adaptive distribution combines role-weighted difficulty with `uniform/N`
minimum probability for every roster member. Its matrix and EMA estimates are
training diagnostics, never the held-out promotion dataset. Missing or changed
opponent identities cannot silently enter a resumed run.

## What the tests establish

Tests exercise rank ties, oracle/public gradient isolation, selected private
inputs, defensive missingness and exact native ron charges, pair-mass/riichi
conversion, confirmation/audit separation, discovery budgets, league floors,
immutable publication, small fused PPO updates, exact serialized CPU Adam/RNG
continuation, critic-only warm-up, critic prefit, fresh native reanalysis,
ranker fitting and actual table play, supervised distillation, and actor reload.

None of those tests establishes stronger Mahjong play. No trained production
checkpoint is included or replaced. No GPU parity or throughput result should be
claimed from CPU tests. Repeat matched experiments across training seeds and use
fresh seat-balanced whole-game evaluation before any production promotion.
