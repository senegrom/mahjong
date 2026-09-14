# Conservative search-policy fitting

The search-replay pipeline can now preserve the collecting policy's uncertainty
when search agrees with it, and penalize movement away from that frozen policy.
This is an **opt-in fitting objective**, not a stronger search algorithm, an
AlphaZero/MCTS implementation, or an automatic champion-promotion loop.

## Why this change

`SEARCH_TRAINING_REPLAY.md` records that the tested search teachers have not
beaten the underlying policy at the table. It also identifies two fitting risks:
sharpening the actor's preferred move even on agreement rows, and no explicit
policy-drift penalty. Faster or longer training does not resolve either problem.

The conservative mode addresses these fitting risks without claiming to fix the
teacher. It uses the existing completed-game replay, action contract and reward.
No shards are relabelled or rewritten, and `SearchReplay.loss` retains its legacy
behavior for existing callers.

## Run a controlled comparison

Build the native engines and collect completed replay as described in
`SEARCH_TRAINING_REPLAY.md`. Fit a candidate into a new directory:

```sh
python -m neural.train_search \
  --checkpoint combined.pt --replay runs/search-data/round-0001 \
  --out runs/search-fit/conservative-1 \
  --epochs 2 --batch 256 --lr 1e-5 \
  --policy-mode changed --actor-kl 1
```

The KL coefficient `1` is a starting point for an experiment, not a tuned or
proven optimum. For the legacy comparison use the **same starting checkpoint,
replay, seed and fitting budget**, a different new output directory, and
`--policy-mode all --actor-kl 0`. Those remain the defaults, so existing commands
do not silently change their training algorithm. Either control can also be
varied separately for an ablation.

For successive learning rounds, collect fresh replay with the candidate before
fitting again. Repeating epochs over one fixed dataset is not new self-play.

## Exact meaning of the objective

Let `a` be the frozen collecting actor, `t` the stored search-improvement target,
and `p` the current learner, all restricted to the row's legal actions. In
`changed` mode the fitted distribution is:

```text
q = t  when no alias of the selected engine move is an actor-probability maximum
q = a  otherwise

loss = mean_rows[ cross_entropy(q, p) + actor_kl * KL(a || p) ]
       + value_weight * mean_rows[ (value - completed_return)^2 ]
```

An agreement row is **not discarded**: it teaches preservation of the actor's
full distribution. At `p = a` its policy-only gradient is zero rather than a
push towards greater certainty. Value learning, shared network parameters,
dropout and weight decay can still change the policy; preservation is an
objective, not a guarantee that parameters stay fixed.

Action comparison uses the repository's engine-to-policy equivalence classes.
A red/plain policy alias is not a different engine move. Riichi declaration and
its conditional discard are compared at their own decision stages. Any selected
alias tied for maximal actor probability is conservatively treated as agreement;
this intentionally avoids relying on a backend's arbitrary argsort tie order.
It is not an exact reconstruction of a historical tie-breaking choice.

The objective is averaged over **all decision rows**, not the number changed.
Thus all-agreement minibatches are well defined, and a small changed-only shard
cannot outweigh a large agreement shard merely because of batching. The existing
whole-deal validation split and per-row cross-shard weighting are retained.

Illegal logits are removed before normalization. Frozen probabilities are
validated and detached; zero-mass actions have a finite KL contribution. Only
the current minibatch's observations are materialized, and the KL term uses the
stored actor probabilities rather than allocating another frozen network.

The value head remains the evaluator selected by the collecting shard, and the
completed-game hybrid reward is unchanged. This does not align the search
teacher with a pure final-placement objective or add belief supervision.

## Resume and compatibility

```sh
python -m neural.train_search \
  --resume runs/search-fit/conservative-1/latest.pt \
  --replay runs/search-data/round-0001 \
  --out runs/search-fit/conservative-2 --epochs 2
```

Omitted settings inherit the saved values. A policy-objective marker and both
controls are checkpointed and checked on exact continuation. Changing either
requires starting a new experiment with `--checkpoint`, not `--resume`.
A pre-objective search checkpoint can resume only with its exact legacy
`all / 0` objective. Missing controls in a newly marked checkpoint are rejected,
not silently reset. Existing data, optimizer, RNG, runtime and parameter-signature
checks remain in place. The input model and any production champion are untouched.

## Validation and limits

```sh
python -m unittest discover -s neural/tests -p 'test_search_objective.py' -v
python -m unittest discover -s neural/tests -p 'test_train_search.py' -v
```

The new tests cover agreement/change gradients, tied choices, red/plain aliases,
two-stage riichi, masked/nonfinite logits, zero-probability KL, detached targets,
row-weighted shard equivalence, value-head selection, replay immutability,
legacy-loss parity, real AdamW optimization, serialized CPU continuation,
old-checkpoint migration, damaged-state rejection and CLI/runner option wiring.
An additional native test exercises standalone and combined checkpoints through
the production runner and exact continuation when both native engines are
installed. Controlled-network tests are not native gameplay or strength tests.

`actor_kl` is a **soft penalty, not a hard KL cap**. It anchors each replay row to
its collecting actor, not every future generation to one immutable ancestor.
It cannot guarantee stability across arbitrarily many recollection rounds.
The helper exposes cross-entropy, KL and changed-row fraction separately for
diagnostics. Total validation losses from different objectives are not directly
comparable, and neither measures playing strength.

Before promotion, compare the teacher against the unsearched actor and candidates
against the incumbent on fresh, paired/seat-balanced deals using the repository's
existing duel/gate procedures. More confidence in an internally estimated search
gain is not a substitute for those games. No cloud job, training run on a real
checkpoint, benchmark win-rate claim, model deployment or champion replacement
is part of this change.
