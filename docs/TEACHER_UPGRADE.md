# Teacher utility, calls, features and trustworthy diagnostics

This follows `REVIEW_WORLD_INTEGRITY.md`; it preserves those repairs, policy
checkpoints, existing artwork and the production opponent. No new trained model is
published by these source changes. **Rebuild both native Python engines. Search
requires API 4; training remains API 2.**

## What is repaired

- The marginal belief output is interpreted as mass per tile kind, distributed
  among available physical copies. `Belief::even` remains a uniform physical-tile
  prior. Constrained riichi allocation is retained. This is still a proposal, not
  an exact posterior over private histories: readers must be recalibrated for the
  changed proposal and strength must be remeasured. Old likelihood readers are
  not reused automatically: search uses uniform proposal weights unless the
  served reader explicitly declares `reader_proposal_version = 4` after
  retraining/calibration. This is reported in search health diagnostics.
- Non-riichi temporary furiten is reconstructed from sampled waits and public
  events since the last reset, excluding the unresolved discard. Opponents' real
  private flags and deferred caches are not copied into their imagined hands.
  The observer's own known private state is preserved, not cleared. A root claim
  can owe an opponent response before a new public event exists; its sampled
  reaction features are rebuilt explicitly. Later turns use normal event updates.
- `--objective placement --valued-by placement` removes the separate root-hand
  points bonus. A finished match pays only `2.5 - rank`; otherwise the root hand
  ends and a placement head values the next hand's first turn-to-act. Legacy
  `--objective hybrid` retains the existing points-plus-placement objective.
- `--search-calls` compares legal claims using the normal simultaneous-resolution
  rules. Other responses come from the sampled world, not the actual table's
  hidden response queue. Initially this option requires network rollouts.
- `--confirm-worlds N` checks a proposed override against the incumbent in a fresh,
  fixed-budget batch. Discovery values are not pooled into confirmation and a
  rejected proposal is not retried until it passes. This is a model-based paired
  standard-error filter, **not a formal coverage guarantee or strength test**.
- `--extra-candidates N` adds legal action-family alternatives and sampled tail
  actions while retaining the incumbent. `--audit-share P` audits some decisions
  otherwise skipped by `--sure`. Both are explicit experiments, not tuned defaults.
- Inactive/one-candidate games are filtered before world generation. Resampling
  uses each game's search RNG, never the environment RNG. Combined-policy
  continuation skips unused value/belief towers; the logits are tested for parity.
- New placement heads use distinct mean/max feature summaries and, for combined
  networks, Mortal's feature vector. Feature version 1 still loads with its exact
  old duplicated inputs. Version 2 fingerprints the complete supporting network.
- Self-play can label first turn-to-act positions per player/hand. Placement
  `--boundary-only` fitting requires these labels rather than guessing them from
  an old round. Existing stable-seed split and overlap rejection remain in place.
- `worth` now reports game-clustered uncertainty for the decision-weighted mean.
  A single independent game reports unknown uncertainty, not a zero error bar.
  Multiple chairs and repeated split trials within a game are not new games.

## Collect, fit and compare (new output paths)

Create a placement head from the frozen actor that the teacher will actually use:

```sh
python -m neural.selfplay actor.pt --games 256 --seed 10000 --out round.pt
python -m neural.placement actor.pt round.pt --boundary-only --out placement.pt
```

The following is an experimental teacher configuration, not a strength claim:

```sh
python -m neural.collect_search actor.pt --placement-head placement.pt \
  --objective placement --valued-by placement --played-by network --depth -1 \
  --search-calls --worlds 16 --confirm-worlds 64 --extra-candidates 1 \
  --games 16 --seed 20000 --source-revision <40-character-code-commit> \
  --out runs/teacher-replay
python -m neural.train_search --checkpoint actor.pt --replay runs/teacher-replay \
  --policy-mode changed --actor-kl 1 --out runs/teacher-fit --epochs 2
```

The completed replay's version-2 metadata binds the teacher utility, search API,
settings and actual placement-head bytes/features. **The student's existing
hybrid critic is still fitted to completed hybrid returns.** The placement teacher
changes its policy targets; it does not relabel that critic or overwrite the
external placement head. This separation is explicit and checked on load/resume.
Version-1 supervised replay keeps its original loss and semantic contract.

For teacher evaluation use the same flags with `neural.searched` and `--record`.
The Modal `searched` controller exposes the same options and binds them into its
immutable experiment identity before launching its child; no cloud run is started
by installing these changes.
A recording carries the actual pinned head digest and explicit teacher objective.
`worth` is a split-world diagnostic of the recorded candidates; with confirmation
it is not an exact replay of the complete two-stage rule. Only fresh, seed-paired,
seat-balanced games establish playing strength. Do not pool hybrid and placement
teacher targets into one unexplained fit.

The existing actor-KL objective remains numerically unchanged: its coefficient
both changes the fitted mixture and scales the policy contribution relative to
the value loss. It is a soft regularizer, not a trust-region bound. Control that
balance explicitly in ablations rather than crediting the teacher for a changed
loss scale.

## Boundaries

These changes implement the concrete correctness/measurement repairs and usable
teacher options. They do not provide an exact historical hidden-hand posterior,
learned danger/payment calibration, an information-set tree search, or a proven
optimal adaptive allocation. Full posterior modelling, auxiliary danger heads
and further multi-round search allocation remain separate research experiments.
No benchmark gain is inferred from unit tests, and no production promotion is
performed automatically.
