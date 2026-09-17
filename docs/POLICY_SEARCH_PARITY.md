# Acting-policy parity and rollout memory

This follows `REVIEW_BOUNDARY_CONTRACTS.md`. The native search API stays at 5;
training, rewards, existing checkpoints and production model weights are unchanged.

## One acting policy

`neural.policy_inference` defines the acting-policy contract used by ordinary
Mortal-space play, search roots, conditional riichi roots and imagined continuation:

- Scores are ordered descending with stable first-policy-index tie breaking,
  matching ordinary `argmax` before engine-action translation.
- CUDA 46-action policies use bfloat16 autocast, preserving their ordinary-player
  behavior. CPU policies and legacy 78-action policies use float32.
- The helper explicitly establishes its context rather than inheriting unrelated
  ambient autocast. Training forwards retain caller-controlled precision; value
  leaf evaluators keep their existing separate precision contract.

Fixing precision/tie contracts is not a claim that every accelerator kernel is
bitwise identical across arbitrary batch sizes or hardware. CUDA parity tests run
when CUDA is available; CPU-only CI must report that hardware test as skipped.

## Bounded imagined inference

`--rollout-batch` (default 256) bounds policy-forward inputs separately from
`--leaf-batch`, which bounds value leaves. Both ordinary imagined decisions and
conditional riichi discards are encoded, densified and evaluated in chunks.
Conditional copies are created only for the current chunk. All actions retain
the original slot ordering before the native atomic `lookahead_apply` call.
The setting is supported by local search, completed replay collection and the
Modal search controller, and is recorded with the experiment.

The limit is not a cap on total process/GPU memory: model weights, native worlds
and their persistent follower copies also occupy memory. At nonzero temperature,
changing chunk sizes can change random draws; it is part of experiment identity.
Use fixed settings and fresh paired games to compare playing strength.

## Evidence compatibility

New completed supervised replay uses version 3 and explicitly records the acting
precision, tie convention and rollout budget. Existing version-1 and version-2
replay remain loadable under their original contracts. Version-3 metadata cannot
omit its inference contract or be downgraded to a previous teacher format. Mixed
inference contracts cannot silently enter one resumed learner dataset. No old
replay is rewritten, and the student's hybrid value labels are unchanged.

## Browser integration

Main already supplied the fused graph's float32 `legal` input alongside `planes`
and disposed that tensor. That concurrent fix is preserved. A new CI fixture
exports actual small standalone and combined checkpoints, then runs the production
worker source against the shipped reduced ONNX WASM runtime. Only browser message
transport and model-byte fetching are adapted for Node; graph inference, output
heads, masks and disposal are real. Cases include ordinary observations and both
riichi stages built through the native follower. This complements the browser
interaction regressions; it is not a mobile-device performance benchmark.

Run the focused tests with the native engines installed:

```sh
python -m unittest neural.tests.test_policy_search_parity -v
python -m neural.tests.export_worker_fixture /tmp/policy-worker-fixture
node web/tests/policy-export-parity.mjs /tmp/policy-worker-fixture
```

The fixture directory must be new. No external trained checkpoint is needed.
