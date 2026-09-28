# Acting-policy precision and browser parity

This follows `REVIEW_BOUNDARY_CONTRACTS.md`. Training, rewards, existing
checkpoints and production model weights are unchanged. The search whose roots,
continuations and rollout batches this document also covered has since been
removed, from the trainer and then from the engine.

## One acting policy

`neural.policy_inference` defines the precision every player that chooses moves
at a table acts at:

- CUDA 46-action policies use bfloat16 autocast, preserving their ordinary-player
  behavior. CPU policies and legacy 78-action policies use float32.
- The helper explicitly establishes its context rather than inheriting unrelated
  ambient autocast. Training forwards retain caller-controlled precision.

Fixing the precision contract is not a claim that every accelerator kernel is
bitwise identical across arbitrary batch sizes or hardware. CUDA parity tests run
when CUDA is available; CPU-only CI must report that hardware test as skipped.

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
