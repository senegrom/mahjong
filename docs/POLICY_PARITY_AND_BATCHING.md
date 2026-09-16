# Ordinary play, search and the browser share a policy contract

Follow-up to the review of `b40b2d5`, built on `6778666`. The browser already had
the two-input fix and the new network store at this base. Those changes, runtime
binaries, production manifest/model, checkpoints and tile artwork are preserved.
This patch adds durable end-to-end coverage rather than replacing that work.

## Deterministic ties

Both serving adapters and the conditional riichi-discard ranking use stable
sorting. Equal policy logits retain their original policy-index order, matching
ordinary `argmax` play before engine translation and alias deduplication. Search
that is bypassed no longer substitutes a different tied move for its incumbent.
Tests include ordinary discards, red/plain aliases, calls, forced actions, both
riichi questions, and complete native games with deliberately tied logits.

## Shared inference arithmetic

`neural.policy_inference.context` is used by ordinary evaluation adapters, search
roots, conditional roots and policy-only imagined continuation. It overrides and
then restores ambient autocast instead of inheriting a caller's accidental mode.
It does not modify training forwards, gradient settings, weights or training AMP.

`--policy-precision auto` preserves ordinary defaults: Mortal-space policies use
bfloat16 autocast on CUDA and float32 on CPU; legacy engine-action policies use
float32. `--policy-precision float32` or `bfloat16` explicitly selects a mode for
the whole evaluation player. Explicit CPU bfloat16 is supported for experiments;
it is not the default or a claim of better accuracy. Programmatic callers can use
`policy_inference.configure(player, precision, device)` once on the frozen player.
The runtime setting is not silently serialized as a checkpoint weight.

The requested and resolved modes, tie rule and inference-contract version are
recorded with new teacher replay and diagnostic recordings. Modal forwards the
setting and includes it in the immutable experiment identity. Historical replay
is not relabelled as having this contract; new marked data rejects missing or
inconsistent controls. Dataset/resume identities include the teacher contract.
Sibling evidence cannot mix changed/missing precision or batch declarations.

These controls align the algorithmic precision policy; they do not promise
bitwise equivalence across hardware, Torch versions, compiled/uncompiled kernels
or arbitrary batch shapes. Compare matched implementations and measure CUDA
parity on the hardware used for an experiment. The CUDA-specific regression skips
explicitly when no GPU is present. CPU tests exercise real float32 and bfloat16
ordinary/full/fast inference, not only mocked context flags.

## Bounded rollout inference

`--rollout-batch` defaults to **256** and bounds each imagined policy encoding,
dense input and forward, including the separate conditional discard after riichi.
It replaces the hard-coded 4096/8192-row forwards and the unbounded second-stage
forward. The same limit is forwarded to discovery and fresh confirmation, native
engine-plane and Mortal-plane rollout adapters, the collector and Modal.
`--leaf-batch` still separately limits value-leaf/reader work.

This is a per-inference limit, **not a cap on total process or GPU memory**.
Native worlds and follower copies still scale with games, candidates and worlds.
At nonzero rollout temperature, changing batching may also change random draws;
the batch setting is therefore recorded as an experiment setting. Tests assert
that all first/second-stage calls obey small limits and retain action mapping,
including incomplete tail batches and a native search without live-state mutation.

Example (new output directory and fresh evidence):

```sh
python -m neural.collect_search actor.pt --out runs/policy-parity-replay \
  --source-revision <40-character-commit> --games 2 --worlds 8 \
  --policy-precision float32 --rollout-batch 128
```

The native search API stays at **5** and training at **2**; there are no Rust
changes in this patch. Recollect comparisons affected by tie handling or GPU
precision rather than attributing old/new differences to search strength alone.

## Actual exported graph through the worker

```sh
python -m unittest neural.tests.test_policy_parity -v
cd web
node --test tests/policy-worker.test.js
node scripts/policy-export-check.mjs
```

The last command requires the native Python engines, Torch/ONNX dependencies,
locked npm dependencies and Chrome/Chromium (`CHROME_BIN` may name the binary).
The neural CI workflow installs them and runs it. It creates temporary standalone
and combined checkpoints, invokes the production exporter, and serves their graphs
to the actual worker using the committed reduced WASM runtime. Native follower
observations include a riichi root and its conditional discard. Actions, policy
weights, values and belief outputs are compared with the reference, repeatedly.
Only the module-import resolution and test manifest are changed in the temporary
site; no inference/runtime test doubles or production model edits are used.

The synthetic exporter fixtures permit removable Identity operators; actual
compatibility is tested against the committed reduced runtime, not a generic
expanded runtime. They are not published as production exports. Unit tests also
verify mask tensor type/shape and disposal after success and failure, and the
optional memory regression now releases that tensor too.

No trained-model promotion or playing-strength claim follows from these tests.
