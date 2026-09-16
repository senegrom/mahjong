# Ordinary/search inference parity and bounded rollouts

Follow-up to the review through b40b2d5, preserving the subsequent external-model
manifest, model cache and two-input worker support on main. No trained weights,
artwork, native reward/API meanings or production model manifest are replaced.
Search remains native API 5; training remains native API 2.

## One policy selection and precision contract

`neural.inference` owns the evaluation contract. Descending candidate ordering is
stable, so equal scores use the lowest policy index, exactly as ordinary argmax
selection does. Translation retains first-legal-meaning priority, red/plain aliases,
and riichi's independent declaration and conditional-discard decisions.

Ordinary players, search roots, conditional riichi labels and imagined policy
continuations share a precision context: float32 on CPU; bfloat16 autocast on CUDA
for Mortal-layout players, as those ordinary players already used; float32 for
legacy engine-layout players. An outer autocast cannot silently change this policy
context. Training autocast and float32 browser export are not changed. Leaf-value
and likelihood-reader estimation remain separate from the policy precision setting.

New supervised collections record the resolved device type, policy precision and
tie version alongside the existing teacher identity. Diagnostic recordings record
them too, and the cloud experiment hash includes the inference contract. Incomplete
markers are refused. Old unmarked evidence remains unmarked and cannot be pooled
with explicitly marked teacher evidence as though it used the new contract.
Recollect data for comparisons across this inference change.

A shared precision mode does not promise bitwise equality across GPU models,
libraries, compiled/uncompiled execution or arbitrary batch shapes. The CUDA test
runs only on actual CUDA hardware; a CPU pass is not a CUDA benchmark.

## Bounded policy batches

`--rollout-batch` defaults to 256 and independently bounds policy inference rows.
It is separate from `--leaf-batch`, which bounds critic/reader inference. Both
engine and Mortal imagined continuations honor it. Mortal copies are encoded and
densified in bounded groups; conditional riichi copies, encoding and inference use
the same bound instead of processing all riichi rows at once.

The bound is available to `neural.searched`, `neural.collect_search` and the Modal
searched controller. It is validated before search and recorded in the experiment.
It is not a cap on the native world pool, all persistent follower copies, model
weights or total memory. Nonzero-temperature sampling may consume randomness in a
different order when batch size changes; keep the recorded batch fixed for exact
experiment reproduction. Temperature-zero mapping is regression-tested across
small and large bounds.

## Browser/export regression coverage

Main already feeds the legality tensor to the fused graph and disposes it. Added
worker tests cover this feed, float dtype/shape and disposal on success/failure.
A new cross-stack CI check generates disposable standalone and combined checkpoints,
exports them through `neural.export`, then runs the actual worker and repository
reduced WASM runtime in Chrome on native ordinary, call and both riichi-stage
observations. Only the model manifest is replaced with test fixture hashes/URLs;
network download verification, inference, runtime and move selection are real.
It checks action, legal probabilities, value and hand predictions against PyTorch,
then repeats requests in the same worker. No test network ships in the application.

Float export removes redundant intermediate Identity aliases before checking the
operator contract, just as quantised export already did. Named graph outputs are
retained; all existing numerical/output validation and atomic replacement remain.

These tests establish implementation contracts, not stronger playing performance.
No cloud training experiment or automatic champion promotion is part of this change.
