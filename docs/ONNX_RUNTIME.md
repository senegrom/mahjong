# Trained model and reduced runtime

## Source of truth

`web/src/lib/model-manifest.js` identifies the one published remote network:
its source, generation, precision, decoded byte count, stored byte count,
origin, immutable object key, and SHA-256. Inspect those fields for the current
release. `neural.export` checks the graph's operators; the publication tool
`web/scripts/publish-model-r2.mjs` writes the delivery manifest.

No ONNX model is served from `web/public`. Both packaging and offline inventory
creation reject unexpected local ONNX files, including nested files. There is
no empty `MODEL_FILES` registry or `models.sha256` ledger masquerading as a
model verification step.

The packaged runtime is listed in `model-package.js`: the reduced WASM,
patched JavaScript loader, and shared memory-budget module. `copy-runtime.mjs`
checks all sources before copying anything. Vite selects ONNX Runtime's
external-WASM entry so it does not emit a second generic WASM binary.

## Verification and memory

Downloaded and cached model bytes are checked against the manifest's exact
size and digest before inference. Transfers have idle and total deadlines,
and a decoded-byte ceiling. Storage failure can allow online inference but
must never be reported as durable offline readiness.

`memory-budget.js` defines reservation ceilings of 384, 512, and 768 MiB.
Memory begins at 16 MiB and grows only as needed. A classified ceiling failure
releases the worker before retrying unresolved requests with a larger ceiling;
browser reservation refusals and unclassified failures do not trigger unlimited
retries. Original observations and cancellation signals survive bounded retries.

The worker serializes decisions and retains the network. A warm decision does
not re-enter offline preparation. Reset/failure invalidates preparation, and
late replies from an obsolete worker cannot complete a new request. All
inference tensors are disposed after use.

The service worker's status reply includes the verified network URL, size,
and digest. The page skips a second complete model hash only when this identity
exactly matches its own manifest. Old or mismatched replies retain the explicit
verification fallback. This proof comes from the app's service-worker channel,
not a localStorage flag.

## Cache lifetime and upgrades

New model writes use an installation-scoped v2 cache. Installation downloads
and verifies the replacement but never prunes the active version. Normal
waiting-worker activation is retained: there is no forced mid-match upgrade.
After safe activation, verified current and previous model URLs are retained;
older scoped entries are removed. Failed verification, a waiting/installing
worker, or an unreadable retention record prevents pruning.

Legacy v1 bytes are verified and can be copied into the scoped cache without
fetching again. A quota failure during this copy does not invalidate a verified
legacy copy. The unscoped legacy cache is deliberately not blanket-deleted:
its entries may still be needed by another installation on the same origin.
Other installation scopes and unrelated caches are never pruned.

## Build and test

Run `npm ci` and `npm run verify` in `web/`. This is also the CI command. It
includes real inference, legacy preference migration, saved-state, memory,
worker, offline, and browser layout checks. Optional external model/checkpoint
fixtures remain explicitly identified by their individual tests.

The manual reduced-runtime workflow builds from its pinned ONNX Runtime source
and reviewed `reduced-ops.config`, restores the loader's memory hooks, runs the
web suites, and proposes its tested tree by PR. It does not push main. Running
`check-model-operators.py model.onnx` checks an explicit export; with no argument
it validates the operator list only and does not claim to have checked model
bytes. Export-time validation and real browser inference cover the remote model.

See [historical evidence](HISTORY.md) for superseded size measurements and
previous memory ceilings.
