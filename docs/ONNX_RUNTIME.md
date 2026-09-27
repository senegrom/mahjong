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

The JavaScript in `onnxruntime-web` calls into the WASM, and those calls change
between releases, so `web/package.json` pins the package to exactly the ONNX
Runtime release the reduced WASM was built from. `copy-runtime.mjs` refuses to
build when the installed package differs from the release recorded in the
binary, so a Dependabot update of that package alone fails until the runtime
is rebuilt at the new release.

The manual reduced-runtime workflow builds the pinned release from source with
the reviewed `reduced-ops.config`; dispatched with another release, it builds
that one and moves the pin in the same commit. It restores the loader's memory
hooks and runs the web suites against the published network, then pushes the
tested tree to an `automation/reduced-runtime-*` branch and links a compare
page from the job summary: GitHub Actions may not open pull requests in this
repository, and one opened with the workflow token would start no checks. Open
the pull request from that link and merge only after its checks pass. The
workflow never pushes main.

`neural.export` refuses a network that needs an operator the reduced runtime
lacks, and the memory regression loads the published network through that
runtime, so a missing kernel fails before release.

See [historical evidence](HISTORY.md) for superseded size measurements and
previous memory ceilings.
