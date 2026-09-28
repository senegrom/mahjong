# Trained model and reduced runtime

## Source of truth

`web/src/lib/model-manifest.js` identifies the one published remote network:
its source, generation, precision, decoded byte count, stored byte count,
origin, immutable object key, and SHA-256. Inspect those fields for the current
release. `neural.export` checks the graph's operators; the publication tool
`web/scripts/publish-model-r2.mjs` writes the delivery manifest, reading the
precision from the exported weights and refusing to run without the source
checkpoint's name. It uploads with an exact wrangler release, gzipped, under
`models/g<generation>/<sha256>`, which `workers/model-cdn` serves to the page,
and writes the manifest only after the upload, staged and atomically renamed:
a failed upload leaves the previous manifest in place. It expects an export
`neural.export` has already checked; uploading proves nothing about parity or
playing strength.

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

Every stored copy carries the size and SHA-256 its writer verified, as
`X-Mahjong-SHA256` and `Content-Length` headers, the same record the service
worker keeps for its own files. Status checks, the foreground availability
probe and retention read that record; they never read or hash the body. The
bytes given to ONNX Runtime are always read and hashed in full. The service
worker's status reply names its build's network and whether it is stored; the
page accepts that answer only when the identity exactly matches its own
manifest, and otherwise reads its own record. This proof comes from the app's
service-worker channel, not a localStorage flag.

## Cache lifetime and upgrades

The page downloads the network, never the service worker: a worker's install,
activation or message event is abandoned after a few minutes (five in Chrome)
and a half-received body is not kept, so 100 MB on a slow link never finished
and the update was retried, and failed, on every start. Installation and
activation fetch the game alone. The worker saves the few megabytes of
runtime; the page then fetches, verifies and stores the network in an
installation-scoped v2 cache and tells the worker it is saved. A player who
once asked for Trained offline has each later version's runtime and network
saved by the running page while the update waits, or by the update's own page
after it starts. A new version never plays the previous network. Normal
waiting-worker activation is retained: there is no forced mid-match upgrade.

Only the active worker prunes: at activation and after its page has saved its
network. The current and previous model URLs are retained; older scoped
entries are removed. A network that is not yet stored, a waiting/installing
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

The memory regression loads the published network through the reduced runtime
three times over. It needs a local copy named by `MAHJONG_NETWORK`, which
`node scripts/prepare-test-network.mjs <path>` fetches and verifies against the
manifest. CI always supplies it and fails without it; a local run without it
skips that one test.

The JavaScript in `onnxruntime-web` calls into the WASM, and those calls change
between releases, so `web/package.json` pins the package to exactly the ONNX
Runtime release the reduced WASM was built from. `copy-runtime.mjs` refuses to
build when the installed package differs from the release recorded in the
binary, so an update of that package alone can only fail. Dependabot still
proposes it, in a pull request of its own that fails until the runtime is
rebuilt: that pull request is the signal to dispatch the reduced ONNX runtime
workflow with the new release, merge the branch it pushes (which moves the pin
with the rebuilt WASM), and close Dependabot's. The build tools in
`web/runtime/requirements-build.txt` get Dependabot pull requests too, checked
by an install and `pip check`; the next rebuild runs them for real.

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
