# Security and quality checks

The browser game and the Python training extension share a repository but have
separate dependency graphs. An npm audit alone does not cover the PyO3 crates
used by the training extension. `engine/libriichi` is also a standalone Cargo
workspace: the root audit is not a substitute for auditing its own lockfile.

## Reproduce the checks

From the repository root, run `cargo fmt --all --check`,
`cargo clippy --locked --workspace --all-targets -- -D warnings`,
`cargo test --locked --workspace`, `cargo audit`, and
`(cd engine/libriichi && cargo audit)`.
The CI audit tool is pinned to cargo-audit 0.22.2. The build workflow audits
the root graph on every change, and the neural workflow audits the standalone
graph on relevant changes and weekly, with no advisory exemptions.

Build the training extension with `cargo build --locked -p riichi-py` and run
`python3 engine/riichi-py/smoke-test.py`, as CI does on Linux; the script also
finds the library under its Windows and macOS names. This loads the real
extension, checks its observation and legality buffers, and exercises a batch
of legal moves. The Python binding requires Rust 1.83 or newer; CI uses Rust
1.98.1.

In `web/`, run `npm ci` and then `npm run verify`, the command CI runs: it
audits npm dependencies at `--audit-level=low`, builds the WASM engine, lints,
runs Svelte diagnostics (`npm run check`), builds the site and runs the unit,
browser and icon checks. Generated WASM, third-party runtime files, build
output and screenshots are not linted as handwritten application source.

## Dependency and workflow policy

Dependabot checks both Cargo workspaces, at `/` and `/engine/libriichi`, npm
at `/web`, GitHub Actions at `/`, and pip at `/.github/actions/python-engines`,
the exact CPU export and training tools CI installs. It ignores
`onnxruntime-web` and leaves `web/runtime/requirements-build.txt` alone: both
move only with a runtime rebuild (see [ONNX_RUNTIME.md](ONNX_RUNTIME.md)).
The root Python binding uses PyO3
0.29.2 and explicitly retains the previous GIL requirement. The vendored
observation engine has its own dependency versions and must pass its own
audit; a clean root audit makes no claim about that separate lockfile.

Actions are pinned to full upstream commit IDs. The JavaScript actions use
Node 24 internally; the application build still uses Node 22. wasm-pack is
pinned to 0.15.0 and installed with Cargo's locked dependency resolution,
instead of an old Node 20 installer requesting `latest`.

The Pages tar is packaged explicitly and uploaded with the pinned Node 24
artifact action: the upstream `upload-pages-artifact` v4 wrapper still embeds
a Node 20 uploader. Keep the artifact named `github-pages` with an
`artifact.tar` file containing the built site.

Build jobs have only `contents: read`, and checkouts do not persist credentials.
Only the deployment job receives Pages and OIDC write permissions. Deployment
requires the Rust, browser and browser-worker parity jobs, including the root
dependency audit, to succeed. The standalone observation-engine audit is
reported by the separate neural workflow; it is not a cross-workflow Pages
deployment dependency.

## Training and browser-export regressions

After installing both native Python engines and the CPU training/export
dependencies from `.github/workflows/neural.yml`, run:

```sh
python -m unittest discover -s neural/tests -v
```

The build workflow's browser-worker parity job, which Pages deployment
requires, exports tiny real networks and runs them through the production
`policy.worker.js` and the reduced WASM runtime after `npm ci` in `web/`:

```sh
python -m neural.tests.export_worker_fixture /tmp/policy-worker-fixture
node web/tests/policy-export-parity.mjs /tmp/policy-worker-fixture
```

Combined checkpoints include the freeze-mode generator, CPU Torch state and
available CUDA states. Restore happens after model, optimizer and opponent
construction. Legacy checkpoints advance the freeze schedule to their absolute
generation using the supplied seed and mode list, but warn that their missing
Torch stream cannot be reproduced. CPU tests compare uninterrupted training
with serialized, resumed training, including AdamW state and resulting weights.

The browser export contract is Mortal version 4: 1,012 observation planes,
34 tile positions and 46 actions, with named policy, value and hands outputs.
Engine-plane students and mismatched action heads are rejected before touching
the destination. A fusion checkpoint, which the published network is, is
exported whole: Mortal, our network and the head joining them, with the
legality mask as a second `legal` input. Its half alone is refused, and so is
a fusion without `--float32`, since int8 weights blunt its value head.
The exporter checks real positions
from deterministic native-engine play rather than random binary inputs.
All three heads must be finite, have the expected shape and have mean absolute
error at most 10% of the reference standard deviation (with a 0.01 scale floor).
Best legal actions must agree on at least 90% of non-forced test decisions.
These are export acceptance thresholds, not playing-strength measurements.
Missing runtime-operator metadata fails closed unless the explicit
`--allow-any-operator` development option is supplied; that option never bypasses
the observation/action contract or numerical checks. A validated graph replaces
the destination atomically. Tests cover failed-export preservation, successful
replacement, every non-finite head, legal masks, and real ONNX export/inference.

## Test-server regression coverage

The tile-effect fixture server is a loopback-only test tool, not part of the
published game. It reads files before sending success headers, uses explicit
400/403/404/405/500 statuses, and constrains both lexical and symlink-resolved
paths to the intended fixture root. Tests cover successful index/script/tile
loads, missing files, malformed encoding, null bytes, traversal, escaping
symlinks, HEAD requests and unsupported methods.

A successful CodeQL workflow means the scan ran; it does not by itself prove
that every dashboard alert is closed. Review alert records separately rather
than dismissing alerts simply to clear a counter.
