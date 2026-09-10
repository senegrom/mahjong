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
The CI audit tool is pinned to cargo-audit 0.22.2. The neural workflow audits
both graphs on relevant changes and weekly, with no advisory exemptions.

On the Linux CI runner, build the training extension with
`cargo build --locked -p riichi-py` and run
`python3 engine/riichi-py/smoke-test.py`. This loads the real extension, checks
its observation and legality buffers, and exercises a batch of legal moves.
The Python binding requires Rust 1.83 or newer; CI uses Rust 1.98.1.

In `web/`, run:

```sh
npm ci
npm run wasm
npm run lint
npm run check
npm run build
npm run test:unit
npm run test:browser
npm audit --audit-level=low
node scripts/check-icons.mjs
node scripts/check-icons.mjs --built
```

`npm run check` now runs Svelte diagnostics. The former interactive-play
checker is retained as `npm run check:play`. `npm run check:all` includes
both static checks and the existing interactive checks. Generated WASM,
third-party runtime files, build output and screenshots are not linted as
handwritten application source.

## Dependency and workflow policy

Dependabot checks both Cargo workspaces, at `/` and `/engine/libriichi`, npm
at `/web`, and GitHub Actions at `/`. The root Python binding uses PyO3
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
requires the Rust and browser verification jobs, including dependency audits,
to succeed. The standalone training audit is reported by the separate neural
workflow; it is not a cross-workflow Pages deployment dependency.

## Training and browser-export regressions

After installing both native Python engines and the CPU training/export
dependencies from `.github/workflows/neural.yml`, run:

```sh
python -m unittest discover -s neural/tests -v
```

Combined checkpoints include the freeze-mode generator, CPU Torch state and
available CUDA states. Restore happens after model, optimizer and opponent
construction. Legacy checkpoints advance the freeze schedule to their absolute
generation using the supplied seed and mode list, but warn that their missing
Torch stream cannot be reproduced. CPU tests compare uninterrupted training
with serialized, resumed training, including AdamW state and resulting weights.

The browser export contract is Mortal version 4: 1,012 observation planes,
34 tile positions and 46 actions, with named policy, value and hands outputs.
Engine-plane students, mismatched action heads and fusion checkpoints are
rejected before touching the destination. The exporter checks real positions
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
