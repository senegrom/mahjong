# Security and quality checks

The browser game and the Python training extension share a repository but have
separate dependency graphs. An npm audit alone does not cover the PyO3 crates
used by the training extension.

## Reproduce the checks

From the repository root, run `cargo fmt --all --check`,
`cargo clippy --locked --workspace --all-targets -- -D warnings`,
`cargo test --locked --workspace`, and `cargo audit`.
The CI audit tool is pinned to cargo-audit 0.22.2.

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

Dependabot checks the Cargo workspace at `/`, npm at `/web`, and GitHub
Actions at `/`. PyO3 is upgraded from 0.23.5 to the patched 0.29.2 series;
the bindings use the current interpreter-detachment API and explicitly retain
the previous GIL requirement. This avoids remaining on a vulnerable release
or upgrading into another affected release. Relevant upstream advisories are
RUSTSEC-2025-0020, RUSTSEC-2026-0176 and RUSTSEC-2026-0177.

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
to succeed.

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
