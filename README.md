# Riichi

[Play Riichi Mahjong](https://senegrom.github.io/mahjong/) against Beginner,
Club, Trained, or a custom mix of opponents. The shared Rust rules engine
implements the EMA Riichi Competition Rules, 2025 edition, and is compiled
for the browser and Python training tools.

The app includes normal play, Agent watch, a physical-table editor, a guided
physical game, hand review, learning aids, and mjai log export. All approved
Classic, Matisse, Dalí, and Van Gogh artwork remains available. See the
[mode guide](docs/AGENT_MODES.md) and [scoring guide](docs/GUIDED_SCORING.md).

## Run and verify

Install Rust, the `wasm32-unknown-unknown` target, wasm-pack, Node satisfying
`web/package.json`, and Chrome/Chromium. CI pins its tools in
[the build workflow](.github/workflows/ci.yml).

```sh
cd web
npm ci
npm run wasm
npm run dev                 # VITE_LAN=1 exposes the dev server to your LAN
npm run verify              # the complete web verification command used by CI
```

`verify` audits dependencies, builds WebAssembly, runs JavaScript lint and
Svelte diagnostics, builds production assets, and runs unit, browser, offline,
and icon checks. Set `CHROME_BIN` when Chrome is not at the test scripts'
standard Linux paths. No separate dev server is required. `check:all` is an
alias for `verify`; `test:unit` and `test:browser` are useful after a build.

From the repository root, `./check.sh` runs engine formatting, lint, tests,
the Python binding smoke test, arena games, and randomized legal play.
`./check.sh --web` also runs web verification. Rust dependency auditing is a
separate CI step. The older `npm run diagnose:live` scripts inspect an already
running site; they are manual diagnostics, not a substitute for verification.

## Offline play and the trained model

The game and every playable tile graphic are prepared automatically. Selecting
Trained additionally prepares its runtime and the single published network.
The app distinguishes a working resident model from a model durably saved for
an offline restart. Check the displayed offline status before disconnecting.

The authoritative model identity, precision, sizes, and download location are
in [model-manifest.js](web/src/lib/model-manifest.js), not copied benchmark
numbers in this README. The model is remote; do not export ONNX into
`web/public`. See [runtime and delivery](docs/ONNX_RUNTIME.md).

Production copies only runtime artwork from the art workspace. Source crops,
provenance, and design galleries stay in the repository, but duplicate PNGs
and design-only files do not enter the mandatory offline download. Approved
SVG files are copied without recompression or visual changes. Each build
reports core, runtime, remote-model, and excluded-workspace byte totals.

## Engine and training

The `engine/` workspace supplies the CLI, browser, and Python bindings.
`neural/` contains self-play, evaluation, export, checkpoint management, and
cloud training. Training and search behavior are unchanged by web packaging.
See [training controls](docs/TRAINING_CONTROLS.md),
[training safety](docs/TRAINING_SAFETY.md), and
[search replay learning](docs/SEARCH_REPLAY_LEARNER.md).

Dated benchmark narratives and earlier implementation reports are indexed in
[historical evidence](docs/HISTORY.md). They describe the recorded revision,
not necessarily the currently published network or current operating steps.

## Licence

See [LICENSE](LICENSE), the vendored engine notices, and the asset licences.
