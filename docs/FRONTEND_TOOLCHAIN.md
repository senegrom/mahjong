# Frontend toolchain

The compatible frontend set is Svelte 5.57, `@sveltejs/vite-plugin-svelte`
7.3 and Vite 8.2. Exact versions and integrity hashes are in
`web/package-lock.json`; use `npm ci` for a checkout. Svelte was already locked
to 5.57.0 before this migration; its manifest minimum now reflects that.

Use Node 22.13+ in the 22.x line, or Node 24+. Node 20.19+ in the 20.x line
also satisfies the declared package requirements. CI uses Node 22. TypeScript
stays at 5.9.3 because the installed `svelte-check` does not accept TypeScript 7
as a drop-in replacement.

After building the Rust WebAssembly package:

```sh
cd web
npm ci
npm run wasm
npm run lint
npm run check
npm audit --audit-level=low
npm run build
npm run test:unit
npm run test:browser
```

The browser suites serve the production files beneath `/mahjong/`, exercise
saved games, mixed controllers and the actual shipped ONNX model, and check
mobile layouts and tile effects. They also run `npm run test:toolchain`, which
starts Vite development and preview servers, plays real neural-opponent turns,
checks saved-match reloads, and exercises Svelte hot reload in a disposable
fixture. Set `CHROME_BIN` when Chrome/Chromium is not in a standard Linux path.
Reports and screenshots are written to `web/test-results/`.

The explicit `base: './'` and `build.target: 'es2022'` are retained. Vite 8's
changed default browser target therefore does not silently replace this app's
configured JavaScript target. Production CSS and worker behaviour are covered
by the browser tests rather than relying on a successful compilation alone.

Dependabot groups Svelte, its Vite plugin and Vite into one version-update PR,
including major upgrades. Their peer dependencies must be reviewed together;
do not use `--force` or `--legacy-peer-deps` to bypass incompatibilities.
