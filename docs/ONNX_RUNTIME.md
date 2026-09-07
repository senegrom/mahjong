# Reduced ONNX Runtime for the browser

The published opponent stays in **ONNX format** (`web/public/model.onnx`). The
size reduction is in the WebAssembly execution runtime, not the model format.

`web/runtime/` is built from matching ONNX Runtime **v1.29.0** source with
`--include_ops_by_config`. `reduced-ops.config` is generated directly from the
published ONNX model, so unused operator kernels are omitted while normal ONNX
loading remains available. This deliberately does **not** use `--minimal_build`,
because ONNX Runtime's minimal Web build requires ORT-format models.

The Vite build selects onnxruntime-web's external-WASM JavaScript entry. That
prevents the package's generic WASM from being emitted alongside the reduced
runtime. `copy-runtime.mjs` verifies `model.sha256` before copying the runtime;
changing the network without rebuilding its operator set therefore fails the
build instead of producing a broken Trained opponent.

The real-browser suite loads the published model through the reduced runtime,
including after an offline browser restart. `runtime-package.test.js` also fails
if a generic hashed ORT WASM appears in `dist/assets`, if the model hash differs,
or if the reduced runtime is not smaller than the package runtime.
