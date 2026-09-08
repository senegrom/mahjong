# Reduced ONNX Runtime for the browser

The published opponent stays in **ONNX format** (`web/public/model.onnx`). The
size reduction is in the WebAssembly execution runtime, not the model format.

`web/runtime/` is built from matching ONNX Runtime **v1.29.0** source with
`--include_ops_by_config`. `reduced-ops.config` is generated directly from the
published ONNX model, so unused operator kernels are omitted while normal ONNX
loading remains available. The final runtime also disables unused ML, contrib and
generation operators. This deliberately does **not** use `--minimal_build`,
because ONNX Runtime's minimal Web build requires ORT-format models.

The current model needs 15 ONNX operators:

`Add`, `Cast`, `Concat`, `Constant`, `ConvInteger`, `DynamicQuantizeLinear`,
`Gather`, `InstanceNormalization`, `MatMulInteger`, `Mul`, `ReduceMean`, `Relu`,
`Reshape`, `Shape`, and `Unsqueeze`.

The measured production WASM size is **3,807,250 bytes**, down from
**13,961,845 bytes** for the generic `onnxruntime-web` runtime shipped by the
same 1.29 package: a **72.73% reduction**. The matching generated loader is
about 0.02 MB. The 2.4 MB ONNX model itself is unchanged.

The Vite build selects onnxruntime-web's external-WASM JavaScript entry. That
prevents the package's generic WASM from being emitted alongside the reduced
runtime. `copy-runtime.mjs` verifies `model.sha256` before copying the runtime;
changing the network without rebuilding its operator set therefore fails the
build instead of producing a broken Trained opponent.

The generated loader limits shared WASM memory to **192 MiB**, starting at
16 MiB and growing on demand. Its previous 4 GiB maximum could fail at runtime
initialization on iPhones even for Quick's 2.4 MB model: the reservation limit,
not the model download size, was too large. The loader's memory constructor,
reported heap maximum and growth ceiling all use the same limit. The unchanged
WASM binary accepts this smaller imported memory. Preserve this setting when
rebuilding with `onnxruntime_EMSCRIPTEN_SETTINGS` / `MAXIMUM_MEMORY=201326592`.
See the [upstream iOS report](https://github.com/microsoft/onnxruntime/issues/22086).

The worker serializes inference and releases the previous session before loading
a different model. Switching agents preserves the model captured by every pending
turn; cancelled queued requests are removed. Inputs and outputs are disposed after
every inference, and any runtime error discards the worker so Retry can initialize
ORT afresh in Play, Watch and Physical modes.

`runtime-memory.test.js` runs both published models through the actual production
runtime while rejecting shared-memory reservations above 192 MiB. It checks
repeated inference and model switching for stable outputs and heap headroom;
worker and client tests cover cancellation, session release and failure recovery.

The reduced-runtime verification ran the production model through the real
browser suite and passed **93 unit/session/cache tests and 102 browser checks**.
That includes two simultaneous Trained opponents sharing one model load, the
Vite development and production-preview workers, and a cold browser restart with
HTTP cache cleared and network access refused while real Trained opponents keep
playing offline. Source and built Home Screen icons were also revalidated.

The real-browser suite loads the published model through the reduced runtime,
including after an offline browser restart. `runtime-package.test.js` also fails
if a generic hashed ORT WASM appears in `dist/assets`, if the model hash differs,
or if the reduced runtime is not smaller than the package runtime.
