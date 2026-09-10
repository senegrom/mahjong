# Reduced ONNX Runtime for the browser

The published opponent stays in **ONNX format** (`web/public/model-full.onnx`).
The size reduction is in the WebAssembly execution runtime, not the model
format.

`web/runtime/` is built from matching ONNX Runtime **v1.29.0** source with
`--include_ops_by_config`. `reduced-ops.config` is generated directly from the
published ONNX model, so unused operator kernels are omitted while normal ONNX
loading remains available. The final runtime also disables unused ML, contrib and
generation operators. This deliberately does **not** use `--minimal_build`,
because ONNX Runtime's minimal Web build requires ORT-format models.

The current model needs 17 ONNX operators:

`Add`, `Cast`, `Concat`, `Constant`, `ConvInteger`, `DynamicQuantizeLinear`,
`Gather`, `InstanceNormalization`, `MatMulInteger`, `Mul`, `ReduceMax`,
`ReduceMean`, `Relu`, `Reshape`, `Shape`, `Sigmoid`, and `Unsqueeze`.

`ReduceMax` and `Sigmoid` are channel attention's, which every block of the
tower uses. The graph has three outputs — `policy`, `value` and `hands` —
because the page reads all three: the move, what the hand is worth, and what
each opponent is taken to be holding.

The measured production WASM size is **3,807,250 bytes**, down from
**13,961,845 bytes** for the generic `onnxruntime-web` runtime shipped by the
same 1.29 package: a **72.73% reduction**. The matching generated loader is
about 0.02 MB. The 17.6 MB ONNX model itself is unchanged.

The Vite build selects onnxruntime-web's external-WASM JavaScript entry. That
prevents the package's generic WASM from being emitted alongside the reduced
runtime. `copy-runtime.mjs` verifies `models.sha256` before copying the runtime;
changing the network without rebuilding its operator set therefore fails the
build instead of producing a broken Trained opponent.

The generated loader initially limits shared WASM memory to **192 MiB**, starting at
16 MiB and growing on demand. Its previous 4 GiB maximum could fail at runtime
initialization on iPhones even for a 2.4 MB model: the reservation limit,
not the model download size, was too large. The loader's memory constructor,
reported heap maximum and growth ceiling all use the same limit. The unchanged
WASM binary accepts this smaller imported memory.
See the [upstream iOS report](https://github.com/microsoft/onnxruntime/issues/22086).

`memory-budget.js`, copied alongside the loader as `memory-budget.mjs`, supplies
the constructor, heap maximum and growth hook. Preserve these hooks when
rebuilding the loader. If an allocation exceeds the application's ceiling, the
page releases the old worker and retries pending decisions in a fresh worker at
**256 MiB**, then **384 MiB** if needed. These are maximum reservations; actual
memory still starts at 16 MiB and grows on demand. Requests preserve their original
observations, selected models and cancellation signals across retries. No larger
reservation is attempted when the browser itself refuses memory, when the request
exceeds 384 MiB, or for an unclassified runtime error. Those memory failures start
the next runtime back at 192 MiB; any other failure, a timeout or a failed
answer, keeps the ceiling that was found to work, so a retry does not repeat the
whole fail-and-grow cycle. If normal heap growth's spare capacity is refused, the
allocator tries only the exact pages needed before reporting failure.

The worker serializes inference and keeps the page's one network loaded between
decisions, so a table does not reload it on every change of turn: measured on a
desktop, that reload cost a table most of a second at the ninetieth percentile.
Every request captures its own model and cancellation signal, so a match that
ends drops its own turn and no more; cancelled queued requests are removed. Inputs and outputs are disposed after
every inference, and any runtime error discards the worker so Retry can initialize
ORT afresh in Play, Watch and Physical modes.

`runtime-memory.test.js` runs both published models through the actual production
runtime while rejecting shared-memory reservations above 192 MiB. It checks
repeated inference and model switching for stable outputs and heap headroom;
worker and client tests cover cancellation, session release and failure recovery.
`memory-budget.test.js` also exercises an actual 200 MiB allocation through the
shipped WASM: it reports the 192 MiB ceiling and succeeds in a fresh 256 MiB
runtime, while a simulated browser reservation refusal stops expansion. Client
tests verify transferred observation replay and bounded retries.

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
