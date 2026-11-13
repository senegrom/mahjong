# Trained model and reduced runtime

## Source of truth

`web/src/lib/model-manifest.js` lists the published remote networks a player
can pick, the default first, each by its version with its source, generation,
the reward it learned from, precision, decoded byte count, stored byte count,
origin, immutable object key, and SHA-256; `MANIFEST` is the default. It lists
0.1.3000, the default, published on 8 November 2026 and again on 11 November
in the export free of Softplus (below), the same weights and the one entry
with `webgpu: true`, then 0.1.2610 (the default from 30 October), 0.1.1410
(from 21 October), 0.1.810 and 0.1.410, which the site played before it.
0.1.1110, the default from 19 October, was dropped from the list on 9
November. Inspect those fields for the current release.

The reward is what the network's value estimates, and the page names and
explains the value by the reward of the network it plays: `{ "kind": "hybrid" }`
for 810 and 410, a hand's points over 4,000 plus the game's placement bonus,
and `{ "kind": "placement", "alpha": 0.05 }` for 3000, 2610 and 1410, the
bonus the scoreboard promises at the start of the next hand, or after the
game's last hand the bonus itself, plus alpha times the hand's points over
4,000 (`neural/train_combined.py --reward`, `neural/place_model.py`). The page
still words alpha 0, the reward 1110 learned, should a network with it be
listed again.

`neural.export` checks the graph's operators; the publication tool
`web/scripts/publish-model-r2.mjs` writes the delivery manifest, reading the
precision from the exported weights and refusing to run without the source
checkpoint's name or the reward the network learned from, given as its trainer
was given it: `--reward hybrid`, or `--reward placement --alpha <weight>`.
Every network it keeps listed keeps its reward, and it does not write over a
manifest that lists a network without one. It uploads with an exact wrangler
release, gzipped, under `models/g<generation>/<sha256>`, which
`workers/model-cdn` serves to the page, and writes the manifest only after the
upload, staged and atomically renamed: a failed upload leaves the previous
manifest in place. The network it publishes becomes the default and every
network listed before it stays listed; its version is `0.1.<generation>`
unless `--version` names another. It expects an export `neural.export` has
already checked; uploading proves nothing about parity or playing strength.

No ONNX model is served from `web/public`. Both packaging and offline inventory
creation reject unexpected local ONNX files, including nested files.

The packaged runtime is listed in `model-package.js`: the reduced WASM,
patched JavaScript loader, and shared memory-budget module. `copy-runtime.mjs`
checks all sources before copying anything. Vite selects ONNX Runtime's
external-WASM entry so it does not emit a second generic WASM binary.

## WebGPU on desktops

On a desktop with a graphics card a review's batches, and nothing else, can
run through ONNX Runtime Web's WebGPU provider. Play never does: every single
decision goes to the WebAssembly worker (`policy.worker.js`), which is
unchanged, and so does every batch until the card has shown it answers as
that worker does, faster. The card's runtime is onnxruntime-web's own default
build, `ort.min.mjs`, whose WebGPU provider runs on JSEP, with
`ort-wasm-simd-threaded.jsep.mjs` and `.jsep.wasm` (29.5 MB), unchanged (the
package's `ort.webgpu.min.mjs` is the other WebGPU provider, built with
Asyncify, which was not measured here). `copy-runtime.mjs` copies them from
the installed package to a folder named by their contents,
`ort-webgpu/<hash>/` (`WEBGPU_RUNTIME_FILES`), and only the card's worker
(`gpu-batch.worker.js`) loads them.

The WebGPU provider has no Softplus kernel, and Mish exported as it is,
x * tanh(softplus(x)), is a Softplus; each of the 163 in Mortal's encoder ran
on the processor, which held a batch of 64 to 36 positions a second.
`neural.export` writes Mish as `ExportMish`, x * s(1 + t) / (1 + t^2) with
s = sigmoid(x) and t = 1 - s, exact in real arithmetic, from Sigmoid, Sub,
Add, Mul and Div, which both runtimes carry; it refuses a graph that still
holds a Softplus and says `activations: free of Softplus` when none is left.
Its parity check still compares the graph with the network's own Mish. One
export serves both runtimes; the reduced runtime's operator list is
unchanged, and keeps Softplus and Tanh for the networks exported before. On
WebAssembly the new form is about a tenth faster.

The publication tool reads the graph's operators from the export and writes
`"webgpu": true` to the network's manifest entry only when every one is an
operator the WebGPU provider has a kernel for (`WEBGPU_KERNELS`, its
`WEBGPU_OP_RESOLVE_RULES`, pinned, and checked against the installed runtime
by a test), or one that needs none: `Reshape`, `Squeeze`, `Unsqueeze`,
`Flatten` and `Identity`, which only relabel a shape, and `Shape`, `Constant`
and `ConstantOfShape`, which ONNX Runtime computes on the processor by design
(`WEBGPU_WITHOUT_KERNEL`). Anything else, a Softplus or an int8 export's
`DynamicQuantizeLinear`, `MatMulInteger` and `ConvInteger`, writes false. A
network listed without the field never runs on the card.

The page's card (`src/lib/gpu-batches.js`) is untried, starting, ready or
declined:

- It is considered only where the played network's entry says
  `webgpu: true`, the address has no `?gpu=off`, the browser has
  `navigator.gpu`, and the device is no phone or tablet
  (`userAgentData.mobile` or an Android or iOS platform, else the user
  agent, an iPad asking as a Mac by its touch points), and where no decline
  is remembered.
- Its bring-up never delays anything. It starts once WebAssembly has
  answered the page's second batch of two or more positions (the first may
  carry the network's load or a wait behind a move), and that batch, its
  answers and WebAssembly's time for it are the sample. The card's worker
  asks for an adapter and refuses none, a fallback one or a software one
  (SwiftShader, llvmpipe, Microsoft's basic renderer); reads the network
  from this page's Cache Storage, checked against its size and SHA-256, and
  refuses one not saved there, never downloading it; fetches the runtime;
  and opens the session. It then runs the timed batch, the sample's
  positions again and again to sixteen (`TIMED_POSITIONS`, `timedOrder`),
  once to warm up and three times more (`TIMED_RUNS`). Sixteen is about a
  hand's decisions, which is what a review asks of the card in one pass. A
  review asks WebAssembly eight positions at most, and the sample is often
  two or three, the end of a hand, while WebAssembly's time grows with every
  position (about 120 ms each on a desktop) and the card's hardly does (70
  to 150 ms a batch, of two or of sixty-four): timed on a sample of two, a
  GeForce RTX 5070 Ti read as too slow for the reviews it runs a hundred
  times faster.
- Every run must agree with WebAssembly: each move's weight within 0.001 of
  WebAssembly's (`WEIGHT_BOUND`; the largest difference measured on a
  GeForce RTX 5070 Ti over 368 positions was 6.5e-6), and the same best move
  unless WebAssembly's own two best are closer than 0.002. The runs' median
  must be at most half of what WebAssembly takes for as many positions
  (`SPEEDUP`), at the least time a position it has shown, over the sample
  and the batches it answers while the card is brought up: a batch that
  waited its turn in the worker behind another request took longer through
  no fault of WebAssembly's, and its time alone, taken eight times over,
  would let a slow card pass. The whole bring-up has 60 seconds
  (`BRING_UP_MS`). Anything else declines the card and ends its worker.
- Ready, the card answers every batch of two or more positions, one at a
  time, each within 10 seconds (`CARD_BATCH_MS`). A batch that fails there or
  runs out of time declines the card, ends its worker, and is asked of
  WebAssembly through the usual path, so its caller gets an answer or its own
  abort. A caller who stops waiting is not a failure of the card, which runs
  the batch to its end; a batch that then runs out of time is, and declines
  the card as it would with its caller waiting. `resetPolicy` drops the
  callers waiting on the card with its reason, as it drops the WebAssembly
  worker's, and keeps the card's worker.
- A decline about the device (no adapter or a software one, disagreeing,
  slow, a session or a run that failed, a bring-up out of time) is kept in
  localStorage under `mahjong-webgpu-declined:<page folder>`, with the
  runtime folder, the network's SHA-256 and the date: a later visit does not
  try again until a new runtime or a new network comes or a week has passed
  (`DECLINE_LASTS_MS`), and `?gpu=on` forgets it. The week is for a decline
  whose reason passes, another program at work on the card that minute or a
  driver since replaced; trying again costs a device without an adapter one
  question, and any other a bring-up beside the review. A runtime the
  browser could not fetch, offline before it was ever saved, or a network
  not saved on this device, declines the card for this visit only. Storage
  that refuses changes nothing but that. Why the card was declined is said
  in the console once.
- While the card runs a batch, in the bring-up or after, the page's root
  carries `data-card-at-work` (`CARD_AT_WORK`), and under it a dora's sheen
  and the white dragon's reveal hold still (`Tile.svelte`), where they are,
  and go on from there when the batch is answered, a tenth of a second or so
  later. Both move by being painted again, and each frame of either has the
  browser paint the part of the page around it again, every tile's picture
  in it, on the one thread of its graphics process that also runs the card's
  work. With six dora in view in a review that thread was busy for all of
  two seconds traced, the page drew 44 frames a second, and a batch of eight
  took the card 4.5 s, longer than WebAssembly takes; with the sheens still
  it took 70 ms (Chrome on a desktop, 12 November 2026).

`policyRuntime()` in `gpu-batches.js` says 'webgpu' once the card is ready
and 'wasm' otherwise. While it says 'webgpu' a review asks up to 64 positions
in one forward pass (`GPU_BATCH`, `batchMostFor` in `game-review.js`), and
the reasoning about a decision up to 64 tables, instead of eight; the time a
position measured still bounds every pass to about ten seconds, and the
gates, aborts and deadlines are unchanged. The offline panel's Trained AI
line then adds "GPU". The network and its moves are the same; the card's
float arithmetic differs in the last bits, so a near tie can flip. The
reduced runtime's memory ceilings do not apply to the stock WebGPU runtime,
which only desktops load, beside the WebAssembly worker.

`scripts/webgpu-check.mjs` runs both workers of the built site on real
positions, where Chrome gives a real adapter (it skips elsewhere, CI
included), given a network exported without Softplus in
`MAHJONG_WEBGPU_NETWORK`: a first batch and the sample on WebAssembly, the
card's bring-up through the page's own gates, then every position on both.

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
inference tensors are disposed after use. A request's deadline starts when the
worker begins it, which the worker says with a `ready` note, so its wait behind
the requests ahead of it is not counted; one the worker never begins is ended
after its own deadline and those of the requests still ahead of it, counted
again from each answer. The whole-game review asks up to eight positions in one
request (64 once batches run on the graphics card), fewer on a slow device, their planes and masks laid end to end with a
`count`; the worker runs them as one batch and answers each position's best
legal move and weights, and the request's deadline grows with the batch.

Trained AI averages over the suits (`src/lib/suits.js`): a question whose
answer puts under 0.9 on its best move is asked again, with the engine's
planes of the game with characters and circles renamed and the mask's moves
renamed, in a request of its own after the first, and the page plays the
mean of the two answers' logits, worked out from their weights. The worker, its
messages, deadlines and retries are unchanged: every request is still one
forward pass, and a review's batch of renamed positions is never larger
than the batch before it, so the memory ceilings hold. The worker parity
job checks the page's mean against `neural.suits` on a small export.

The last hand's two rules live in the page too (`src/lib/trained-move.js`),
not in the worker: they read the engine's codes and play over the network's
answers, asking nothing more but the tile of a riichi that a declined
self-draw leaves. A Trained opponent's riichi tile is now asked of a preview
of its reach, before anything is recorded, with the same planes as before,
so the worker sees the same requests. The worker parity job holds the page's
rules to the Python wrappers as well.

Every stored copy carries the size and SHA-256 its writer verified, as
`X-Mahjong-SHA256` and `Content-Length` headers, the same record the service
worker keeps for its own files. Status checks, the foreground availability
probe and retention read that record; they never read or hash the body. The
bytes given to ONNX Runtime are always read and hashed in full. Each page
names its player's pick in its messages; the service worker's status reply
names the network that pick resolves to in its build's list and whether it is
stored, and the page accepts that answer only when the identity exactly
matches the network it plays, and otherwise reads its own record. This proof
comes from the app's service-worker channel, not a localStorage flag.

## Cache lifetime and upgrades

The page downloads the network, never the service worker: a worker's install,
activation or message event is abandoned after a few minutes (five in Chrome)
and a half-received body is not kept, so 100 MB on a slow link never finished
and the update was retried, and failed, on every start. Installation and
activation fetch the game alone. The worker saves the few megabytes of
runtime; the page then fetches, verifies and stores the network in an
installation-scoped v2 cache and tells the worker it is saved. A player who
once asked for Trained offline has each later version's runtime and network
saved by the running page while the update waits, after the page's own
network and never beside another network download, or else by the update's
own page after it starts, once per visit while connected. The update's
network is the one its page will play for this player: the waiting worker
reads the pick against its own list, and answers with its default when the
player picked nothing or a version it no longer lists. The updates that
made 1110, 1410, 2610 and 3000 the default are the exceptions: their pages
drop every pick saved before them (settings record versions 3, 4, 5 and 6),
which their waiting workers cannot tell from the running page's message, so
for a player who had picked an older network (before 1110, 0.1.410, or
0.1.810 picked before #139 made picking the default no pick; before 1410,
0.1.810 or 0.1.410; before 2610, 0.1.1110, 0.1.810 or 0.1.410; before 3000,
0.1.1410, 0.1.1110, 0.1.810 or 0.1.410) nothing new is saved ahead, and the
update's own page saves the new default once it runs, while connected. A new
version never plays a network it does not list, even one an earlier version
left saved.
Normal waiting-worker activation is retained: there is no forced mid-match
upgrade.

Only the active worker prunes: at activation and after its page has saved its
network. Every model its build lists, and the current and previous model
URLs, are retained; older scoped entries are removed. A network that is not
yet stored, a waiting/installing worker, or an unreadable retention record
prevents pruning.

The unscoped legacy v1 cache is no longer read: it could only ever hold
generation 36. It is deliberately not blanket-deleted either: its entries may
still be needed by another installation on the same origin. Other installation
scopes and unrelated caches are never pruned.

## Build and test

Run `npm ci` and `npm run verify` in `web/`; CI runs the same checks in
parallel jobs. It includes real inference, legacy preference migration,
saved-state, memory, worker, offline, and browser layout checks. Optional
external model/checkpoint fixtures remain explicitly identified by their
individual tests.

The memory regression loads the default network through the reduced runtime
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
hooks and runs the web suites against the default network, then pushes the
tested tree to an `automation/reduced-runtime-*` branch and links a compare
page from the job summary: GitHub Actions may not open pull requests in this
repository, and one opened with the workflow token would start no checks. Open
the pull request from that link and merge only after its checks pass. The
workflow never pushes main.

`neural.export` refuses a network that needs an operator the reduced runtime
lacks, and the memory regression loads the default network through that
runtime, so a missing kernel fails before release.

See [historical evidence](HISTORY.md) for superseded size measurements and
previous memory ceilings.
