# Historical evidence

Use the README and the other documents in this directory for current operating
instructions. The following immutable snapshots preserve earlier measurements,
design decisions, and test reports without presenting them as current claims:

- [Training narrative and benchmarks before the September 2026 cleanup](https://github.com/senegrom/mahjong/blob/a920079f77285602db824175f0e4b42e123c2349/README.md)
- [Earlier reduced-runtime measurements and memory limits](https://github.com/senegrom/mahjong/blob/a920079f77285602db824175f0e4b42e123c2349/docs/ONNX_RUNTIME.md)
- [Earlier mode implementation and regression report](https://github.com/senegrom/mahjong/blob/a920079f77285602db824175f0e4b42e123c2349/docs/AGENT_MODES.md)
- [The project plan with its status of 6 September 2026](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/PLAN.md):
  the status narrative, the milestone table and the planning of the
  architecture, web app, AI opponents, evaluation and compute. PLAN.md now
  keeps only its rules and engine reference. Much of the rest was planned
  and never built, or has been superseded: the status table's 192 by 10
  int8 network (the manifest names a float32 fusion of 116 MB), WebGPU
  (the policy worker runs onnxruntime-web's WASM backend), the Strong and
  Expert tiers (the app has Beginner, Club and Trained), and a
  `docs/RULES_DECISIONS.md` that was never written. The search of milestone
  M4 measured no better than the plain policy in every variant tried, and
  it was removed in September 2026, with its imagined worlds and the reader
  trained on them.

Validate a numerical claim in any of these against the commit, model
identity and experiment it describes before treating it as a present-day
result.

## Review reports

Reports of review repairs, removed on 28 September 2026 once what they still
said truly had moved into the living documents, chiefly
[TRAINING_SAFETY.md](TRAINING_SAFETY.md), [ONNX_RUNTIME.md](ONNX_RUNTIME.md),
[AGENT_MODES.md](AGENT_MODES.md) and [UI_RELIABILITY.md](UI_RELIABILITY.md).
Several describe code removed since: the search and its imagined worlds, the
reader of hidden hands, a candidate/champion gate, and red-five aliases.

- [TRAINING_HARDENING.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/TRAINING_HARDENING.md):
  training hardening reconciled after PR #40: strict native actions,
  `.previous` checkpoints and learner guards.
- [TRAINING_REVIEW_REPAIRS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/TRAINING_REVIEW_REPAIRS.md):
  the training review repairs of 12 September 2026: cloud opponents,
  auxiliary learning, schemas and exploration bookkeeping.
- [REVIEW_BOUNDARY_CONTRACTS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/REVIEW_BOUNDARY_CONTRACTS.md):
  physical-table claim matching and the reader's declarations in checkpoints.
- [REVIEW_FIXES_2026_09.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/REVIEW_FIXES_2026_09.md):
  physical wall accounting, deal seeds independent of the round size,
  unsaved-play identity and CI's real-network memory test.
- [REVIEW_MODEL_DELIVERY.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/REVIEW_MODEL_DELIVERY.md):
  verified network transfer, the publisher, scoring explanations and
  saved-result compatibility.
- [EVALUATION_RELIABILITY.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/EVALUATION_RELIABILITY.md):
  bot ownership in the native arena, pooled tie rewards and interrupted
  replay publication.
- [TRAINING_CORRECTNESS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/TRAINING_CORRECTNESS.md):
  completed-game rewards, ties, incomplete games and replay publication.
- [POLICY_SEARCH_PARITY.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/POLICY_SEARCH_PARITY.md):
  the acting policy's precision and the browser worker parity fixture.

## Merged documents

Documents whose current content now lives in another one; the link is the
last version of each as a file of its own.

- [TRAINING_CONTROLS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/TRAINING_CONTROLS.md):
  bounded updates on training API 2 and the review findings they answered
  (opponent path collisions, forced BF16 recording, abandoned prefetchers,
  padded baseline batches). The policy update controls are in
  [TRAINING_SAFETY.md](TRAINING_SAFETY.md).
- [CUSTOM_OPPONENTS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/CUSTOM_OPPONENTS.md):
  custom tables, now a section of [AGENT_MODES.md](AGENT_MODES.md). It said
  saves of formats 1 to 3 replay recorded neural answers; any save from before
  format 5 holding a trained move is refused.
- [HAND_RESULTS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/HAND_RESULTS.md)
  and [DISCARD_HINTS.md](https://github.com/senegrom/mahjong/blob/b34d169ad6341b273866970b93a9793ffef8c2fb/docs/DISCARD_HINTS.md):
  hand results, waits, mjai accounting and discard previews, now sections of
  [UI.md](UI.md). HAND_RESULTS.md called save format 3 current; it is 6.
