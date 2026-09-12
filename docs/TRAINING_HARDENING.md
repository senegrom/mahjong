# Training hardening reconciled after PR #40

This follow-up is built from `main` at
`09b800ce71247a6cfdfc1fb60228d2943ee20fd8`, which includes the final PR #40
repairs. It supersedes the earlier stacked #41 implementation, not those
newer main-branch fixes. The reconciliation retains both histories without
force-pushing or reverting the merged safety work.

## Conflict-resolution decisions

- Keep `neural.checkpoints` as the single checkpoint implementation, including
  staged CPU validation and the newer cloud snapshot publication logic. Do not
  reintroduce the competing `neural.checkpoint` module.
- Keep the tile-specific fusion belief head and **main's exact migration of
  Adam moments and step counts**. Do not replace that migration with #41's
  earlier moment-reset implementation.
- Keep main's cloud invocation isolation, replay validation/recovery, current
  observation adapters, terminal tie rewards, and Mortal RNG resume ordering.
- Keep main's fixed-size minibatch contract and `optimizer_updates` /
  `checkpoint_generation` log fields. A short rollout still fails explicitly;
  the earlier #41 eager-remainder alternative is intentionally not carried over.
- Keep current-model network-only search baselines; only actual native search
  needs the engine-only layout restriction. No search-history approximation is
  introduced.

## Remaining changes carried forward

### Strict native actions and independent simulation randomness

`Arena.step` now validates every live action before changing any table. An
invalid later batch row cannot leave earlier games advanced. Illegal moves
raise `ValueError` rather than silently substituting a pass or first legal
move. Finished rows remain ignored; `strict=False` explicitly opts into legacy
lenient behavior and is not used by the trainers or evaluators.

Hypothetical sampling uses a separate per-table RNG from real dealing. The
benchmark no longer samples unused hands just to reproduce an old RNG side
effect. Extra simulation work alone must not change subsequent real hands.
This does not make policy-dependent game trajectories identical.

Rebuild and reinstall `riichi_py`: the affected Python entry points require
`TRAINING_API_VERSION == 2`. PPO logs/checkpoints record that version. Resuming
an older environment resets only smoothed/best benchmark history, keeping the
model, optimizer, generation and sampling state. Remeasure competing models
under the same version; old same-seed measurements are not interchangeable.
Existing checkpoint or replay files are not relabelled or deleted.

### Previous checkpoints without losing validation

The existing `checkpoints.atomic_save` still serializes, flushes, and safely
validates the new staged checkpoint before publishing. It additionally copies
the old live bytes into `<filename>.previous` using a separate atomic rename.
The backup is single-writer and local; it is not a distributed transaction or
a guarantee that an already-corrupt old checkpoint is valid. Recovery remains
an explicit operator choice, not silent rollback. `keep_previous=False` omits
the backup where a caller deliberately does not need it.

An interruption after either rename must not delete its destination. POSIX
directory fsync requests durability; Windows does not have the same portable
power-loss guarantee. Hard exits may leave harmless uniquely named partial
files. Cloud publication retains its existing independently validated copies,
archives and run-isolation behavior; local `.previous` files are not implicitly
restored from or published to the volume.

Redundant final PPO saves are removed, so a normal completed run keeps the
preceding saved generation rather than overwriting its backup with a duplicate
of the last one. A no-work invocation does not create a new learned checkpoint.

### Learner and legacy-distillation guards

The combined value baseline reads detached policy features; auxiliary losses
train their heads without reshaping either policy backbone. The belief head
and its checkpoint/optimizer migration remain main's implementation.

PPO entry points reject missing requested input checkpoints, invalid game or
measurement counts, malformed opponent/freeze configuration, and nonfinite
optimizer gradients. Singleton statistics remain finite. The standalone
78-action trainer rejects an incompatible 46-action learner before collection.

Legacy `distil.collect` now has a checked step budget and refuses unfinished
games. It shares the existing native-search layout guard, rejects partial
loading of combined/Mortal checkpoints, and uses the correct CPU device for
measurement. This script still trains action/hand labels, **not completed-game
value targets**; its documentation no longer implies otherwise.

## Validation and remaining work

Run unfiltered workspace tests/Clippy and the complete neural regression suite
with both native engines freshly built. Added regressions cover strict actions,
claim-window preservation, unchanged real deals despite extra imaginary-world
calls, checkpoint interruption on both sides of both renames, staged validation,
configuration/environment metadata, auxiliary gradients, and real small-model
training/resume with controlled collected batches. Keep every existing #40 test.

This follow-up does not claim stronger play or completion of AlphaZero-style
training. Modern information-consistent search, sound world weighting, search
improvement measurements, policy-distribution plus outcome learning, and a
candidate/champion opponent-population gate remain separate work. Regenerate
replay data known to contain prior incorrect rewards; preserve diagnostic copies.

The existing post-publication SIGINT regression now targets the live checkpoint
rename explicitly rather than the first rename (now the backup). Its assertion
that the new live generation survives is retained, and it additionally checks
the preceding generation and the exact set of surviving files. No regression
test is removed or skipped. Checkpoint validation retains main's generation-
returning API; Mortal resume retains its sampling_state payload field.
