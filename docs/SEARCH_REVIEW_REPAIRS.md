# Search, recording and saved-game review repairs

These repairs preserve the current training reward (`reward_version = 1`):

```
root hand's score change / 4000 + final placement bonus
```

## One objective at every search depth

A hybrid-return critic is valid only while the search is still inside its root
hand. Adding that critic at a later hand's first decision also adds the later
hand's expected score change, which is not in the original decision's target.

Search now banks the root hand's score change exactly once. A world that crosses
that boundary continues under the rollout policy to the end of the match and
adds **only terminal placement**, including tied placements and dealer repeats.
The club and network rollout paths follow this same rule. Negative network depth
bypasses the root-hand critic and obtains terminal placement by playing on.

This is compatible with existing checkpoints, but crossing a hand boundary can
be substantially more expensive than the previous incorrect bootstrap. It is
not a claim that the rollout policy estimates optimal continuation. A separately
trained, versioned placement-only head now replaces the terminal rollout on
request: `neural.placement` fits one to the placement part of self-play returns
(`neural.selfplay --out round.pt` keeps a round's positions and labels), and
`neural.searched --valued-by placement --placement-head head.pt` stops every
world at the hand boundary (`boundary` on `lookahead_begin`/`leaves_from`,
`SEARCH_API_VERSION = 2`), banks the root hand, and asks the head where the
standings lead from the searching player's first decision of the next hand. The
head records a fingerprint of the features it was fitted to and is refused
beside any other network. An existing hybrid head cannot do this merely by
changing its label.

Rebuild `riichi-py`: search now requires `SEARCH_API_VERSION = 3` (see
`REVIEW_WORLD_INTEGRITY.md` for independent-world recording format 3), independently of the
unchanged training API. Recordings carry `search_backup_version = 2`. Earlier
recordings remain available for diagnostic inspection but are refused as targets
by `Recorded` and therefore the sibling-head and teaching trainers. Do not relabel
old recordings as corrected data; collect them again.

## Strict imagined decisions

The complete multi-game action vector is validated before any lookahead advances.
An illegal action, short vector or long vector raises an error without changing
any world's state or event cursor. Search no longer substitutes a pass or the
first legal action. Unfinished or broken continuations are errors rather than
silently omitted evidence. Ordinary gameplay's explicitly lenient API is unchanged.

## Bounded leaf evaluation

`--leaf-batch` (default 256; also available on the cloud controller) bounds copied
Mortal states, encoding, densification, device transfers and inference together.
Terminal slots do not reach a critic. The search retains scalar results in the
native slot order. The convenience `served.leaves()` API remains for small
callers; the actual search uses `served.leaf_batches()`.

## Immutable recording publication

A local recording directory now contains:

```
current.json                 # one atomic pointer
snapshots/<snapshot-id>/     # immutable arrays, metadata and checksum manifest
```

Readers use `neural.recordings.resolve_recording()` once, then read every array
from that pinned snapshot. `validate_snapshot()` checks sparse and dense schemas,
legality, finite candidate values, checksums, reward version and completeness.
Failed writes or copies leave the previous pointer intact; the writer never
removes historical snapshots. Final publication works even when the final row
count equals the last progress snapshot's row count.

Each cloud attempt gets its own experiment ID, bound to the copied checkpoint's
SHA-256/generation and the complete search configuration, with a unique suffix
for repeated attempts. `experiment.json` retains this provenance. `progress/`
contains partial snapshots. Only a zero-exit subprocess with a validated complete
recording can publish `complete/`. `result.json` retains the exit status and output;
an unsuccessful child raises an error and cannot replace an accepted experiment.
Use an attempt's `complete/` directory as a training input, not `progress/` or its
parent. Both teaching and sibling-head cloud consumers resolve this format.

Immutable snapshots intentionally retain history. Periodic local saves and
separate experiments can consume additional storage; remove obsolete experiment
folders explicitly only after deciding which records to retain.

## Legacy implicit passes

Legacy saves that omitted opponent passes after a human call are migrated with
the current named Mortal pass (45), checked against `opponent_mask_mortal()`.
Explicit old-schema opponent actions are not blindly reinterpreted as new IDs.

## Regression coverage

Rust tests cover root-hand reward accounting through terminal placement and
repeats, plus side-effect-free action rejection. Python tests cover cross-game
native batch atomicity, bounded leaf construction, missing histories, recording
write/copy/pointer failures, configuration identity and actual cloud-controller
failure paths. JavaScript tests exercise a complete implicit-pass legacy restore,
resave/restore round trip and rejection when a pass is unavailable.
