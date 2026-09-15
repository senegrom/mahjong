# World, confidence and auxiliary-artifact integrity

This follow-up repairs the six findings against `b801a3f`.

## Publicly constrained riichi proposals

A declared opponent is assigned a legal waiting concealed body before unconstrained
hands are dealt. Random weighted component orders cover ordinary hands, seven pairs
and both thirteen-orphans waits. Concealed quads count as sets. All constrained
opponents share a backtracking allocator, so one player's proposal cannot strand
another. The allocator reads public melds/declarations, tile availability and the
searcher's own hand, never the real concealed identities. A hidden drawn tile is
sampled separately where necessary. Exhausted waits remain legal; a fifth copy in
the player's own hand/melds does not. Riichi furiten is reconstructed from the
sampled waits and public discards, excluding an unresolved latest claim window.

The constructive sampler is a proposal, **not an exact posterior over histories**.
Its shape distribution has changed; the likelihood reader should be retrained and
playing strength remeasured against the new proposal. Impossible public constraints
fail instead of falling back to the actual hidden hand or an invalid random deal.
Joint backtracking can cost more in tightly constrained positions. When the last
constrained hand must consume exactly the remaining pool, its body is checked
directly instead of enumerating its decompositions again. A tight multi-hand pool
uses exact cover across all hands, assigning the scarcest tile first rather than
committing an entire early hand. Costly trial orders may restart after rolling back
all reservations; a final exhaustive attempt retains completeness. Neither path
consults hidden identities or turns a search-budget failure into an invalid deal.

## Independent-world confidence

The Python engine boundary coalesces repeated proposal IDs before either club or
network rollouts. Their resampling masses are added, with no second application of
the reader likelihood ratio. Each unique proposal gets one continuation, including
when rollout temperature is nonzero. Confidence therefore counts independent
proposals, not copies. Two proposals repeated 16 times each still cannot clear the
native three-proposal minimum. Equal-valued but independently drawn proposal IDs
remain separate; identity is not guessed from tile equality.

Recordings now use format 3 and retain `world_weights.npy`, one mass per independent
proposal. The offline selector uses the same weighted comparison, including positive
zero-error differences, and shared TSV fixtures test Python/native parity. Sibling
precision uses weighted independent-world errors. A single independent world has
zero precision, not maximum confidence; weighted fitting refuses a dataset with no
estimable independent-world uncertainty rather than normalizing zero weights.
Old format-2 pointers remain resolvable for diagnostics, but training readers require
new format-3 records: old records cannot establish which repeats were independent.
Backed-up reward semantics remain version 2, and training reward/API semantics
remain unchanged.

Rebuild `riichi-py` for **SEARCH_API_VERSION=3**. Existing policy checkpoints remain
loadable. Recollect search labels rather than relabelling old confidence evidence.

## Reader observation contracts

The serving contract reports the world reader's input planes or explicit uniform
fallback. Both engine and Mortal observation readers receive their own public
planes, and concealed-hand proposals retain their relative-seat ordering.
Reconstruction, device transfer and reader evaluation are bounded by the leaf batch
limit. Nonfinite reader output fails instead of producing silent uniform evidence.

## Placement splits and atomic files

Collected rounds retain their real initial seed. Placement splits use unsigned
64-bit environment seeds, independent of input order or shard boundaries. Duplicate
and overlapping seed ranges are refused, including copies at different paths.
Rounds missing seed provenance are not silently assigned fictitious new identities.
This conservatively also rejects repeated deals collected under a different actor;
any intentional reweighting requires a separately designed dataset contract.

Placement heads, sibling heads and self-play rounds serialize to same-filesystem
staging files, validate their own schema, sync, then atomically replace the output.
A serialization/validation/pre-rename failure preserves the old bytes; a signal
following rename can expose only the complete new artifact. Validation does not
consume the training random stream, and metadata cannot replace protected fields.
These remain single-writer APIs. Version-1 rounds use NumPy/pickle storage and must
come from a trusted collector, not an untrusted download.

Tests include live accepted declarations, all waiting-shape families, concealed
quads, exhausted waits, joint constraints, conservation and hidden-input independence;
rollback of interrupted ordinary and exact-cover allocations; native repeated-proposal
paths; reader layout/batch checks; copied/overlapping round rejection and near-u64
seed boundaries; and serialization/validation/rename faults.
