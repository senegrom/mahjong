# Custom tables

Choose **Opponents → Custom table** to pick Beginner, Club or Trained separately
for the Left, Opposite and Right players, then press **Start custom game**.
Cancel leaves the running match unchanged. Starting a different table confirms
abandonment when the existing match has progress. The quick all-opponents
selector remains available; **Edit opponents** reopens a custom setup.

Each opponent's panel displays its type. Assignments belong to fixed player
identities, not East/South/West/North winds, and stay with those people through
dealer changes, repeat hands, the final result and reload. New game reuses the
current assignments. A custom table can contain two or more Trained opponents;
they share the existing model and worker, not separate downloads or servers.

Only Trained players request inference. Beginner and Club use their respective
native bots for both turns and calls. All eligible responses are gathered before
claim priority is resolved, including native and trained competing Ron claims.
The human cannot implicitly pass another opponent's decision.

When inference fails, **Retry trained opponent** keeps the table unchanged.
**Use Club for … opponent only** changes just the pending trained player, without
redealing or discarding already submitted claims. The separate all-Trained
fallback changes every remaining Trained player to Club, never a Beginner.
Fallback decisions are recorded and replayed exactly like other match commands.

Save format 4 records the original three assignments. Formats 1 (unversioned),
2 and 3 continue to replay their original uniform type and recorded neural
answers; migration ignores only additive identity metadata (and the derived
fields already migrated by earlier releases). Tile, phase, score and legal
choice validation is unchanged. New saves also validate controller identity.

Verification is part of the normal unit, native and browser suites. It covers
all 27 combinations, mixed simultaneous wins, pending-claim recovery, complete
matches, legacy restoration, cancellation, mobile layouts, and two Trained
players using the actual published model with one shared model load.
