# Completed-game training and replay integrity

Self-play credits hand payments after every arena step, including a step on
which only external opponents move. A completed hand clears all of its
pending decisions before another hand can contribute payments. Placement
rewards are added only after **every game is finished**.

## Ties and measurements

`neural.outcomes` is shared by self-play, the heuristic benchmark, duplicate
arena, direct duels and search evaluation. Equal scores receive the average
of their occupied ranks and the average of the corresponding placement
rewards. A tie for first splits one win among the tied winners: two winners
receive 0.5 each; a four-way tie gives placement 2.5, zero placement reward
for the default reward vector, and 0.25 wins per player. The `wins` metric is
therefore fractional first-place credit, not a count of outright wins.

Older measurements broke ties by array index. Do not interpret differences
between old and new tie-affected metrics as policy improvements alone.

The self-play, score-only measurement, duel and search loops reject an
unfinished batch with `IncompleteGamesError`. The default safety budget is
still 4,000 steps. Callers may supply a larger positive `max_steps`, but may
not use provisional scores as terminal training targets. Even an arena with
no pending decisions must report completion before its scores are accepted.
A game that finishes on the last permitted step is accepted.

## Replay publication and recovery

The replay ring is a **single-writer** cache; do not open the same directory
from multiple training processes. Each pushed batch uses a fresh generation
directory, never overwriting arrays referenced by the current manifest.
Array files are flushed before the new generation is published. `ring.json`
is written to a temporary file, flushed and atomically replaced. Old batches
are removed only after publication. POSIX directories are also synced; on
Windows, where portable directory syncing is unavailable, atomic replacement
still protects against process interruption, not every possible power-loss
or filesystem failure.

Loading checks row counts and sparse-array consistency, and warns rather
than sampling incomplete batches. Unreferenced generations from interrupted
writes are reclaimed. Existing legacy replay slots remain readable and are
replaced safely as the ring advances. No model checkpoints are deleted.

**These changes cannot repair previously generated wrong rewards or detect
every legacy mixed write.** Equal-sized legacy arrays can look structurally
valid despite belonging to different rounds. Rebuild the replay cache from
new self-play when using data from an affected run, especially after an
interrupted legacy write. Preserve an old cache separately when it is needed
for diagnosis; do not relabel it as corrected data.

## Browser move probabilities

Live trained advice and historical review sum the policy mass of equivalent
five-tile actions. This game's red-five aliases and ordinary-five action name
the same ordinary discard. Both the advised and played weight use that same
sum. Additional unnameable kan choices stay unscored. Riichi remains a
first-stage declaration probability; its candidate discards do not represent
independent first-stage probabilities. Selection still follows the original
best network action, preserving the existing training/inference contract.

## Regression checks

The normal neural workflow discovers the outcome, native-ledger, truncation
and replay-interruption tests. Replay tests fail writes at all eight array
boundaries and terminate a subprocess immediately before and after manifest
publication. The web tests exercise all three five-tile aliases against the
actual WASM, and the browser hand-review check independently sums the real
worker outputs for its expected displayed probability.
