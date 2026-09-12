# Bounded updates and checkpoint selection

This change is based on merged training API 2 (`fd614d79`); rebuild native
engines when upgrading from an earlier API. It does not change the reward
contract, overwrite checkpoints, or claim a measured increase in playing strength.

## Review findings addressed

- The main cloud trainer copied every opponent to its basename. For example,
  `family-a/latest` and `family-b/latest` both became `opponents/latest.pt`;
  both supplied paths then loaded family B. Every cloud trainer now allocates
  distinct numbered paths, and missing explicitly requested opponents raise
  an error rather than silently changing the training population.
- Mortal and combined rollout recording forced CUDA BF16 even when the trainer
  did not request AMP. The caller now owns precision for both riichi stages,
  matching the later policy-loss forward. CPU context/propagation tests do not
  substitute for a GPU determinism test.
- An abandoned prefetch consumer left its producer blocked on a full queue,
  retaining minibatches. Prefetchers now support explicit cancellation and
  context management; every training/replay consumer closes them on all exits.
  Closing joins the thread after its current finite preparation finishes. It
  cannot cancel an arbitrary nonterminating user-supplied preparation function.
- Baseline evaluation padded to fixed batches of 4,096 or 8,192 even when a
  rollout or configured training batch was tiny. `--baseline-batch` now bounds
  the forward. Without an override it uses `--batch`, capped at rollout size.
  The final chunk is still padded to that size for compiled shape stability.

## Policy update controls

All three self-play trainers accept `--target-kl` and `--baseline-batch`.
The cloud training functions and main cloud launcher forward these settings.
The KL guard is opt-in (`0` keeps the previous update budget); for example:

```sh
python -m neural.train --rounds 1 --games 1 --channels 8 --blocks 1 \
  --batch 32 --epochs 2 --replay-steps 1 --measure-games 1 \
  --baseline-batch 16 --target-kl 0.02 --out runs/smoke
```

A CPU smoke configuration is a correctness check, not a competitive policy.
Choose production settings using held-out comparisons. New validation rejects
nonfinite learning rates/weights and invalid clipping/precision parameters
before constructing a learner. Existing full-minibatch and zero-update guards
remain in place.

For sampled log-probability change `d = new_log_prob - old_log_prob`, the new
diagnostic is `mean(expm1(d) - d)`. It is nonnegative; under old-policy samples
and equal legal support its expectation is `KL(old || new)`. The old signed
`approx_kl` field remains for log compatibility; new fields are
`sampled_kl_last`, `sampled_kl_max`, `target_kl`, and `kl_early_stop`.

If the guard exceeds the threshold, the pending optimizer step is not applied
and remaining PPO epochs stop. Updates already made are not rolled back; this
is a sampled minibatch safeguard, not a hard full-policy KL guarantee. The main
trainer may still run its detached auxiliary replay pass. A round stopped
before any update fails explicitly rather than publishing a learned generation.
Controls are recorded under `training_controls` in checkpoints for provenance;
command-line settings remain authoritative and must be supplied on resume.
Checking each minibatch synchronizes a scalar from CUDA and adds overhead.

PPO clipping alone does not guarantee a small policy change; early stopping is
also used in the original Spinning Up implementation:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

## Bidirectional candidate/champion gate

`best.pt` is still selected by the existing smoothed heuristic benchmark. It is
not automatically a validated champion. Use the new separate command before
promoting a candidate:

```sh
python -m neural.gate runs/candidate/latest.pt runs/champion.pt \
  --games 512 --seed 9200001 --attempt 1 --confidence 0.95 > gate-report.json
```

The gate copies and validates both inputs before evaluation so a running
trainer cannot change the compared bytes. Reports include SHA-256 hashes,
generations, seed range, native API version and the underlying measurements.
No checkpoint is replaced or promoted automatically; a training agent can read
`promote` and archive the report before explicitly publishing a winner.

It plays both candidate-versus-three-champions and champion-versus-three-
candidates, each in all four seats. `--games N` therefore means **8N actual
games but only N seed groups for statistical inference**, not 8N independent
samples. Per-seed placement improvement is `(reverse - forward) / 2` in
`[-1.5, 1.5]`. The one-sided Hoeffding lower bound subtracts
`3 * sqrt(log(1 / alpha_i) / (2N))` from its mean. Identical policies, tiny
samples, or zero empirical variance alone cannot produce a confident promotion.
This deliberately conservative test can require many games to establish a
small edge. `--minimum-edge` can require a practically meaningful improvement;
`--minimum-deals` defaults to 128. Truncated games are errors, not evidence.

Repeated attempts spend `alpha_i = (1-confidence)/(i*(i+1))`. These allocations
sum to `1-confidence`. The caller must maintain an increasing attempt counter,
preselect the comparison settings, and use fresh held-out seeds independent of
candidate training and previous attempts. The command does not maintain a
seed registry or enforce this external experimental protocol. Reusing seeds,
resetting the counter, or tuning candidates on the test set voids the claimed
error control. Pseudorandom seeds are the simulator's operational approximation
to independent draws. A pass establishes evidence for this symmetric matchup,
not superiority over arbitrary opponents, a league or human players.

Modern information-consistent search and an automatically managed champion/
opponent population remain separate work. Current-history native lookahead
retains the existing explicit unsupported-layout diagnostic.
