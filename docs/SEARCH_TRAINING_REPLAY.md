# Current-lineage search replay

This is an opt-in bridge from search to supervised learning for the current
Mortal-v4 observation / 46-action network, including `combined.Combined`.
It does not modify the PPO trainers, cloud launchers, production checkpoints,
existing diagnostic recordings, or promotion policy.

## Collect a completed round

Build/install both Python engines using the repository's normal training setup.
From a checkout of this branch, collect a small pilot round:

```sh
python -m neural.collect_search path/to/combined.pt \
  --source-revision "$(git rev-parse HEAD)" \
  --out runs/search-replay/round-0001 \
  --games 2 --seed 20260913 --worlds 8 --candidates 4 --pool 1 \
  --played-by network --depth 0 --valued-by critic
```

Supply the revision of the code actually used to collect the round, and rebuild
native engines when required. The CLI copies the checkpoint into a validated
private snapshot and hashes that snapshot before loading it. All four seats
search with that frozen actor. Each collection uses one fixed actor; the Python
API assumes its caller does not mutate the network concurrently.

The default is policy-guided shallow search. `--depth -1` uses the existing
end-of-hand continuation mode. Neither setting has an implied playing-strength
guarantee. Search quality must be measured separately. A two-game pilot is a
correctness check, not enough data to establish learning progress.

The destination must not exist. No training shard is written until **all games
finish** and the reward ledger settles every decision. A step-budget failure
raises an error rather than treating current scores as final outcomes.

## What version 1 means

Observations are the exact adapter inputs used by the actor, stored as the
existing sparse `Planes` representation. Each example has the engine legal
mask and selected engine action, the corresponding 46-action mask, frozen
actor probabilities, frozen policy target, completed-game return, game/player
identity, engine step, and decision stage.

An executed riichi produces two linked examples: the ordinary observation with
a declaration target, then the captured conditional observation with a tile
target. Both get the same realized return. An imagined but unchosen declaration
does **not** produce a conditional training row. Discard aliases share target
mass rather than each receiving a duplicate of it. A move that cannot be
expressed under the policy's first-legal-meaning translation is rejected; no
fallback action is silently substituted into the labels.

The policy target is

```text
target = (1 - improve) * frozen_actor + improve * selected_action_aliases
```

The selected action's aliases share its mass in the actor's existing
proportions; an all-zero alias group uses equal shares. `--improve` defaults to
0.5. This is the repository's shallow-search action-improvement idea, **not tree
visit counts** and not a measured search sampling distribution. Executed actions
come from the search decision, not from sampling this target.

The reward is deliberately unchanged and explicitly versioned:

```text
reward version 1 = points moved in the decision's hand / 4000
                   + final placement bonus
```

This remains the existing hybrid objective, not pure final-placement utility.

## What the teacher was measured to be worth

Collect with these settings and you are copying a teacher that has been
measured, on this lineage, to play worse than the student it is copied
from. The example above is one ply valued by the critic, which placed
2.76 against a level of 2.50 over two hundred deals. Playing the worlds
out to the end of the match with the club heuristic instead, which the
corrected search does at `--depth -1`, placed 2.574 over a hundred deals
in the cloud and 2.616 over another hundred on a desktop: about a tenth
of a placement worse than not searching, at two and a half standard
errors together. That same search reports gaining 0.028 of a unit a
decision by its own reckoning, with the worlds split so that one half
decides and the other scores (`neural.worth`), which is why an
estimator's confidence in itself settles nothing.

No configuration of this search has yet been measured to beat the policy
at the table. Until one is, a shard collected here is material for
studying the pipeline, not supervision known to be worth learning.

Two further cautions for whoever fits this:

- The improvement target is applied at every decision, including the
  thirteen in fourteen where a margin-gated search agrees with the actor.
  Asking there for more mass on the move the actor already chose is a
  demand to sharpen rather than to keep, and sharpening is what took
  three earlier lineages backwards (`neural.teach --rows changed` teaches
  only the decisions the search overrode for that reason).
- There is no leash to the frozen actor in the loss. The `(1 - improve)`
  term anchors a single fit, but it does not hold across rounds of
  collect-and-fit the way the KL in `neural.train_combined` does.
Changing the objective requires a new supported reward contract and new data;
these shards must not be relabelled or silently mixed with another objective.

The manifest identifies the actor and all four opponent copies by SHA-256,
the collecting source revision, training API version, observation/action/reward
contracts, seed range, and all search settings, including rollout temperature
and value-head selection. These are caller-provided provenance in the Python
API; the CLI computes the actor hash from its immutable input snapshot.

## Load and use the supervised loss

`SearchReplay.load` verifies file hashes, exact array shapes/dtypes, legal
probabilities, frozen targets, finite returns, and riichi linkage. Legacy
`neural.searched.Recording` directories are not training shards: they lack the
required masks, policy probabilities and realized outcomes. No automatic
conversion is attempted.

For a small learning smoke test with an already loaded current-lineage network:

```python
from pathlib import Path
import numpy as np
import torch
from neural import contract, zoo
from neural.search_replay import SearchReplay

replay = SearchReplay.load(Path("runs/search-replay/round-0001"))
net = contract.unwrap(zoo.load_player(Path("path/to/combined.pt"), "cpu"))
if hasattr(net, "set_mode"):
    net.set_mode("none")
net.train()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
rows = np.arange(min(32, replay.metadata["rows"]), dtype=np.int64)
optimizer.zero_grad(set_to_none=True)
loss = replay.loss(net, rows, device="cpu", value_weight=0.5)
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
```

The loss trains the same value-head selection used during search. It masks
illegal logits before cross-entropy and prevents accidental `(N, 1)` value-target
broadcasting. It does not create a PPO ratio from search-selected actions.
The caller still owns minibatch scheduling, optimizer state, checkpointing and
held-out evaluation; this example deliberately publishes no checkpoint.

## Publication and validation

Shard creation is exclusive: existing directories are never overwritten. Arrays
are flushed and synced before the checksummed completion manifest is atomically
published. A write interruption can leave an incomplete directory, which the
reader refuses. Keep it for diagnosis or collect into a different new directory.
Directory fsync is used where supported; this is not a distributed multi-file
transaction or a multiwriter replay ring.

Run the focused regressions with:

```sh
python -m unittest discover -s neural/tests -p 'test_search_replay.py' -v
```

The contract, controlled-collector, loss, corruption and publication tests can
run without native engines. Four additional tests use both real engines to
check the production action mapping, actual combined-network gradients, a
complete game round trip, and rejection/cleanup of truncated multi-candidate
search. Repository CI installs those engines. A passing smoke test is not
playing-strength evidence and does not validate CUDA/AMP/compiled parity.

Still separate work: useful-search evaluation, better action-ranking targets,
population refresh, a bounded multi-round search replay window, an automated
actor/learner cycle, objective alignment, and statistically powered promotion.
