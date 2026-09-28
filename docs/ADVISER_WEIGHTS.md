# Adviser weights and value labels

This repair changes presentation and retains distributions already returned by
advisory inference. It does not change the playing policy, masks, selected
actions, tie-breaks, model, training, saves or inference request count.

## Selection is independent of display

The selected action is still the network's original answer. A riichi answer
still causes exactly the existing second question, whose chosen tile remains
the played tile. Neither grouping/sorting the displayed rows nor multiplying
probabilities is used to select a move. Built-in agents still make exactly
one `agent_pick` call and have no invented policy percentages.

`weightsByChoice` retains its first-stage `weight` field for existing callers.
Riichi candidates additionally carry `conditionalWeight`, taken from the
already evaluated post-declaration distribution and the engine's own action
translation. A missing conditional evaluation is `null`, not zero. No extra
inference is requested to score a riichi the agent did not choose.

`groupPolicyChoices` creates one display row for the riichi declaration, with
its conditional discards underneath. It does not mutate the input array and
keeps original choice objects for the existing confirmation callbacks. The
blue border and explicit Agent choice label follow `analysis.choice`, not the
largest displayed weight. The grouped declaration itself is not a playable
button: a real discard must still be selected and confirmed.

Example: declaration 80%, conditional discards 75%/25%, ordinary discard 20%.
The first-stage rows show 80%/20%; the nested, separately labelled distribution
shows 75%/25%. They do not appear as 80%/80%/20%, and no 60% joint weight is
used to rerank the selected move. Historical review labels both stages too.

Current main disables red-five aliases. This PR leaves that mask and all
translation semantics unchanged. The synthetic many-to-one regression only
checks that aggregation cannot override the backend's selected action; it
does not claim aliases are legal in current play.

## Value units

The current exported model predicts a mixed training reward:

    hand-point change / 4,000 + final placement bonus

The UI labels this Estimated training reward and explains its units in visible
text as well as its tooltip. It is not a predicted finishing place, a gain in
places, or a win probability. The value still comes from the initial position,
not the second-stage riichi response. No value or policy is rescaled.

## Coverage

`adviser-presentation.test.js` exercises the actual evaluator/reviewer with a
controlled network and translator: unchanged selected actions and request
counts, no joint-probability reranking, grouping without mutation, absent/zero
conditional weights, retained initial value, unscored kans and cancellation.
`adviser-rendering.test.js` compiles the production components for client and
server and checks grouped markup, selected borders, disabled action buttons,
explicit unevaluated stages, built-in labels, reward units and review labels.
The existing real-WASM policy/review tests and browser suite remain in CI.

Save compatibility (review item 1) is intentionally deferred.
