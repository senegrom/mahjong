"""Self-play: many tables at once, all four places played by one network.

Every decision is a training example. What a decision was worth is only
known later, so each one waits: when the hand ends, the points that changed
hands are credited to the decisions that led to it, and when the game ends
the placement is credited to every decision in it. That is the whole reward
signal, and it is the game's own, not a hand-made one.

What the network sees at each decision comes from `observe.Views`: Mortal's
planes for the current lineage, the engine's own for older networks, so a
table can seat both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

import riichi_py

from . import zoo
from .observe import Planes, Views

POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS
ORACLE_PLANES = riichi_py.ORACLE_PLANES
HANDS = riichi_py.HANDS
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES

# What a hand moved, brought to about the size of the placement term below
# so neither drowns the other. A big hand is worth a few tenths.
HAND_SCALE = 1.0 / 4000.0

# What the game itself was worth, by the place it ended in. This is the
# winner bonus of the rules, and using it rather than the final score matters
# for a reason that is easy to miss: the game term is added to every one of a
# player's decisions, so whatever it is, it carries no information about any
# single decision and only adds spread. A raw score of about 30,000 points
# adds a large offset and a spread of thousands; the place adds a bounded
# number that is zero on average across the table, which is the smallest
# honest way to say what the game is for. The numbers live in the engine,
# which needs them too when a search plays an imagined world to the end of
# the game.
PLACEMENT_VALUE = tuple(riichi_py.PLACEMENT_VALUE)


@dataclass
class Batch:
    """What one round of self-play produced."""

    #: The planes of every decision, kept sparse: a decision is 34,408
    #: values of which about one in eighteen is not zero, and a round of
    #: them dense would be tens of gigabytes. A minibatch is made dense on
    #: the card. Empty when the network that played sees the engine's
    #: planes, which nothing trains on any more.
    observations: Planes
    legal: torch.Tensor
    actions: torch.Tensor
    #: What the three opponents were holding at each decision, as a
    #: distribution over the 34 kinds for each. The label for the head that
    #: reads a table, and never shown to the network when it chooses.
    #: Named apart from `hands`, which counts how many were played.
    held: torch.Tensor
    #: What the deciding seat could not see at each decision: the opponents'
    #: concealed tiles, the draws to come and the hidden indicators, as 0/1
    #: planes kept in bytes. For the oracle critic, which only trains; the
    #: network is never shown this when choosing.
    oracle: torch.Tensor
    #: What the proposal imagined the opponents held at each decision, dealt
    #: from the network's own belief about them: the reader's negatives,
    #: against the real hands the oracle planes carry.
    imagined: torch.Tensor
    returns: torch.Tensor
    log_probs: torch.Tensor
    games: int
    hands: int
    decisions: int
    final_scores: np.ndarray
    hand_results: list[int] = field(default_factory=list)
    #: Where the round's wall time went, in seconds by part: the engine
    #: and the follower, the encoder, the network, the seated others, and
    #: the bookkeeping. For finding what to make faster.
    timing: dict[str, float] = field(default_factory=dict)


def imagine(arena, beliefs: np.ndarray) -> bytes:
    """One imagined world per game from the network's beliefs, as the
    hidden-hand planes. The beliefs cross as bytes where the engine takes
    them so: a list of a hundred thousand floats a step cost seconds a
    round to build and read."""
    if hasattr(arena, "imagined_hands_bytes"):
        return arena.imagined_hands_bytes(np.ascontiguousarray(beliefs, dtype=np.float32).tobytes())
    return arena.imagined_hands(beliefs.reshape(-1).tolist())


def gather(blocks: list[np.ndarray]) -> torch.Tensor:
    """Stacks a round's blocks into one tensor, freeing each block as it
    is copied, so the peak is the round itself and one block over."""
    total = sum(len(block) for block in blocks)
    out = np.empty((total, *blocks[0].shape[1:]), dtype=blocks[0].dtype)
    at = 0
    for index in range(len(blocks)):
        block = blocks[index]
        out[at : at + len(block)] = block
        at += len(block)
        blocks[index] = None
    blocks.clear()
    return torch.from_numpy(out)


@torch.no_grad()
def play(
    net,
    games: int,
    seed: int,
    device: str = "cuda",
    bot_places: list[int] | None = None,
    greedy: bool = False,
    max_steps: int = 4000,
    amp: bool = False,
    opponents: list | None = None,
    opponent_share: float = 0.0,
) -> Batch:
    """Plays `games` games to the end and returns every decision made.

    With `amp` the network's forward passes run in bfloat16, which is
    plenty for choosing a move and about half the arithmetic.

    `opponents` are older checkpoints, of either kind. In that share of
    games one seat is played by one of them, drawn at random, and nothing
    that seat does is recorded: PPO trains on what the learner's own policy
    did. The point is that four copies of one network playing only each
    other have nothing to be robust to, and this run has been measured
    getting worse against outside policies while getting better against
    fixed weak ones. With no opponents given, every path below is the one
    that ran before.
    """
    net.eval()
    for other in opponents or []:
        other.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=bot_places or [])
    kinds = {net.kind} | {other.kind for other in opponents or []}
    views = Views(arena, games, kinds)
    recording = net.kind == "mortal"

    # Which player, if any, an older checkpoint holds in each game, and
    # which checkpoint it is. Fixed for the game, so a seat does not change
    # hands mid-hand.
    foreign_player = np.full(games, -1, dtype=np.int64)
    foreign_which = np.zeros(games, dtype=np.int64)
    if opponents and opponent_share > 0:
        picker = np.random.default_rng(seed ^ 0x0DDBA11)
        taken = picker.random(games) < opponent_share
        foreign_player[taken] = picker.integers(0, 4, size=int(taken.sum()))
        foreign_which[taken] = picker.integers(0, len(opponents), size=int(taken.sum()))

    # One block per step, holding the live games' rows in the order the
    # decisions are numbered below: a round is a few hundred blocks rather
    # than a few hundred thousand arrays, which the heap handles.
    observations: list[Planes] = []
    legal_masks: list[np.ndarray] = []
    held: list[np.ndarray] = []
    oracle: list[np.ndarray] = []
    imagined: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    # Which decisions are still waiting to learn what they were worth.
    pending: list[list[list[int]]] = [[[] for _ in range(4)] for _ in range(games)]
    everything: list[list[list[int]]] = [[[] for _ in range(4)] for _ in range(games)]

    hands = 0
    steps = 0
    clock = time.perf_counter
    timing = {"engine": 0.0, "encode": 0.0, "network": 0.0, "opponents": 0.0, "other": 0.0}
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        began = clock()
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        # The arena has settled: every game either owes a decision or is
        # over, and everything up to that is in its log. Read it.
        views.advance()

        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        truth = np.frombuffer(arena.opponent_hands(), dtype=np.float32)
        truth = truth.reshape(games, OPPONENTS, POSITIONS)
        hidden = np.frombuffer(arena.oracle(), dtype=np.float32)
        hidden = hidden.reshape(games, ORACLE_PLANES, POSITIONS)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        their_choice = np.zeros(games, dtype=np.int64)
        timing["engine"] += clock() - began
        began = clock()

        # Who owes each game's decision, as a person rather than a seat:
        # the seats move between hands and the players do not.
        deciding = np.where(
            live, players[np.arange(games), np.minimum(seats, 3)].astype(np.int64), -1
        )
        # Games whose pending decision belongs to an older checkpoint are
        # answered separately and never recorded.
        theirs = live & (deciding == foreign_player) & (foreign_player >= 0)
        index = np.nonzero(live & ~theirs)[0]
        timing["other"] += clock() - began
        began = clock()
        if theirs.any():
            for which in np.unique(foreign_which[theirs]):
                rows = np.nonzero(theirs & (foreign_which == which))[0]
                other = opponents[int(which)]
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"
                ):
                    their_choice[rows] = zoo.choose(
                        other, views, rows, deciding[rows], mask[rows], device
                    )
        timing["opponents"] += clock() - began
        began = clock()
        if not len(index):
            arena.step(their_choice.tolist())
            timing["engine"] += clock() - began
            continue
        choice = their_choice.copy()
        if hasattr(net, "decide"):
            # A learner in an action space of its own (see
            # `mortal_learner`): it answers the table in ours and records
            # its decisions itself, possibly more than one per row, and
            # keeps its own account of the time.
            picked, records = net.decide(views, index, deciding[index], mask[index], greedy)
            choice[index] = picked
            observations.append(records.planes)
            legal_masks.append(records.masks)
            record_actions = records.actions
            record_log_probs = records.log_probs
            record_slots = records.slots
            began = clock()
        else:
            if recording:
                sparse = views.sparse(index, deciding[index])
                timing["encode"] += clock() - began
                began = clock()
                batch_planes = sparse.dense(device)
            else:
                sparse = None
                batch_planes = views.dense(net.kind, index, deciding[index], device)
            batch_mask = torch.from_numpy(mask[index]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
                logits, _value, guessed = net.everything(batch_planes, batch_mask)
            logits = logits.float()
            distribution = torch.distributions.Categorical(logits=logits)
            # What the network believes the opponents hold, so the engine
            # can imagine one world per game from it: the reader's
            # negatives, the hands the proposal deals that were not the
            # real ones.
            beliefs = np.zeros((games, HANDS), dtype=np.float32)
            beliefs[index] = (
                torch.softmax(guessed.float(), dim=2).reshape(len(index), HANDS).cpu().numpy()
            )
            proposed = np.frombuffer(imagine(arena, beliefs), dtype=np.float32)
            proposed = proposed.reshape(games, HIDDEN_HANDS_PLANES, POSITIONS)
            chosen = logits.argmax(dim=1) if greedy else distribution.sample()
            chosen_log_prob = distribution.log_prob(chosen)
            record_actions = chosen.cpu().numpy()
            record_log_probs = chosen_log_prob.cpu().numpy()
            record_slots = np.arange(len(index))
            choice[index] = record_actions

            # Copies, not views: a view would keep the whole step's buffer
            # alive until the round is gathered at the end.
            if sparse is not None:
                observations.append(sparse)
            legal_masks.append(mask[index].copy())
            held.append(truth[index].copy())
            oracle.append(hidden[index].astype(np.uint8))
            imagined.append(proposed[index].astype(np.uint8))
            timing["network"] += clock() - began
            began = clock()

        for record in range(len(record_actions)):
            game = int(index[record_slots[record]])
            seat = int(seats[game])
            person = int(players[game][seat])
            step_index = len(actions)
            actions.append(int(record_actions[record]))
            log_probs.append(float(record_log_probs[record]))
            rewards.append(0.0)
            pending[game][person].append(step_index)
            everything[game][person].append(step_index)
        timing["other"] += clock() - began
        began = clock()

        arena.step(choice.tolist())

        ended = np.frombuffer(arena.hand_ended(), dtype=np.uint8)
        if ended.any():
            results = np.frombuffer(arena.hand_result(), dtype=np.int32).reshape(games, 4)
            for game in np.nonzero(ended)[0]:
                hands += 1
                for person in range(4):
                    value = float(results[game][person]) * HAND_SCALE
                    for step_index in pending[game][person]:
                        rewards[step_index] += value
                    pending[game][person] = []
        timing["engine"] += clock() - began

    # A learner that decides for itself kept its own account; fold it in
    # and clear it for the next round.
    own = getattr(net, "timing", None)
    if isinstance(own, dict):
        for name, value in own.items():
            timing[name] = timing.get(name, 0.0) + value
            own[name] = 0.0

    # The placement, which is what the game is actually for, reaches every
    # decision that player made.
    final_scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4)
    places = (-final_scores).argsort(axis=1).argsort(axis=1)
    for game in range(games):
        for person in range(4):
            value = PLACEMENT_VALUE[int(places[game][person])]
            for step_index in everything[game][person]:
                rewards[step_index] += value

    decisions = len(actions)
    if decisions == 0:
        raise RuntimeError("self-play produced no decisions")

    return Batch(
        observations=Planes.cat(observations),
        legal=gather(legal_masks),
        actions=torch.tensor(actions, dtype=torch.int64),
        # A learner of the zoo's kind records none of the labels the
        # auxiliary heads want, having no such heads.
        held=gather(held) if held else torch.zeros(0),
        oracle=gather(oracle) if oracle else torch.zeros(0),
        imagined=gather(imagined) if imagined else torch.zeros(0),
        returns=torch.tensor(rewards, dtype=torch.float32),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        games=games,
        hands=hands,
        decisions=decisions,
        final_scores=final_scores.copy(),
        timing=timing,
    )


@torch.no_grad()
def measure(
    net, games: int, seed: int, device: str = "cuda", amp: bool = False
) -> dict[str, float]:
    """Plays the network against three heuristic opponents.

    The network takes place 0 at every table; the other three places are the
    benchmark. What comes back is the average placement, where 1.0 would be
    winning every game and 4.0 losing every one, and the average final score.

    It plays its best move rather than sampling, because that is what the
    web app does. Measuring sampled play would mix how well the network has
    learned with how much exploration noise is on top of it, and then the
    checkpoint kept as best would be chosen partly on that noise.

    This is a score-only loop rather than `play(..., greedy=True)`: it does
    not fetch oracle/truth labels, construct rewards, or retain a round-sized
    training batch. It deliberately still asks the belief head for one
    imagined world per decision. `Arena::imagined_hands` advances the same
    per-table RNG that later deals the next hand, so keeping that one side
    effect makes a fixed benchmark seed produce exactly the same games as the
    historical path while the large allocations disappear.
    """
    net.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[1, 2, 3])
    views = Views(arena, games, {net.kind})
    hands = 0
    steps = 0
    while not arena.all_finished() and steps < 4000:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        views.advance()

        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        index = np.nonzero(live)[0]
        deciding = players[index, seats[index]]

        choice = np.zeros(games, dtype=np.int64)
        if hasattr(net, "choose"):
            # A player of the zoo's kind chooses for itself.
            choice[index] = net.choose(views, index, deciding, mask[index])
            arena.step(choice.tolist())
            hands += int(np.frombuffer(arena.hand_ended(), dtype=np.uint8).sum())
            continue
        batch_planes = views.dense(net.kind, index, deciding, device)
        batch_mask = torch.from_numpy(mask[index]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            logits, _value, guessed = net.everything(batch_planes, batch_mask)
        logits = logits.float()

        # Preserve the old evaluator's RNG consumption exactly. The imagined
        # hands themselves are not needed for scoring, so they are discarded
        # immediately rather than retained with every decision.
        beliefs = np.zeros((games, HANDS), dtype=np.float32)
        beliefs[index] = (
            torch.softmax(guessed.float(), dim=2).reshape(len(index), HANDS).cpu().numpy()
        )
        imagine(arena, beliefs)

        choice[index] = logits.argmax(dim=1).cpu().numpy()
        arena.step(choice.tolist())
        hands += int(np.frombuffer(arena.hand_ended(), dtype=np.uint8).sum())

    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4).copy()
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return {
        "placement": float(order[:, 0].mean()),
        "score": float(scores[:, 0].mean()),
        "wins": float((order[:, 0] == 1).mean()),
        "hands": hands,
    }
