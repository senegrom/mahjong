"""Self-play: many tables at once, all four places played by one network.

Every decision is a training example. What a decision was worth is only
known later, so each one waits: when the hand ends, the points that changed
hands are credited to the decisions that led to it, and when the game ends
the placement is credited to every decision in it. That is the whole reward
signal, and it is the game's own, not a hand-made one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import riichi_py

PLANES = riichi_py.PLANES
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

    observations: torch.Tensor
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
) -> Batch:
    """Plays `games` games to the end and returns every decision made.

    With `amp` the network's forward passes run in bfloat16, which is
    plenty for choosing a move and about half the arithmetic.
    """
    net.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=bot_places or [])

    observations: list[np.ndarray] = []
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
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break

        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        truth = np.frombuffer(arena.opponent_hands(), dtype=np.float32)
        truth = truth.reshape(games, OPPONENTS, POSITIONS)
        hidden = np.frombuffer(arena.oracle(), dtype=np.float32)
        hidden = hidden.reshape(games, ORACLE_PLANES, POSITIONS)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)

        index = np.nonzero(live)[0]
        batch_planes = torch.from_numpy(planes[index]).to(device)
        batch_mask = torch.from_numpy(mask[index]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            logits, _value, guessed = net.everything(batch_planes, batch_mask)
        logits = logits.float()
        distribution = torch.distributions.Categorical(logits=logits)
        # What the network believes the opponents hold, so the engine can
        # imagine one world per game from it: the reader's negatives, the
        # hands the proposal deals that were not the real ones.
        beliefs = np.zeros((games, HANDS), dtype=np.float32)
        beliefs[index] = (
            torch.softmax(guessed.float(), dim=2).reshape(len(index), HANDS).cpu().numpy()
        )
        proposed = np.frombuffer(arena.imagined_hands(beliefs.reshape(-1).tolist()), dtype=np.float32)
        proposed = proposed.reshape(games, HIDDEN_HANDS_PLANES, POSITIONS)
        chosen = logits.argmax(dim=1) if greedy else distribution.sample()
        chosen_log_prob = distribution.log_prob(chosen)

        choice = np.zeros(games, dtype=np.int64)
        chosen_cpu = chosen.cpu().numpy()
        log_prob_cpu = chosen_log_prob.cpu().numpy()
        choice[index] = chosen_cpu

        for slot, game in enumerate(index):
            seat = int(seats[game])
            person = int(players[game][seat])
            step_index = len(actions)
            # Copies, not views: a view would keep the whole step's buffer
            # alive, seven megabytes for five hundred tables, until the
            # round is stacked at the end.
            observations.append(planes[game].copy())
            legal_masks.append(mask[game].copy())
            held.append(truth[game].copy())
            oracle.append(hidden[game].astype(np.uint8))
            imagined.append(proposed[game].astype(np.uint8))
            actions.append(int(chosen_cpu[slot]))
            log_probs.append(float(log_prob_cpu[slot]))
            rewards.append(0.0)
            pending[game][person].append(step_index)
            everything[game][person].append(step_index)

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
        observations=torch.from_numpy(np.stack(observations)),
        legal=torch.from_numpy(np.stack(legal_masks)),
        actions=torch.tensor(actions, dtype=torch.int64),
        held=torch.from_numpy(np.stack(held)),
        oracle=torch.from_numpy(np.stack(oracle)),
        imagined=torch.from_numpy(np.stack(imagined)),
        returns=torch.tensor(rewards, dtype=torch.float32),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        games=games,
        hands=hands,
        decisions=decisions,
        final_scores=final_scores.copy(),
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
    """
    batch = play(
        net,
        games=games,
        seed=seed,
        device=device,
        bot_places=[1, 2, 3],
        greedy=True,
        amp=amp,
    )
    scores = batch.final_scores
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return {
        "placement": float(order[:, 0].mean()),
        "score": float(scores[:, 0].mean()),
        "wins": float((order[:, 0] == 1).mean()),
        "hands": batch.hands,
    }
