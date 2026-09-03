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

# Points are large numbers; these bring both signals to about the same size
# so neither drowns the other.
HAND_SCALE = 1.0 / 4000.0
PLACEMENT_SCALE = 1.0 / 20000.0


@dataclass
class Batch:
    """What one round of self-play produced."""

    observations: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
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
) -> Batch:
    """Plays `games` games to the end and returns every decision made."""
    net.eval()
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=bot_places or [])

    observations: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
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
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)

        index = np.nonzero(live)[0]
        batch_planes = torch.from_numpy(planes[index]).to(device)
        batch_mask = torch.from_numpy(mask[index]).to(device)
        logits, _value = net(batch_planes, batch_mask)
        distribution = torch.distributions.Categorical(logits=logits)
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
            observations.append(planes[game])
            legal_masks.append(mask[game])
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
    for game in range(games):
        for person in range(4):
            value = float(final_scores[game][person]) * PLACEMENT_SCALE
            for step_index in everything[game][person]:
                rewards[step_index] += value

    decisions = len(actions)
    if decisions == 0:
        raise RuntimeError("self-play produced no decisions")

    return Batch(
        observations=torch.from_numpy(np.stack(observations)),
        legal=torch.from_numpy(np.stack(legal_masks)),
        actions=torch.tensor(actions, dtype=torch.int64),
        returns=torch.tensor(rewards, dtype=torch.float32),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        games=games,
        hands=hands,
        decisions=decisions,
        final_scores=final_scores.copy(),
    )


@torch.no_grad()
def measure(net, games: int, seed: int, device: str = "cuda") -> dict[str, float]:
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
    )
    scores = batch.final_scores
    order = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return {
        "placement": float(order[:, 0].mean()),
        "score": float(scores[:, 0].mean()),
        "wins": float((order[:, 0] == 1).mean()),
        "hands": batch.hands,
    }
