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

from .training_safety import require_training_engine

from . import zoo
from .observe import Planes, Views
from .outcomes import placement_rewards, placements, require_finished, validate_budget, win_shares

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

#: What the network is being asked to maximise, named and numbered.
#:
#: The target is a hybrid and should be read as one rather than as final
#: placement with a detail attached:
#:
#:     z = (points the decision's hand moved) / 4000  +  placement bonus
#:
#: The first term is credited only within the hand it belongs to; the
#: second reaches every decision that player made in the game. So a hand
#: worth four thousand points weighs as much in the target as a whole place
#: at the table. That is a deliberate choice — it gives a decision
#: something to learn from long before the game ends — but it means
#: training and evaluation are not measuring the same thing: `neural.duel`
#: and `measure` report placement alone.
#:
#: The number is here so that a checkpoint records which objective it was
#: trained against. A saved round carries it, and a trainer must refuse a
#: round from a different one rather than mix two objectives in one replay:
#: the returns would not be on one scale and nothing would say so.
#:
#: 1 — hand points over four thousand, plus the placement bonus.
REWARD_VERSION = 1


def explore(logits: torch.Tensor, legal: torch.Tensor, epsilon: float, rng) -> tuple:
    """A move from the policy, or now and then a legal one at random, and
    the probability the *behaviour* gave whatever came out.

    Why this exists. The value head only ever sees positions the policy
    actually reached, and a search asks it what a move the policy would not
    have chosen is worth. Those positions are off the distribution it was
    trained on, its error there is not noise but a bias that correlates
    with the move being considered, and averaging over more imagined worlds
    cannot remove it because every world shares it. Widening where
    self-play goes is the direct attack: the value head is shown the
    positions the search will ask about.

    Why the probability recorded is the policy's own and not the mixture's.
    The first version wrote down the mixture's, (1 - epsilon) pi(a) +
    epsilon / legal, on the argument that PPO divides by the probability
    the behaviour gave the move. Two blocks eroded on it the same way and
    the arithmetic says why: a move the policy gave 1e-4 was recorded near
    0.007, PPO's ratio for it began near 0.014, far below the clip, and
    there the clipped objective has no gradient for a negative advantage
    and a full one for a positive -- so a bad forced move was never pushed
    down and a lucky one was pushed up. The policy's own unlikely moves
    suffered the same. Now every move is recorded at pi(a): given the coin
    said "policy", the move is a draw from pi and that is the right ratio;
    a forced move is flagged, and the trainer keeps it out of the policy
    gradient altogether (see `train_combined`), while the value, hands and
    reader terms still see the position it led to, which was the point.
    """
    distribution = torch.distributions.Categorical(logits=logits)
    chosen = distribution.sample()
    if epsilon > 0:
        count = legal.sum(dim=1).clamp(min=1)
        forced = torch.from_numpy(rng.random(len(chosen))).to(logits.device) < epsilon
        if forced.any():
            # A legal move chosen evenly: the cumulative count of legal
            # entries reaches the drawn rank exactly at the wanted one.
            draw = torch.from_numpy(rng.random(len(chosen))).to(logits.device)
            rank = (draw * count).floor().clamp(max=count - 1).long()
            walk = legal.long().cumsum(dim=1) - 1
            picked = (walk == rank.unsqueeze(1)) & legal
            instead = picked.float().argmax(dim=1)
            chosen = torch.where(forced, instead, chosen)
        return chosen, distribution.log_prob(chosen), forced
    return chosen, distribution.log_prob(chosen), torch.zeros_like(chosen, dtype=torch.bool)


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
    #: Which objective `returns` was built against, so a trainer can refuse
    #: a round that was not built against the one it is learning.
    reward_version: int = REWARD_VERSION
    #: Which decisions were a legal move taken at random rather than the
    #: policy's own choice. The positions that follow one are the positions
    #: a search asks the value head about, so a diagnostic can weigh the
    #: critic's error on them apart from the rest.
    explored: torch.Tensor | None = None
    #: Which of the seated others took a place in each game, by index into
    #: the list given, or -1 where the learner held all four. Kept so a
    #: round can say who it played rather than only how it did on average.
    seated: np.ndarray | None = None
    #: How the learner placed against each of them, one row a player. Never
    #: summed: improving against your own recent past while losing to a
    #: fixed reference is specialisation, and an average hides it.
    matchups: list[dict] = field(default_factory=list)
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
    population=None,
    explore_share: float = 0.0,
    want_oracle: bool = False,
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
    require_training_engine()
    validate_budget(games, max_steps)
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
    #: Which seated player took a place in each game, or -1 for a table of
    #: the learner alone. The same thing `foreign_which` says, but valid
    #: only where somebody was actually seated, which is what a report of
    #: who-played-whom needs.
    seated_in = np.full(games, -1, dtype=np.int64)
    # Its own stream, so how much the round wanders cannot change what is
    # dealt, and turning exploration on or off leaves the games alone.
    wanderer = np.random.default_rng(seed ^ 0x3A17_9E55)
    if opponents and opponent_share > 0:
        picker = np.random.default_rng(seed ^ 0x0DDBA11)
        if population is not None:
            seated_in = population.seat(games, opponent_share, picker)
            taken = seated_in >= 0
            foreign_which[taken] = seated_in[taken]
        else:
            taken = picker.random(games) < opponent_share
            foreign_which[taken] = picker.integers(0, len(opponents), size=int(taken.sum()))
            seated_in[taken] = foreign_which[taken]
        foreign_player[taken] = picker.integers(0, 4, size=int(taken.sum()))

    # One block per step, holding the live games' rows in the order the
    # decisions are numbered below: a round is a few hundred blocks rather
    # than a few hundred thousand arrays, which the heap handles.
    observations: list[Planes] = []
    legal_masks: list[np.ndarray] = []
    held: list[np.ndarray] = []
    oracle: list[np.ndarray] = []
    imagined: list[np.ndarray] = []
    wandered: list[np.ndarray] = []
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
    # A learner that keeps its own account may have been deciding
    # elsewhere since, in a measurement; this round starts from nothing.
    own = getattr(net, "timing", None)
    if isinstance(own, dict):
        for name in own:
            own[name] = 0.0
    def settle() -> int:
        """Pays every hand that ended on the step just taken, and says how
        many did.

        Called after every `arena.step`, including the steps on which only
        foreign opponents decided. A hand can end on any of them, the
        engine forgets that it ended as soon as the next step is taken, and
        the decisions waiting on it are the learner's whoever made the last
        move. Missing one does not only lose that reward: the entries stay
        pending and are paid by the hand after, which credits a decision
        with points from a hand it was not part of.
        """
        ended = np.frombuffer(arena.hand_ended(), dtype=np.uint8)
        if not ended.any():
            return 0
        results = np.frombuffer(arena.hand_result(), dtype=np.int32).reshape(games, 4)
        counted = 0
        for game in np.nonzero(ended)[0]:
            counted += 1
            for person in range(4):
                value = float(results[game][person]) * HAND_SCALE
                for step_index in pending[game][person]:
                    rewards[step_index] += value
                pending[game][person] = []
        return counted

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
        # Every deciding player's view in one call, for whoever asks below.
        everyone = np.nonzero(live)[0]
        views.prepare(everyone, deciding[everyone])
        timing["encode"] += clock() - began
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
            # A step nobody recorded a decision on can still end a hand,
            # and the decisions waiting on that hand are owed its points
            # exactly as much as if the learner had made the last move.
            hands += settle()
            timing["engine"] += clock() - began
            continue
        choice = their_choice.copy()
        if hasattr(net, "decide"):
            # A learner in an action space of its own (see
            # `mortal_learner`): it answers the table in ours and records
            # its decisions itself, possibly more than one per row, and
            # keeps its own account of the time.
            # The precision of the rollout is the trainer's choice, made
            # here and nowhere inside: the probabilities recorded are the
            # ones the learning forward will reproduce.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
                picked, records = net.decide(
                    views, index, deciding[index], mask[index], greedy,
                    explore_share=0.0 if greedy else explore_share, wanderer=wanderer,
                )
            choice[index] = picked
            observations.append(records.planes)
            legal_masks.append(records.masks)
            record_actions = records.actions
            record_log_probs = records.log_probs
            record_slots = records.slots
            # What the three opponents were really holding, one row for each
            # decision recorded rather than one for each row of the table: a
            # riichi is two decisions from the one position, and the reading
            # of the hands is trained on both.
            held.append(truth[index][record_slots].copy())
            # The oracle's planes on this path too, when somebody is going
            # to read them. The fusion decides through here, so without
            # them its oracle critic sees nothing at all and cannot be
            # trained or even measured -- but nothing trains it yet, and
            # they are about a kilobyte a decision, which is most of a
            # gigabyte on a large round. Collected on request rather than
            # by default.
            if want_oracle:
                oracle.append(hidden[index][record_slots].astype(np.uint8))
            wandered.append(getattr(records, "forced", np.zeros(len(record_slots), dtype=bool)))
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
            if greedy:
                chosen = logits.argmax(dim=1)
                chosen_log_prob = distribution.log_prob(chosen)
                was_forced = torch.zeros_like(chosen, dtype=torch.bool)
            else:
                chosen, chosen_log_prob, was_forced = explore(
                    logits, batch_mask, explore_share, wanderer
                )
            record_actions = chosen.cpu().numpy()
            record_log_probs = chosen_log_prob.cpu().numpy()
            record_slots = np.arange(len(index))
            choice[index] = record_actions
            wandered.append(was_forced.cpu().numpy())

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
        hands += settle()
        timing["engine"] += clock() - began

    require_finished(arena, steps=steps, context="self-play")

    # A learner that decides for itself kept its own account; fold it in
    # and clear it for the next round.
    own = getattr(net, "timing", None)
    if isinstance(own, dict):
        for name, value in own.items():
            timing[name] = timing.get(name, 0.0) + value
            own[name] = 0.0

    # A round that ran out of steps has games still in progress, and their
    # final scores are whatever the table happened to hold when the clock
    # stopped. Turning those into placements labels an unfinished game as a
    # finished one and teaches the network from it. The limit is a guard
    # against a hand that will not end, not a budget to be spent, so
    # reaching it is a fault to be reported rather than worked around.
    if not arena.all_finished():
        unfinished = int((np.frombuffer(arena.seats(), dtype=np.uint8) != 0xFF).sum())
        raise RuntimeError(
            f"self-play stopped after {steps} steps with {unfinished} of {games} games "
            "unfinished; their placements would be invented, so the round is refused"
        )

    # Every game is over, so every hand ended, so every decision waiting on
    # a hand was paid by it and nothing can still be waiting. An entry left
    # here is the sign of a hand whose ending was not noticed: it would have
    # gone unpaid and then been paid by the hand after, crediting a decision
    # with points from a hand it took no part in. The check is cheap and the
    # fault is silent, which is exactly when to keep one.
    stranded = sum(len(pending[game][person]) for game in range(games) for person in range(4))
    if stranded:
        raise RuntimeError(
            f"{stranded} decisions were still waiting on a hand that had already ended; "
            "a hand's end was missed and its points would have been credited to the wrong hand"
        )

    # The engine counted the hands too, and it cannot miss one. If the two
    # counts differ then an ending went by unnoticed, and the decisions
    # waiting on that hand were paid by the hand after it — credited with
    # points from a hand they took no part in. Nothing else shows it: the
    # decisions are paid, once each, by the wrong hand, and every total
    # still adds up.
    truly = int(np.asarray(arena.hands_done(), dtype=np.int64).sum())
    if truly != hands:
        raise RuntimeError(
            f"the engine finished {truly} hands and the collector noticed {hands}; "
            f"{truly - hands} endings went by unseen, so their points reached the "
            "decisions of the hand that followed"
        )

    # The placement, which is what the game is actually for, reaches every
    # decision that player made.
    final_scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4)
    bonuses = placement_rewards(final_scores, PLACEMENT_VALUE)
    for game in range(games):
        for person in range(4):
            value = float(bonuses[game, person])
            for step_index in everything[game][person]:
                rewards[step_index] += value

    decisions = len(actions)
    if decisions == 0:
        raise RuntimeError("self-play produced no decisions")

    # How the learner placed in each game, so the round can say who it
    # played and how it did against each of them rather than only how it
    # did on average. Places are per person; the learner holds every seat
    # a foreign player did not, so its own placement is the mean of those.
    places = placements(final_scores)
    learner_place = np.zeros(games, dtype=np.float64)
    for game in range(games):
        mine = [person for person in range(4) if person != foreign_player[game]]
        learner_place[game] = float(np.mean([places[game][person] for person in mine]))
    against: list[dict] = []
    if population is not None and opponents:
        from .population import matchups as _matchups

        against = _matchups(seated_in, learner_place, population.members)

    return Batch(
        seated=seated_in,
        matchups=against,
        observations=Planes.cat(observations),
        legal=gather(legal_masks),
        actions=torch.tensor(actions, dtype=torch.int64),
        # A learner of the zoo's kind records none of the labels the
        # auxiliary heads want, having no such heads.
        held=gather(held) if held else torch.zeros(0),
        oracle=gather(oracle) if oracle else torch.zeros(0),
        imagined=gather(imagined) if imagined else torch.zeros(0),
        explored=gather(wandered) if wandered else None,
        returns=torch.tensor(rewards, dtype=torch.float32),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        games=games,
        hands=hands,
        decisions=decisions,
        final_scores=final_scores.copy(),
        timing=timing,
    )


@torch.no_grad()
def evaluate_games(
    net, games: int, seed: int, device: str = "cuda", amp: bool = False,
    max_steps: int = 4000, place: int = 0,
) -> tuple[np.ndarray, int]:
    """Plays the network against three heuristic opponents.

    The network takes the fixed player identity `place`, not a wind: seats
    rotate between hands. Return final scores by player and the hand count,
    without retaining training records. Raw Mortal-space networks are adapted
    before either observation or action masks reach their forward method.

    It plays its best move rather than sampling, because that is what the
    web app does. Measuring sampled play would mix how well the network has
    learned with how much exploration noise is on top of it, and then the
    checkpoint kept as best would be chosen partly on that noise.

    This is a score-only loop rather than `play(..., greedy=True)`: it does
    not fetch oracle/truth labels, construct rewards, or retain a round-sized
    training batch. Imagined worlds use an independent native RNG, so
    benchmarking has no need to generate unused hidden-hand proposals.
    """
    require_training_engine()
    validate_budget(games, max_steps)
    if isinstance(place, bool) or not isinstance(place, (int, np.integer)) or not 0 <= place < 4:
        raise ValueError("place must be a player index from 0 to 3")
    if getattr(net, "speaks_mortal", False):
        net = zoo.MortalSpacePlayer(net, device)
    net.eval()
    arena = riichi_py.Arena(games=games, seed=seed,
                            bot_places=[player for player in range(4) if player != place])
    views = Views(arena, games, {net.kind})
    hands = 0
    steps = 0
    while not arena.all_finished() and steps < max_steps:
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
        if np.any(deciding != place):
            raise RuntimeError("The arena exposed a heuristic opponent's decision; rebuild riichi_py")

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
            logits, _value, _guessed = net.everything(batch_planes, batch_mask)
        logits = logits.float()

        choice[index] = logits.argmax(dim=1).cpu().numpy()
        arena.step(choice.tolist())
        hands += int(np.frombuffer(arena.hand_ended(), dtype=np.uint8).sum())

    require_finished(arena, steps=steps, context="measurement")
    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, 4).copy()
    return scores, hands


@torch.no_grad()
def measure(
    net, games: int, seed: int, device: str = "cuda", amp: bool = False,
    max_steps: int = 4000,
) -> dict[str, float]:
    """Score player 0 against three independent heuristic opponents."""
    scores, hands = evaluate_games(net, games, seed, device, amp, max_steps)
    order = placements(scores)
    return {
        "placement": float(order[:, 0].mean()),
        "score": float(scores[:, 0].mean()),
        "wins": float(win_shares(scores)[:, 0].mean()),
        "hands": hands,
    }
