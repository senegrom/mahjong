"""Does the network play better when it thinks ahead?

The network supplies three things and the engine does the rest: an order
over the moves worth trying, what it takes the opponents to be holding, and
what a position is worth. The engine imagines the worlds and makes the
moves; the network values every position that results in one pass; the
engine picks. Nothing is played out to the end by a heuristic, which is
what the first version did and what measured worse than not searching.

Between the candidate move and the position that is valued, the other
seats have to be moved by somebody. By default that is the club heuristic;
with `--played-by network` it is the network itself, which is what the
opponents are in self-play and what the strong searchers do, the engine
handing every decision the imagined worlds are waiting on to the policy in
one batch at a time. With the network moving them, `--depth` lets it play
the searching player's own next turns too before the position is valued,
in the manner of OLSS's policy-guided depth.

The comparison is the same deals four times over with the searching player
in each chair, and its error bar comes from the deals rather than the four
seatings, for the reason set out in `arena.py`: the seatings share their
deals and are nowhere near independent.

    python -m neural.searched E:/tmp-claude/mahjong/big-run/best.pt --games 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import statistics
import sys
import time

import numpy as np
import torch

from .outcomes import placements as tied_placements, require_finished, validate_budget, win_shares

import riichi_py

from . import contract as contract_module
from . import worlds as worlds_module
from . import zoo
from .contract import UnsupportedSearchLayout
from .training_safety import require_training_engine, require_search_engine

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS
HANDS = riichi_py.HANDS
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES
SEATS = 4


def require_native_search(net) -> None:
    """The lookahead in which the network moves the other seats asks the
    engine for its own observations at every decision inside the search,
    and those are the engine's planes in our moves. Only a network of that
    lineage can answer them. The club-played search is served more widely:
    see `neural.contract`, whose `UnsupportedSearchLayout` this raises."""
    require_training_engine()
    planes = getattr(net, "planes", PLANES)
    planes = planes() if callable(planes) else planes
    actions = getattr(net, "actions", ACTIONS)
    if planes != PLANES or actions != ACTIONS or getattr(net, "kind", "engine") != "engine":
        raise UnsupportedSearchLayout(
            f"Native lookahead requires {PLANES} engine planes and {ACTIONS} actions; "
            f"got {planes} planes, {actions} actions ({getattr(net, 'kind', 'unknown')}). "
            "Hypothetical leaves have no Mortal event history. Use neural.arena or the "
            "network-only baseline for current checkpoints; do not pad or relabel engine planes."
        )


@torch.no_grad()
def play_lookahead(net, arena, *, device="cuda", temperature=0.0, passes=8000):
    """Plays every decision the lookaheads are waiting on with the policy
    until none is left: the network moving the other seats inside the
    search, and the searching player's own turns beyond the first when a
    depth was asked for. Its best move at temperature zero, a sample
    otherwise. Returns how many passes of the policy it took; a slot still
    waiting after `passes` is given up on and does not count."""
    require_native_search(net)
    taken = 0
    while taken < passes:
        planes_bytes, masks_bytes, count = arena.lookahead_owed()
        if count == 0:
            break
        taken += 1
        planes = np.frombuffer(planes_bytes, dtype=np.float32).reshape(count, PLANES, POSITIONS)
        masks = np.frombuffer(masks_bytes, dtype=np.uint8).reshape(count, ACTIONS).astype(bool)
        actions = np.empty(count, dtype=np.int64)
        step = 8192
        for start in range(0, count, step):
            rows = slice(start, start + step)
            logits, _value = net(
                torch.from_numpy(planes[rows]).to(device),
                torch.from_numpy(masks[rows]).to(device),
            )
            if temperature > 0:
                odds = torch.softmax(logits.float() / temperature, dim=1)
                picked = torch.multinomial(odds, 1).squeeze(1)
            else:
                picked = logits.argmax(dim=1)
            actions[rows] = picked.cpu().numpy()
        arena.lookahead_apply(actions.tolist())
    return taken


@torch.no_grad()
def search_with_value_head(
    net,
    arena,
    ranked,
    belief_flat,
    *,
    worlds,
    candidates,
    margin,
    hurried,
    device="cuda",
    pool=4,
    played_by="club",
    depth=0,
    temperature=0.0,
    valued_by="critic",
    health=None,
    served=None,
    leaf_batch=contract_module.DEFAULT_LEAF_BATCH,
):
    """One searched decision for every live game, valued by the network.

    `ranked` is the network's move order per game, best first; `belief_flat`
    is its belief about the opponents' hands, one row of HANDS per game.

    The hidden hands are weighed, not only sampled. The engine imagines
    `pool` times `worlds` worlds from the belief's per-tile marginals,
    which is only a proposal; the reader says of each how much more likely
    its hidden hands are than the proposal made them; the `worlds` most
    plausible are kept with those weights and the rest dropped. The engine
    makes the moves in the kept worlds, the network values every resulting
    position in a single pass, and the engine picks, keeping the first move
    unless another beats it by `margin` standard errors of the weighted
    world-by-world difference.

    `played_by` says who moves the other seats between the candidate move
    and the leaf: the club heuristic, or the network itself. With the
    network, `depth` is how many of the searching player's own turns it
    plays before the position is valued, and `temperature` whether those
    moves are its best (zero) or sampled. `valued_by` names the head that
    judges the leaves.
    """
    served = served or contract_module.serve(net)
    require_search_engine()
    if type(leaf_batch) is not int or leaf_batch <= 0:
        raise ValueError("leaf_batch must be a positive integer")
    if played_by == "network" and served.contract.reads != "mortal":
        require_native_search(net)
    games = len(ranked)
    # A stream of its own for choosing worlds, so which ones are drawn does
    # not depend on, or disturb, anything else. The seed moves with the
    # position so two identical calls agree and successive ones do not.
    seed_for_worlds = 0x51ED ^ (len(ranked) * 1_000_003) ^ int(sum(map(len, ranked)))
    efficiency: list[float] = []
    distinct: list[int] = []
    hands_bytes, counts = arena.imagine(belief_flat, worlds=pool * worlds)
    total = sum(counts)
    kept = [[] for _ in range(games)]
    weights = [[] for _ in range(games)]
    if total:
        weighs = served.contract.reads == "engine" and hasattr(net, "read_plausibility")
        if weighs:
            hands = np.frombuffer(hands_bytes, dtype=np.float32)
            hands = hands.reshape(total, HIDDEN_HANDS_PLANES, POSITIONS)
            public = np.frombuffer(arena.observations(), dtype=np.float32)
            public = public.reshape(games, PLANES, POSITIONS)
            game_of = np.repeat(np.arange(games), counts)
            plausible = np.empty(total, dtype=np.float32)
            step = 4096
            for start in range(0, total, step):
                rows = slice(start, start + step)
                position = torch.from_numpy(public[game_of[rows]]).to(device)
                shown = torch.from_numpy(hands[rows]).to(device)
                plausible[rows] = net.read_plausibility(position, shown).float().cpu().numpy()
        else:
            # The reader that weighs a world reads our own planes beside the
            # hands, and a network on Mortal's planes has no such reader.
            # Its worlds count evenly, which is what a reader that had
            # learned nothing would give: the unweighted estimator.
            plausible = np.zeros(total, dtype=np.float32)
        offset = 0
        # Drawn in proportion to the reader's weights rather than taken from
        # the top of them: see `neural.worlds`. Keeping the likeliest worlds
        # and renormalising throws away the mass below the cut, and in this
        # game the hand that decides whether a discard was a mistake is
        # usually the unlikely one.
        picker = np.random.default_rng(seed_for_worlds)
        for game, count in enumerate(counts):
            if count == 0:
                continue
            scores = plausible[offset : offset + count]
            offset += count
            if len(ranked[game]) < 2:
                # Nothing to compare: no worlds are kept, the engine builds
                # this game no lookahead, and its one move comes straight
                # back. Rolling the other seats' single choices out to the
                # end of the hand was most of the search's cost and none of
                # its answer.
                continue
            chosen = worlds_module.resample(scores, worlds, picker)
            kept[game] = chosen.kept
            weights[game] = chosen.weights
            efficiency.append(chosen.efficiency)
            distinct.append(chosen.distinct)
    if played_by == "network":
        # A depth below zero bypasses the root-hand critic. After the
        # root hand ends, the network plays to terminal placement without
        # adding later hands' score changes to this decision's reward.
        arena.lookahead_begin(
            ranked, kept, weights, candidates=candidates, depth=max(depth, 0),
            until_hand_ends=depth < 0,
        )
        # The seats inside the lookahead are moved by the network on the
        # planes it reads: the engine's own, or Mortal's through copies
        # of every seat's state kept in step (`MortalServed`).
        if served.contract.reads == "mortal":
            served.play_lookahead(arena, device=device, temperature=temperature)
        else:
            play_lookahead(net, arena, device=device, temperature=temperature)
        planes_bytes, counts, _settled, _wanted = arena.lookahead_leaves()
    else:
        planes_bytes, counts, _settled, _wanted = arena.leaves_from(
            ranked, kept, weights, candidates=candidates, hurried=hurried
        )
    total = sum(counts)
    if total == 0:
        return arena.decide([], margin, ranked)
    # The leaves as *this* network reads them. The engine writes its own
    # ninety-seven planes for every one, which is right for a network of
    # its own lineage and useless for one that reads Mortal's thousand and
    # twelve: that one gets a copy of the seat's real state advanced by the
    # events its imagined world invented. Root and continuation go through
    # the same contract, because a search that serves one correctly and the
    # other some other way is measuring a network that does not exist.
    valued = np.zeros(total, dtype=np.float32)
    for slots, chunk in served.leaf_batches(
        arena, planes_bytes, counts, device, wanted=_wanted, batch_size=leaf_batch,
    ):
        values = served.value(chunk, head=valued_by).float().cpu().numpy()
        if values.shape != (len(slots),) or not np.isfinite(values).all():
            raise ValueError("The search critic must return one finite value per wanted leaf")
        valued[slots] = values
        del chunk
    if health is not None and efficiency:
        # How much of the proposal the weights actually used, and how many
        # distinct worlds survived. An efficiency near zero means the search
        # is running on a handful of worlds whatever `worlds` was asked for,
        # and its margin is measuring the spread of those few.
        health.setdefault("efficiency", []).extend(efficiency)
        health.setdefault("distinct", []).extend(distinct)
    return arena.decide(valued.tolist(), margin, ranked)


@torch.no_grad()
def play(
    net,
    games: int,
    seed: int,
    searcher: int | None,
    worlds: int,
    candidates: int,
    margin: float,
    pool: int = 4,
    hurried: bool = True,
    device: str = "cuda",
    played_by: str = "club",
    depth: int = 0,
    temperature: float = 0.0,
    valued_by: str = "critic",
    served=None,
    leaf_batch: int = contract_module.DEFAULT_LEAF_BATCH,
    max_steps: int = 4000,
    recording: "Recording | None" = None,
    sure: float = 1.0,
    health: dict | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Plays `games` games out and returns the final scores.

    With a `recording`, every searched decision is added to it: the root
    as the network read it, the candidates and what each came to in every
    world, beside the policy's choice and the search's. Roots are kept
    sparse, so only a network on Mortal's planes can be recorded.

    `searcher` is the place that thinks ahead, or None for nobody. Everyone
    else plays the network's first choice, so the only thing that differs
    between the two arms is whether that choice was checked.

    `sure` is how sure the policy must be of its first move, as its
    probability, for that move to be taken at its word without a search;
    one searches every decision of the searcher's. Only its own decisions
    with two or more legal moves cost worlds at all: the other seats' and
    the forced ones are answered straight from the order. `health`, when
    given, is filled with how many own decisions there were (`own`) and
    how many were taken sure (`sure`).
    """
    require_training_engine()
    validate_budget(games, max_steps)
    if searcher is None:
        from . import duel
        player = zoo.MortalSpacePlayer(net, device) if getattr(net, "speaks_mortal", False) else net
        scores = duel.table(player, player, games, seed, 0, device, max_steps=max_steps)
        return scores, (0, 0)
    net.eval()
    # Refused here, before an arena exists, if nothing can serve it: a
    # search that cannot build the planes a network reads must not write
    # a tally that looks like a measurement.
    served = served or contract_module.serve(net)
    if played_by == "network" and served.contract.reads != "mortal":
        require_native_search(net)
    arena = riichi_py.Arena(games=games, seed=seed, bot_places=[])
    # The network that answers, out of whatever player wrapped it, and
    # for one reading Mortal's planes the follower its root is built from;
    # the leaves are copied from the same states.
    net = served.net
    if served.contract.reads == "mortal":
        from .observe import Views

        views = Views(arena, games, {served.contract.reads})
        contract_module.remember_follower(arena, views.observer.follower)
    else:
        views = None
    # How much of each proposal the reader's weights actually used. A
    # search whose worlds all come from a handful of proposals is not
    # searching the number of worlds it was asked for.
    health = {} if health is None else health
    steps = 0
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        if not (seats != 0xFF).any():
            break

        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(games, ACTIONS).astype(bool)
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)
        live = seats != 0xFF
        rows = np.nonzero(live)[0]
        deciding = players[rows, np.minimum(seats[rows], 3)].astype(np.int64)
        if views is not None:
            views.advance()
            views.prepare(rows, deciding)

        # The root as this network reads it, through the same contract as
        # the leaves below: its order over our moves, with a reach's second
        # question asked wherever one is legal (`contract.root_order`).
        order, logits, _value, guessed = contract_module.root_order(
            served, arena, views, rows, deciding, mask, device
        )
        belief = np.zeros((games, HANDS), dtype=np.float32)
        belief[rows] = torch.softmax(guessed.float(), dim=2).reshape(len(rows), HANDS).cpu().numpy()
        # How sure the policy is of its first move: the probability it put
        # on it, in whichever moves it answers in.
        top = np.ones(games, dtype=np.float32)
        top[rows] = np.nan_to_num(
            torch.softmax(logits.float(), dim=1).max(dim=1).values.cpu().numpy(), nan=1.0
        )

        if searcher is None:
            choice = order[:, 0].tolist()
        else:
            # Only the searching player's own turns are searched, and only
            # those where the policy hesitates and has a choice; the others
            # take the network's first move, which is what `ranked` gives
            # back when a game is not theirs to think about. A single
            # candidate costs no worlds (`search_with_value_head`), and the
            # candidates are the legal head of the order: the order's tail
            # names every move so a caller may read only its head.
            ranked = []
            for game in range(games):
                seat = int(seats[game])
                own = seat != 0xFF and int(players[game][seat]) == searcher
                thinking = own and top[game] < sure
                if own:
                    health["own"] = health.get("own", 0) + 1
                    if not thinking:
                        health["sure"] = health.get("sure", 0) + 1
                if thinking:
                    first = order[game][:candidates]
                    first = first[mask[game][first]]
                else:
                    first = order[game][:1]
                ranked.append([int(index) for index in first] or [int(order[game][0])])
            choice = search_with_value_head(
                net,
                arena,
                ranked,
                belief.reshape(-1).tolist(),
                worlds=worlds,
                candidates=candidates,
                margin=margin,
                hurried=hurried,
                device=device,
                pool=pool,
                played_by=played_by,
                depth=depth,
                temperature=temperature,
                valued_by=valued_by,
                health=health,
                served=served,
                leaf_batch=leaf_batch,
            )
            if recording is not None:
                if views is None:
                    raise UnsupportedSearchLayout(
                        "a recording keeps the root as sparse Mortal planes; this network "
                        "reads the engine's, which nothing here records"
                    )
                judgements = arena.judgements()
                for at, game in enumerate(rows):
                    judged = judgements[game]
                    if len(judged) < 2:
                        continue
                    root, _own = views.sparse_and_masks(rows[at : at + 1], deciding[at : at + 1])
                    recording.add(
                        root, judged, int(order[game][0]), int(choice[game]),
                        int(searcher), int(game), steps, sure=float(top[game]),
                        legal=mask[game],
                    )
                recording.checkpoint()
        arena.step(list(choice))

    require_finished(arena, steps=steps, context="search evaluation")
    scores = np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()
    if sure < 1.0 and health.get("own"):
        print(
            f"  sure: {health.get('sure', 0)} of {health['own']} own decisions were taken at "
            f"the policy's word, its first move above {sure:.2f}",
            file=sys.stderr,
            flush=True,
        )
    if health.get("efficiency"):
        mean = float(np.mean(health["efficiency"]))
        print(
            f"  worlds: the reader's weights used {mean:.1%} of each proposal on average, "
            f"{float(np.mean(health['distinct'])):.1f} distinct worlds kept",
            file=sys.stderr,
            flush=True,
        )
    return scores, arena.search_tally()


def placements(scores: np.ndarray, place: int) -> np.ndarray:
    return tied_placements(scores)[:, place]


class Recording:
    """What a search saw and judged, decision by decision, on Mortal's
    planes: the root, the candidates in the network's order, what each
    came to in every world, and the move the policy would have made
    beside the one the search made. Written as one sparse block of roots
    and arrays alongside, padded to the widest candidate list and world
    count with -1 and NaN.

    The raw material for a critic that ranks siblings: every row is a
    position with several moves tried in the same worlds, which is what
    self-play never shows a value head.

    Given a `folder` and `meta`, `checkpoint()` writes what is recorded so
    far there every `every` seconds, marked incomplete, so a search killed
    mid-run leaves its decisions behind rather than nothing; `save` at the
    end writes the whole, marked complete.
    """

    def __init__(self, folder=None, meta: dict | None = None, every: float = 600.0) -> None:
        self.roots: list = []
        self.candidates: list[list[int]] = []
        self.values: list[list[float]] = []
        self.per_world: list[list[list[float]]] = []
        self.policy: list[int] = []
        self.search: list[int] = []
        self.chair: list[int] = []
        self.game: list[int] = []
        self.step: list[int] = []
        #: How sure the policy was of its first move there: its probability.
        self.sure: list[float] = []
        #: What the engine allowed at the root, in our moves: what a policy
        #: taught from this recording is asked under, and leashed under.
        self.legal: list[np.ndarray] = []
        self.folder = Path(folder) if folder is not None else None
        self.meta = dict(meta or {})
        self.every = every
        self.saved_at = time.perf_counter()
        self.saved_rows = 0

    def __len__(self) -> int:
        return len(self.candidates)

    def add(
        self, root, judged, policy: int, search: int, chair: int, game: int, step: int,
        sure: float = 1.0, legal=None,
    ) -> None:
        self.roots.append(root)
        self.candidates.append([int(index) for index, _value, _worlds in judged])
        self.values.append([float(value) for _index, value, _worlds in judged])
        self.per_world.append([[float(w) for w in worlds] for _index, _value, worlds in judged])
        self.policy.append(int(policy))
        self.search.append(int(search))
        self.chair.append(int(chair))
        self.game.append(int(game))
        self.step.append(int(step))
        self.sure.append(float(sure))
        self.legal.append(
            np.array(legal, dtype=bool, copy=True) if legal is not None else np.ones(ACTIONS, dtype=bool)
        )

    def checkpoint(self) -> bool:
        """Writes what is recorded so far to the folder given at
        construction, when there is one, something new, and the interval
        has passed."""
        if self.folder is None or len(self) == self.saved_rows:
            return False
        if time.perf_counter() - self.saved_at < self.every:
            return False
        self.save(self.folder, self.meta, complete=False)
        return True

    def save(self, folder, meta: dict, complete: bool = True) -> None:
        from .recordings import write_snapshot

        write_snapshot(Path(folder), self._write, {**meta, "complete": bool(complete)})
        self.saved_at = time.perf_counter()
        self.saved_rows = len(self)

    def _write(self, folder: Path, meta: dict) -> None:
        from .observe import Planes

        rows = len(self)
        width = max((len(c) for c in self.candidates), default=0)
        worlds = max((len(w) for c in self.per_world for w in c), default=0)
        candidates = np.full((rows, width), -1, dtype=np.int64)
        values = np.full((rows, width), np.nan, dtype=np.float32)
        per_world = np.full((rows, width, worlds), np.nan, dtype=np.float32)
        for at in range(rows):
            n = len(self.candidates[at])
            candidates[at, :n] = self.candidates[at]
            values[at, :n] = self.values[at]
            for k, w in enumerate(self.per_world[at]):
                per_world[at, k, : len(w)] = w
        # `cat` frees the blocks it joins, so the joined block takes their
        # place: compact, and still here for the next write.
        joined = Planes.cat(self.roots)
        self.roots = [joined]
        joined.save(folder, "root")
        np.save(folder / "candidates.npy", candidates)
        np.save(folder / "values.npy", values)
        np.save(folder / "per_world.npy", per_world)
        np.save(folder / "policy.npy", np.asarray(self.policy, dtype=np.int64))
        np.save(folder / "search.npy", np.asarray(self.search, dtype=np.int64))
        np.save(folder / "chair.npy", np.asarray(self.chair, dtype=np.int64))
        np.save(folder / "game.npy", np.asarray(self.game, dtype=np.int64))
        np.save(folder / "step.npy", np.asarray(self.step, dtype=np.int64))
        np.save(folder / "sure.npy", np.asarray(self.sure, dtype=np.float32))
        np.save(folder / "legal.npy", np.stack(self.legal) if self.legal else np.zeros((0, ACTIONS), dtype=bool))
        (folder / "meta.json").write_text(json.dumps({**meta, "rows": rows}, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=200, help="deals per chair")
    parser.add_argument("--seed", type=int, default=90_210)
    parser.add_argument("--worlds", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--pool",
        type=int,
        default=4,
        help="how many worlds are imagined for each one kept: the reader "
        "weighs the pool and keeps the most plausible. A pool of one is "
        "the sampled search, with no weighing at all",
    )
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--leaf-batch", type=int, default=contract_module.DEFAULT_LEAF_BATCH,
                        help="maximum leaves reconstructed, densified and valued together")
    parser.add_argument(
        "--played-by",
        choices=("club", "network"),
        default="club",
        help="who moves the other seats between the candidate move and the "
        "position that is valued: the club heuristic, or the network itself",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="with the network moving the other seats, how many of the "
        "searching player's own turns it plays before the position is "
        "valued; zero values the next one in this hand. Below zero bypasses "
        "the critic. Any crossed-hand world plays on to terminal placement",
    )
    parser.add_argument(
        "--valued-by",
        choices=("critic", "public", "mean"),
        default="critic",
        help="which head judges the leaves: the critic with a tower of its "
        "own, the public head on the policy tower's pooled features, or "
        "their mean. Every generation measures all three on a fresh round "
        "before updating them, under public_error, oracle_error and "
        "critic_error in the training log",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="how the network's moves inside the lookahead are drawn: its "
        "best at zero, a sample of its policy above",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where the network runs. The card by default, and the "
        "processor when there is none, which is slow but lets a command "
        "line be tried while a run holds the card",
    )
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="a directory to write every searched decision to: the root as "
        "the network read it, the candidates and what each came to in "
        "every world, the policy's choice and the search's",
    )
    parser.add_argument(
        "--chair",
        type=int,
        default=-1,
        help="one chair to play rather than all four, so the chairs of one "
        "measurement can run side by side; the deals are the same, so "
        "their per-deal placements pair up afterwards",
    )
    parser.add_argument(
        "--sure",
        type=float,
        default=1.0,
        help="how sure the policy must be of its first move, as its "
        "probability, for the move to be taken at its word without a "
        "search: one searches every own decision; nine tenths searches "
        "only where the policy hesitates, which is where a search has "
        "ever changed anything, at a fraction of the cost",
    )
    parser.add_argument(
        "--save-every",
        type=float,
        default=600.0,
        help="seconds between partial writes of the recording, so a run "
        "killed halfway leaves its decisions behind",
    )
    args = parser.parse_args()

    # Pin the bytes before loading or recording a digest. A trainer may
    # replace the caller's latest.pt while this long measurement runs.
    from .checkpoints import copy_checkpoint

    with tempfile.TemporaryDirectory(prefix="mahjong-searched-") as folder:
        checkpoint = Path(folder) / "checkpoint.pt"
        generation = copy_checkpoint(Path(args.checkpoint), checkpoint)
        _run(args, checkpoint, generation)


def _run(args, checkpoint: Path, generation: int | None) -> None:
    # Loaded whole, then asked what it reads and answers in. The server
    # that comes back builds the planes for the root and for every imagined
    # continuation alike; a network nothing here can serve is refused by
    # name rather than approximated with another's planes.
    net = zoo.load_player(checkpoint, args.device, args.channels, args.blocks)
    served = contract_module.serve(net, str(args.checkpoint))
    print(json.dumps({"contract": served.contract.describe()}), flush=True)

    per_chair = []
    per_deal = []
    asked = overrode = own = taken_sure = 0
    chairs = [args.chair] if 0 <= args.chair < SEATS else list(range(SEATS))
    from .recordings import digest_file

    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest_file(checkpoint),
        "checkpoint_generation": generation,
        "temperature": args.temperature,
        "leaf_batch": args.leaf_batch,
        "device": args.device,
        "save_every": args.save_every,
        "contract": served.contract.describe(),
        "games": args.games,
        "seed": args.seed,
        "worlds": args.worlds,
        "pool": args.pool,
        "candidates": args.candidates,
        "margin": args.margin,
        "played_by": args.played_by,
        "depth": args.depth,
        "valued_by": args.valued_by,
        "sure": args.sure,
        "chairs": chairs,
    }
    recording = Recording(args.record, meta, every=args.save_every) if args.record is not None else None
    for chair in chairs:
        began = time.perf_counter()
        health: dict = {}
        scores, tally = play(
            net,
            games=args.games,
            seed=args.seed,
            searcher=chair,
            worlds=args.worlds,
            candidates=args.candidates,
            margin=args.margin,
            pool=args.pool,
            device=args.device,
            played_by=args.played_by,
            depth=args.depth,
            temperature=args.temperature,
            valued_by=args.valued_by,
            served=served,
            leaf_batch=args.leaf_batch,
            recording=recording,
            sure=args.sure,
            health=health,
        )
        asked += tally[0]
        overrode += tally[1]
        own += health.get("own", 0)
        taken_sure += health.get("sure", 0)
        got = placements(scores, chair)
        print(
            f"chair {chair}: placement {got.mean():.3f} over {args.games} games, "
            f"search changed {tally[1]} of {tally[0]} decisions, "
            f"{health.get('sure', 0)} of {health.get('own', 0)} own decisions taken sure, "
            f"{time.perf_counter() - began:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        per_chair.append(
            {
                "chair": chair,
                "placement": float(got.mean()),
                "score": float(scores[:, chair].mean()),
                "wins": float(win_shares(scores)[:, chair].mean()),
            }
        )
        per_deal.append(got.astype(float))

    if recording is not None:
        recording.save(args.record, meta)
        print(f"recorded {len(recording)} searched decisions to {args.record}", file=sys.stderr, flush=True)

    overall = statistics.fmean(row["placement"] for row in per_chair)
    paired = np.stack(per_deal).mean(axis=0)
    error = float(paired.std(ddof=1) / (len(paired) ** 0.5))
    edge = 2.5 - overall
    sigmas = edge / error if error else 0.0

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "worlds": args.worlds,
                "pool": args.pool,
                "played_by": args.played_by,
                "valued_by": args.valued_by,
                "depth": args.depth,
                "temperature": args.temperature,
                "candidates": args.candidates,
                "margin": args.margin,
                "sure": args.sure,
                "games_total": args.games * len(chairs),
                "chairs": chairs,
                "device": args.device,
                "placement": overall,
                "standard_error": error,
                "difference_from_level": edge,
                "standard_errors": sigmas,
                "by_chair": per_chair,
                # One figure a deal: the four chairs' placements on that
                # deal, averaged. Two arms run at the same seed share their
                # deals exactly, so subtracting these elementwise gives a
                # paired difference whose error is the honest one.
                "by_deal": [round(float(value), 4) for value in paired],
                # Asked counts the decisions with two or more legal moves
                # that were searched; the other seats' and the forced ones
                # cost nothing and are not in it.
                "overrides": f"{overrode} of {asked}, {100.0 * overrode / max(asked, 1):.1f}%",
                "own_decisions": own,
                "taken_sure": taken_sure,
                "verdict": (
                    "searching helps"
                    if sigmas > 2
                    else "searching hurts"
                    if sigmas < -2
                    else "not settled either way, which needs more games"
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
