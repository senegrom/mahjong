"""Teaches the network the moves its own search came to.

This is the policy improvement half of the AlphaZero idea, in the form this
game allows. The network proposes an order over the moves and a belief
about the hidden hands; the search makes the top few moves in worlds drawn
from that belief and has the network's own value head judge what results;
whichever move survives that is a better move than the one proposed, or the
search is worth nothing. Training the network towards it makes the next
proposal better. This native-layout script now learns frozen policy-improvement
targets and completed-game returns using the evaluator selected for search.
Search quality must still be established independently; these targets are not
tree visit counts, and current Mortal-layout training is a separate path.

It also solves the practical problem with search, which is that half a
second a decision is hopeless in a browser. A network taught the search's
answers plays them without searching.

    python -m neural.distil --resume E:/tmp-claude/mahjong/big-run/best.pt \
        --rounds 200 --games 24 --out E:/tmp-claude/mahjong/distilled
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import atomic_save
from .auxiliary_training import validate_auxiliary_options, require_supervised_rows
from .training_batches import require_updates
from .search_learning import improvement_targets, masked_policy_loss
from torch import nn

import riichi_py

from . import ledger
from . import worlds as worlds_module
from .model import from_payload
from .selfplay import measure
from .searched import require_native_search, UnsupportedSearchLayout
from .outcomes import require_finished, validate_budget
from .training_safety import TRAINING_API_VERSION, require_training_engine

PLANES = riichi_py.PLANES
POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS
OPPONENTS = riichi_py.OPPONENTS
HANDS = riichi_py.HANDS
HIDDEN_HANDS_PLANES = riichi_py.HIDDEN_HANDS_PLANES


@torch.no_grad()
def search_with_value_head(
    net, arena, ranked, belief_flat, *, worlds, candidates, margin, hurried, device="cuda",
    pool=4, health=None, valued_by="critic",
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
    """
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
            chosen = worlds_module.resample(scores, worlds, picker)
            kept[game] = chosen.kept
            weights[game] = chosen.weights
            efficiency.append(chosen.efficiency)
            distinct.append(chosen.distinct)
    planes_bytes, counts, _settled, _wanted = arena.leaves_from(
        ranked, kept, weights, candidates=candidates, hurried=hurried
    )
    total = sum(counts)
    if total == 0:
        return arena.decide([], margin, ranked)
    planes = np.frombuffer(planes_bytes, dtype=np.float32).reshape(total, PLANES, POSITIONS)
    # Every slot is valued, including the few that want no value; the engine
    # adds what it settled itself, ignores the rest, and that is cheaper
    # than gathering.
    valued = np.empty(total, dtype=np.float32)
    step = 8192
    for start in range(0, total, step):
        chunk = torch.from_numpy(planes[start : start + step]).to(device)
        valued[start : start + step] = net.value_only(chunk, head=valued_by).float().cpu().numpy()
    if health is not None and efficiency:
        # How much of the proposal the weights actually used, and how many
        # distinct worlds survived. An efficiency near zero means the search
        # is running on a handful of worlds whatever `worlds` was asked for,
        # and its margin is measuring the spread of those few.
        health.setdefault("efficiency", []).extend(efficiency)
        health.setdefault("distinct", []).extend(distinct)
    return arena.decide(valued.tolist(), margin, ranked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--games", type=int, default=24, help="tables per round")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--worlds", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument(
        "--hurried",
        action="store_true",
        default=True,
        help="run the other players round to the next decision without "
        "counting acceptance, which is most of what they cost",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument(
        "--improve", type=float, default=0.5,
        help="how much of the policy's own mass moves onto the move the "
             "search preferred. One is the old hard label; below it the "
             "target keeps the policy's opinion about everything the "
             "search did not rank. This search gives one improved move per "
             "position rather than the visit counts a tree would, so there "
             "are no counts to use and none are invented",
    )
    parser.add_argument(
        "--value-weight", type=float, default=0.5,
        help="how hard the critic is pulled towards what the searched "
             "games actually paid. At zero the evaluator is never "
             "corrected against the outcomes of the play its own opinions "
             "produced, which is what this loop was missing",
    )
    parser.add_argument("--valued-by", choices=("critic", "public", "mean"), default="critic")
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--measure-games", type=int, default=384)
    parser.add_argument("--seed", type=int, default=4_040_404)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def collect(net, args, seed: int, device: str) -> tuple[np.ndarray, ...]:
    """Plays a round where every decision is searched, keeping the answers.

    Every seat searches, so a round yields four times the positions a single
    searching seat would, and the play the positions come from is the play
    the network will actually meet.
    """
    require_native_search(net)
    net.eval()
    max_steps = getattr(args, "max_steps", 4000)
    validate_budget(args.games, max_steps)
    arena = riichi_py.Arena(games=args.games, seed=seed, bot_places=[])
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    held: list[np.ndarray] = []
    proposed: list[int] = []
    actor_policies: list[np.ndarray] = []
    # What each decision turned out to be worth. Without this the value
    # head learns nothing here at all: the search picks its moves using the
    # critic, the policy learns those picks, and the critic is never told
    # what any of it came to. That is not a loop, it is a network agreeing
    # with itself.
    account = ledger.Ledger(args.games)
    rows: list[int] = []

    steps = 0
    for _step in range(max_steps):
        steps += 1
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break

        planes = np.frombuffer(arena.observations(), dtype=np.float32)
        planes = planes.reshape(args.games, PLANES, POSITIONS)
        mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
        mask = mask.reshape(args.games, ACTIONS).astype(bool)
        truth = np.frombuffer(arena.opponent_hands(), dtype=np.float32)
        truth = truth.reshape(args.games, OPPONENTS, POSITIONS)

        logits, _value, guessed = net.everything(
            torch.from_numpy(planes).to(device),
            torch.from_numpy(mask).to(device),
        )
        actor_policy = torch.softmax(logits.float(), dim=1).cpu().numpy()
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        belief = torch.softmax(guessed, dim=2).reshape(args.games, HANDS).cpu().numpy()

        ranked = [
            [int(index) for index in order[game][: args.candidates]]
            for game in range(args.games)
        ]
        chosen = search_with_value_head(
            net,
            arena,
            ranked,
            belief.reshape(-1).tolist(),
            worlds=args.worlds,
            candidates=args.candidates,
            margin=args.margin,
            hurried=args.hurried,
            device=device,
            valued_by=getattr(args, "valued_by", "critic"),
        )

        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(args.games, 4)
        for game in np.nonzero(live)[0]:
            observations.append(planes[game])
            masks.append(mask[game])
            labels.append(int(chosen[game]))
            held.append(truth[game])
            proposed.append(int(order[game][0]))
            actor_policies.append(actor_policy[game].copy())
            person = int(players[game][min(int(seats[game]), 3)])
            rows.append(account.open(int(game), person))

        arena.step(list(chosen))
        account.settle(arena)

    require_finished(arena, steps=steps, context="search distillation")
    returns = account.close(arena)
    return (
        np.stack(observations),
        np.stack(masks),
        np.array(labels, dtype=np.int64),
        np.stack(held),
        np.array(proposed, dtype=np.int64),
        returns[np.asarray(rows, dtype=np.int64)],
        np.stack(actor_policies),
    )


def main() -> None:
    args = parse_args()
    validate_auxiliary_options(args)
    require_training_engine()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    payload = torch.load(args.resume, map_location=device, weights_only=True)
    if "combined" in payload or "model" not in payload:
        raise UnsupportedSearchLayout("Legacy distillation cannot load a Mortal/combined checkpoint as a standalone model")
    net = from_payload(payload, device, args.channels, args.blocks)
    require_native_search(net)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"device {device} | {net.channels}x{net.blocks} "
        f"| {net.parameter_count() / 1e6:.2f}M parameters | resumed {args.resume}",
        flush=True,
    )

    best_placement = float("inf")
    smoothed = None
    for round_index in range(args.rounds):
        began = time.time()
        planes, masks, labels, truth, proposed, returns, actor_policies = collect(
            net, args, args.seed + round_index * 977, device
        )
        require_supervised_rows(len(labels))
        played = time.time() - began
        observations = torch.from_numpy(planes).to(device)
        legal = torch.from_numpy(masks).to(device)
        targets = torch.from_numpy(labels).to(device)
        wanted_hands = torch.from_numpy(truth).to(device)
        wanted_returns = torch.from_numpy(returns).to(device)
        if wanted_returns.shape != targets.shape or not torch.isfinite(wanted_returns).all():
            raise ValueError("Outcome targets must be finite vectors aligned with decisions")
        wanted_policy = improvement_targets(torch.from_numpy(actor_policies).to(device),
                                            targets, legal, args.improve)

        # How often the search disagreed with what the network proposed. If
        # this is zero there is nothing to learn and the search is a no-op;
        # if it is most of the time the margin is not doing its job.
        changed = float((labels != proposed).mean())

        net.train()
        total_loss = 0.0
        seen = 0
        updates = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(len(targets), device=device)
            for start in range(0, len(targets), args.batch):
                picks = order[start : start + args.batch]
                if picks.numel() < 2:
                    continue
                logits, value, guessed = net.everything(
                    observations[picks], legal[picks]
                )

                # Fixed search-time targets do not move with the student between epochs.
                loss = masked_policy_loss(logits, wanted_policy[picks], legal[picks])
                value = net.value_only(observations[picks], head=args.valued_by)

                # And the critic learns what the searched games really came
                # to. Without this the evaluator is never corrected against
                # the outcomes of the play its own opinions produced, which
                # is the difference between a loop and a network agreeing
                # with itself.
                loss = loss + args.value_weight * nn.functional.mse_loss(
                    value, wanted_returns[picks]
                )

                # The table-reading head keeps learning here too: the label
                # is exact and the search depends on it.
                mine = wanted_hands[picks]
                holding = mine.sum(dim=2) > 0
                log_guess = torch.log_softmax(guessed, dim=2)
                reading = -(mine * log_guess).sum(dim=2)
                loss = loss + (reading * holding).sum() / holding.sum().clamp(min=1)

                optimiser.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite distillation loss; no optimizer step was applied")
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                optimiser.step()
                updates += 1
                total_loss += loss.item() * picks.numel()
                seen += picks.numel()

        require_updates(updates)
        record = {
            "optimizer_updates": updates,
            "valued_by": args.valued_by,
            "round": round_index,
            # What made this round, so a replay of it can be told apart
            # from one made by another actor under other rules.
            "actor": str(args.resume),
            "reward_version": ledger.REWARD_VERSION,
            "improve": args.improve,
            "search": {
                "worlds": args.worlds,
                "candidates": args.candidates,
                "margin": args.margin,
                "hurried": bool(args.hurried),
            },
            "positions": int(len(targets)),
            "search_changed": round(changed, 4),
            "loss": round(total_loss / max(seen, 1), 4),
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
        }

        if (round_index + 1) % args.measure_every == 0 or round_index == 0:
            against = measure(net, games=args.measure_games, seed=9_000 + round_index, device=device)
            record.update(
                {
                    "placement": round(against["placement"], 3),
                    "score": round(against["score"], 1),
                    "win_rate": round(against["wins"], 3),
                }
            )
            smoothed = (
                against["placement"]
                if smoothed is None
                else against["placement"] / 3 + smoothed * 2 / 3
            )
            record["smoothed"] = round(smoothed, 3)
            saved = {
                "model": net.state_dict(),
                "generation": round_index + 1,
                **net.payload_fields(),
                "training_api_version": TRAINING_API_VERSION,
                "reward_version": ledger.REWARD_VERSION,
                "valued_by": args.valued_by,
                "placement": against["placement"],
            }
            atomic_save(saved, args.out / "latest.pt")
            if smoothed < best_placement:
                best_placement = smoothed
                atomic_save(saved, args.out / "best.pt")
                record["best"] = True

        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    print("distilling finished", flush=True)


if __name__ == "__main__":
    main()
