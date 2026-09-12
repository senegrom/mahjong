"""Re-heads one of our networks into Mortal's action space.

Ours cuts the game into 78 moves and Mortal's into 46. Almost all of the
difference is riichi: one action of Mortal's stands for our thirty-four
"declare and discard this tile", because there the declaration is a move of
its own and the tile is asked for afterwards. While the two spaces differ,
joining the networks means mapping one onto the other, and that mapping
throws the declaration tile away.

So the trunk is kept and only the last layer is replaced, by one that
answers in Mortal's space, and it is taught to say what the old head said:

  * the old head's distribution over our 78, mapped onto the 46, which
    sums the whole riichi block onto Mortal's single reach and the three
    kinds of kan onto its one kan;
  * and, wherever the network actually declares riichi, the tile it chose,
    from the position in which the reach is already declared. That is the
    second question Mortal is asked, and it is where the declaration tile
    now lives.

Nothing but the two policy heads trains, so this is a change of clothes
rather than a change of player, and a duel against the original should say
so.

    python -m neural.rehead --teacher w1012-run/latest.pt --out runs/rehead
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
from torch import nn

import riichi_py

from . import zoo
from .model import MORTAL_ACTIONS, PolicyValueNet, from_payload, shape_of
from .observe import Planes, Views

POSITIONS = riichi_py.POSITIONS
ACTIONS = riichi_py.ACTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="the network to re-head")
    parser.add_argument("--resume", type=Path, default=None, help="a student this wrote before")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--games", type=int, default=64, help="tables per round")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", type=Path, default=Path("E:/tmp-claude/mahjong/rehead"))
    return parser.parse_args()


def student_from(teacher: PolicyValueNet, device: str) -> PolicyValueNet:
    """The same network with a head that answers in Mortal's space. Every
    weight but the two policy heads is the teacher's own."""
    student = PolicyValueNet(
        teacher.channels, teacher.blocks, teacher.planes, teacher.attention, MORTAL_ACTIONS
    ).to(device)
    kept = {
        name: value
        for name, value in teacher.state_dict().items()
        if not name.startswith(("policy_tiles.", "policy_pooled."))
    }
    missing, unexpected = student.load_state_dict(kept, strict=False)
    unexpected = [name for name in unexpected]
    if unexpected:
        raise SystemExit(f"the teacher has weights the student cannot take: {unexpected[:4]}")
    if any(not name.startswith(("policy_tiles.", "policy_pooled.")) for name in missing):
        raise SystemExit(f"the student is missing more than its head: {missing[:4]}")
    return student


@torch.no_grad()
def collect(
    teacher: PolicyValueNet,
    student: PolicyValueNet,
    games: int,
    seed: int,
    device: str,
    temperature: float,
) -> tuple[Planes, np.ndarray, np.ndarray, int]:
    """Plays a round with the teacher at every seat, keeping each position
    as the student sees it, the moves it may make there, and what the
    teacher would have said about them in Mortal's space."""
    arena = riichi_py.Arena(games=games, seed=seed)
    views = Views(arena, games, {"mortal"})
    means = torch.from_numpy(zoo.MEANS_BY_OURS).to(device)
    blocks: list[Planes] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    declared = 0

    for _step in range(4000):
        seats = np.frombuffer(arena.seats(), dtype=np.uint8)
        live = seats != 0xFF
        if not live.any():
            break
        views.advance()
        rows = np.nonzero(live)[0]
        players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, 4)
        who = np.array([players[game][seats[game]] for game in rows])
        legal = np.frombuffer(arena.legal_mask(), dtype=np.uint8).reshape(games, ACTIONS)
        legal = legal.astype(bool)

        # One encoding serves both: they read the same planes.
        views.prepare(rows, who)
        sparse, _own = views.sparse_and_masks(rows, who)
        planes = sparse.dense(device)
        ours = torch.from_numpy(legal[rows]).to(device)
        logits, _value = teacher(planes, ours)
        spread = torch.softmax(logits.float() / temperature, dim=1)

        # The same weight, said in Mortal's words: the riichi block lands
        # whole on its single reach, the three kans on its one kan. The
        # mapping is not one to one, since our discard of a five is meant
        # both by Mortal's plain five and by its red one, so what Mortal
        # may actually do here decides what the weight is allowed to land
        # on before it is normalised.
        allowed = zoo.translatable(legal[rows])
        # Several Mortal aliases can mean the same engine move (red/plain
        # fives). Split that move's mass rather than duplicating it when the
        # authoritative mask allows both aliases.
        allowed_tensor = torch.from_numpy(allowed).to(device)
        multiplicity = allowed_tensor.float() @ means.T
        target = ((spread / multiplicity.clamp(min=1)) @ means) * allowed_tensor
        total = target.sum(dim=1)
        usable = allowed.any(axis=1) & (total.cpu().numpy() > 1e-6)
        if usable.any():
            keep = np.nonzero(usable)[0]
            picked_rows = torch.from_numpy(keep).to(device)
            blocks.append(sparse.rows(keep))
            masks.append(allowed[keep])
            kept = target[picked_rows] / total[picked_rows].unsqueeze(1)
            targets.append(kept.cpu().numpy().astype(np.float32))

        picked = torch.distributions.Categorical(probs=spread).sample().cpu().numpy()
        choice = np.zeros(games, dtype=np.int64)
        choice[rows] = picked

        # Where it declared riichi, the tile it chose is the answer to the
        # second question, asked from the state in which the reach stands.
        second = np.nonzero((picked >= zoo.RIICHI_DISCARD) & (picked < zoo.TSUMO))[0]
        if len(second):
            declared += len(second)
            follower = views.observer.follower
            pairs = [(int(rows[i]), int(who[i])) for i in second]
            for game, player in pairs:
                follower.tell(game, player, json.dumps({"type": "reach", "actor": player}))
            indptr, indices, values, _after_masks = follower.encode(pairs)
            after = Planes.from_follower(indptr, indices, values)
            tiles = legal[rows][second, zoo.RIICHI_DISCARD : zoo.TSUMO]
            after_allowed = np.zeros((len(second), MORTAL_ACTIONS), dtype=bool)
            after_allowed[:, :POSITIONS] = tiles
            block = spread[torch.from_numpy(second).to(device),
                           zoo.RIICHI_DISCARD : zoo.TSUMO].cpu().numpy()
            block = block * after_allowed[:, :POSITIONS]
            total = block.sum(axis=1, keepdims=True)
            good = total[:, 0] > 1e-6
            if good.any():
                after_target = np.zeros((len(second), MORTAL_ACTIONS), dtype=np.float32)
                after_target[:, :POSITIONS] = block / np.maximum(total, 1e-6)
                blocks.append(after.rows(np.nonzero(good)[0]))
                masks.append(after_allowed[good])
                targets.append(after_target[good])

        arena.step(choice.tolist())

    if not blocks:
        return Planes.empty(), np.zeros((0, MORTAL_ACTIONS), bool), np.zeros((0, MORTAL_ACTIONS), np.float32), 0
    return Planes.cat(blocks), np.concatenate(masks), np.concatenate(targets), declared


def main() -> None:
    args = parse_args()
    validate_auxiliary_options(args)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    payload = torch.load(args.teacher, map_location=device, weights_only=True)
    teacher = from_payload(payload, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if teacher.speaks_mortal:
        raise SystemExit("the teacher already speaks Mortal's space; there is nothing to re-head")

    if args.resume is not None:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        student = PolicyValueNet(**shape_of(resumed)).to(device)
        student.load_state_dict(resumed["model"])
        print(f"student resumed from {args.resume}", flush=True)
    else:
        student = student_from(teacher, device)

    # Only the head learns: the trunk, the value and the reader are the
    # teacher's and stay exactly as they are.
    head = list(student.policy_tiles.parameters()) + list(student.policy_pooled.parameters())
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in head:
        parameter.requires_grad_(True)
    optimiser = torch.optim.AdamW(head, lr=args.lr, weight_decay=1e-4)
    print(
        f"device {device} | {student.channels}x{student.blocks} | "
        f"{sum(p.numel() for p in head) / 1e3:.1f}k parameters in the new head",
        flush=True,
    )

    for round_index in range(args.rounds):
        began = time.time()
        planes, masks, targets, declared = collect(
            teacher, student, args.games, args.seed + round_index * 977, device, args.temperature
        )
        played = time.time() - began
        require_supervised_rows(int(len(masks)))
        legal = torch.from_numpy(masks).to(device)
        wanted = torch.from_numpy(targets).to(device)

        student.train()
        total_loss, agreed, seen = 0.0, 0, 0
        updates = 0
        for _epoch in range(args.epochs):
            order = torch.randperm(len(masks), device=device)
            for start in range(0, len(masks), args.batch):
                picks = order[start : start + args.batch]
                if picks.numel() < 2:
                    continue
                logits, _value = student(
                    planes.rows(picks.cpu().numpy()).dense(device), legal[picks]
                )
                weight = wanted[picks]
                # A masked logit is negative infinity and carries no weight;
                # zero times that is not zero, so take the log only where
                # the teacher put something.
                log_odds = torch.log_softmax(logits.float(), dim=1)
                log_odds = torch.where(weight > 0, log_odds, torch.zeros_like(log_odds))
                loss = -(weight * log_odds).sum(dim=1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite auxiliary loss; no optimizer step was applied")
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head, 1.0, error_if_nonfinite=True)
                optimiser.step()
                updates += 1
                total_loss += loss.item() * picks.numel()
                agreed += int((logits.argmax(dim=1) == weight.argmax(dim=1)).sum())
                seen += picks.numel()

        require_updates(updates)
        record = {
            "optimizer_updates": updates,
            "round": round_index,
            "positions": int(len(masks)),
            "declarations": declared,
            "loss": round(total_loss / seen, 4),
            "agreement": round(agreed / seen, 4),
            "seconds": round(time.time() - began, 1),
            "play_seconds": round(played, 1),
        }
        atomic_save(
            {"model": student.state_dict(), "generation": 0, **student.payload_fields()},
            args.out / "latest.pt",
        )
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    print("re-heading finished", flush=True)


if __name__ == "__main__":
    main()
