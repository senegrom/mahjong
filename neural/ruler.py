"""A cheaper, tighter ruler: every table of a duel in one arena, and a second
challenger against the same incumbent on the same deals.

`neural.duel` plays a set of deals four times, one seating after another,
each at a table of its own. A table's last few hundred steps are played by
fewer and fewer games, and each such step still pays the engine, the encoder
and two networks their full price for a handful of rows; the four tables pay
that tail four times over. Here all four seatings of every deal sit in one
arena, each table dealt from its deal's own seed (`riichi_py.Arena(seeds=...)`),
so the engine steps them together and the tails overlap; each step's deciding
players are encoded in two calls rather than eight, the incumbent's on a
worker while the challengers decide (see `Views.prepare`); and every network
plays with its weights cast to bfloat16 once (`policy_inference.precast`),
which leaves every answer as it was.

A second challenger (`--also`) sits at four more tables of every deal, so the
two meet the same incumbent on the same deals in the same seats and their
difference is read deal by deal. What a challenger gains or loses on a deal
is mostly the deal's luck, and two close networks share most of it: the
published rulers of 810 and 910 against published Mortal correlate at
+0.628 deal by deal, so the difference of the pair carries 0.6 of the error
of the two read apart (see `neural.looks` for reading a series of such
calls). Each arm's games are the ones a ruler of that challenger alone
would play.

How each network is asked decides whether the games are the old duel's. On
the card a batch of another size can be worked through by other kernels,
which in bfloat16 can move an answer in its last bits, and a near tie can
turn on a bit (see `play_graphs`). So:

- `--questions seating`, the default, asks each network about each seating's
  tables apart, in the very batches `neural.duel` asked it, and plays the old
  duel's games bit for bit: the same placements, deal by deal. With
  `--graphs`, a size asked often is answered from a CUDA graph recorded at
  that very size, which replays the kernels the eager forward chose there
  and so changes no answer (see `ExactGraphs`).
- `--questions fixed` asks each network once a step about all of its tables,
  cut into pieces of exactly `--rows` rows, the last padded with blank rows.
  Every answer then depends on its own row alone: the incumbent answers a
  position the same at every table and every step, two challengers that are
  one network play the same games until a decision of theirs differs, and
  answering from CUDA graphs (`--graphs`), leaving rows with one move unasked
  (`--skip-forced`) and asking a reach's tile with the reach
  (`--preview-reach`) change no answer at all. Its games are not the old
  duel's, so a series measured this way is compared only with itself.

    python -m neural.ruler challenger.pt incumbent.pt --graphs --games 1000 --seed 9100000
    python -m neural.ruler a.pt incumbent.pt --also b.pt --questions fixed --graphs \\
        --skip-forced --preview-reach --games 1000 --seed 9100000

The report is the duel's, with `play` saying how it played and where the
time went (see `duel.Split`); with `--also`, one entry a challenger under
`arms` and their `difference`, positive when the first is the stronger.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

import riichi_py

from . import play_graphs, zoo
from .duel import Split, play_record, verdict
from .outcomes import paired_summary, placements, require_finished, validate_budget, win_shares
from .training_safety import TRAINING_API_VERSION, require_training_engine

ACTIONS = riichi_py.ACTIONS
SEATS = 4
#: How the networks are asked; see the module's notes.
QUESTIONS = ("seating", "fixed")
#: The one size every question is asked at with `--questions fixed`. About
#: where a joined player's forward on the card stops being bound by its
#: launching (see `play_graphs.ROWS`), so a step's thousands of rows are
#: worked through about as fast as in one batch, and a handful of rows at
#: the end of the arena costs one padded piece.
ROWS = 256


def require_seeded_arena() -> None:
    """The ruler deals one deal to several tables of an arena, which the
    engine allows from `ARENA_SEEDS_API_VERSION` 1."""
    if getattr(riichi_py, "ARENA_SEEDS_API_VERSION", 0) < 1:
        raise RuntimeError(
            "Rebuild and reinstall riichi_py: the ruler deals each deal to several tables of one "
            "arena, which needs riichi_py.Arena(seeds=...)"
        )


@dataclass(frozen=True)
class Layout:
    """Where each table of a ruler sits in its arena: challenger by
    challenger, each one's four seatings one after another, each seating's
    tables in the order of their deals. Table `(arm * 4 + place) * deals +
    deal` is dealt from `seed + deal`, with challenger `arm` at player
    `place` and the incumbent at the other three, so every deal is played
    at `4 * arms` tables, as `neural.duel` plays it at four."""

    arms: int
    deals: int

    @property
    def games(self) -> int:
        return self.arms * SEATS * self.deals

    def seeds(self, seed: int) -> list[int]:
        # Reduced as the engine's own seed + i wraps, so a table dealt from
        # this list is the table an arena of `seed` deals at index `deal`.
        return [(seed + deal) % (1 << 64) for _ in range(self.arms * SEATS) for deal in range(self.deals)]

    def block(self, arm: int, place: int) -> slice:
        """The tables of challenger `arm` seated at `place`."""
        start = (arm * SEATS + place) * self.deals
        return slice(start, start + self.deals)

    @property
    def arm_of(self) -> np.ndarray:
        return np.repeat(np.arange(self.arms), SEATS * self.deals)

    @property
    def place_of(self) -> np.ndarray:
        return np.tile(np.repeat(np.arange(SEATS), self.deals), self.arms)

    @property
    def block_of(self) -> np.ndarray:
        return np.repeat(np.arange(self.arms * SEATS), self.deals)


class FixedRows:
    """`forward(planes, mask)` asked only ever at one size of batch: a
    question is cut into pieces of `rows` rows, the last padded as a CUDA
    graph's is (`play_graphs.Graphed`), with planes of nought and every move
    open, and the answers' first rows kept. No row of these networks reads
    another's, so a row's answer is then the same whatever else is asked
    with it and however many: the kernels are always the ones chosen for
    `rows`. The forward answers a tensor or a tuple of them, one row each for
    each row asked, as every deciding forward of a player of Mortal's moves
    does."""

    def __init__(self, forward, rows: int) -> None:
        if type(rows) is not int or rows <= 0:
            raise ValueError("rows must be a positive whole number")
        self.forward = forward
        self.rows = rows
        #: How many pieces were asked, and how many of their rows were padding.
        self.pieces = 0
        self.padded = 0

    def __call__(self, planes: torch.Tensor, mask: torch.Tensor):
        n = planes.shape[0]
        if n == 0:
            return self.forward(planes, mask)
        answers = []
        for start in range(0, n, self.rows):
            piece_planes = planes[start : start + self.rows]
            piece_mask = mask[start : start + self.rows]
            short = self.rows - piece_planes.shape[0]
            if short:
                piece_planes = torch.cat([piece_planes, piece_planes.new_zeros((short, *planes.shape[1:]))])
                piece_mask = torch.cat([piece_mask, piece_mask.new_ones((short, *mask.shape[1:]))])
            answer = self.forward(piece_planes, piece_mask)
            kept = self.rows - short
            if isinstance(answer, torch.Tensor):
                answers.append(answer[:kept])
            else:
                answers.append(tuple(None if part is None else part[:kept] for part in answer))
            self.pieces += 1
            self.padded += short
        if isinstance(answers[0], torch.Tensor):
            return torch.cat(answers)
        return tuple(None if parts[0] is None else torch.cat(parts) for parts in zip(*answers))


class ExactGraphs:
    """`forward(planes, mask)` answered, for a question of at most
    `largest` rows, from a CUDA graph recorded at the very size asked, once
    that size has been asked `after` times; every other question eagerly.

    A graph replays the kernels the forward chose eagerly at its size, on
    the same inputs, so its answer is the eager answer bit for bit (see
    `play_graphs.Graphed`), and a question the old duel asked about a
    handful of rows costs one launch instead of a thousand small ones: the
    last steps of every table, and a challenger's quarter of a seating's
    rows, are such questions. A size asked only once or twice is not worth
    recording, and a big question keeps the card busy for longer than its
    launching takes."""

    def __init__(self, forward, module: nn.Module, largest: int = 192, after: int = 3) -> None:
        self.eager = forward
        self.graphed = play_graphs.Graphed(forward, module, tuple(range(1, largest + 1)))
        self.largest = largest
        self.after = after
        self.asked: dict[int, int] = {}

    def __call__(self, planes: torch.Tensor, mask: torch.Tensor):
        n = planes.shape[0]
        self.asked[n] = self.asked.get(n, 0) + 1
        if n > self.largest or self.asked[n] < self.after:
            return self.eager(planes, mask)
        return self.graphed(planes, mask)


def ask_from_exact_graphs(player, device: str, largest: int = 192, after: int = 3):
    """`player`, on the card in bfloat16, answering the questions it is
    asked often from graphs at their own sizes (see `ExactGraphs`): a
    joined player's deciding forward, or a Mortal's or one of ours's in
    Mortal's moves. One of ours over the engine's moves, which plays in
    float32, is left as it is, as `play_graphs.graphed` leaves it."""
    if torch.device(device).type != "cuda":
        return player
    if hasattr(player, "deciding"):
        player.deciding = ExactGraphs(player.decision, player, largest, after)
    elif isinstance(getattr(player, "net", None), nn.Module) and hasattr(player, "forward"):
        player.forward = ExactGraphs(player.forward, player.net, largest, after)
    return player


def forwards_of(player, kind: type = None) -> list:
    """The forwards `player` decides with that are cut into fixed rows or
    answered from graphs at their own sizes (see `ask_in_fixed_rows` and
    `ask_from_exact_graphs`), or only those of `kind`."""
    found = []
    for name in ("deciding", "forward"):
        forward = getattr(player, "__dict__", {}).get(name)
        if isinstance(forward, kind or (FixedRows, ExactGraphs)):
            found.append(forward)
    return found


def ask_in_fixed_rows(player, device: str, rows: int, graphs: bool = False):
    """`player`, every question it is asked put to its network in pieces
    of `rows` rows (see `FixedRows`): a joined player's deciding forward
    (`Combined.deciding`), a Mortal's or one of ours's in Mortal's moves
    (`forward`), or one of ours over the engine's moves, itself a module.
    With `graphs`, on the card in bfloat16, the pieces are answered from a
    CUDA graph recorded at that one size (see `play_graphs.graphed`), which
    answers what the forward answers eagerly at that size, bit for bit."""
    if graphs:
        play_graphs.graphed(player, device, amp=True, rows=(rows,))
    if hasattr(player, "deciding"):
        player.deciding = FixedRows(player.deciding or player.decision, rows)
    elif isinstance(getattr(player, "net", None), nn.Module) and hasattr(player, "forward"):
        player.forward = FixedRows(player.forward, rows)
    elif isinstance(player, nn.Module):
        player.forward = FixedRows(player.forward, rows)
    else:
        raise ValueError(f"{type(player).__name__} has no network to ask in fixed rows")
    return player


def seat(
    path: Path | str,
    device: str,
    channels: int | None = None,
    blocks: int | None = None,
    questions: str = "seating",
    rows: int = ROWS,
    graphs: bool = False,
    skip_forced: bool = False,
    preview_reach: bool = False,
):
    """A player from a checkpoint, as a ruler seats it: asked only for
    moves, so with no gradients, its convolutions' and linear layers'
    weights cast to bfloat16 once where it plays on the card in bfloat16
    (`zoo.play_in_bfloat16`, bit for bit the answers autocast gives), and,
    with `questions` "fixed", asked in pieces of `rows` (see
    `ask_in_fixed_rows`) with the switches that then change no answer.
    Asked by seating, `graphs` answers the questions it is asked often
    from graphs at their own sizes, which changes no answer either (see
    `ExactGraphs`)."""
    validate_questions(questions, rows, graphs, skip_forced, preview_reach)
    if str(path) == zoo.CLUB:
        raise ValueError("the heuristic player plays inside the arena and cannot be seated at a ruler")
    player = zoo.load_player(path, device, channels, blocks)
    player.eval()
    # A Mortal of the zoo, or one of ours in its moves, holds its network
    # as `net`; a joined player, or one of ours in our moves, is one.
    network = getattr(player, "net", player)
    if isinstance(network, nn.Module):
        network.requires_grad_(False)
    player = zoo.play_in_bfloat16(player, device)
    if questions == "fixed":
        ask_in_fixed_rows(player, device, rows, graphs)
        if skip_forced and hasattr(player, "skip_forced"):
            player.skip_forced = True
        if preview_reach and hasattr(player, "preview_reach"):
            player.preview_reach = True
    elif graphs:
        ask_from_exact_graphs(player, device)
    return player


def validate_questions(questions: str, rows: int, graphs: bool, skip_forced: bool,
                       preview_reach: bool) -> None:
    if questions not in QUESTIONS:
        raise ValueError(f"questions must be one of {', '.join(QUESTIONS)}")
    if questions == "fixed" and (type(rows) is not int or rows <= 0):
        raise ValueError("rows must be a positive whole number")
    if questions != "fixed" and (skip_forced or preview_reach):
        # Each changes the batches a network is asked in, and with them the
        # last bits of answers, unless every batch has one size.
        raise ValueError("skip-forced and preview-reach change answers unless the questions are fixed")


def groups(rows: np.ndarray, block_of: np.ndarray, apart: bool) -> list[np.ndarray]:
    """`rows`, ascending, as one question, or, `apart`, one a seating's
    block of tables, in the order the blocks sit: what `neural.duel` asked
    at each of its tables."""
    if not len(rows):
        return []
    if not apart:
        return [rows]
    return np.split(rows, np.flatnonzero(np.diff(block_of[rows])) + 1)


@torch.no_grad()
def play(
    challengers: list,
    incumbent,
    deals: int,
    seed: int,
    device: str = "cuda",
    max_steps: int = 4000,
    questions: str = "seating",
    timing: dict | None = None,
) -> np.ndarray:
    """Plays every table of a ruler in one arena (see `Layout`) and returns
    the final scores, a row a table.

    Every network plays its best move. Each step the challengers' deciding
    players are encoded here and the incumbent's on the worker meanwhile;
    the challengers are asked first, then the incumbent, each about its own
    rows: in `questions` "seating", each seating's block apart, as the old
    duel's tables asked; in "fixed", all of them at once, the player cutting
    them into pieces itself. With `timing`, a dict, adds where the time went
    to it (see `duel.Split`)."""
    require_training_engine()
    require_seeded_arena()
    validate_budget(deals, max_steps)
    if questions not in QUESTIONS:
        raise ValueError(f"questions must be one of {', '.join(QUESTIONS)}")
    if not challengers:
        raise ValueError("a ruler needs at least one challenger")
    layout = Layout(len(challengers), deals)
    for player in (*challengers, incumbent):
        player.eval()
    games = layout.games
    arm_of, place_of, block_of = layout.arm_of, layout.place_of, layout.block_of
    apart = questions == "seating"
    clock = Split(timing)
    arena = riichi_py.Arena(games=games, seeds=layout.seeds(seed))
    views = clock.views_for(arena, games, {player.kind for player in (*challengers, incumbent)})
    # Only a player that reads Mortal's planes is served the follower's
    # encodings; one of ours over the engine's planes reads the engine's.
    encoded = [player.kind == "mortal" for player in challengers]
    incumbent_encoded = incumbent.kind == "mortal"
    steps = 0
    while not arena.all_finished() and steps < max_steps:
        steps += 1
        with clock("engine"):
            seats = np.frombuffer(arena.seats(), dtype=np.uint8)
            live = seats != 0xFF
        if not live.any():
            break
        with clock("follow"):
            views.advance()

        with clock("engine"):
            mask = np.frombuffer(arena.legal_mask(), dtype=np.uint8)
            mask = mask.reshape(games, ACTIONS).astype(bool)
            players = np.frombuffer(arena.seat_players(), dtype=np.uint8).reshape(games, SEATS)

        # Who owes each live game's decision, as a person rather than a
        # seat: the seats move between hands and the players do not.
        index = np.nonzero(live)[0]
        deciding = np.full(games, -1, dtype=np.int64)
        deciding[index] = players[index, seats[index]]
        mine = deciding[index] == place_of[index]
        theirs = [index[mine & (arm_of[index] == arm)] for arm in range(layout.arms)]
        others = index[~mine]

        # The challengers' views here, in one call, and the incumbent's on
        # the worker while the challengers decide: its moves do not depend
        # on theirs, and nothing is drawn at random, so the order changes
        # nothing. A reach told to the follower waits for the worker first.
        wanted = [rows for rows, served in zip(theirs, encoded) if served and len(rows)]
        ahead = [player for player, served in zip(challengers, encoded)
                 if served and getattr(player, "preview_reach", False)]
        with clock("encode"):
            if wanted:
                rows = np.sort(np.concatenate(wanted))
                views.prepare(rows, deciding[rows])
                if ahead:
                    ready = rows[mask[rows, zoo.RIICHI_DISCARD:zoo.TSUMO].any(axis=1)]
                    views.prepare(ready, deciding[ready], after_reach=True)
            if incumbent_encoded and len(others):
                aside = bool(wanted)
                views.prepare(others, deciding[others], aside=aside)
                if getattr(incumbent, "preview_reach", False):
                    ready = others[mask[others, zoo.RIICHI_DISCARD:zoo.TSUMO].any(axis=1)]
                    views.prepare(ready, deciding[ready], after_reach=True, aside=aside)

        choice = np.zeros(games, dtype=np.int64)
        for player, rows_of_arm in zip(challengers, theirs):
            for rows in groups(rows_of_arm, block_of, apart):
                with clock("challengers", len(rows)):
                    choice[rows] = zoo.choose(player, views, rows, deciding[rows], mask[rows], device)
        with clock("encode"):
            # Whatever of the incumbent's encoding the challengers did not hide.
            views.wait()
        for rows in groups(others, block_of, apart):
            with clock("incumbent", len(rows)):
                choice[rows] = zoo.choose(incumbent, views, rows, deciding[rows], mask[rows], device)

        with clock("engine"):
            arena.step(choice.tolist())

    require_finished(arena, steps=steps, context="ruler")
    if timing is not None:
        timing["steps"] = timing.get("steps", 0) + steps
    return np.frombuffer(arena.final_scores(), dtype=np.int32).reshape(games, SEATS).copy()


def arm_report(scores: np.ndarray, layout: Layout, arm: int) -> dict:
    """One challenger's share of a ruler, reported as `neural.duel` reports
    a duel, from the same arithmetic on the same scores: the placement by
    seat, and by deal over the four seatings, with the error from the
    spread across deals."""
    per_seat = []
    per_deal = []
    for place in range(SEATS):
        table = scores[layout.block(arm, place)]
        got = placements(table)[:, place]
        per_seat.append(
            {
                "place": place,
                "placement": float(got.mean()),
                "score": float(table[:, place].mean()),
                "wins": float(win_shares(table)[:, place].mean()),
            }
        )
        per_deal.append(got.astype(float))
    overall = float(statistics.fmean(row["placement"] for row in per_seat))
    paired, error = paired_summary(per_deal)
    return {
        "games_per_seat": layout.deals,
        "games_total": layout.deals * SEATS,
        "placement": overall,
        "standard_error": error,
        "by_seat": per_seat,
        "by_deal": paired.tolist(),
    }


def difference(first: dict, second: dict, scores: np.ndarray | None = None,
               layout: Layout | None = None) -> dict:
    """The first challenger's edge less the second's, deal by deal, and its
    error from the spread of that difference across deals. Positive when
    the first is the stronger. `unpaired_error` is what the two errors
    would give read apart, and `correlation` is the deals' own: how much
    of the luck the two arms shared. With the scores, `same_games` counts
    the tables whose final scores the two arms matched exactly, a sign of
    how many games were played the same to the end."""
    a = np.asarray(first["by_deal"], dtype=np.float64)
    b = np.asarray(second["by_deal"], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("the two challengers were not played on the same deals")
    by_deal = b - a
    n = len(by_deal)
    error = float(by_deal.std(ddof=1) / n ** 0.5) if n > 1 else 0.0
    spread = n > 1 and a.std() > 0 and b.std() > 0
    report = {
        "edge": float(by_deal.mean()),
        "standard_error": error,
        "unpaired_error": float((first["standard_error"] ** 2 + second["standard_error"] ** 2) ** 0.5),
        "correlation": float(np.corrcoef(a, b)[0, 1]) if spread else None,
        "same_deals": int((a == b).sum()),
    }
    if scores is not None and layout is not None:
        same = 0
        for place in range(SEATS):
            same += int((scores[layout.block(0, place)] == scores[layout.block(1, place)]).all(axis=1).sum())
        report["same_games"] = same
    report["verdict"] = difference_verdict(report)
    return report


def difference_verdict(report: dict) -> str:
    """One call's reading of a difference, at two standard errors; a
    series of calls is read by `neural.looks`, by its pre-registered rule."""
    edge, error = report["edge"], report["standard_error"]
    if error == 0:
        return "the same placements on every deal" if edge == 0 else "not enough deals to say"
    sigmas = edge / error
    size = f"{edge:+.4f} placement, {sigmas:+.1f} standard errors"
    if sigmas > 2:
        return f"the first challenger is stronger: {size}"
    if sigmas < -2:
        return f"the second challenger is stronger: {size}"
    return f"not settled either way in this call: {size}"


def ruler(
    challengers: list,
    incumbent,
    deals: int,
    seed: int,
    device: str = "cuda",
    max_steps: int = 4000,
    questions: str = "seating",
    timing: dict | None = None,
    how: dict | None = None,
) -> dict:
    """Plays a ruler and reports it: as `neural.duel` reports a duel for one
    challenger, and for two, an entry each under `arms` and their
    `difference`. With `timing`, a dict, the report says how it played and
    where the time went, as `play`, with `how` among it."""
    began = time.perf_counter()
    scores = play(challengers, incumbent, deals, seed, device=device, max_steps=max_steps,
                  questions=questions, timing=timing)
    seconds = time.perf_counter() - began
    layout = Layout(len(challengers), deals)
    arms = [arm_report(scores, layout, arm) for arm in range(layout.arms)]
    result = {"training_api_version": TRAINING_API_VERSION, "seed": seed}
    if len(arms) == 1:
        result.update(arms[0])
    else:
        result["games_per_seat"] = deals
        result["arms"] = arms
        if len(arms) == 2:
            result["difference"] = difference(arms[0], arms[1], scores, layout)
    if timing is not None:
        result["play"] = play_record(timing, seconds, path="ruler", **(how or {}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("challenger", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--also", type=Path, default=None,
                        help="a second challenger, seated against the same incumbent on the same deals")
    parser.add_argument("--games", type=int, default=1000, help="deals per seating")
    parser.add_argument("--seed", type=int, default=555_000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--questions", choices=QUESTIONS, default="seating",
                        help="seating: the old duel's batches and games, bit for bit; "
                             "fixed: every question in pieces of --rows rows")
    parser.add_argument("--rows", type=int, default=ROWS, help="with --questions fixed, the size of every piece")
    parser.add_argument("--graphs", action="store_true",
                        help="answer from CUDA graphs: in fixed rows, the pieces from one graph; asked by "
                             "seating, the sizes asked often from a graph each at its own size; "
                             "neither changes an answer")
    parser.add_argument("--skip-forced", action="store_true",
                        help="with --questions fixed, leave rows with one move open unasked")
    parser.add_argument("--preview-reach", action="store_true",
                        help="with --questions fixed, ask a reach's tile in the question that chooses it")
    parser.add_argument("--timing", action="store_true",
                        help="report how it played and where the time went, as `play`")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # The width and depth a network of ours is rebuilt at only where its
    # checkpoint does not say, as the oldest do not (see `zoo.load_player`).
    parser.add_argument("--channels", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--incumbent-channels", type=int, default=192)
    parser.add_argument("--incumbent-blocks", type=int, default=10)
    args = parser.parse_args()
    validate_questions(args.questions, args.rows, args.graphs, args.skip_forced, args.preview_reach)

    how = {"questions": args.questions, "graphs": args.graphs}
    if args.questions == "fixed":
        how.update(rows=args.rows, skip_forced=args.skip_forced, preview_reach=args.preview_reach)
    options = {name: getattr(args, name) for name in ("questions", "rows", "graphs", "skip_forced", "preview_reach")}
    began = time.perf_counter()
    paths = [args.challenger] + ([args.also] if args.also is not None else [])
    challengers = [seat(path, args.device, args.channels, args.blocks, **options) for path in paths]
    incumbent = seat(args.incumbent, args.device, args.incumbent_channels, args.incumbent_blocks, **options)
    loaded = time.perf_counter() - began
    timing = {} if args.timing else None
    result = ruler(challengers, incumbent, args.games, args.seed, device=args.device, max_steps=args.max_steps,
                   questions=args.questions, timing=timing, how=how)
    if timing is not None:
        # Loading and, with --graphs in fixed rows, recording, before the first move.
        result["play"]["loading"] = round(loaded, 3)
        seated = (*challengers, incumbent)
        fixed = [forward for player in seated for forward in forwards_of(player, FixedRows)]
        if fixed:
            result["play"]["pieces"] = sum(forward.pieces for forward in fixed)
            result["play"]["padded_rows"] = sum(forward.padded for forward in fixed)
        exact = [forward for player in seated for forward in forwards_of(player, ExactGraphs)]
        if exact:
            # Recorded while playing, and so inside the play's own time.
            result["play"]["graphs"] = sum(len(forward.graphed.graphs) for forward in exact)
            result["play"]["graph_seconds"] = round(sum(forward.graphed.seconds for forward in exact), 3)
            result["play"]["from_graphs"] = sum(forward.graphed.replays for forward in exact)
    if "arms" in result:
        for arm, path in zip(result["arms"], paths):
            arm["challenger"] = str(path)
            arm["verdict"] = verdict(arm)
    else:
        result["challenger"] = str(args.challenger)
        result["verdict"] = verdict(result)
    result["incumbent"] = str(args.incumbent)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
