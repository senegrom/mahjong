"""Characters and circles renamed: a second view of the questions a player
hesitates over, and the mean of the two answers.

Under our rules characters and circles are interchangeable. There are no
red fives, and the one yaku that names a suit, All Green, names bamboo;
the dora's wrap, every other yaku and the scoring treat the two alike.
Renaming every 1m-9m as 1p-9p and every 1p-9p as 1m-9m turns a game into
another game, just as likely and played the same way, so the best move of
one is the renamed best move of the other. Bamboo cannot be swapped so.
A network knows the symmetry only as well as it has learned it: g810 picks
another best move in the two namings of 4% of its decisions with two or
more moves. Asking both namings and playing the mean of the two answers
takes that accident out of a decision. In the screen of 15 October, at
the ruler's table, it changed g810's move in 1.9% of those decisions, 3.2
a game, and published Mortal, asked over both namings too, preferred the
new move a little more often than a matched placebo predicts (52% against
45%).

The second view is libriichi's own encoding of the renamed game, not the
game's planes permuted. Its efficiency lookahead (rows 889-890 and 959-1011
of Mortal's v4 planes) breaks ties by tile index, characters before
circles, so there the renamed game's planes differ from the game's planes
renamed, in about half the questions, by up to 0.07. The renamed game's
own encoding is a game's encoding like any other, the kind of planes every
network here learned from (Mortal's own augmentation renames events before
encoding them, too); the permuted planes are no game's. A second follower
(`observe.Observer`) is fed every event of the tables it follows with its
tiles renamed (`rename_line`), and told every reach the first is told, so
every question, a reach's tile included, is asked of the renamed game
itself.

Which average: the mean of the two masked logits, the one the screen
measured and published Mortal judged. Its best move is the best of the
two policies multiplied together, a move both namings must support, and
each answer's own normalisation cancels, so no weights need working out.
Averaging the two weights instead would have played another move in only
4 questions at each of the screen's two tables.

Which questions: only those whose plain answer hesitates, its top weight
below `GATE`. At 0.9 that is about 18% of the questions, and it keeps 155
of the 156 changes the mean makes when every question is asked twice; no
decision the plain view made with 0.99 or more changed. The renamed
questions are put to the network in a forward of their own, after the
plain one, so the plain questions are asked in the very batches a plain
player is asked in; asked in fixed rows on the card (see `neural.ruler`),
a table of each then plays the same game until a decision of the
averaging player changes.

It is off unless asked for: only a ruler seats a player that averages
(`neural.ruler --average-suits`), so self-play, training and the watcher's
duel play as they did. A question asked again costs a second encoding and
a second forward, for about a fifth of the player's rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import numpy as np

#: Tile kind `k` of a game is kind `KINDS[k]` of the renamed game: 1m-9m
#: (0-8) and 1p-9p (9-17) trade places, and bamboo and honours stay.
KINDS = np.concatenate([np.arange(9, 18), np.arange(0, 9), np.arange(18, 34)])
#: Mortal's move `m` of a game is move `MOVES[m]` of the renamed game: the
#: discards by kind as `KINDS`, then the red fives' discards (34-36: 5mr,
#: 5pr, 5sr), whose characters and circles trade places too although our
#: tables deal none, and the rest, riichi, the calls, a win, an abortive
#: draw and pass, which name no tile.
MOVES = np.concatenate([KINDS, [35, 34], np.arange(36, 46)])

#: A tile of characters or circles as an mjai line names one: a digit and
#: its suit in quotes, and an `r` after a red five. Every tile in a line is
#: a quoted name of its own, and nothing else in one is a quoted digit and
#: an `m` or a `p`.
TILE = re.compile(r'"[1-9][mp]r?"')
RENAMED = {f'"{number}{suit}{red}"': f'"{number}{other}{red}"'
           for number in "123456789" for suit, other in (("m", "p"), ("p", "m")) for red in ("", "r")}

#: The plain answer's top weight below which a question is asked of the
#: renamed game too (see the module's notes).
GATE = 0.9


def _renamed(tile: re.Match) -> str:
    return RENAMED[tile[0]]


def rename_line(line: str) -> str:
    """One mjai event, a line of JSON, with every tile of characters named
    as the circle of its number and every circle as the character."""
    return TILE.sub(_renamed, line)


def rename_games(lines: list[list[str]], games) -> list[list[str]]:
    """The events of the games named, each game's renamed, in that order:
    what a follower of those games is fed (see `observe.Observer`)."""
    return [[rename_line(line) for line in lines[game]] for game in games]


def swap_moves(values: np.ndarray) -> np.ndarray:
    """Anything over Mortal's forty-six moves, its last axis renamed: a
    game's mask as the renamed game's, or the renamed game's answers as
    the game's. The same call maps either way."""
    return np.asarray(values)[..., MOVES]


def top_weight(values: np.ndarray) -> np.ndarray:
    """Each row's largest weight once its logits are made weights, a move
    masked out at minus infinity weighing nothing."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return 1.0 / np.exp(values - values.max(axis=1, keepdims=True)).sum(axis=1)


@dataclass
class Averaging:
    """How a player averages over the two namings, and what it has done
    since it was seated: how many rows it was asked, how many of them it
    asked the renamed game too, and how many of those the mean gave
    another best move. The rows are questions, a reach's tile and a
    preview of one included, not decisions."""

    gate: float = GATE
    asked: int = 0
    again: int = 0
    changed: int = 0

    def report(self) -> dict:
        return asdict(self)


def averaging(player) -> Averaging | None:
    """How `player` averages over the two namings, or None if it does not."""
    return getattr(player, "suit_average", None)


def averaged(ask, account: Averaging):
    """`ask(views, who, fresh, allowed, ahead=())`, a player's question in
    Mortal's moves (see `zoo.choose_in_mortal_space`), asking its hesitant
    rows of the renamed game too and answering the mean of the two answers
    there. The renamed rows are asked in a question of their own, from the
    views' renamed game (`observe.Views.renamed`), with the same `fresh`,
    each preview of a reach (`ahead`) previewed there too, and the moves
    allowed renamed; the answers are renamed back before the mean."""

    def both(views, who, fresh, allowed, ahead=()):
        values, masks = ask(views, who, fresh, allowed, ahead)
        account.asked += len(values)
        again = np.flatnonzero(top_weight(values) < account.gate)
        if not len(again):
            return values, masks
        renamed = getattr(views, "renamed", None)
        if renamed is None:
            raise RuntimeError("this table follows no renamed game: a player that averages over the suits is "
                               "seated at a ruler, whose views follow its tables renamed too (neural.ruler)")
        mine, previews = again[again < len(who)], again[again >= len(who)] - len(who)
        answers, _own = ask(renamed, [who[row] for row in mine], fresh, swap_moves(np.asarray(allowed)[again]),
                            [ahead[row] for row in previews])
        mean = (values[again] + swap_moves(answers)) / 2
        account.again += len(again)
        account.changed += int((mean.argmax(axis=1) != values[again].argmax(axis=1)).sum())
        values[again] = mean
        return values, masks

    return both


def average(player, gate: float = GATE):
    """`player`, from now on asking each question whose plain answer
    hesitates (its top weight below `gate`) of the renamed game too, and
    answering the mean of the two answers' logits there; how it averages,
    and what it has done, is kept as `suit_average`. Only a player whose
    answers are a policy's logits over Mortal's moves can: a joined player
    or one of ours in Mortal's moves. A published Mortal answers values,
    whose weights say nothing of how sure it is."""
    from . import combined, zoo

    if not isinstance(player, (combined.Combined, zoo.MortalSpacePlayer)):
        raise ValueError(f"{type(player).__name__} answers no policy in Mortal's moves: only a joined player or one "
                         "of ours in Mortal's moves can average over the suits")
    if not 0.0 <= gate <= 1.0:
        raise ValueError("the gate is a weight, from nought to one")
    if averaging(player) is not None:
        raise ValueError("the player already averages over the suits")
    account = Averaging(float(gate))
    player._ask = averaged(player._ask, account)
    player.suit_average = account
    return player
