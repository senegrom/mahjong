"""Who else is at the table, and how often.

Self-play against nothing but yourself teaches you to beat yourself. The
failure is specific and this project has seen it: a policy drifts somewhere
peculiar, its three opponents drift with it because they *are* it, and the
peculiarity never costs anything because nobody at the table punishes it.
Placement against the heuristic bots does not catch it either — that table
compresses real differences several-fold, and one number cannot say which
opponent a network got worse against.

So the trainers seat other players, the checkpoints `--opponents` names,
and this says who they are and how often each sits down. The roster is
written to the run's log, and every round reports how the learner did
against each member it met, so a run records who was at its tables.

A member is one of two kinds:

- **reference** — a fixed outside player that never changes, so a number
  measured against it this month means the same as last month (see
  `REFERENCES`). A checkpoint named on the command line that is one of
  these keeps its note and its weight.
- **recent** — any other checkpoint named on the command line, such as
  one of the lineage's own archives, seated as often as a weight of one
  says.

A member's share is how often it takes a seat, not how often it wins.

Nothing here promises that beating one member implies beating another.
That is the reason for keeping them apart in the report: a policy that
improves against its own recent past while losing to the fine-tuned Mortal
has specialised, and one averaged number would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Member:
    """One player who may be seated, and what it is."""

    #: Where the checkpoint lives, as the volume names it.
    name: str
    #: reference, or recent for any other checkpoint named on the command line.
    role: str
    #: What this player is, for whoever reads the log a month from now.
    note: str
    #: How often it takes a seat, relative to the others.
    weight: float = 1.0
    #: The chance it is in the pool at all in any one round. Below one, it
    #: comes and goes at irregular intervals (see `Population.for_round`).
    presence: float = 1.0


#: The name that seats the engine's Club-tier heuristic player: no
#: checkpoint, a player the arena plays itself (`zoo.ClubPlayer`).
CLUB = "club"


#: Players that do not change, so a reading against them keeps its meaning.
#:
#: The fine-tuned Mortal earns its place for a reason worth writing down:
#: it is the strongest thing in the repository and the fusion contains it,
#: so it is simultaneously the hardest opponent available and the exact
#: baseline the fusion has to beat to justify existing. Duelled at one
#: table, the fusion has been level with it. A network that cannot beat the
#: Mortal inside itself has not learned anything the Mortal did not already
#: know, and seating it makes that visible every round rather than only
#: when somebody runs a duel.
REFERENCES: tuple[Member, ...] = (
    Member(
        name="mortal-run/latest",
        role="reference",
        note="Mortal fine-tuned on our rules by self-play; the strongest "
        "player here, and the one the fusion is built around",
        weight=2.0,
    ),
    Member(
        name="zoo/mortal_298k",
        role="reference",
        note="the published Mortal, trained on Tenhou's rules from human "
        "games; the outside yardstick that owes nothing to this project",
        weight=1.5,
    ),
    Member(
        name="w1012-run/mortal-space",
        role="reference",
        note="our own network on Mortal's planes, speaking Mortal's moves; "
        "the other half of the fusion",
        weight=1.0,
    ),
    Member(
        name=CLUB,
        role="reference",
        note="the engine's Club-tier heuristic player, the benchmark the "
        "placement figure is read against: a style no network here has. In "
        "about three rounds in four, and then, beside published Mortal and "
        "one older checkpoint seated by player at one half, at about one "
        "table in seven: one table in ten over the run",
        weight=0.2,
        presence=0.75,
    ),
)


@dataclass
class Population:
    """The roster, and how a round's tables are made from it."""

    members: list[Member] = field(default_factory=list)

    @classmethod
    def from_paths(cls, paths) -> Population:
        """A roster from checkpoints named on a command line.

        Launchers copy a checkpoint to a local file and pass that path, so
        `mortal-run/latest` arrives as `.../mortal-run--latest.pt`. The
        original name is recovered from it, which is what lets a member
        keep the role and the note it has in `REFERENCES` instead of
        becoming an anonymous entry that the log cannot explain.
        """
        known = {member.name: member for member in REFERENCES}
        members = []
        for path in paths:
            stem = Path(str(path)).stem
            name = stem.replace("--", "/")
            member = known.get(name)
            members.append(
                member
                if member is not None
                else Member(name, "recent", "a checkpoint named on the command line", 1.0)
            )
        return cls(members=members)

    def describe(self) -> list[dict]:
        """The roster as it should appear in a run's log: who was available
        to be seated, in what role, and how often."""
        total = sum(member.weight for member in self.members) or 1.0
        return [
            {
                "name": member.name,
                "role": member.role,
                "share": round(member.weight / total, 3),
                **({"presence": member.presence} if member.presence < 1 else {}),
                "note": member.note,
            }
            for member in self.members
        ]

    def for_round(self, seed: int) -> Population:
        """The roster as one round seats it. A member whose presence is
        below one sits the round out with the rest of that chance, drawn
        from the round's seed, so it comes and goes at irregular intervals
        rather than taking a thin share of every round; one sitting out
        weighs nought."""
        rng = np.random.default_rng(seed ^ 0x5EA7_ED00)
        return Population(members=[
            member if member.presence >= 1 or rng.random() < member.presence
            else replace(member, weight=0.0)
            for member in self.members
        ])

    def seat(self, games: int, share: float, rng: np.random.Generator) -> np.ndarray:
        """Which member sits in each game, or -1 for a table of the
        learner alone.

        `share` is how many of the games have anyone else at all. Within
        those, a member is drawn by its weight. One foreign seat a game:
        the learner holds the other three, so every table still gives three
        seats' worth of its own decisions to learn from, and the foreign
        player is something to be measured against rather than a crowd to
        be drowned by.
        """
        chosen = np.full(games, -1, dtype=np.int64)
        if not self.members or share <= 0 or not self.weights().any():
            return chosen
        taken = rng.random(games) < share
        count = int(taken.sum())
        if not count:
            return chosen
        chosen[taken] = rng.choice(len(self.members), size=count, p=self.weights())
        return chosen

    def weights(self) -> np.ndarray:
        """How often each member is drawn, as probabilities."""
        weights = np.array([member.weight for member in self.members], dtype=np.float64)
        total = weights.sum()
        # Nought throughout when every member sits the round out.
        return weights / total if total > 0 else weights


def mixed_tables(games: int, share: float, weights, rng: np.random.Generator) -> np.ndarray:
    """Which member holds each player of each game, or -1 for the learner.

    Each player is, independently, somebody else with chance `share`, the
    member drawn by `weights`; a table the draw filled with others gives one
    player, chosen at random, back to the learner. So one table can hold the
    learner, published Mortal and an old checkpoint at once, which one
    foreign seat a game never did: the lineage that learned only against
    itself got better at beating its ancestors and no better at beating
    published Mortal.
    """
    if not 0 < share < 1:
        raise ValueError("share must be above nought and below one")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or (weights < 0).any() or not weights.sum() > 0:
        raise ValueError("weights must be one nonnegative weight a member, not all nought")
    weights = weights / weights.sum()
    outside = rng.random((games, 4)) < share
    full = outside.all(axis=1)
    if full.any():
        outside[np.nonzero(full)[0], rng.integers(0, 4, size=int(full.sum()))] = False
    owner = np.full((games, 4), -1, dtype=np.int64)
    count = int(outside.sum())
    if count:
        owner[outside] = rng.choice(len(weights), size=count, p=weights)
    return owner


def matchups(seated: np.ndarray, placements: np.ndarray, members: list[Member]) -> list[dict]:
    """How the learner did against each member it met, kept apart.

    `seated[game]` is the member in that game or -1, and `placements[game]`
    is the learner's own average placement there; with mixed tables
    `seated` has a column for each player (see `mixed_tables`), and a
    member's row counts every game it sat in. Reported one row a
    member, never summed: improving against your own recent past while
    losing to a fixed reference is specialisation, and a single average is
    exactly what hides it.
    """
    seated = np.asarray(seated)
    table = seated if seated.ndim == 2 else seated[:, None]
    rows = []
    for index, member in enumerate(members):
        met = (table == index).any(axis=1)
        played = int(met.sum())
        if not played:
            continue
        rows.append(
            {
                "name": member.name,
                "role": member.role,
                "games": played,
                "placement": round(float(placements[met].mean()), 4),
            }
        )
    alone = (table < 0).all(axis=1)
    if alone.any():
        rows.append(
            {
                "name": "itself",
                "role": "self",
                "games": int(alone.sum()),
                "placement": round(float(placements[alone].mean()), 4),
            }
        )
    return rows
