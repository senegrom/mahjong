"""Who else is at the table, and how often.

Self-play against nothing but yourself teaches you to beat yourself. The
failure is specific and this project has seen it: a policy drifts somewhere
peculiar, its three opponents drift with it because they *are* it, and the
peculiarity never costs anything because nobody at the table punishes it.
Placement against the heuristic bots does not catch it either — that table
compresses real differences several-fold, and one number cannot say which
opponent a network got worse against.

So the trainer seats other players, and this says which. Passing a list of
paths on the command line was the old way; it made the population a
property of whoever typed the launch line, and nothing in the run recorded
who had been at the table.

The roster has four kinds of member:

- **champion** — the network that last passed `neural.promote`. Beating the
  current best is the point, so it gets the largest share.
- **recent** — checkpoints from the last few blocks of this lineage, which
  keep the policy honest against what it was very recently.
- **older** — checkpoints from further back, which catch a policy that is
  going round in circles: losing to your own great-grandparent is a clear
  signal and a plain self-play loop never sees it.
- **reference** — fixed outside players that never change, so a number
  measured against them this month means the same as last month. The
  published Mortal and the network the browser used to ship are both here.

A member's share is how often it takes a seat, not how often it wins. The
shares are documented rather than tuned: the champion often, the fixed
references regularly enough to keep their readings meaningful, the older
checkpoints seldom.

Nothing here promises that beating one member implies beating another.
That is the reason for keeping them apart in the report: a policy that
improves against its own recent past while losing to the fine-tuned Mortal
has specialised, and one averaged number would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Member:
    """One player who may be seated, and what it is."""

    #: Where the checkpoint lives, as the volume names it.
    name: str
    #: champion, recent, older or reference.
    role: str
    #: What this player is, for whoever reads the log a month from now.
    note: str
    #: How often it takes a seat, relative to the others.
    weight: float = 1.0


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
)


@dataclass
class Population:
    """The roster, and how a round's tables are made from it."""

    members: list[Member] = field(default_factory=list)

    @classmethod
    def around(
        cls,
        champion: str | None = None,
        recent: list[str] | None = None,
        older: list[str] | None = None,
        references: tuple[Member, ...] = REFERENCES,
    ) -> Population:
        """The usual roster: the champion, its recent past, its distant
        past, and the fixed references."""
        members: list[Member] = []
        if champion:
            members.append(
                Member(champion, "champion", "the last checkpoint to pass the gate", 3.0)
            )
        for name in recent or []:
            members.append(Member(name, "recent", "a recent checkpoint of this lineage", 1.0))
        for name in older or []:
            members.append(Member(name, "older", "an older checkpoint of this lineage", 0.5))
        members.extend(references)
        return cls(members=members)

    def names(self) -> list[str]:
        return [member.name for member in self.members]

    def describe(self) -> list[dict]:
        """The roster as it should appear in a run's log: who was available
        to be seated, in what role, and how often."""
        total = sum(member.weight for member in self.members) or 1.0
        return [
            {
                "name": member.name,
                "role": member.role,
                "share": round(member.weight / total, 3),
                "note": member.note,
            }
            for member in self.members
        ]

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
        if not self.members or share <= 0:
            return chosen
        taken = rng.random(games) < share
        count = int(taken.sum())
        if not count:
            return chosen
        weights = np.array([member.weight for member in self.members], dtype=np.float64)
        weights = weights / weights.sum()
        chosen[taken] = rng.choice(len(self.members), size=count, p=weights)
        return chosen


def matchups(seated: np.ndarray, placements: np.ndarray, members: list[Member]) -> list[dict]:
    """How the learner did against each member it met, kept apart.

    `seated[game]` is the member in that game or -1, and `placements[game]`
    is the learner's own average placement there. Reported one row a
    member, never summed: improving against your own recent past while
    losing to a fixed reference is specialisation, and a single average is
    exactly what hides it.
    """
    rows = []
    for index, member in enumerate(members):
        met = seated == index
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
    alone = seated < 0
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
