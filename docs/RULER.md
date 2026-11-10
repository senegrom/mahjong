# The ruler

How one network is measured against another: what a ruler is, the faster way
of playing it, two challengers on the same deals, a challenger that averages
its moves over characters and circles, one that leaves the last hand's riichi
undeclared where its tile keeps the player first through every win, one that
follows another rule of the last hand, and the rule a series is read by, so
that a clear difference stops early and a small one is given the seeds it
needs.

## What a ruler is

A duel (`neural.duel`) seats a challenger at one place of a table and an
incumbent at the other three, plays every deal four times with the challenger
in each seat, and reports the challenger's placement. Two identical players
average 2.50, so the edge is 2.5 less the placement, positive when the
challenger is stronger; each deal gives one figure, its placement over the four
seatings, and the error comes from the spread across deals.

A ruler is a series of duels on Modal, one call a seed of 1,000 deals: the
published rulers are 20 seeds from 9,100,000, every 100,003, 80,000 games in
all, against published Mortal (`zoo/mortal_298k`). Each call took about 21
minutes on an L40S, so a ruler was about 7 L40S-hours and $17-20.

## The faster play path: `neural.ruler`

`neural.duel` plays a call's four seatings one after another, each at a table
of 1,000 games. A table's last few hundred steps are played by fewer and fewer
games, and each step still pays the engine, the encoder and two networks their
whole fixed price; the four tables pay that tail four times. `neural.ruler`
plays all four seatings of every deal in one arena: `riichi_py.Arena(seeds=...)`
deals each table from its deal's own seed, so a table dealt from `seed + i`
plays exactly the game the old duel's arena deals at its table `i`. In that
arena:

- the engine steps every table together, and the four seatings' tails overlap;
- each step's deciding players are encoded in two calls rather than eight, the
  challengers' here and the incumbent's on a worker while the challengers
  decide (`Views.prepare`);
- every network plays with its weights cast to bfloat16 once
  (`policy_inference.precast`), which leaves every answer bit for bit as
  autocast gave it.

### Two ways of asking

On the card a batch of another size can be worked through by other kernels,
which in bfloat16 can move an answer in its last bits, and a near tie can turn
on a bit (see `neural/play_graphs.py`). So how each network is asked decides
whether the games are the old duel's:

- `--questions seating`, the default, asks each network about each seating's
  tables apart, in the very batches `neural.duel` asked it. It plays the old
  duel's games bit for bit: the same placement on every deal. A ruler played
  this way continues the published series. With `--graphs`, a size of
  question asked often (three times, up to 512 rows) is answered from a CUDA
  graph recorded at that very size. A graph replays the kernels the eager
  forward chose at its size, so the answers stay the old duel's, and the
  small questions that end every table cost one launch instead of a
  thousand small ones.
- `--questions fixed` asks each network once a step about all of its tables,
  cut into pieces of exactly `--rows` rows (256), the last padded with blank
  rows. On the card every answer then depends on its own row alone, so the
  incumbent answers a position the same at every table and every step, and
  answering from a CUDA graph (`--graphs`), leaving rows with one move unasked
  (`--skip-forced`) and asking a reach's tile with the reach
  (`--preview-reach`) change no game. Its games are not the old duel's, so a
  series measured this way is compared only with series measured the same way:
  re-measure the reference (910, say) once with the same questions.

`--timing` adds `play` to the report: the wall time and its parts (the engine,
the follower, encoding on the playing thread, each side's questions, what the
worker encoded meanwhile), the steps and the decisions asked of each side. The
old duel takes `--timing` too and plays the same games with it.

### What the desktop shows

On this desktop's card (an RTX 5070 Ti), the same rows asked at another batch
size moved the last bits of every answer of both published Mortal and
fusion-long's 810, while at one size, the rows shuffled or put beside other
rows, nothing moved. On the processor, in float32, shuffling alone moved two
or three rows of 64, so the fixed rows' guarantee is the card's. A graph
recorded at the very size asked answered every row bit for bit as the eager
forward, for published Mortal and 910, at each of the fourteen sizes tried
from 1 to 512. At 1,000 deals a network may record a graph for most sizes up
to 512 as its tables finish, each in about a tenth of a second on the desktop
(14 in 1.2 s); the timing check says how many the L40S recorded and in how
long.

The arena and its follower take about 0.1-0.4 MB of host memory a table and
do not grow over a game (600 tables played to the end: 1.43-1.44 GB
throughout), so even a paired call's 8,000 tables add only about 2 GB.

The roadmap's check, at a desktop's scale: 910 against published Mortal on
seed 9,100,000, 8 deals a seating (the deals of the published ruler's first
call's first eight), on a desktop whose processors were shared with another
session's browser. The times are long, and at 8 deals the networks' small
questions weigh far more than at 1,000:

| way | placement | play | engine, follower, encoding | questions | same deals as the old duel |
|---|---|---|---|---|---|
| old duel | 1.9688 +- 0.1666 | 511 s | 15, 17, 73 s | 405 s | |
| ruler, seating | 1.9688 +- 0.1666 | 342 s | 2, 3, 11 s (18 aside) | 326 s | 8 of 8, and every seat |
| ruler, seating, graphs | 1.9688 +- 0.1666 | 70 s | 2, 0.5, 23 s (26 aside) | 44 s | 8 of 8, and every seat |
| ruler, fixed 256, no switch | 1.9688 +- 0.2029 | 142 s | 1, 0.3, 12 s (25 aside) | 128 s | 5 of 8 |
| ruler, fixed 256, every switch | 1.9688 +- 0.2029 | 74 s | 3, 1, 28 s (30 aside) | 41 s | 5 of 8; all 8 as the fixed ruler with no switch |

Asked by seating, the ruler asks as many questions as the old duel, one a
seating, so without graphs it gains only what the arena and the encoding gain;
with graphs at the sizes asked (14 recorded, in 1.2 s), 7,136 of its 7,167
questions were answered from them. With 810 as the challenger the old duel
and the seating ruler read the same too, 1.9375 +- 0.1398, and a paired call
of 810 against itself in fixed rows placed both arms the same on all 32
tables, a difference of exactly nought.

How much a batch's shape decides: on those eight deals the published call, on
an L40S at tables of 1,000, placed as the desktop's old duel, at tables of 8,
on only 4 of them, and as the desktop's fixed ruler, whose pieces of 256 rows
are nearer the published call's batches, on 7.

At 1,000 deals a question holds hundreds of rows for most of a call, graphs
answer only questions of up to 512, and the encoder's work, the same in every
way of playing, weighs more. So the desktop's ratios are not the cloud's; the
timing check below measures those.

## Two challengers on the same deals

`--also` seats a second challenger at four more tables of every deal, against
the same incumbent in the same seats. The report has an entry for each under
`arms`, each exactly what a ruler of that challenger alone would report, and
their `difference`: the first's edge less the second's, deal by deal, with the
error from the spread of that difference.

What a challenger gains or loses on a deal is mostly the deal's luck, and two
close networks share most of it. The published rulers of 810 and 910 against
published Mortal, played in separate calls on the same deals, correlate at
+0.628 deal by deal, and 25.6% of the deals were placed the same by both; the
difference of the pair reads -0.0022 +- 0.0035 against +- 0.0058 for the two
rulers read apart.

Asked by seating, a paired call is two old duels at once, each arm placing
exactly as a duel of its challenger alone. Asked in fixed rows, the incumbent
answers a position the same at both arms' tables, so a table's games stay the
same until a decision of the challengers' differs: in the old duel a real
difference at one table changed the batches every other table of its seating
was asked in, and with them, now and then, a near tie elsewhere. On the
desktop, 910 and 810 against published Mortal at 8 deals played 13 of 32
games the same to the end in fixed rows and 11 of 32 asked by seating, too
few deals to say how much that adds. Two challengers that are one network play
every game the same and differ by exactly nought, which is the check that the
pairing is sound.

## Averaging over the suits

Under our rules characters and circles are interchangeable: there are no red
fives, and All Green, the one yaku that names a suit, names bamboo. Renaming
every 1m-9m as 1p-9p and back turns a game into another game just as likely,
so a player's best move in one is the renamed best move in the other. A
network has learned that only so far: g810 picks another best move in the two
namings of 4% of its decisions. `--average-suits` has a challenger ask each
question it hesitates over (its plain answer's top weight below 0.9) of the
renamed game too, and play the mean of the two answers' logits (`neural.suits`
says why that average and that gate).

The second view is libriichi's own encoding of the renamed game, not the
game's planes permuted: the efficiency lookahead breaks ties by tile index,
characters before circles, so the two differ in about half the questions. A
second follower follows the averaging challenger's tables with every event
renamed and is told every reach the first is told, so a reach's tile is asked
of the renamed game as well. The renaming gives, to the bit, the planes Mortal's
own augmentation gives at every decision of a game (`test_suits`). The renamed
questions go to the network in a forward of their own, after the plain ones, so
the plain questions are asked exactly as a plain player's are; nothing else
changes, and without the option nothing at all.

The screen of 15 October measured it on 14,269 of g810's real questions: at
the ruler's table the mean changed g810's move in 1.93% of its decisions with
two or more moves (95% 1.64-2.23%), 3.2 a game; 17.7% of questions were asked
again. Published Mortal, its values averaged over both namings, preferred the
new move in 52% of the changes against a matched placebo's 45%, so the gain is
likely small: about +0.003 to +0.005 placement. Put to the same stored
questions on the desktop's card in float32, as the screen asked them, the
option made exactly the screen's changes, 155 of 155 at the ruler's table and
101 of 101 at the Club table, and asked 17.7% of the questions again.

A paired call of one checkpoint averaging and plain, asked in fixed rows on
the card, plays each deal the same at both arms' tables until a decision of
the averaging arm changes, and its plain arm plays exactly the games of a call
with two plain arms. On the desktop at 16 deals a seating (810 against
published Mortal, every switch on), the averaging arm asked 19% of its rows
again, the mean changed 185 of its 10,987 rows (2.9 a game), 26 of the 64
games ended with the same scores at both arms' tables, and the call took 99 s
where two plain arms took 77 s: the questions asked again are a fifth of the
arm's rows but, at so few tables, nearly a whole padded piece a step. At 1,000
deals the averaging arm should cost a fifth to a third more than a plain one.
Each arm that averages says under `suits` how many rows it was asked, how many
of them it asked again, and how many moves the mean changed.

The site's Trained AI averages the same way, wherever the page asks it
(`web/src/lib/suits.js`, [Agent modes](AGENT_MODES.md)). The second state
lives in the page's engine too (`engine/riichi-wasm`): four more of Mortal's
states, told the same events renamed by `Event::augment`, and told only when
a renamed view is asked, so a table without a trained player pays nothing.
A review rebuilds a decision's renamed view from the hand's deal, as it does
the plain one. The page asks a hesitant question's renamed view in a request
of its own after the plain one, and works the mean out from the two
answers' weights: the square root of each move's two weights multiplied,
normalised, is the softmax of the mean of the two logits. CI's worker
parity job checks the page against `suits.averaged` on a small export's
PyTorch twin (`neural/tests/export_worker_fixture.py`).

### The series: 810 averaging against 810 plain

Deploy first: the app must hold `ruler(average_suits=)`, and Modal must take
launches. Then, with the Modal client's Python from the repository root:

```sh
python -m neural.ruler_cloud series g810-suits history/gen-00810 zoo/mortal_298k \
    --also history/gen-00810 --average-suits first --run fusion-long --expected 0.004 \
    --questions fixed --graphs --skip-forced --preview-reach --local E:/tmp-claude/mahjong/rulers
```

Every call plays 810 averaging (the first challenger) and 810 plain (the
second) against published Mortal on the same 1,000 deals, so the series reads
the edge averaging gives, deal by deal, positive when it helps. An expected
effect of 0.004 registers the small-effect rule: ten seeds a look, four looks,
forty seeds from 9,100,000. The 16 deals on the desktop put the spread of the
paired difference roughly where it is for 810 and 910 (0.46 a deal against
0.48), so by the table under "Reading a series" the rule finds +0.004 about a
third of the time and +0.006 about three times in five: a level reading would
not rule a gain of that size out. A series registered with `--average-suits`
keeps it; `looks g810-suits` reads it later, and names the first arm "averaged
over the suits".

## All-last dama

South 4 is the last hand: there is no extension round and no agari-yame, so a
win by anyone but the dealer ends the game, and the places are then the
points. A non-dealer for whom riichi would add no win, and whose every win
without it leaves them alone in first, gains no place by declaring through
any win of their own, and pays a thousand points and the news that the hand
is ready. The networks declare there all the same: in the probe of 29 October,
810 and 1410 declared riichi at every gap tried. `--all-last-dama` has a
challenger leave such a riichi undeclared (`neural.all_last`): in South 4,
where its best move is riichi and it is not the dealer, it asks the engine
about the tile the reach would throw, and if that tile, thrown without
declaring, keeps it first through every win, it throws the tile and declares
nothing.

The engine answers (`riichi_py.Arena.all_last_dama`, from
`engine/riichi-core/src/all_last.rs`), tile kind by tile kind, with its own
scorer and its own settlement: the hand as the plain discard leaves it, every
live wait (a copy left that the player cannot see) won by ron off each of the
three others, unless the discard leaves the hand furiten, and by tsumo, each
win settled as a real one is, counters and the bets on the table included.
The tile keeps first (code 1) only if every such win leaves the player strictly
ahead of all three: a tie shares the uma (EMA 2025 section 3.7.1). Otherwise
the reach stands, and the code says why: a live wait with no yaku without
riichi (2), a win that leaves another player level or ahead (3), or no live
wait (4). The answer holds for the rest of the hand, since until the player
wins the others' points can only fall; two winners on one discard are not
modelled. The dealer is never asked: a dealer's win keeps the deal, and South
4 is played again. The engine's tests play every such win through the real
flow and get the same codes. In forty games of four Club players, 67 turns
were a non-dealer's in South 4 with riichi open, and 9 of them had a tile that
keeps first (`cargo test --release -p riichi-core club_games_counted --
--ignored --nocapture` counts 20,000 games).

The tile is the reach's own: the best of the legal riichi tiles in the reach's
answer, asked with the first question (`--preview-reach`), or the one tile
that keeps the hand ready when `--skip-forced` left the reach unasked. Not the
network's best move once riichi is put aside, which would set a reach with
one tile against silence with another. Where the rule fires, the discard of
that tile becomes the row's only finite answer, so the player discards it, no
second question is asked and no follower is told a reach the table never
makes. So the rule needs `--questions fixed` and `--preview-reach`, and is
refused without them: asked by seating, a firing at one table would change
the batches the rest of its seating is asked in.

It asks the network nothing more, and only where it fires does a game change.
A paired call of one checkpoint with the rule and plain, in fixed rows on the
card, should therefore play every table the same at both arms but those where
the rule fired, and the ruler checks exactly that: each table whose final
scores differ between the arms must be one the rule fired at
(`all_last_check` in the report, with any violations by deal and seat). That
holds only of one network played twice, so the rule is played only so: the
ruler, `modal_app.ruler` and the launcher refuse it unless `--also` names the
challenger's own checkpoint and both arms average alike (no `--average-suits`,
or `both`). The check cannot see everything: the rule fires only in South 4,
so at a table it fired at, a pairing that broke earlier in the game hides
behind the firing and is put down to the rule. It is a condition the pairing
must meet, not a proof that it held. A report with violations is written and
kept as any other, and the ruler ends as usual; the launcher stops the series
on it (see below). The arm says under `all_last` what the rule did: `riichi`,
the decisions it judged; `fired`, those it threw the tile undeclared at;
`tables`, the arena's tables it fired at; `hands`, how many hands; `kept`, by
code, why it let the reach stand at the rest; and `riichi_after_firing`, the
reaches it let stand later in a hand it had fired in, when the tile of a later
turn no longer kept first.

### The series: 2610 with the rule against 2610 plain

The rule is new engine code and a new option of `modal_app.ruler`, so it is
played only on an app deployed from a checkout that holds it under a name of
its own: a small file outside the checkout that renames `modal.App` while it
imports `neural.modal_app`, deployed from the checkout, makes one on the same
volume while `mahjong-train` keeps the code its runs restart with. Never on
`mahjong-train` itself, the live training app: a series must play the rule of
the checkout it measures, and deploying that checkout as `mahjong-train`
would change the code the live runs restart with (a `mahjong-train` deployed
before the rules would also fail every such call). So the launcher refuses
`--all-last-dama` unless `--app` names another app, keeps the app in the
series, and refuses to resume the series on another. Before it registers
anything it looks the app's ruler up, and stops if there is none; a call that
fails on an argument the app's code does not take, from an app deployed from
an older checkout, is never spawned again, since every call there fails
alike, and the series is then registered again under a new tag. Then, with
the Modal client's Python from the repository root:

```sh
python -m neural.ruler_cloud series alllast-dama-g2610 history/gen-02610 zoo/mortal_298k \
    --also history/gen-02610 --all-last-dama first --run fusion-place-a005 \
    --questions fixed --graphs --skip-forced --preview-reach \
    --looks 1 --per-look 20 --bound 2.0 --app mahjong-rules \
    --local E:/tmp-claude/mahjong/rulers
```

Every call plays 2610 with the rule (the first challenger) and 2610 plain (the
second) against published Mortal on the same 1,000 deals. `--looks 1
--per-look 20 --bound 2.0` registers one look after twenty seeds, read at z
above 2.0, in place of four looks read at Pocock's bound: the rule is kept
only if that one look reads the first arm the stronger. `looks
alllast-dama-g2610` names the first arm "with all-last dama". Stop the app
once the series is read (`modal app stop mahjong-rules --yes`).

Every report is held to the check registered with the series: the ruled arm
fired at least once, the plain arm says nothing of the rule, and
`all_last_check` names no violation. The first report that fails stops the
series as invalid: it is kept, no seed is spawned again (a seed that passed
once it was spawned again would hide the fault), the calls already spawned
end on their own, and `looks`, of either module, refuses to read the folder.

The series of 30 October read 2610 with the rule the stronger at its one
look: +0.0010 +- 0.0002 placement over 20,000 deals (z +5.08). The rule
judged 15,779 riichi decisions and threw the tile undeclared at 1,091 of
them, at 940 tables; it let the reach stand at 6,237 for a live wait with no
yaku without riichi, at 8,451 for a win that would leave another player level
or ahead, and at none for a dead wait. 911 tables ended with other scores
than the plain arm's, every one a table the rule fired at, and 126 deals were
placed otherwise, 88 better and 38 worse. The rule weighs only the player's
own wins: not declaring also changes how the others play, who see no reach,
and leaves the player free to change its hand later, so the series measures
the net. The project keeps the rule. The training plays it too, in the
learner's own self-play, at the tile each reach draws, when told
(`train_combined --last-hand-rules`, see docs/TRAINING_SAFETY.md, "The last
hand's rules in self-play"), and so does the site's Trained AI
(`web/src/lib/trained-move.js`, [Agent modes](AGENT_MODES.md)).

## Late-game rules

The plateau diagnosis of 30 October found the policy settled: 81% of its
decisions with more than one move open sit above 0.999, and policy gradient
does not move a settled habit. An exact rule, measured by a paired ruler,
can. Three more rules of the last hand, each measured by a series of its own
with `--late-rule`, one rule a series and never beside `--all-last-dama`, were
played as the all-last dama rule is: the first challenger follows the rule and
the second, the same checkpoint, plays plain; in fixed rows with the reach's
tile asked with the reach (`--preview-reach`), both arms averaging alike;
checked the same way, every table whose final scores differ between the arms
being one the rule fired at (`late_rule_check`); and only on an app named with
`--app`. Each rule asked the engine (engine/riichi-core/src/all_last.rs) about
the rows whose best move is its kind, each row a game and a player by place,
and the engine worked the answer out on copies of the hand with its own legal
moves, scoring, settlement, tenpai predicate and the tiles the table shows
safe. Each answered on what the table shows, never on another player's hidden
hand or the tiles under the dora indicators, so that it could be played as it
is at a real table. Where a rule fired, the move it played became the row's
only finite answer, or, for a declined self-draw, the win was struck from the
row, so the player played it and no follower was told anything the table does
not do.

The project keeps one of the three, `decline-last` (`neural.late_rules`),
still played as above, and dropped the other two, `wide-dama` and
`dealer-noten` (see the series below). Their code, the engine's questions and
bindings, the players and their tests, was removed after #215, the pull
request that holds it. Their sections below keep what each rule was, and the
series what each measured; nothing in this repository plays either now.

The arm says under `late_rule` what the rule did: `rule`; `judged`, the
decisions the engine judged; `fired`, those the rule changed; `tables` and
`hands`, where it fired; `kept`, why it let the move stand at the rest; and
`kept_after_firing`, the moves it let stand in a hand it had fired in. Every
report of a series must say the rule it was registered with, meet the rule's
floor, keep the plain arm plain and pass the check; the first that does not
stops the series as invalid, as for the all-last dama rule. The floor is
`fired` at least once for `decline-last`, which finds something to change in
any call of a thousand deals. The series of 31 October held `wide-dama` to the
same floor, and `dealer-noten`, whose question comes to the challenger only as
the dealer of South 4 near the end of the wall and may find no move to change
in a call, only to `judged` at least once. A rule the code no longer names is
held to `judged` alone, so the two dropped rules' kept series still read: the
`comparing` line names the first arm as registered ("with wide dama"), a
verdict by the rule ("with the late rule wide-dama").

### Riichi that buys no place: `wide-dama` (dropped)

Dropped: it adds nothing measurable over the all-last dama rule (see the
series below).

2610 and 2820 declare riichi in South 4 whatever the places would be. The
all-last dama rule leaves a riichi undeclared only where the player keeps
first through every win; this one wherever riichi would change no place. The
engine (`riichi_py.Arena.all_last_wide_dama`, in #215) takes the dama
question's wins, every live wait won without riichi by ron off each other
player, unless furiten, and by tsumo, and wins each again with riichi declared
on the same tile: the thousand points paid and the bet in the pool, from which
the winner takes it back as a real win does, a double riichi in the player's
first turn with the first set of turns unbroken. What riichi adds is not known
when it is declared, so the riichi win is scored at the least riichi adds,
riichi alone, and at the most, riichi, the one-shot, and under each dora
indicator an ura indicator of the kind the winning hand holds most copies of.
A place is how many others end ahead of the player and how many level, since a
tie shares the uma (EMA 2025 section 3.7.1). Riichi buys no place (code 1)
where every win's place is the plain win's at both bounds; a larger win only
moves the winner up and the payers down, so the two bounds cover every riichi
win between them. Otherwise the code says why the reach stands: a live wait
with no yaku without riichi (2), a win riichi would end in a better place (3),
or no live wait (4). A tile that keeps first is always one whose riichi buys
no place. Not counted, with riichi and without alike: a win on the last tile,
on a quad's replacement or by robbing a quad, and a new kan dora, any of which
could carry one of the two over a line and not the other; and two winners on
one discard. And unlike the all-last dama code, which holds for the rest of the
hand, this one holds at the moment of the decision: a riichi bet another
player places later, which the winner collects, or a new kan dora moves the
lines a win's place is read against, with riichi and without. `kept` counts
`no_yaku`, `buys_a_place` and `dead`.

### Declining a win that leaves the player last: `decline-last`

Given the 14,269 questions 810 was asked in its games against Club players
and against published Mortal, which the screen of 15 October kept (see
"Averaging over the suits"), 2820 declined none of the 233 wins among them,
62 of which came in South 3 or 4; and in a round of 8,536 learner seat-games
it took 219 South 4 wins that left it alone in last (2.6%). A non-dealer's
win in South 4 ends the game, so a win that leaves the player strictly below
all three others is a fourth place for certain, and where it is the only
win, declining it can place them no lower. The engine
(`riichi_py.Arena.all_last_decline`, asked at a turn or in a claim window,
which asks its seats one at a time) plays the win on a copy through its own
`act` or `resolve_calls`, so the counters, every bet on the table and the bet
of a riichi declared with the very tile won on, which goes back to its owner
(EMA 2025 section 3.3.10), are settled as a real win settles them: code 1
where the player ends alone in last, 2 where they do not. A riichi winner's
ura dora are turned only once the win is declared, so the engine never reads
them: it plays a riichi win at the most they could add, under each dora
indicator an ura indicator of the kind the winning hand holds most copies of.
A larger win only lifts the winner and lowers the payers, so code 1 means
alone in last whatever lies under the indicators. The dealer is never asked: a
dealer's win keeps the deal. A declined ron is the pass; a declined self-draw
is the best other move of the same answer, a reach among them, whose tile is
then the preview's. The code is the player's win alone. Another player may win
on the same discard (EMA 2025 section 3.3.1), and the rule cannot see it:
whether another hand can win on the tile is hidden, and at a real table the
answers to a discard are given at once, so a rule that read either could not
be played honestly. With a second winner, declining can cost: a dealer who
wins too keeps the deal, and the declined ron only loses its points; a second
non-dealer has the discarder pay twice, which can lift the player off last.
Rare, and not modelled. `kept` counts `not_last`; `kept_after_firing` counts
the wins taken later in a hand the rule declined one in.

### The leading dealer's last discard: `dealer-noten` (dropped)

Dropped: it reads level (see the series below).

In a probe of 31 October, 2610 as the South 4 dealer kept its hand waiting on
its last discard at every lead tried, 3,000, 5,000, 8,000 and 12,000, in
hand-built positions with one tile left to draw, the position the dealer's
last draw leaves when nobody has called: the dealer waiting on 1m-4m, with a
South to throw that keeps the wait and an East that breaks it, both
completely safe, every other copy of each being in sight. A dealer waiting at
the exhaustive draw keeps the deal and South 4 is played again; one not
waiting passes it, and the game ends.

The engine (`riichi_py.Arena.all_last_dealer_noten`, in #215) answers at the
dealer's own discard in South 4 with at most three tiles left to draw: South,
West and North draw them in turn and the dealer draws no more, so the hand the
discard leaves is the hand the draw finds. A call that hands the dealer
another turn (with three tiles left, South claiming the dealer's discard lets
the dealer draw the last) meets the question again at that turn's discard. It
answers only for a dealer more than 4,000 ahead of every other player: for
each legal discard, 1 where it leaves the hand waiting by the very predicate
the draw is settled by; 2 where it leaves it not waiting and the tile is safe
against each of the three others, as the table shows it (in their own pond,
which makes them furiten on it, or thrown since their riichi and not won on);
3 where it leaves it not waiting and the tile is not. Any waiting player who
is not furiten may win on the dealer's discard (on the last discard of all,
even without another yaku), and a ron could cost the dealer first: in a
reviewer's example a dealer on 40,100 threw a 5p that left it not waiting,
South, on 36,000 and waiting on 2p-5p, won on it, and the dealer ended second.

If nobody wins, the lead outlasts the hand. Nobody's points rise but by a win:
a riichi another player declares (legal while a tile is left to draw) costs
its declarer a thousand, and its bet, which no win then collects, goes to the
leader at the end of the game; and the noten payments move at most 4,000
between the dealer and another player (a lone waiting player takes 3,000 while
the dealer pays 1,000; three take 1,000 each while the dealer pays 3,000). So a
dealer more than 4,000 ahead who ends the hand not waiting ends the game alone
in first. A win by another player after the discard ends the game the same
whatever the dealer holds, but for what a dealer left waiting could still do:
win on a discard yet to come, which would only play South 4 again, no place
yet (the ruler measures placement alone), though it would end the hand before
anyone else could win in it; or win beside another player on one discard and
keep the deal. Neither is modelled, and the second is rare. Nor is a claim on
the tile thrown, which, with tiles left to draw, changes who draws them, and so
who might yet win. Nor a quad: with four or more tiles left at the dealer's
discard the question is not put, but a quad another player declares later
moves a live tile to the dead wall and can take the dealer's last draw, so
that discard, the dealer's real last, goes unasked.

Where the dealer's best move keeps the hand waiting (a discard coded 1, or a
riichi, which always waits) or throws an unsafe tile that leaves it not
waiting (coded 3), the rule plays the discard the answer rates highest of
those coded 2, if there is one. Where the best move it replaces is an unsafe
tile, the rule does not weigh what it gives up: a ron on that tile would have
ended the game at once, possibly with the dealer still first, where the safe
tile lets the hand run on. A riichi row needs no tile to be judged: it is
decided within the first question, so where the rule fires the reach's preview
goes unread and no reach is made. A riichi dealer may throw only the tile
drawn, which keeps the hand waiting, so the rule never fires for one. `kept`
counts `already_safe_noten` (the best move was already a discard coded 2) and
`no_safe_noten`; `kept_after_firing` counts a move kept at a later discard of
the same hand, which only a call brings.

### The series: 2610 with each rule against 2610 plain

Each series was played as the all-last dama rule's was, on an app deployed
from a checkout that held the rules, `mahjong-rules`, never `mahjong-train`,
one series a rule, under the tags `late-wide-g2610`, `late-decline-g2610` and
`late-noten-g2610`. Every call played 2610 with the rule and 2610 plain
against published Mortal on the same 1,000 deals.

A series of the kept rule is registered as `late-decline-g2610` was, with the
Modal client's Python from the repository root. Deploy `mahjong-rules` from
the checkout being measured before every series. One deployed before the late
rules takes no `late_rule` and fails every call: the series stops at its first
seeds, which are never spawned again, and the tag is used up and must be
registered anew. One deployed from an earlier version of the rules plays that
version without any error.

```sh
python -m neural.ruler_cloud series late-decline-g2610 history/gen-02610 zoo/mortal_298k \
    --also history/gen-02610 --late-rule decline-last --run fusion-place-a005 \
    --questions fixed --graphs --skip-forced --preview-reach \
    --looks 1 --per-look 20 --bound 2.0 --app mahjong-rules \
    --local E:/tmp-claude/mahjong/rulers
```

`looks late-decline-g2610` names the first arm "declining last-place wins". A
pilot of one call on the desktop, before anything is spent:

```sh
python -m neural.ruler <2610.pt> <mortal_298k.pt> --also <2610.pt> --late-rule decline-last \
    --games 500 --seed 6600000 --questions fixed --graphs --skip-forced --preview-reach --timing
```

The engine's tests play every win and declination through the real flow and
the table, and check the codes at every decision of Club games (`cargo test
--release -p riichi-core club_games_counted -- --ignored --nocapture` counts
20,000 of them). In those 20,000 games, 2,012 of the 18,623 wins open to a
non-dealer in South 4 would have left the player alone in last, and 868 more,
riichi wins, only with the ura dora as dealt, which the rule never reads.
Before the dropped rules' code was removed, the same count found that 19.2% of
the 28,719 last-hand riichi decisions had a tile whose riichi buys no place
(14.7% one that keeps first), and that a leading dealer's own discard with at
most three tiles left came 1,052 times, 230 of them with a safe discard that
leaves the hand not waiting (51 with one that keeps it waiting as well); of
their 7,007 legal discards, 795 keep the hand waiting, 322 leave it not
waiting on a safe tile and 5,890 on an unsafe one, 537 of which another player
could have won on when played. They are the Club players' figures, not a
network's: how often each rule fires for 2610 is the series' to say.

The series of 31 October, each one look after twenty seeds of 1,000 deals,
read at z above 2.0:

| rule | series | reading | judged | fired | tables fired at | tables ended otherwise |
|---|---|---|---|---|---|---|
| `wide-dama` | `late-wide-g2610` | with the rule the stronger, +0.0011 +- 0.0002 (z +4.99) | 16,108 | 2,017 | 1,550 | 1,487 |
| `decline-last` | `late-decline-g2610` | with the rule the stronger, +0.0009 +- 0.0001 (z +7.25) | 19,011 | 1,409 | 1,037 | 1,037 |
| `dealer-noten` | `late-noten-g2610` | level, -0.00001 +- 0.00001 (z -1.00) | 950 | 106 | 106 | 28 |

No report named a violation: every table that ended with other scores than
the plain arm's was one the rule fired at.

`wide-dama` fires at every table the all-last dama rule fires at: the 940
tables of the dama series are all among its 1,550. The two series played the
same seeds and their plain arms placed alike on every deal, so their ruled
arms, read against each other deal by deal, say what the widening adds: over
the 20 seeds, each deal's placement by the dama series' ruled arm less its
placement by the wide series' (the first arm's `by_deal` in each report),
positive where the widening placed better, averages +0.0001 over the 20,000
deals, with an error of 0.0001 from the spread across deals (z +0.99) and
0.00008 from the spread of the 20 seeds' means (z +1.29); 53 deals were placed
otherwise, 30 better and 23 worse. The widening adds nothing measurable over
the dama rule.

Every table `decline-last` fired at ended with other scores, and 65 deals were
placed otherwise, 64 better and one worse. `dealer-noten` changed the move at
106 of the 950 decisions it judged; at 774 there was no safe discard that
leaves the hand not waiting, and at 70 the best move was one already. 28 of
its 106 tables ended with other scores, and one game's place changed, for the
worse: at seed 10,100,030, deal 928 (table 3928), the dealer ended a place
lower with the rule than without it.

Each rule was measured alone, 2610 with it against 2610 plain at published
Mortal's table; the all-last dama rule and `decline-last` were never measured
together. The project keeps those two, the dama rule (+0.0010) and
`decline-last` (+0.0009). The training plays both together, decline first, in
the learner's own self-play when told (`train_combined --last-hand-rules
all-last-dama decline-last`, see docs/TRAINING_SAFETY.md, "The last hand's
rules in self-play"), and so does the site's Trained AI
(`web/src/lib/trained-move.js`, [Agent modes](AGENT_MODES.md)).
`wide-dama` adds nothing measurable over the dama rule (+0.0001 +- 0.0001),
and `dealer-noten` reads level (-0.00001 +- 0.00001: one deal of 20,000 placed
worse). So both were dropped.

## Networks measured with the rules: `--last-hand-rules`

Networks are measured against each other with the kept rules, never plain:
the site plays every network with them, and the networks differ by a few
thousandths, less than a plain series can tell apart from what the rules
change. With `--last-hand-rules`, every player at the table, each challenger
and the incumbent, follows both rules as the site plays them: decline-last is
asked first and the all-last dama rule around it, because a declined
self-draw can leave a riichi as the move, which dama must judge. Nothing else
of the play changes.

It needs fixed questions and the reach's tile asked with the reach, as each
rule does alone, and takes neither `--all-last-dama` nor `--late-rule`, which
measure one rule against plain play. No pair is checked: the checks of those
series hold only of one network played twice against a plain incumbent.
Instead, every report must show that each challenger (`all_last` and
`late_rule`) and the incumbent (`incumbent_rules`) followed both rules and
that each rule judged at least one decision of each (`looks.rules_problem`); a
report that does not stops the series. A series names each player "with the
rules".

It is played on an app deployed from a checkout that holds it, never on
`mahjong-train`, as each rule's series is. A head-to-head series of two
networks, the first in one seat against three of the second:

```sh
python -m neural.ruler_cloud series ruled-g3500-vs-g3000 history/gen-03500 history/gen-03000 \
    --last-hand-rules --run fusion-place-ruled \
    --questions fixed --graphs --skip-forced --preview-reach \
    --expected 0.006 --app <the app> --local E:/tmp-claude/mahjong/rulers
```

## Reading a series: `neural.looks`

A clear difference shows long before the twentieth seed and a level one never
does. Reading a series after every seed and stopping at the first reading past
two standard errors would find a difference that is not there about one time
in four, so the rule is written down before the first seed is spawned
(`design.json`, beside the reports):

- The seeds are read in their registered order, seed `k` being 9,100,000 +
  `k` x 100,003; a look covers the first seeds only and waits for every one of
  them. A failed call is run again with its own seed, never replaced.
- A look is taken after every 5 seeds, four looks, 20 seeds; or after every 10,
  40 seeds, when the effect looked for is 0.006 placement or less.
- At a look, z is the mean per-deal difference over every deal so far over its
  standard error, from the spread across deals.
- The comparison stops at the first look where |z| > 2.361, Pocock's bound for
  four looks: with no difference, the chance that any of the four crosses it is
  5% two-sided, as one look at 1.96 would be. The side z points to is the
  stronger. No crossing by the last look reads level.

That is the usual rule. `--looks`, `--per-look` and `--bound` register
another before the first seed (see "Running one"), as the rule series above
did: one look after 20 seeds, read at |z| > 2.0, about 4.6% two-sided.
`design.json` keeps it, and `looks` names a bound other than Pocock's for the
looks "the bound registered".

The difference is the one the series was registered with (see "Running one"):
the first challenger's edge less the second's with `--also`, the challenger's
edge less that of another series' challenger with `--against`, and the
challenger's edge over the incumbent with neither. Once a second player is
named, the incumbent never decides when a series stops.

How often a difference between two rulers of one lineage is found, at the
spread of the 810 and 910 pair (0.0153 a seed), and by which look:

| difference | 20 seeds: found | stopped by looks 1-4 | 40 seeds: found | stopped by looks 1-4 |
|---|---|---|---|---|
| 0.004 | 17% | 4, 8, 13, 17% | 31% | 6, 14, 23, 31% |
| 0.006 | 34% | 7, 16, 25, 34% | 61% | 13, 30, 47, 61% |
| 0.010 | 76% | 18, 42, 61, 76% | 97% | 38, 73, 90, 97% |
| 0.020 | 100% | 71, 97, 100, 100% | 100% | 96, 100, 100, 100% |

Replayed through the rule, from the published rulers' own per-deal results:

| comparison | looks | reading |
|---|---|---|
| 910 against 810, both against published Mortal, deal by deal | 4 | level, -0.0022 +- 0.0035 (z -0.62) |
| 516 against 466, head to head (the trial of 8 October) | 4 | level, -0.0029 +- 0.0032 (z -0.91) |
| 910 against 610, both against published Mortal | 4 | level, +0.0045 +- 0.0042 (z +1.07) |
| 410 against the old site's gen 36, head to head | 4 | 410 the stronger, +0.0104 +- 0.0038 (z +2.77) |
| 910 against published Mortal | 1 | 910 the stronger, +0.0486 +- 0.0082 (z +5.95) |

The last stops after five seeds, a quarter of the ruler.

## Running one

Everything below needs the deployed app to hold `ruler` and `duel(timing=)`:
deploy first. Run the launcher with the Modal client's Python from the
repository root; it needs nothing but the client. A series spawns one look's
seeds at a time and the next only if the rule says to go on, and can be
stopped and run again.

A series stops on the comparison it was registered with, so a series that
compares two generations names both when it is registered. The generations
measured lately are all far ahead of published Mortal (910 crossed the bound
at the first look, z +5.95): a series of one of them against Mortal stops
after five seeds, and two such series read against each other afterwards
never get past their first look. There are two ways to name the second.

With `--also`, both challengers are played in every call, against the same
incumbent on the same deals, and the rule reads the first against the
second, deal by deal:

```sh
python -m neural.ruler_cloud series g1010-g910 history/gen-01010 zoo/mortal_298k \
    --also history/gen-00910 --run fusion-long --expected 0.006 \
    --questions fixed --graphs --skip-forced --preview-reach --local E:/tmp-claude/mahjong/rulers
```

With `--against`, one challenger is played and the rule reads it against the
challenger of a published ruler, deal by deal on the same seeds. Asked by
seating, a series plays the old duel's games, so the published ruler stands
for the second arm and only the new challenger is paid for. The published
ruler is kept as a series from its call file first, while Modal still holds
its calls (about a fortnight: the ruler of 110, spawned on 29 September, was
gone by 14 October):

```sh
python -m neural.ruler_cloud keep g910 E:/tmp-claude/mahjong/modal-call-duelmany-fusionlong-g910-vs-mortal.txt \
    --local E:/tmp-claude/mahjong/rulers
python -m neural.ruler_cloud series g1010-vs-g910 history/gen-01010 zoo/mortal_298k --against g910 \
    --graphs --run fusion-long --local E:/tmp-claude/mahjong/rulers
```

The kept ruler must sit in the same `--local` folder and hold a report for
every seed the rule reads, of as many deals, against the same incumbent, and
asked the same way: a ruler measured the old way counts as asked by seating.
So `--against` a published ruler reads the usual 20 seeds; an effect of 0.006
or less, read over 40, is measured with `--also`. A series that could not be
read so is refused before anything is registered, as is one that asks with
`--skip-forced` or `--preview-reach` by seating, where every call would fail.

`--looks`, `--per-look` and `--bound` register another rule than the usual
(see "Reading a series"): `--looks 1 --per-look 20 --bound 2.0`, as the rule
series above were registered, reads one look after twenty seeds at |z| > 2.0,
and a bound left out is Pocock's for the looks. `--app` names the deployed app
the calls are spawned on, `mahjong-train` unless another is named, which the
series then keeps; the rules of the last hand are played only on another.

`looks <tag>` mirrors what the volume kept (`rulers/<tag>/seed-<seed>.json`)
and prints the looks so far, by the comparison the series was registered
with. `looks <tag> --against <other>` reads it against another series instead,
and says so when its rule was registered for another comparison, since then
its looks did not decide when it stopped. Both must be series of single
rulers: a series registered with `--also` holds each challenger's
placements and none of its own, and is read only as its first challenger
against its second (`looks <tag>`), so `--against` on either side of it is
refused. When the local folder holds that series, it is refused before
anything is read; when only the volume does, once a report of it is
mirrored, before any is compared (with none on the volume yet there is
nothing to compare, and the looks are empty). The two must also have
played at one table, as registration with `--against` requires: the same
incumbent (of the same run, for an archive or the latest or best of a run's
own), asked the same way, the rules of the last hand at both or neither, and
as many deals a seed, each taken from its registration or, for a folder never
registered, from its reports. A pair that did not is refused before any look
is read. `python -m neural.looks <folder>` reads a local folder of reports,
the old duel's included; `--against <folder>` reads it against another,
held to the same table.

At the spend limit Modal refuses a launch, which the launcher asks again
later, or takes it and leaves the call in its queue with no container. A
call in the queue is waiting, not playing: the launcher says when one starts
waiting, again every hour it waits, and when it leaves the queue, and waits
with it; it never spawns such a call again, since it has not failed. Neither
ends a series, however long the limit holds: it goes on by itself once the
limit is raised.

A call Modal cannot be asked about at all is another matter. When every
poll of it for an hour raises ConnectionError, ServiceError, InternalError
or ResourceExhaustedError, the call may be playing or lost, and nothing says
which, so the launcher stops and names the call and the error. Run the same
command again to resume: the series asks about the call again, and never
spawns its seed again while the call may still be playing. The hour is one
of polls, a minute apart: a call left unasked for more than a quarter of an
hour, as the seeds after one whose launch Modal refuses are, starts its hour
again at its next poll, since what Modal could not answer before says
nothing of the time between.

## The timing check

Before the port is trusted, one call of 1,000 deals on seed 9,100,000 is
played four ways, 910 against published Mortal:

```sh
python -m neural.ruler_cloud timing history/gen-00910 zoo/mortal_298k --run fusion-long \
    --local E:/tmp-claude/mahjong/rulers/timing
```

It spawns the old duel with `timing=True`, the ruler asked by seating without
graphs and with them, and the ruler in fixed rows of 256 with every switch,
and waits for the four (about $3). It keeps their reports and prints a block
for each, always in that order: placement, play time and its parts, how many
deals each ruler placed as the old duel did, and how much faster it was; then
the verdict. The old duel should read 2.4110 +- 0.0183, as the published
ruler's first seed did if nothing in the engine or the image has changed
since 14 October. The calls are written down as they are spawned: if Modal
refuses a launch partway, as at the spend limit, the same command run again
later spawns only the rest.

Each report is kept in the folder as its call finishes, written whole or
not at all, so a kill mid-write leaves none cut short; one that cannot be
read all the same is said, and its call asked about again. A call that
fails, as an old duel does when it dies of signal 9, stops the check with
its reason. The same command run again spawns again only the ways whose call
failed and says so; a call that finished, waits in Modal's queue or still
plays is kept, and the failed calls are written down under `failed` in
`timing-calls.json`. A call Modal cannot be asked about for an hour stops
the check as it stops a series, and the same command run again asks about
it again.

A seating ruler must place exactly as the old duel on all 1,000 deals, or it
has changed games and must not continue the published series. Without
graphs it asks the old duel's very batches, so one that changed games there
did it by its arena or its weights cast once, and no seating ruler may
continue the series; one that changed games only with graphs did it by its
graphs, and the series is continued without `--graphs`. The roadmap keeps
the port only if a seating ruler that kept the games is at least 1.5 times
as fast. The fixed ruler plays other games, so its placement differs within
its error. The parts say where the rest of the time goes: the encoder's work
is the same in every way of playing, so a call whose old duel spends most of
its time encoding cannot be made much faster by asking differently.
