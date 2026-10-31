# The ruler

How one network is measured against another: what a ruler is, the faster way
of playing it, two challengers on the same deals, a challenger that averages
its moves over characters and circles, and the rule a series is read by, so
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
nothing to compare, and the looks are empty). `python -m neural.looks
<folder>` reads a local
folder of reports, the old duel's included.

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
