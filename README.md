# Riichi

Riichi mahjong in the browser, played by the **EMA Riichi Competition Rules,
2025 edition** (in force since 1 January 2026), against opponents that learn
the game from self-play on the very same engine.

**[Play it](https://senegrom.github.io/mahjong/)** against three heuristic opponents.

The rules live once, in Rust. That crate is compiled to WebAssembly for the
browser and, later, to a Python extension for training, so the game a person
plays and the game the opponents were trained on cannot drift apart.

## What works today

- **The rules.** Tiles, hands, shanten with waits and acceptance, every
  reading of a complete hand, scoring with yaku, minipoints, limits and
  payments, the wall and dead wall, a full hand with calls and quads, and a
  full game with rounds, counters and uma.
- **A heuristic opponent** that plays for speed, only opens a hand that can
  still be declared, and folds against a declared riichi.
- **A browser game** against three of them, with the tile art, discard rows,
  called sets and optional hints, on a desktop or a phone.
- **Played by keyboard or mouse.** Arrow keys move along the hand and Enter
  throws the marked tile, the numbers throw one directly, and every tile
  carries its name for a screen reader.
- **Learning aids**: how far the hand is from a wait, what it is waiting on
  and how many of each are still unseen, the dora in hand, which tiles
  cannot deal into a declared riichi, and a furiten warning. A panel under
  the header explains each of those to somebody meeting them for the first
  time.
- **A post-game review** that takes each of your decisions again and shows
  what it traded: how far the move left the hand from complete, how many
  tiles would still have improved it, and whether it could have dealt in.
- **Logs in the mjai format**, one JSON object per line, which replayers,
  reviewers and other people's bots read.
- **A command-line arena and fuzzer** for checking the engine at scale.

- **Training from self-play**, with a warm start that imitates the heuristic
  player and a clipped actor-critic loop that continues from it.
- **A search that imagines the hands it cannot see and asks the network
  what results.** A policy head answers from the position in front of it
  and never looks ahead; this deals the unseen tiles into the other three
  hands and the wall, makes each candidate move, runs the other players
  round to the next decision, and has the network's value head judge the
  position that results, across hundreds of worlds and every core. A move
  has to beat the player's own choice by two standard errors of the
  world-by-world difference before it is taken, because taking the best of
  several noisy estimates otherwise keeps the luckiest candidate rather
  than the best. An earlier version played the worlds out with a heuristic
  instead, and measured worse than not searching once it was allowed to
  override: a biased judge, sharpened.
- **A network that reads the table.** The worlds are not dealt evenly, which
  would assume opponents discard at random. They chose what to throw, so
  what is left is what they wanted to keep, and the network has a head that
  learns that from the discards. Self-play knows the true hands, so the
  label is free and exact.
- **The hidden hands are weighed, not only sampled.** Dealing from that
  head's per-tile marginals is only a proposal: it knows nothing of shape,
  and deals a hand of thirteen strays as readily as one a turn from
  winning. So a second network, the reader, is shown the position with the
  real hidden hands and with hands the proposal imagined, and learns to
  tell them apart; what it learns is how much more likely a set of hidden
  hands is than the proposal made it. The search imagines a pool of worlds,
  has the reader weigh them, keeps the plausible ones with their weights,
  and decides by weight, which is what the strongest published searchers do
  with their hidden information. In training a second critic, the oracle,
  sees the hidden tiles outright and gives the policy gradient a quieter
  baseline; the policy never sees them.
- **The network moves the other seats inside the search.** Between the
  candidate move and the position that gets valued, the other three
  players have to act, and who they are is an assumption. The heuristic
  player is one answer and the wrong one against a network: in self-play
  the opponents are the network. So the engine hands every decision its
  imagined worlds are waiting on back to the caller, thousands at a time,
  and the policy answers them in one pass; the same handle lets the
  network play the searching player's own next turns before the position
  is valued, which is how a policy buys depth without a tree.
- **And the search is off, because it does not pay.** Every version of it
  measures worse than not searching, against a level of 2.50: 2.73 with
  heuristic rollouts, 2.52 with the critic on sampled worlds, 2.54 on
  weighed ones, 2.54 played by the club heuristic and 2.57 with the value
  head that predicts the return best. Those last two differ by 0.025 at
  0.3 standard errors, so the evaluator is not the variable.

  What settles it is that the heads separate candidate moves at four times
  the margin the rule demands. The search is not guessing, it is choosing
  confidently and wrongly, and the two-standard-error margin cannot catch
  that: it measures disagreement between imagined worlds, and every world
  agrees with the same systematic error. The likely cause is that the
  search asks what a position is worth after a move the policy would never
  make, while the value heads have only ever seen positions the policy did
  reach. If the line is revived, the thing to change is what those heads
  are trained on, not what the search evaluates.
- **The trained opponent in the browser**, as ONNX in a worker beside the
  rules in WebAssembly, so a whole game runs offline.

## Where the trained opponent stands

Stronger than the heuristic bot, and published.

Average placement over a few hundred games carries a standard error of
about 0.05, which is the size of the improvement being looked for. So the
arena plays the same deals four times with the network in each seat, and
takes its error bar from the deals rather than from the four seatings: the
seatings share their deals, and a network indistinguishable from the bots
would play the same four games and place summing to exactly ten every time.
Run against itself the arena therefore reports plus or minus nothing, which
is both correct and a check that the estimator understands the design.

The published network, over 10,000 games against three heuristic players:

| | |
|---|---|
| placement | 2.443 |
| error | 0.011 |
| difference from level | +0.057, or 5.2 standard errors |

A network no better than those bots averages 2.5. Reproduce it with
`python -m neural.arena <checkpoint> --games 2500`.

This figure says a network beats the heuristic players. It does not say
which of two networks is better, and it is much too coarse to: the
saturation is severe. Two checkpoints of the larger network being trained
score 2.4635 and 2.4288 here, a difference of 0.035, and when they play
each other the later one wins by 0.23 placement at thirteen standard
errors. Both beat the bots easily, so how much better one is than the
other barely shows.

Two networks are therefore ranked by sitting them at one table. One takes
a place and the other the remaining three, in the same games, four times
over with the challenger in each seat; two identical networks return
exactly 2.50 with no error, which is the check that the estimator is not
inventing precision. `python -m neural.duel challenger.pt incumbent.pt`.
Run it in both directions, because sitting alone against three of a kind
could in principle be a handicap of its own, and only the reverse duel
rules that out.

A larger network, 320 channels by 20 blocks, is being trained to replace
this one and has not earned the place yet. Against the bots the two are
level, 2.4288 against 2.4247. At one table the published network wins by
0.06, from both directions, so it keeps the browser. The larger one gains
about 0.23 every thirty-five generations by that measure, so the gap is
small and closing.

It reaches the browser as 2.4 MB of int8 weights in a worker beside the
rules in WebAssembly, so a whole game runs offline. That published network
is 192 channels by 10 blocks; the one training now is 320 by 20, 12.6M
parameters and about fifty megabytes, far too big for a phone and meant to
be distilled down once it is worth distilling. Quantising left the
best move unchanged on every position tested. It answers in 38 milliseconds
at the median and 41 at the ninetieth percentile, where the plan asks for
under 200.

Still to come: replays in the browser and a measured game against Mortal.
The plan is in [docs/PLAN.md](docs/PLAN.md).

## How the rules are checked

The scorer has more places to be quietly wrong than any other part of these
rules, and a test suite only checks what its author thought of. So every
hand is scored twice: once here, and once by the MIT-licensed `mahjong`
library, which was itself validated against millions of hands from Tenhou.

Over **one million random winning hands**, the two agree on han and
minipoints except in three places, and all three are the rules rather than
faults:

| hands | difference | why |
|---|---|---|
| 1,015 | a pair of both the seat and round wind | EMA 4.1.1, new in 2025: worth 2 minipoints, not the 4 the older convention gives |
| 26 | two yakuman in one hand | EMA 4.2: yakuman are not cumulative |
| 2 | a yakuman read as three identical sequences | EMA 3.4.3: score to the highest possibility, and with no counted yakuman the sequences cap at sanbaiman while the yakuman pays more |

**Unexplained disagreements: none.** Reproduce it with:

```bash
cargo run -p riichi-cli --release -- dump --games 1000000 --seed 20260903 > hands.jsonl
pip install mahjong
python engine/riichi-cli/differential.py hands.jsonl
```

The log is checked the same way. A test plays fifty whole games and rebuilds
every hand from its events alone, then compares the rebuilt hands, called
sets, scores and riichi against what the engine holds, so an event the log
forgets to write shows up as a hand that has drifted.

## Running it

```bash
./check.sh                      # everything the workflow checks, before pushing
cargo test --workspace          # the rules, with their tests
cargo run -p riichi-cli -- hand --seed 1        # one hand, move by move
cargo run -p riichi-cli --release -- arena --games 200   # bots, with statistics
cargo run -p riichi-cli --release -- fuzz --games 500    # random legal play
cargo run -p riichi-cli --release -- log --games 1     # a game as an mjai log
cargo run -p riichi-cli --release -- dump --games 1000 # scored hands, for the check above

cd web
npm install
npm run wasm                    # build the engine for the browser
npm run dev                     # play at the address printed
npm run check:all               # play it in a real browser and check what it did
```

`check:all` runs six checks against a running copy: a hand played to its
end, the keyboard and the tile names a screen reader reads, the learning
aids, the post-game review, the saved log, and a whole hanchan through to
the final standings. Each takes a real browser at real speed, because a
headless run on a virtual clock reports a loading network as a hang.

`npm run wasm` needs the WebAssembly target and `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
npm install -g wasm-pack
```

## Training

Build the engine for Python once, with `maturin develop --release` inside
`engine/riichi-py`, and Mortal's engine the same way inside
`engine/libriichi` (vendored from the Mortal project under its AGPL
licence; its encoder is what the network sees, see
`engine/libriichi/NOTICE.md`), then:

```bash
python -m neural.imitate --rounds 400 --out runs/clone
python -m neural.train --generations 4000 --resume runs/clone/latest.pt --out runs/play
python -m neural.export runs/play/best.pt web/public/model.onnx
```

The warm start teaches the network the heuristic player's moves, which saves
it rediscovering that tiles which go together should be kept; self-play then
improves on them. Progress is reported as average placement against three
heuristic opponents, where 2.5 is even and lower is better.

From September 2026 the network sees Mortal's observation, 1012 planes
built by Mortal's own engine from the game our engine plays, rather than
the engine's ninety-seven: waits, furiten, every discard in order and an
efficiency lookahead are in it, and a network that had to learn those from
the raw position plateaued 0.09 behind the published one. Each block of
the tower also pools the whole line, Mortal's channel attention. A
checkpoint records which planes it sees, and a table serves each network
its own, so the new lineage is measured against the old at one table
(`python -m neural.duel new.pt old.pt`). The browser keeps the older
network until Mortal's encoder runs there.

The game offers the **Trained** tier only when `web/public/model.onnx` is
present, so a checkout without one simply shows the two heuristic tiers.

`node scripts/play-check.mjs <url>` plays the game in a real browser and
reports the console, the moves and any failure. It is the only way to test
the parts that load a runtime, since a headless screenshot with a virtual
clock races ahead of the work and reports a hang that is not there.

## Layout

```
engine/riichi-core   the rules: tiles, shanten, scoring, a hand, a game, a bot
engine/riichi-wasm   WebAssembly bindings for the browser
engine/riichi-cli    arena and fuzzer
web/                 the browser game (Svelte, Vite)
docs/                the plan and the rulebooks the tests cite
```

## The rules, and how they are checked

Every rule a program can decide is implemented as written and tested against
the text, with the section number in the test. Seven of the rulebook's ten
scoring examples are tests checked against its printed payments; so are the
2025 changes, such as riichi needing only one tile left in the wall and four
han thirty minipoints paying a mangan.

Rules that only a referee can decide, such as call timing at a physical
table, dead hands, chombo and etiquette, are out of scope: the software never
lets a player make the corresponding mistake.

The fuzzer plays only actions the engine offered and checks after every one
that no hand holds a fifth copy of a tile, that the tiles in play stay within
a set, and that points are only moved, never made.

## Credits and licence

The code is AGPL-3.0-or-later. The tile drawings in `web/public/tiles` are by
[FluffyStuff](https://github.com/FluffyStuff/riichi-mahjong-tiles) and are in
the public domain (CC0). The rulebooks in `docs/rules` are published by the
[European Mahjong Association](https://www.mahjong-europe.org) under
CC BY-NC-SA 4.0.
