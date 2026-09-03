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

It reaches the browser as 2.4 MB of int8 weights in a worker beside the
rules in WebAssembly, so a whole game runs offline. Quantising left the
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
`engine/riichi-py`, then:

```bash
python -m neural.imitate --rounds 400 --out runs/clone
python -m neural.train --generations 4000 --resume runs/clone/latest.pt --out runs/play
python -m neural.export runs/play/best.pt web/public/model.onnx
```

The warm start teaches the network the heuristic player's moves, which saves
it rediscovering that tiles which go together should be kept; self-play then
improves on them. Progress is reported as average placement against three
heuristic opponents, where 2.5 is even and lower is better.

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
