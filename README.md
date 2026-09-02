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
  called sets, keyboard play and optional hints.
- **A command-line arena and fuzzer** for checking the engine at scale.

Still to come: the log format, the neural opponents, the review tool. The
plan is in [docs/PLAN.md](docs/PLAN.md).

## Running it

```bash
cargo test --workspace          # the rules, with their tests
cargo run -p riichi-cli -- hand --seed 1        # one hand, move by move
cargo run -p riichi-cli --release -- arena --games 200   # bots, with statistics
cargo run -p riichi-cli --release -- fuzz --games 500    # random legal play

cd web
npm install
npm run wasm                    # build the engine for the browser
npm run dev                     # play at the address printed
```

`npm run wasm` needs the WebAssembly target and `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
npm install -g wasm-pack
```

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
