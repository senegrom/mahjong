# Rules and engine reference

The rules this game implements, and how the rules engine that implements them
is built and tested. This file began as the project plan; its status
narrative and its planning of the web app, the AI opponents and the
milestones are kept in [historical evidence](HISTORY.md).

## 1. Authoritative rules

The rules are the **EMA Riichi Competition Rules, 2025 edition (version 1.1,
August 2025)**, in force since 1 January 2026. They supersede the 2016
edition: <http://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf>.
Both PDFs are kept under `docs/rules/` as the reference the tests cite.

Every rule that a program can decide is implemented exactly as written and
tested against the text; every rule that only a referee can decide (call
timing at a physical table, dead hands, chombo, etiquette, tournament
sessions) is out of scope because the software never lets a player make the
corresponding mistake.

### 1.1 Rules card (what the engine enforces)

Setup and flow

- 136 tiles, four of each of 34 kinds. No red fives, no flowers, no jokers.
- Four players, 30,000 points each. A game is a hanchan: East round then South
  round, each at least four hands. No extension round, no agari-yame, no
  bankruptcy (scores go negative and play continues).
- Seat and round winds rotate exactly as in sections 2.1 to 2.2 and 3.4.5:
  the dealer keeps the deal on a win or on tenpai at an exhaustive draw.
- Dead wall of 14 tiles: four replacement tiles, then dora, kan dora and ura
  dora indicators. After each quad the last live tile joins the dead wall.
  Dora chains: 9 to 1, red to white to green to red, east to south to west to
  north to east.
- Deal: East starts with 14 tiles and does not draw on the first turn.
- Counters: plus one after a dealer win and after every exhaustive draw;
  cleared when a non-dealer wins and the dealer does not. Each counter adds
  300 to a win by discard or 100 from each opponent on self-draw, paid to
  every winner when several win.

Calls and turns

- Chii only from the player on the left; pon and kan from anyone.
- Priority: a win beats any set call; pon or kan beats chii. The software
  gives every player a decision window on each discard and resolves by that
  priority, which is the only faithful rendering of the timing rules.
- Several players may win on the same discard. The discarder pays each
  winner in full, including counters. Riichi bets: winners who declared
  riichi get their own bet back; all other bets on the table go to the winner
  first in turn order after the discarder.
- Swap-calling is illegal: after a chii or pon the engine never offers the
  claimed tile, nor the other end of a claimed sequence, as a discard.
- Quads: claimed, extended and concealed; the dealer of the dead wall reveals
  a kan dora before the replacement draw. At most four quads per hand. No quad
  after drawing the last live tile; a quad with one tile left leaves only a
  replacement draw. A concealed quad can be robbed only for Thirteen Orphans.
- The last live discard can be claimed only for a win.

Riichi and furiten

- Riichi needs a concealed tenpai hand and at least one tile left in the wall
  (2025 change). Bet 1,000. A furiten player may declare riichi.
- After riichi: the hand cannot change except a concealed quad on the drawn
  tile that keeps the waits identical and whose three tiles can only be read
  as a triplet in every completed hand (section 6.7.1 examples become tests).
- Ippatsu is lost when any set is claimed or any quad is declared, concealed
  ones included. Double riichi is 2 han, replaces riichi, combines with
  ippatsu.
- Furiten: permanent while any wait is among the player's own discards
  (including discards others claimed; a tile used to extend a triplet does
  not count); temporary after passing a winning discard or a robbable quad
  until the next draw or claim; permanent for the rest of the hand after
  riichi. Furiten never blocks tsumo.
- Tenpai: a hand waiting only on a fifth copy is noten; a hand whose waits
  are all visible elsewhere is still tenpai. Noten penalty 3,000 in total,
  split as in section 3.4.2. Riichi players must reveal; a tenpai player may
  declare noten. The guided physical-scoring UI offers this; the engine's own
  exhaustive draw does not (see 1.2).

Scoring

- Han from yaku plus dora, kan dora and, for riichi hands only, ura dora.
- Fu: 25 fixed for Seven Pairs; otherwise 20, plus 10 for a win by discard
  with a concealed hand, plus 2 for self-draw except with pinfu, plus set
  values (2/4/8/16 melded, doubled concealed, doubled again for terminals and
  honours), plus 2 for a dragon or seat or round wind pair (still 2 when the
  pair is both seat and round wind, a 2025 change), plus 2 for an edge,
  closed or pair wait, plus 2 for open pinfu. Round up to 10.
- Base value fu × 2^(han + 2), capped at 2,000 (so 4 han 30 fu and 3 han 60
  fu are mangan, a 2025 change). Limits: mangan 5 han, haneman 6 to 7, baiman
  8 to 10, sanbaiman 11 or more, yakuman. Yakuman are not cumulative and
  there is no counted yakuman. Payments round up to 100; the dealer receives
  and pays double shares.
- The highest-scoring reading of a winning hand is chosen automatically.
- Liability: the player who fed the third dragon set or fourth wind set pays
  the whole yakuman on self-draw and half on another player's discard;
  counters are paid by the discarder only.
- Yaku, 2025 classification (closed value, minus one han when open where
  marked): riichi, ippatsu, fully concealed self-draw, pinfu, pure double
  sequence (closed), all simples (open allowed), each dragon or seat or round
  wind triplet, after a quad, robbing a quad, under the sea, under the river
  (1 han); double riichi, seven pairs (closed), mixed triple sequence*, pure
  straight*, half outside hand* (must contain honours), triple triplet, three
  concealed triplets, three quads, all triplets, little three dragons, all
  terminals and honours (2 han); twice pure double sequence (closed), half
  flush*, full outside hand* (3 han); blessing of man (5 han, combines with
  nothing); full flush* (6 han); the thirteen yakuman of section 4.2.6 with
  their conditions (four concealed triplets by discard only on a pair wait,
  no concealed quad in nine gates, heaven or earth). Asterisks lose one han
  open.
- End of game: subtract 30,000, add uma 15,000 / 5,000 / −5,000 / −15,000
  with ties splitting the pooled places; leftover riichi bets go to the
  winner, split on a tie with decimals rounded down.

### 1.2 What the engine deliberately leaves out

Audited against chapters 1 to 4 of the rulebook in September 2026. Every
rule software can decide is implemented and tested, with these exceptions,
each of which is a referee's judgement rather than a decidable rule:

- **Dead hands (3.3.14) and chombo (3.4.6).** Both are penalties for
  mistakes the software does not let a player make: it never deals the wrong
  number of tiles, never offers an illegal call, and never lets a hand be
  declared that is not a win. There is therefore nothing to declare dead and
  nothing to re-deal.
- **Declaring noten while waiting (3.4.2).** A player at a table may keep a
  waiting hand to themselves at an exhaustive draw; the engine's own
  exhaustive draw never allows it and counts every waiting hand as tenpai.
  Only the guided physical-scoring UI, which records a real table, lets a
  player without riichi declare noten.
- **Call timing (3.3.1).** A physical table resolves claims by who spoke
  first; software cannot reproduce that and does not try. Every player gets
  the same window on a discard and claims are settled by the rulebook's
  priority, which is what "if it's unclear whether calls are simultaneous or
  not, consider they are" amounts to.
- **Riichi needs 1,000 points in hand.** The rules let a player borrow
  sticks and keep playing below zero (4.1.4, 5.6). The engine requires the
  bet up front, which is the common house reading and simpler to show.
- **The deal and the wall are abstracted.** Tiles are dealt thirteen at a
  time rather than in blocks of four, and the wall is a shuffled sequence
  rather than a broken square. Under a shuffled wall, where the wall is
  broken changes nothing, so the break is not modelled and no log records
  a dice roll.

Everything else in those chapters is implemented, including the parts most
easily got wrong: the dead wall's composition, all three quads and the
replacement draw, liability for feeding a yakuman, furiten in each of its
three forms, the concealed quad a riichi player may declare, robbing a quad,
the noten penalty split, multiple winners, counters, dealer rotation, every
row of the minipoint table, the base-value cap that makes four han thirty a
mangan, and every yaku's han with its open-hand penalty.

### 1.3 Rule variants

The engine implements EMA 2025 only and takes no rule-set parameter. WRC or
Tenhou variants (red fives, abortive draws, kazoe yakuman, different uma)
would have to be added to the game logic itself. Anything other than EMA 2025
would be a clearly labelled practice option, never the default.

## 2. Rules engine (`riichi-core`)

The rules exist once, in Rust, and are compiled to WebAssembly for the browser
(`engine/riichi-wasm`) and to a Python extension for training
(`engine/riichi-py`), so the AI trains on exactly the code people play
against, and a scoring bug fixed once is fixed everywhere. `engine/riichi-cli`
plays random games and arenas and writes logs. `engine/libriichi`, Mortal's
engine vendored under its AGPL licence, only encodes positions into the
network's observation planes; the rules and the legal moves stay ours.

Design

- Tiles as 0..33 indices; hands as 34-count arrays. A hand is an explicit
  state machine whose phases are draw, act, call window and over
  (`game.rs`); `table.rs` carries the game across hands: rounds, the deal
  moving on, and uma.
- Deterministic: a seeded RNG builds the wall, so any game replays exactly
  from its seed and action list. Every action is validated against the
  legal-action list, never trusted.
- Legal actions per player per phase: discard (with tsumogiri flag), riichi
  with discard, chii (which sequence), pon, three quad kinds, ron, tsumo,
  pass.
- Shanten is exact: each suit's readings are found by backtracking and
  cached, and a small dynamic program combines the four, for the ordinary
  shape, Seven Pairs and Thirteen Orphans; the same module gives waits and
  acceptance counts for the hints. Winning-hand decomposition enumerates
  every reading so the scorer can take the maximum.
- Scoring returns a full breakdown (yaku list with han, fu items with reasons,
  limit name, payments per player) because the UI shows it.
- Logs in the mjai JSON event format, the de facto standard for riichi bots,
  so replays, external reviewers and third-party bots can read our games.

Testing (the engine is only as good as this)

- Rules are tested by name, citing their sections. Scoring examples 1 to 5,
  7 and 10 of section 4.3 are literal tests in `score.rs` and example 8 in
  `agari.rs`; examples 6 and 9 have no test yet. The four invalid-quad
  hands of section 6.7.1 are tests in `tests/riichi_quads.rs`, and
  `tests/every_yaku.rs` builds a hand for every yaku the rulebook lists.
- Differential scoring: `engine/riichi-cli/differential.py` scores the hands
  `riichi-cli dump` writes again with the MIT-licensed `mahjong` Python
  library (validated against millions of Tenhou hands), its optional rules
  set to EMA (no red fives, kiriage mangan, no counted yakuman, no double
  yakuman). Every disagreement is either a documented EMA-specific rule or a
  bug; one million random winning hands left none unexplained.
- `tests/mjai_replay.rs` rebuilds every hand from its logged events alone and
  compares the result with what the engine holds.
- Fuzzing: `riichi-cli fuzz` plays games of random legal choices and checks
  after every step that no tile is in play five times and that points are
  only ever moved; CI plays 500 such games on every run.

Sources: the EMA rules page and 2025 PDF at mahjong-europe.org, the
riichi.wiki summary of the EMA rules, the `mahjong` Python library (MIT), and
the Mortal project (mjai format).
