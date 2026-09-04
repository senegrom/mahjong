# Riichi Mahjong: project plan

A browser riichi mahjong game with a beautiful, usable table, faithful to the
European Mahjong Association rules, and AI opponents trained AlphaZero-style
from self-play on the same rules engine the humans play against.

Status: 3 September 2026. The engine, the browser game and the training
loop are built and published at <https://senegrom.github.io/mahjong/>; what
is left is the strength of the trained opponent.

| milestone | where it stands |
|---|---|
| M0, the rules engine | done. Every rules-card item has a citing test, and one million random winning hands were scored here and by an independent library with no unexplained disagreement. Logs are written in the mjai format and a test rebuilds every hand from its own events. |
| M1, the browser game | done but for replays. A whole game is playable, by mouse or keyboard, with the learning aids, a post-game review, and the hand saveable as an mjai log. Published from Actions. |
| M2, the training loop | done. The network is 0.057 placement ahead of the heuristic bot over ten thousand duplicate deals, which is 5.2 standard errors. |
| M3, neural tiers in the browser | done. The trained tier is published: 2.4 MB of int8 weights in a worker beside the rules, answering in 38 milliseconds at the median where the plan asks for under 200. The review and the learning aids are there; a replay is not. |
| M4, search and Mortal | search is built the AlphaZero way: worlds imagined from what a seat can see and drawn from what the network believes the opponents hold, each candidate made in each, the position that results valued by the network's value head in one batched pass, and a move taken only when it beats the player's own choice by two standard errors of the world-by-world difference. The earlier rollout evaluator measured worse than not searching as the world count rose (2.503 at ten, 2.581 at two hundred, overrides climbing from 2.7% to 3.7%), which is what a biased judge does when sharpened. The value head is being trained by self-play; the measurement follows. Mortal not started. |

The honest summary of the AI: a warm start on the heuristic player reaches
its level, and self-play has passed it. The gain is real but not large, and
the reason is the shape of the reward rather than a bug: a whole game's
result reaches every one of the hundreds of thousands of decisions in it,
so the signal per decision is thin. Four times the games per update is what
turned a flat run into a rising one.

The arena exists so that any claim about strength has to be shown rather
than asserted, and it has already cost one claim of mine. Its error bars
were taken from the four seatings, which share their deals and so are not
independent; correcting that turned four standard errors into one and eight
tenths, and the number that eventually justified publishing came from ten
thousand games rather than from a better story about two thousand.

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
  declare noten (the UI offers this).

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
  waiting hand to themselves at an exhaustive draw; the engine always shows
  it. Worth offering in the interface later, since it is a real decision.
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
  rather than a broken square. The dice are still rolled and logged so a
  replay can show the table, and under a shuffled wall the two are
  equivalent.

Everything else in those chapters is implemented, including the parts most
easily got wrong: the dead wall's composition, all three quads and the
replacement draw, liability for feeding a yakuman, furiten in each of its
three forms, the concealed quad a riichi player may declare, robbing a quad,
the noten penalty split, multiple winners, counters, dealer rotation, every
row of the minipoint table, the base-value cap that makes four han thirty a
mangan, and every yaku's han with its open-hand penalty.

### 1.3 Rules kept parameterised

The engine takes a `RuleSet` value so that WRC or Tenhou variants (red fives,
abortive draws, kazoe yakuman, different uma) can be added later without
touching game logic. Version 1 ships only `ema2025`; anything else is a
clearly labelled practice option, never the default.

## 2. Architecture: one rules engine everywhere

The single most important decision: **the rules exist once, in Rust, and are
compiled to WebAssembly for the browser and to a Python extension for
training.** The AI is trained on exactly the code the humans play against,
and a scoring bug fixed once is fixed everywhere.

```
riichi-core   (Rust crate: tiles, wall, state machine, shanten, agari, scoring)
   ├── riichi-wasm   (wasm-bindgen)  → web app engine + client-side validation
   ├── riichi-py     (PyO3/maturin)  → batched environment for self-play
   └── riichi-cli    (Rust)          → random games, log replay, fuzzing
web/          TypeScript + Vite + Svelte app, ONNX Runtime Web for the AI
neural/       PyTorch training, Modal Functions (H100), evaluation arena
docs/         rules PDFs, this plan, design notes, results
```

Rust 1.98 and Node 24 are already installed; `wasm-pack` and `maturin` are
the only additions.

## 3. Rules engine (`riichi-core`)

Design

- Tiles as 0..33 indices; hands as 34-count arrays; a game is an explicit
  state machine with phases (deal, draw, act, call window, kan replacement,
  win resolution, exhaustive draw, hand end, game end).
- Deterministic: a seeded RNG builds the wall, so any game replays exactly
  from its seed and action list. Every action is validated against the
  legal-action list, never trusted.
- Legal actions per player per phase: discard (with tsumogiri flag), riichi
  with discard, chii (which sequence), pon, three quad kinds, ron, tsumo,
  declare tenpai or noten, pass.
- Shanten and waits by the standard per-suit decomposition tables (one table
  for a 9-number suit, one for the 7 honours), which also give acceptance
  counts for hints and for the efficiency oracle. Winning-hand decomposition
  enumerates every reading so the scorer can take the maximum.
- Scoring returns a full breakdown (yaku list with han, fu items with reasons,
  limit name, payments per player) because the UI shows it.
- Logs in the mjai JSON event format, the de facto standard for riichi bots,
  so replays, external reviewers and third-party bots can read our games.

Testing (the engine is only as good as this)

- Every numbered rule in the card above becomes at least one named test that
  cites its section; the ten scoring examples of section 4.3 and the four
  invalid-quad examples of section 6.7.1 are literal tests.
- Differential scoring: one million random winning hands scored by the Rust
  engine and by the MIT-licensed `mahjong` Python library (validated against
  26 million Tenhou hands), with its optional rules set to EMA (no red fives,
  kiriage mangan, no counted yakuman, no double yakuman). Every disagreement
  is either a documented EMA-specific rule or a bug.
- Property tests: tile conservation across a whole game, phase invariants,
  score sums always zero apart from riichi sticks on the table, replay from
  seed reproduces the log.
- Fuzzing: the CLI plays millions of random-policy games under debug asserts.

## 4. Web application

Stack: TypeScript, Vite, Svelte 5, the WASM engine, a Web Worker running the
AI (ONNX Runtime Web on WebGPU, WASM fallback), deployed as a static site to
GitHub Pages, installable as a PWA and fully offline. No server in version 1.

Table and interaction

- A real table view: own hand at the bottom, opponents' discards, melds and
  riichi sticks in their seats, the wall count, dora indicators, counters,
  round and seat winds, scores, all readable at a glance on desktop and in
  portrait on a phone.
- Tiles are SVG from the CC0 `riichi-mahjong-tiles` set, with light and dark
  table themes, and animated draws, discards, calls and score payments
  (honouring reduced-motion settings).
- Discard by click or tap, keyboard for everything (number keys and arrows
  select a tile, letters for chii, pon, kan, riichi, ron, tsumo), call prompts
  with a configurable timer and clear defaults, auto-options like real
  clients (auto-pass calls, tsumogiri after riichi, auto-win, auto-sort).
- Score screens with the full breakdown: yaku, han, fu with reasons, limit
  name, who pays what, counters and riichi sticks, and the game-end sheet
  with uma. Every hand is stored and can be replayed step by step.
- Accessibility: ARIA roles and live announcements of every draw, discard
  and call; suits are distinguishable by shape, not colour alone; strong
  focus states; screen-reader tile names; scalable layout.
- English UI with Japanese yaku names alongside, following the rulebook's
  own naming.

Learning aids (toggleable, never on by default in a rated game)

- Shanten count and acceptance list for the current hand, waits and their
  remaining copies, furiten warning, dora highlight.
- Defence view: safe tiles against each riichi (genbutsu, suji) and the AI's
  own danger estimates.
- Post-game review: the AI's preferred move at each of your decisions, with
  its win, deal-in and value estimates, and a "why" line (efficiency, value,
  or defence).

## 5. AI opponents

### 5.1 What "AlphaZero-style" means here

AlphaZero proper needs perfect information and two players. Riichi has four
players, hidden tiles and random draws, so the recipe is adapted while
keeping its spirit: **no human data, self-play only, a policy-value network
improved by search-quality targets, and evaluation by arena.** The concrete
choices:

- Actor-critic self-play (PPO or V-trace) over batched games, all four seats
  played by the current network, replay window and paced learner exactly as
  in the connect4 loop.
- Oracle-guided critic: during training the value head may see the hidden
  tiles (opponents' hands, the wall) as extra features that are annealed away,
  a proven variance reducer for mahjong; the policy never sees them.
- Reward: final game result including uma, plus per-hand score changes as an
  annealed shaping term. The value head sees round, scores, counters and
  riichi sticks so it can trade a hand's value against placement.
- Search-improved targets as the second stage: at the actor, sample hidden
  tiles consistent with public information, evaluate each legal action by
  short rollouts or one-ply expectation under the sampled worlds, and train
  the policy toward the improved distribution, Gumbel-style, as the connect4
  actors do with their two-ply scores.
- Exact oracles for the parts that are exactly solvable, the analogue of the
  connect4 perfect tables: the single-hand tile-efficiency problem (which
  discard maximises the chance to reach tenpai or win within the remaining
  draws) has an exact dynamic-programming solution over the 34-count vector
  and is used both as an auxiliary training target and as a held-out
  blunder test on efficiency-only positions.

### 5.2 Network

- Input: per-player observation as channels over the 34 tile kinds (own
  hand counts, melds, each player's discards with order, tsumogiri and
  riichi-tile flags, dora indicators, visible-tile counts, riichi states,
  furiten, waits of the own hand) plus a scalar block (scores, seat, round,
  counters, sticks, tiles left).
- Body: 1-D residual network over the 34 positions, 10 to 20 million
  parameters at first, scaled once the loop is stable.
- Heads: policy over about 46 masked actions (34 discards plus tsumogiri,
  riichi, three chii shapes, pon, three quad kinds, ron, tsumo, pass), value,
  and auxiliary heads that predict each opponent's tenpai state and waits
  (which is what defence needs) and the own hand's shanten.
- Export to ONNX for the browser; the same weights run in the app and on
  Modal.

### 5.3 Difficulty tiers in the app

| tier | engine |
|---|---|
| Beginner | efficiency-only heuristic, no defence, noisy discards |
| Club | heuristic bot: shanten and acceptance efficiency, value awareness, riichi decision, defence by genbutsu and suji |
| Strong | neural policy, greedy |
| Expert | neural policy with sampled-world search at each decision |

The heuristic bot ships first and stays as a fixed benchmark and as a
sparring partner that keeps early self-play from collapsing into nonsense.

### 5.4 Evaluation

- Arena with duplicate deals: the same seeded walls are played with the
  candidate in every seat and the opponents permuted, which removes most
  of the luck. Metrics per candidate: average placement, win rate, deal-in
  rate, average win value, riichi rate, and Elo against previous
  generations and the heuristic bot.
- External benchmark: Mortal, the open-source riichi AI, speaks mjai, so it
  can sit at our tables locally. It was trained on Tenhou rules, so results
  under EMA rules are indicative rather than exact, but beating it is the
  first serious milestone.
- Efficiency blunder rate against the exact oracle on held-out positions.
- Human play: you, with the review tool showing where the AI disagrees.

### 5.5 Compute plan

Reuse the connect4 infrastructure as is: Modal Functions only, never
sandboxes; H100 actors that write compressed shards to the volume; an H100
learner that trains one generation per call from the newest checkpoint over
a replay window; the local driver with pacing, the single-driver guard,
stop files, and the generation watcher. The Rust environment runs on the
actor's CPU cores with network evaluation batched on its GPU, which is the
standard shape for mahjong self-play; a fully tensorised GPU environment is
a stretch goal if profiling shows the CPU side is the bottleneck. The local
RTX 5070 Ti is for development, smoke tests and the browser export.

## 6. Milestones

| # | deliverable | acceptance |
|---|---|---|
| M0 | `riichi-core` complete with tests, differential scoring, CLI random games, mjai logs | every rules-card item has a citing test; zero unexplained scoring disagreements over one million hands |
| M1 | playable web app vs the heuristic bot, tiles, mobile layout, score screens, replays, Pages deploy | a full hanchan playable on a phone; accessibility audit passes |
| M2 | training loop live: PyO3 env, network, learner and actors on Modal, arena | a neural generation beats the heuristic bot in duplicate-deal arena |
| M3 | neural tiers in the browser, review tool, learning aids | Strong tier runs in the browser under 200 ms per decision on a laptop |
| M4 | search targets, scaling, Mortal benchmark, optional online play | measured win over Mortal in arena; decision on multiplayer |

Milestones are sequential in their acceptance but overlap in work: the app
(M1) and the loop (M2) both start as soon as the engine's API is stable.

## 7. Risks and open questions

- Rules interpretation: a handful of EMA wordings need a decision for
  software (for example which reading counts as "delayed" in the call
  window); each such decision is written down in `docs/RULES_DECISIONS.md`
  with the section it interprets.
- Self-play from zero may plateau at strong-club rather than expert level;
  the mitigations are the oracle critic, the exact efficiency targets, the
  heuristic sparring partner, and, if a properly licensed set of human game
  logs is available, a supervised warm start (not planned by default).
- Browser inference cost for the Expert tier; the Strong tier is the
  guaranteed fallback.
- Choices to confirm: the app's name and visual theme, and whether online
  multiplayer is wanted at all.

Decided: **Svelte for the interface** (2026-09-02). A mahjong table keeps far
more state than a Connect Four grid, and Svelte compiles components away, so
the shipped page stays small next to the tile art, the WebAssembly engine and
the network. The cost is a build step, which the connect4 project deliberately
avoided.

## 8. Immediate next steps

1. Repository skeleton: Rust workspace, web app scaffold, `docs/rules/` with
   both PDFs, CI that builds the engine, runs the tests and deploys Pages.
2. `riichi-core` tiles, wall, deal and the turn state machine with the
   rules-card tests, then shanten and agari tables, then scoring with the
   differential harness.
3. The heuristic bot and the CLI arena, so the app has an opponent on day
   one of M1.

Sources: the EMA rules page and 2025 PDF at mahjong-europe.org, the
riichi.wiki summary of the EMA rules, the `riichi-mahjong-tiles` repository
(CC0), the `mahjong` Python library (MIT), and the Mortal project (mjai
format).
