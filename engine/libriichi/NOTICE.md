# libriichi, vendored from Mortal

This directory is the `libriichi` crate from Mortal, by Equim, copied from
<https://github.com/Equim-chan/Mortal> at commit `0cff2b5` on 7 September
2026, under its AGPL-3.0-or-later licence (see LICENSE). This project is
AGPL-3.0-or-later as well.

This project uses it for its encoder alone, which gives the network its
observation, Mortal's own thousand and twelve planes, from the mjai events
our rules engine writes as it plays. Two parts read it:

- the trainer in `neural/`, through the `Follower` below, built as the
  `libriichi` Python module. A published Mortal plays at our tables on
  those same planes (`neural/zoo.py`);
- the site's engine, `engine/riichi-wasm`, which builds the crate without
  its default features, as a plain Rust library, for the browser.

Our engine stays the authority on the rules and on which moves are legal;
this crate reads the game and describes it.

Local changes, all under the same licence:

- `src/follow.rs`, new: a `Follower` that holds four player states for each
  of many games, is fed each game's events and encodes the deciding
  players' observations in parallel without the GIL, sparse. Its `tell`
  gives one player an event ahead of the table (its own reach, so a Mortal
  choosing riichi can be asked which tile follows), and that player skips
  the table's copy. `encode` takes `after_reach`, which previews the
  player's own reach on a copy, and `half`, which hands the values back as
  float16. Registered in `src/lib.rs` as the `follow` submodule.
- `src/lib.rs`: the submodules are registered under the short module name,
  because maturin installs the native module inside a package.
- `Cargo.toml`: its own workspace and release profile; the benches are not
  vendored, so the bench target is gone. `pyproject.toml`, new: how maturin
  builds it, with the default features off (no mimalloc) and `pymod` on.
- The `pymod` feature split, so that without Python the crate builds for
  wasm32-unknown-unknown as the state machine, the mahjong algorithms and
  the observation encoder. `Cargo.toml` makes every Python- or native-only
  dependency optional under `pymod`, keeps ahash off `runtime-rng` without
  it, and gives the bins `required-features`. `build.rs` asks for PyO3's
  link arguments only with it. `src/lib.rs` gates the Python-only modules
  and the module function on it, and makes `consts` public. In
  `src/consts.rs`, `src/mjai/mod.rs` and
  `src/state/{action,getter,mod,obs_repr,player_state}.rs` the PyO3
  attributes, imports and module registration are gated, and the Python
  getters and methods move into `#[pymethods]` blocks of their own that
  forward to the plain Rust ones; `player_state.rs` also gates the two
  mjai JSON entry points that only Python and the tests call, and
  `src/rankings.rs` lets a plain build leave `player_by_rank` unread, as
  only the Python half reads it.
- The port to PyO3 0.29, with the renames PyO3 0.26 made: `PyObject` is
  now `Py<PyAny>`, `Python::with_gil` is `Python::attach` and
  `allow_threads` is `detach`, in `src/agent/{mjai_log,mortal,py_agent}.rs`,
  `src/arena/{one_vs_three,two_vs_two}.rs` and `src/mjai/bot.rs`, and
  `src/lib.rs` declares the module `gil_used = true`. Every `#[pyclass]`
  that derives `Clone`, in `src/arena/{one_vs_three,two_vs_two}.rs`,
  `src/dataset/{gameplay,grp}.rs`, `src/stat.rs` and
  `src/state/{action,player_state}.rs`, asks for `from_py_object`, so it
  stays extractable by value now that PyO3 0.28 has deprecated deriving
  that unasked.
- `src/dataset/grp.rs` no longer has `Grp::len` and `is_empty`, which
  nothing in the crate called and the Python class never exposed.
- Newer dependencies: `Cargo.toml` takes PyO3 and numpy 0.29 and pyo3-log
  0.13 for that port, then ndarray 0.17, rand and rand_chacha 0.10, sha3
  0.12 and indicatif 0.18, and adds the `half` crate, with numpy's `half`
  feature, for the follower's float16. `Cargo.lock` follows, and
  Dependabot keeps it current: it watches this crate's lockfile beside the
  root one.
- `src/state/golden.rs`, new, with `testdata/encoder-golden`: the encoder's
  golden test. Whole games recorded from this project's arena are replayed
  through the player states, and at every decision the planes, the mask and
  the lookahead's table, and the same with the player's reach previewed
  where our rules allow one, are held to digests the encoder gave before;
  so are a few hands put to the lookahead directly, with its other options.
  `neural/tests/test_encoder_golden.py` holds the `Follower`'s output for
  the same games to digests of its own. A change meant to be exact has to
  reproduce them to the bit. `src/state/mod.rs` registers the test, and
  `Cargo.toml` optimises the test profile, in which the games replay in
  seconds where an unoptimised build took minutes.
- The efficiency lookahead, about twice as fast with every value the same
  to the bit. `src/algo/shanten.rs` gains `Near`, the shanten of every hand
  one tile away from a base hand: it looks the base's four groups up once
  and combines each group's three others once, so that each neighbour costs
  one lookup and one entry of a product where `calc_all` looked up and
  combined all four groups again, and its comment says why each answer is
  exactly `calc_all`'s; `near_is_calc_all` compares the two on random hands
  (`NEAR_HANDS` sets how many). `src/algo/sp/state.rs` asks it in the scans
  over the tiles that could be thrown or drawn, which made some eleven
  thousand shanten calculations an observation; with
  `sp_reproduce_cpp_ver` they still ask `calc_normal` afresh.
  `draw_without_tegawari_slow` in `src/algo/sp/calc.rs` sums its draw
  probabilities with the turns swapped, each sum still taking the same
  terms in the same order.
