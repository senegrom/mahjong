# libriichi, vendored from Mortal

This directory is the `libriichi` crate from Mortal, by Equim, copied from
<https://github.com/Equim-chan/Mortal> at commit `0cff2b5` on 7 September
2026, under its AGPL-3.0-or-later licence (see LICENSE). This project is
AGPL-3.0-or-later as well.

It is built as its own Python module, `libriichi`, and used two ways: its
encoder gives the network its observation, Mortal's own thousand and twelve
planes, from the mjai events our rules engine writes as it plays; and its
mjai interface lets Mortal itself sit at our tables. Our engine stays the
authority on the rules and on which moves are legal; this crate reads the
game and describes it.

Local changes, all under the same licence:

- `src/follow.rs`, new: a `Follower` that holds four player states for each
  of many games, is fed each game's events and encodes the deciding
  players' observations in parallel without the GIL, sparse. It can also
  tell one player an event ahead of the table (its own reach, so a Mortal
  choosing riichi can be asked which tile follows), skipping the table's
  copy. Registered in `src/lib.rs` as the `follow` submodule.
- `src/lib.rs`: the submodules are registered under the short module name,
  because maturin installs the native module inside a package.
- `Cargo.toml`: its own workspace and release profile; the benches are not
  vendored, so the bench target is gone. `pyproject.toml`, new: how maturin
  builds it, with the default features off (no mimalloc).
