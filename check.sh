#!/usr/bin/env bash
# Everything the workflow checks, so a push does not have to find out.
#
#   ./check.sh          the rules engine
#   ./check.sh --web    the browser build as well, which needs npm
set -euo pipefail
cd "$(dirname "$0")"

echo "== formatting"
cargo fmt --all --check

echo "== lints"
cargo clippy --workspace --all-targets -- -D warnings

echo "== tests"
cargo test --workspace

echo "== a few games, to be sure the engine still plays"
cargo build --release -p riichi-cli
./target/release/riichi-cli arena --games 20 --seed 1 >/dev/null
./target/release/riichi-cli fuzz --games 50 --seed 1

if [ "${1:-}" = "--web" ]; then
  echo "== the browser build"
  cd web
  npm run wasm
  npm run build
fi

echo "all clear"
