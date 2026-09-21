#!/usr/bin/env bash
# Rules-engine checks; --web also runs the same complete web verification as CI.
# Dependencies must already be installed. Browser checks use CHROME_BIN when set.
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  ""|--web) ;;
  *) echo "Usage: $0 [--web]" >&2; exit 2 ;;
esac

echo "== formatting"
cargo fmt --all --check
echo "== lints"
cargo clippy --locked --workspace --all-targets -- -D warnings
echo "== tests"
cargo test --locked --workspace
echo "== Python binding"
cargo build --locked -p riichi-py
python3 engine/riichi-py/smoke-test.py
echo "== arena and randomized legal play"
cargo build --locked --release -p riichi-cli
./target/release/riichi-cli arena --games 20 --seed 1
./target/release/riichi-cli fuzz --games "${FUZZ_GAMES:-500}" --seed "${FUZZ_SEED:-1}"

if [ "${1:-}" = "--web" ]; then
  (cd web && npm run verify)
fi

echo "all requested checks passed (Rust dependency audit remains a separate CI step)"
