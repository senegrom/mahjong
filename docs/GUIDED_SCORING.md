# Guided physical hand settlement

Guided physical games score wins and exhaustive draws using the existing Rust
rules engine. The adapter calls the same win and draw resolvers as normal play;
it does not simulate unknown hands or draw placeholder wall tiles.

## At the table

Record your ron/tsumo choice, or choose the actual winner under **Someone won /
hand ended**. A ron must refer to the pending discard or robbable kan. A tsumo
must refer to the current player's draw. Exhaustive draws retain the last
actual discard and happen only after its response window closes.

The result form uses your recorded hand. For an unknown winning opponent,
enter their revealed **standing concealed tiles**, excluding the winning tile
and called sets. An unknown opponent's tsumo additionally requires their
actual winning draw. All simultaneous ron winners must be selected. Confirm
that opponents have not incurred temporary or riichi furiten by passing a win;
that history cannot be recovered from previously hidden hands. The engine
still checks the winning shape, yaku and furiten from the recorded discards.

For any riichi winner, enter every actual ura indicator, one per dora indicator,
even when the bonus is zero. Indicators and all revealed hands share the physical
four-copy budget. The common ron tile is counted once, not once per winner.
Dora and ura are displayed separately; neither can supply the required yaku.
The result retains actual indicators even when hints are off.

For an exhaustive draw, select each tenpai declaration and reveal any unknown
hand being declared tenpai. Riichi players must show a valid waiting hand.
Non-riichi players may declare noten. The Rust engine applies the noten split.

**Calculate hand settlement** previews the yaku, han, fu calculation, limit and
four-seat ledger without changing any balance. **Apply settlement** recomputes
and applies it once. The result determines dealer continuation, next honba and
remaining riichi sticks. You cannot advance an unpaid scored hand or override
its dealer continuation.

## Accounting and persistence

The settlement column includes win/draw transfers, honba, awarded riichi sticks,
and any refund for ron on a riichi declaration discard. The whole-hand column
also includes the riichi bets deducted earlier in the hand. This distinction
prevents an own riichi bet from appearing as an extra 1,000-point profit.

Multi-ron and responsibility payments use the existing EMA resolver. Each
winning riichi player's own bet is returned, then the closest winner collects
the remaining pool. The pool is awarded once. A draw leaves the pot on the table.

A saved result contains the pre-settlement position, the entered physical facts
and the calculated ledger. Reload verifies it with Rust without paying again.
Undo restores the pre-settlement balances and pot; another undo restores the
winning decision. Direct score edits cannot silently alter a pending or applied
scored result. Use Undo to correct the recorded table first.

Older finished saves remain explicitly manual and are never retrospectively
paid. **Other hand end** remains an explicitly labelled manual path for referee
adjudications, penalties and results outside normal win/exhaustive settlement.
It does not pretend to compute yaku or transfers.

## Regression checks

After installing the repository toolchain and dependencies:

```sh
cargo fmt --all --check
cargo test --locked --workspace
cd web
npm run wasm
npm run lint
npm run check
npm run build
npm run test:unit
node scripts/guided-game-check.mjs
```

`guided-settlement-state.test.js` isolates the bookkeeping boundary with a mock
scorer. `guided-settlement.test.js` uses real WASM for ron, dealer/nondealer
tsumo, fu exceptions, dora/ura, no-yaku and furiten rejection, declaration-bet
refunds, multi-ron, first-turn yaku, chankan, rinshan, responsibility payments,
and every exhaustive-draw split. The production guided browser checks cover
preview/apply, reload, undo, next hand and narrow-screen ura display with hints
off. Chromium checks are not physical-iPhone or Safari certification.
