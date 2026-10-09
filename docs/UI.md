# Gameplay interface

The table UI keeps the rules engine and match state separate from presentation. Desktop uses a larger table surface with clearer opponent zones; compact phone layouts keep the round, opponent type and status visible while moving secondary controls into the settings sheet.

When **Hints and markings** is enabled, every tile in the human hand has a small number directly below the face showing how many copies of that tile are still unseen from public information. The count includes the player's own concealed copies and subtracts visible unclaimed discards, called sets and exposed dora indicators exactly once. Zero is highlighted as exhausted and one as scarce. Turning Hints off removes the counts together with the other learning aids.

Calls are presented as a staged discard followed by the legal response choices. On phones, hand results appear in a large bottom sheet with the result value first, then the winning hand, yaku and score changes; the table can be revealed again without losing the result. Reduced-motion preferences suppress nonessential transitions.

The 7-by-2 phone hand layout is intentional: it preserves large, reliable touch targets instead of squeezing fourteen tiles into one row. The newly drawn tile remains slightly separated, including its hint count, and carries no border merely for being newly drawn.

## Tile faces

Classic and Matisse have a picture for every tile. Dalí and Van Gogh are still
being painted: a tile the chosen set has no approved artwork for shows its name
in text on the ivory face instead, so it plainly waits for its picture and never
borrows another set's. In the hand the number stands over its suit, or an
honour's first word over its second, in the words the tile is announced with.
On the smaller faces of discard rows the first line stands alone and larger,
and the colour of a number tells its suit: red characters, blue circles, green
bamboo. Where even an honour's word would be too small to read, as in agent
watch or on a phone, it is given in letters: E, S, W and N for the winds, and
Wh, G and R for the dragons. The name stays upright on a tile turned for riichi
and carries the dora ring and shine and both discard marks; a white dragon
written out has no dragon under its shine. Nothing is downloaded or saved
offline for these tiles, and each becomes its picture as soon as its artwork is
approved.

## Discard previews and readiness markings

With **Hints and markings** enabled, each legal discard is analysed using the
engine's read-only `discard_hint` query. The selected tile's wait panel and all
readiness borders use the same cached, per-decision results. Selecting another
tile changes the hypothetical discard; no move is played or saved. A selection
leaving no waits clears the previous selection's waits rather than showing
waits for a different hand. Touch/mouse previews use **Select before discarding**;
the keyboard marker also previews the focused discard.

- **Silver:** discarding that tile leaves one shanten, one step from a ready hand.
- **Gold:** discarding that tile leaves tenpai, a ready hand (zero shanten).

These describe the resulting shape, not a promise of a yaku or permission to
declare riichi. Open hands also receive readiness markings. Only legal discards
can have these borders: pending calls, disabled tiles, hidden faces, melds and
completed-hand displays do not acquire them. The existing dora/ura-dora foil and
red border, safety and selection cues are retained; overlapping marks use the
existing striped ring so no cue hides another.

A newly drawn tile keeps its accessible “just drawn” label and is separated by
a gap on desktop, phones and landscape layouts, but has no border merely for
being drawn. It can still receive a readiness, dora, safety or selection border
when that condition independently applies. Mobile grids reserve room for the
gap while keeping the drawn tile the same size as the rest of the hand.

## Discard rows

A riichi declaration lies sideways in its row. Two more marks are of different
kinds, a shade and a fade, so a tile can carry both:

- **A darker face:** the tile was thrown straight from the draw (tsumogiri), as
  Tenhou and Mahjong Soul show it. A tile thrown from the hand looks normal.
- **See-through:** another player claimed the tile for a call. The whole tile,
  dora ring included, lets the table show through, so on the felt it reads
  green while staying legible.

The marks appear on the table, in the inspection of all discards, in agent
watch and in the guided game's remembered table; the physical editor's tile
beside each discard shows the boxes ticked for it. They record what happened
at the table rather than give advice, so they stay when Hints and markings is
switched off. Each tile's name, read by screen readers and shown on hover, says
the same: claimed, riichi declaration, discarded from the draw. The shade and
the fade are the `--from-draw-shade` and `--claimed-opacity` tokens in
`app.css`, and the tile guide in the settings has a line and a swatch for each.

## Hand results

After a win by a player who declared riichi, the table shows a labelled
**Ura-dora** indicator row alongside the ordinary dora indicators. Each winning
hand also shows its own ordinary and ura-dora indicators and separate bonus
counts. These are result information, so the indicators remain visible when
Hints and markings is switched off. Repeated indicators retain separate slots.
Ura-dora are never exposed during play, on an exhaustive draw, or for a winner
who did not declare riichi. In a multiple-winner result the bonus belongs only
to eligible winners. A zero ura-dora bonus still shows the revealed indicators;
a yakuman result explains that dora do not add to its value.

Score explanations show the reading the scorer chose, set by set, in normal and
physical play alike, with the winning tile shown once. A triplet completed by
ron is shown open and one completed by tsumo concealed, and each dragon or wind
yaku names its own tile.

### Waits and keyboard play

A drawn tile is not included when deriving the standing hand's waits. In riichi,
this keeps the locked wait stable across unrelated draws. Before riichi the
interface distinguishes the wait before the draw from the wait after a selected
discard. Discard previews are read-only queries checked against legal actions.

Arrow keys skip disabled hand tiles, and the marker follows successful focus.
After a keyboard discard, the hand regains focus for the next decision unless
the player has deliberately focused another control. Browser shortcuts and
other focused controls remain unaffected.

### Final-hand review and saves

Opening final standings does not start another hand, so it does not clear the
final hand's event history or an already open review. Review final hand and
Save final hand remain available below the standings, including after reload.

The whole game can be saved as one mjai log as well,
`riichi-game-<date>-<time>.mjai.jsonl`: Save game so far on the score screen
between hands, Save whole game on the final standings, where the log closes
with end_game, and Export game in the settings during play. On a phone the
final hand's result sheet covers the standings, so it offers Save whole game
itself. The log opens at East 1 and holds every finished hand, numbered as the
hand's own log numbers its players. The hand being played is never in it,
since its deal shows every player's tiles; it joins the log when it ends.
Nothing is added to the saved match for this: a restore replays every command
from the first deal, so a reloaded match exports from East 1 again whatever
format it was saved in. A win names its ura indicators both as
`uradora_markers`, the original protocol's name, and as `ura_markers`, which
Mortal and its log validator read.

Matches are saved in format 6. An older save is replayed from its recorded
legal commands and checked against all authoritative state; only what was added
or repaired since (derived waits, result indicators, claimed tiles, player
identities and score attribution) is left out of that comparison. An
incompatible or corrupt save fails closed rather than being silently replaced;
[UI reliability](UI_RELIABILITY.md) lists the saves refused outright.

### MJAI accounting

Settlement deltas describe only that individual event. A reader first processes
reach_accepted's 1,000-point payment, then each hora or ryukyoku delta once.
Multiple winners have separate incremental settlements and intermediate score
balances; their changes are not duplicated. `Hand::deltas()` intentionally
remains the cumulative change for the score screen. The native log replay test
reconciles every settlement against the preceding balance and sums all events,
including riichi payments, against the hand's total.

## Verification

`npm run verify` in `web/` covers these behaviours: real-WASM discard previews
and Tile rendering, riichi waits, disabled-tile navigation, consecutive keyboard
turns, saved-state migration, ura-dora indicators with hints off, final-hand
review, history and export, and settlements that balance event by event.
Its browser checks run in Chromium, including narrow portrait and landscape
sizes; they are not physical-iPhone or Safari testing.
