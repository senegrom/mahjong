# Agent modes

Use the mode bar or **Game settings → Game mode** on the smallest screens.
Normal Play, Watch, Physical, and Guided modes keep their own lifecycles;
opening a direct Watch/Physical/Guided link does not overwrite a regular match.

## Advisers and weights

Beginner and Club are built-in agents. Trained uses the single network named
in the model manifest. The current physical adapter exposes the observation,
legal-mask, and action-conversion methods needed for Trained advice. Controls
are enabled when that capability and the network are available, including
after reconnect or download without reloading. Older adapters lacking those
methods show an explanation and keep Trained disabled rather than inventing
an observation. A saved choice is not silently replaced with Club.

Trained weights are normalized preferences over legal network actions, not
win probabilities. Built-in agents mark the selected move instead of displaying
invented percentages. Additional legal kans that the network cannot name stay
available but unscored. Advice for physical play uses the table data entered by
the player; enter visible history accurately, and leave unknown hands empty.

## Watch

Choose the followed agent and each other seat independently, then **Start
watching**. Assignments follow players when seat winds rotate. Pause to inspect
weights or use Auto play at the chosen pace. The pause-at-hand-end option waits
for the watcher to deal the next hand; disabling it permits automatic continuation.

**Play this choice** applies the displayed decision. Choosing another legal
move asks for confirmation, pauses autoplay, and applies only that original
decision. Cancelling leaves the position unchanged. Leaving Watch cancels its
outstanding work. Watched games start fresh and do not replace regular saves.
Hints and markings apply here too; blue marks the recommendation and combines
with dora, readiness, and safety markings.

## Physical table editor

Select the seat being analysed and enter its concealed hand, including any
drawn tile. Keep unknown hands empty. Tile notation is `123m456p789s11z`:
`m` means characters, `p` circles, `s` bamboo, and `1z` through `7z` mean
East, South, West, North, White, Green, Red. Tap entered tiles to remove them.

Set the current round, scores, sticks, wall count, dora indicators, melds, and
discards. Scores are after paid riichi declarations. Discard order starts at
zero across the whole table; retain claimed discards and mark their claim.
For chii enter the lowest tile and Left as the source. Concealed kan uses Self.
Each completed kan needs its additional indicator.

For a draw decision identify the tile already included in hand. After a set
call identify the tile just claimed. For a response identify the discarding
seat and pending tile; kan-robbery windows have separate kinds. **Show agent
weights** validates the table and analyses it. Editing cancels stale advice.
Record the move actually played, even when it differs from the suggestion.

**Record the next physical move** adds a known hand's actual draw or records
a discard. An unknown opponent's otherwise-unrecorded draw is counted with its
discard. After exceptional turns correct the wall count to the actual table.
**Add an earlier discard** edits visible history without advancing play.

Calls consume only held tiles and retain the claimed discard. Newly recorded
sets remember exactly which tile was claimed, so chii rotates the correct
tile even after save/reload. Legacy manually entered sets without this metadata
remain readable; the display does not guess a new claimed tile. After kan,
enter the real replacement tile and newly exposed indicator. No random wall
or unknown hand is manufactured. Passed wins record furiten; kan robbery
preserves ippatsu until the kan stands and its replacement is recorded.

**Undo edit / move** reverses tile edits, recorded moves, selectors, checkboxes,
and numeric fields. Consecutive typing in one focused numeric field forms one
undo step; changing fields or leaving the field ends that step. Clear table is
undoable for a readable draft. Undo history is window-local; the latest draft
is persisted separately from normal Play. Conflicting edits in another window
pause this editor and offer **Reload saved table**. Unreadable saves are
preserved until explicitly cleared. Storage warnings mean persistence failed.

## Guided physical game

Start from points and seat, enter exactly 13 starting tiles, then the real dora
indicator. East also supplies its extra dealer tile at the first draw prompt.
The guide asks for opponents' discards and your actual draws in turn order,
then offers advice and records the choice selected. Choose the adviser in the
mode's selector; availability follows the capability rules above.

**Record suggested move** plays the suggestion. Another legal choice asks for
confirmation. After a discard or pass, report other calls or choose **No other
calls · continue**. Calls wait for higher-priority responses before committing.
Chii asks for its lowest tile; the offered tile is recorded automatically for
both your calls and opponents' calls. Undo and saved snapshots retain it.
Pon/chii is followed by a discard without an extra draw.

For concealed or added kan, use the kan entry at the opponent-turn prompt.
Resolve robbery before recording the additional indicator and replacement.
Replacement draws count once; chained kans repeat this sequence. Invalid
counts, directions, and stale actions cannot alter the saved game.

**Someone won / hand ended** opens the result flow. Known results can use the
Rust scorer after the missing winning-hand and indicator data are entered;
the guide does not invent unseen winning hands or payments. Apply a validated
settlement once, then continue to the next hand with scores and seat winds
rotated together. Manual hand-end results remain explicitly manual. See
[Guided scoring](GUIDED_SCORING.md) for the settlement inputs and safeguards.

**Undo last step** and resume work after reload. A new guided game requires
confirmation and replaces only the guided save. Cross-window conflicts,
unreadable records, and storage failures are handled explicitly as in the
physical editor; a late analysis cannot modify a newer position.

## Developer checks

`npm run verify` in `web/` is the canonical local and CI verification command.
It includes real browser interactions and the saved-state, cancellation,
claimed-tile, grouped-Undo, and offline regressions. Historical implementation
reports are preserved in [the evidence index](HISTORY.md).
