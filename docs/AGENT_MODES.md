# Agent modes

Use the mode bar, or **Game settings → Game mode** on the smallest phone screens.

## Watch

Select **Agent watch**, choose the followed agent and each of the other three
players, then **Start watching**. Quick and Strong can sit at the same table.
Agent assignments stay with the same players as their seat winds change.

With **Auto play** off, the table pauses at each followed player's decision,
including calls. **Play this choice** applies the exact displayed choice and
advances the other players to the next decision. Auto play handles the next
hand as well; the pace controls the pause between decisions. **Show choice
weights** works with either setting. Leaving Watch stops its outstanding work.
Your regular saved match remains separate; watched games start fresh.

The percentages sum to 100% over the trained policy's legal action space. The
highest weight is the action played; a 70% weight is not a 70% chance of winning.
Built-in agents have rules and random tie-breaks, rather than a learned policy,
so their chosen move is marked **Selected**. The trained action space can name
only the first legal kan of each kind. A second legal kan remains available to
record in Physical play and is labelled **Unscored**.

## Physical table

1. Select **Physical agent play** and the seat to analyse. Seats use the current
   hand's East/South/West/North winds, so change them when the dealer changes.
2. Enter the concealed hand, including any drawn tile. Other concealed hands
   may be left empty; they are never supplied to the agent's observation.
   Use the tile palette or notation such as `123m456p789s11z`. Tap an entered
   hand tile to remove it. `m` is characters, `p` circles, `s` bamboo, and
   `1z`–`7z` are East, South, West, North, White, Green and Red.
3. Set the round, scores, counters, wall and indicators. Enter all visible
   called sets and discards. For a chii, select its lowest tile and Left as
   the source. Concealed kan uses Self. A completed kan needs another dora
   indicator, including when the indicator repeats an earlier one.
4. Discard order starts at zero across the whole table. When entering an
   existing position seat by seat, edit the order numbers to reflect play.
   Mark claimed discards (still needed for furiten), the riichi declaration
   tile, and discards taken directly from the draw. Scores and riichi sticks
   in the editor are the current amounts, after declarations have been paid.
5. For your turn choose the drawn tile already in the hand, or the tile just
   claimed for a chii/pon. For a response choose the discarding seat and its
   latest unclaimed tile. Kan robbery has separate added/concealed settings;
   enter the announced kan but only indicators already revealed.
6. Select the adviser and **Show agent weights**. Invalid positions explain
   what to correct. Any edit clears the old analysis and cancels its request.
   You can record any legal choice, including a different one from the agent.

**Record the next physical move** adds a real draw to a known 13-tile shape,
or records a discard for any seat. It removes a discarded tile from a known
hand; for an unknown hand, recording a discard also counts that player's
otherwise unrecorded draw. After a call or exceptional turn, set the live-wall
count to the actual table's count. **Add an earlier discard** only edits the
history, without changing the concealed hand or current decision.

Recording a pass leaves the tile available for the other seats' responses;
passing ron marks furiten. It never invents the others' responses. Calls move
the held tiles into a set and retain the claimed discard. After kan, enter the
actual replacement tile and newly exposed indicator. No random tiles are
drawn, no hidden hands are filled, and no opponents play automatically.

**Undo edit / move** restores the previous recorded edit or move. Direct field
edits remain editable in place. **Clear table** starts an empty draft and can
also be undone. The latest physical draft is saved separately from the regular
game on this device; it is restored when returning to the mode. A recorded win
marks the hand finished; settle the physical scores and enter the next hand.

## Validation

`web/tests/agent-modes.test.js` compares manually reconstructed live positions
with the exact engine observations and legal masks; checks calls, furiten,
riichi, quads and impossible inputs; and plays watched mixed tables across hand
boundaries. It also checks masked softmax and cancellation of a disposed watch.
