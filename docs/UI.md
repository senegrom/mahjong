# Gameplay interface

The table UI keeps the rules engine and match state separate from presentation. Desktop uses a larger table surface with clearer opponent zones; compact phone layouts keep the round, opponent type and status visible while moving secondary controls into the settings sheet.

When **Hints and markings** is enabled, every tile in the human hand has a small number directly below the face showing how many copies of that tile are still unseen from public information. The count includes the player's own concealed copies and subtracts visible unclaimed discards, called sets and exposed dora indicators exactly once. Zero is highlighted as exhausted and one as scarce. Turning Hints off removes the counts together with the other learning aids.

Calls are presented as a staged discard followed by the legal response choices. A quad that may be robbed is staged as the kan it is, with Ron offered as robbing it, and a robbed quad's result and hand history say it was robbed. On phones, hand results appear in a large bottom sheet with the result value first, then the winning hand, yaku and score changes; the table can be revealed again without losing the result. The value is what the hand is worth, written from the engine's payments: a ron's total, a dealer's self-draw as what each other player pays ("4,000 all"), and any other self-draw as each non-dealer's share and the dealer's ("2,000 / 4,000"). Reduced-motion preferences suppress nonessential transitions.

The 7-by-2 phone hand layout is intentional: it preserves large, reliable touch targets instead of squeezing fourteen tiles into one row. The newly drawn tile remains slightly separated, including its hint count, and carries no border merely for being newly drawn.

## Tile faces

Classic and Matisse have a picture for every tile. Dalí and Van Gogh are still
being painted: a tile the chosen set has no approved artwork for shows its name
in text on the ivory face instead, so it plainly waits for its picture and never
borrows another set's. In the hand the number stands over its suit, or an
honour's first word over its second, in the words the tile is announced with.
On faces of 40px or less, as in discard rows, dora indicators and the physical
editor's entered tiles, a number stands over its suit's letter as tile notation
writes it (m characters, p circles, s bamboo), or beside it on a tile turned on
its side, so a suit never rests on its colour alone; the colours (red
characters, blue circles, green bamboo) only repeat it. On the same faces an
honour is given in letters, as large as they fit: E, S, W and N for the winds,
and Wh, G and R for the dragons; its first word alone was too small there to
read at a glance. Every name keeps half a pixel clear of a claimed tile's
border and its ivory hairline, and is the same size whether its tile is
claimed or not and upright or turned, so the border never cuts it and a row's
names match; the letters on a phone's rows, a white dragon's Wh and a hand's
second words are sized to fit that room. The name stays upright on a tile
turned for riichi and carries the dora ring and shine and both discard marks; a
white dragon written out has no dragon under its shine, and with reduced motion
its still shine rests on a corner rather than across the name. Nothing is
downloaded or saved offline for these tiles, and each becomes its picture as
soon as its artwork is approved.

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

A riichi declaration lies sideways in its row: the face is laid out lying down,
so its edge falls on whole pixels like an upright face's, and only the picture
on it is turned. Two more marks are of different kinds, a shade and a border,
so a tile can carry both. Both lie on one layer over the face, clipped to its
rounded edge, so they line up with the face either way up:

- **A darker face:** the tile was thrown straight from the draw (tsumogiri), as
  Tenhou and Mahjong Soul show it. A veil of black keeps 72% of every colour,
  so the picture still reads. A tile thrown from the hand looks normal.
- **A dark green border:** another player claimed the tile for a call. The
  border is solid and lies on the rim of the face, inside a dora ring, on a
  riichi tile as on any other. A one-pixel hairline of the tile's ivory lies
  inside it: the bamboo of every artist's set is painted to its edges in
  nearly the border's green, and without the hairline the border ran into the
  painting and the tile looked unclaimed. On an ivory face the hairline
  disappears and the green alone stands out. The face stays solid and keeps
  its colours, so a darker face always means a tile from the draw.

The marks appear on the table, in the inspection of all discards, in agent
watch and in the guided game's remembered table; the physical editor's tile
beside each discard shows the boxes ticked for it. They record what happened
at the table rather than give advice, so they stay when Hints and markings is
switched off. Each tile's name, read by screen readers and shown on hover, says
the same: claimed, riichi declaration, discarded from the draw. The shade and
the border's colour are the `--from-draw-shade` and `--claimed-border` tokens
in `app.css`, and the tile guide in the settings has a line and a swatch for
each.

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

Matches are saved in format 7. A save of format 5 or 6 is replayed from its
recorded legal commands and checked against all authoritative state; only what
was added or repaired since (score attribution, a win's value as numbers and
the words for a robbed quad) is left out of that comparison. Before format 7 a
declared quad left the turn's drawn tile held apart while the quad could be
robbed and after it was, even where the quad had taken that tile. So at a call
or a result an older save's concealed tiles are compared as one set, without a
tile whose four copies are all in a quad, and its waits are left out. An
incompatible or corrupt save fails closed rather than being silently replaced.
A save from before format 5, written between 5 and 10 September 2026, is no
longer resumed; [UI reliability](UI_RELIABILITY.md) says what the page tells
its player.

### MJAI accounting

Settlement deltas describe only that individual event. A reader first processes
reach_accepted's 1,000-point payment, then each hora or ryukyoku delta once.
Multiple winners have separate incremental settlements and intermediate score
balances; their changes are not duplicated. `Hand::deltas()` intentionally
remains the cumulative change for the score screen. The native log replay test
reconciles every settlement against the preceding balance and sums all events,
including riichi payments, against the hand's total.

## Whole-game review

**Review whole game** on the final standings, there again after a finished
match is reloaded, puts every decision the player made in every hand to Trained
AI, and leads with the few where the player's move departed furthest from its
choice. The hand review reviews one hand and is gone at the next deal; this one
is rebuilt for the whole game. On a phone the final hand's result is a sheet
over the standings, so the sheet offers the review too, as it offers the
game's file. The sheet steps aside whenever the review opens, as it does for
the hand's review, whichever button opened it: its own, or the standings'
own, which it covers but a keyboard or a screen reader can still reach. **Back
to the standings** then lands on the standings, uncovered, with focus on their
own **Review whole game**, and the result is a tap away on its chip, which a
keyboard reaches after the standings' buttons. The chip goes as it brings the
result back, so focus goes to the result, as when it first appeared, rather
than to the page. Up to 480 pixels wide the sheet's buttons take their short
labels, **Review hand**, **Save hand**, **Review game** and **Save game**.
Measured in Chromium, they then take two rows over the result from 360 pixels
up with Windows' fonts and from 390 with one as wide as Verdana, and three on
narrower screens.

**How it is built.** Nothing is added to the saved match, and its format is
unchanged. The finished match is replayed from its saved commands, from the
first deal, on an engine of the review's own; the replay must end exactly where
the match did, or the review refuses to start. At the end of each hand the
review asks the questions the hand review asks (`review-policy.js`), eight
positions a forward pass: Mortal's planes for each decision as it was made,
then, where the network declares riichi, which tile it would discard. A move
with one legal choice, such as the discard after riichi, is not asked and is
left out of every figure.

**How decisions are ranked.** A decision is flagged when the player's move had
under 5% of Trained AI's weight while its own choice had 60% or more, so a
network that is not decisive flags nothing. Riichi is judged in two stages, as
the weights are shown everywhere: when the player and the network both
declared, the tiles are compared by the discard weights given riichi;
otherwise the move itself is compared, the declaration carrying its
declaration weight. Flagged decisions are ranked by the gap between the two
weights, widest first, and equal gaps in the order they were played. The first
five are listed, the rest behind **Show all**. The list says where each one
was (the hand, with its honba where the hand repeats one, so no two hands
share a name, and the player's decision number) and the weights of the
player's move and of Trained AI's own choice, but not what that choice was.
The list is ranked afresh as each hand is reviewed, and keyboard focus on a
flagged decision stays on it however the list moves, also while the player is
in another window or tab, which takes focus from the page but leaves it on the
decision for when the player comes back: should the hands reviewed since rank
it out of the first five, every flagged decision is shown rather than its
button being taken away, and a status line under the heading says where it now
stands ("this one is now 7th"), since a list should not be reshuffled under
someone reading it without their being told. The line says the place again
whenever the place changes, but not how many are flagged: that grows with every
hand reviewed while the place stays, so a count said once would go stale, and
one kept current would be read out again at every hand. The line goes when
focus moves on.

**Replay and choose again.** A flagged decision opens at the position it was
made from: the table as it stood, the player's hand, and what happened since
their previous decision. **From the deal**, **Previous** and **Next** step
through the hand a decision of the player's at a time, saying whether each
earlier move matched Trained AI; the step's status says the move as well, so a
screen reader hears it with each step. A step that disables the button pressed
hands keyboard focus on rather than dropping it: at the deal to **Next**, and
at the decision to the hand, whose description asks for the choice, as play
gives the hand focus when a turn comes. **Choose again** does the same, since
the answer it closes held the button. At the flagged decision the player chooses
again, a tile from the hand or a move below it, before anything is revealed,
or asks for **Trained AI's answer** outright. An answer to another's discard
leaves no tile to throw, so there the description sends the player to the
moves below the hand, which **Tab** reaches next. The answer lists the weight of
every move, the riichi discards under their declaration, marking Trained AI's
choice, the move made in the game and the new one. It leads with the five
heaviest moves and those it marks; **Show every move** shows the rest and
stays, as **Show the leading moves**, so focus stays on it. For a turn, a table sets
out what each of those moves leaves, counted as the hand review counts it:
the hand left, the unseen tiles that improve it, whether the tile could deal
into a declared riichi, and its worth as dora (the engine's `review_measure`).
Nothing is played on from there, and the saved match is never touched. **Back
to the list** returns focus to the decision opened, wherever the hands
reviewed meanwhile have ranked it: if they have ranked it out of the first
five, every flagged decision is shown, and the same status line says where it
went. The line stays in the page, silent, while a decision is open: a screen
reader hears a status only once it is in the accessibility tree, and the list
comes back around the line in the same moment that it speaks.

**Agreement and its trend.** The figure is the share of decisions with a
choice where the player's move was Trained AI's. When every hand is reviewed,
the game enters a trend kept in this browser only (`riichi.review-trend.v1`,
the last 50 games; a browser that keeps nothing still reviews, without a
trend). The figure is shown against the decision-weighted average of up to ten
earlier games, with a line of the last twelve and a table of the same. The
change is stated, not coloured: matching Trained AI is not the same as playing
well, since the network has habits of its own. The review is judged by the
version of Trained AI the page plays, which the player can pick in the game
settings, and the review and the trend name it. Another version judges
differently, so each game is kept with the version that judged it, a game
judged again by another version is kept beside its first entry, and the
figure, the line and the table take only the games of the version playing
now; a note counts the others, which come back when their version is picked
again.

**Measured on played games.** On 16 October 2026, 50 games were played with
the Beginner bot in the player's seat and 50 with the Club bot, each against
three Trained players, the page's default table, and reviewed by the page's own
review code with Trained 0.1.810 and these thresholds; on the four games also
reviewed in the page, the two gave the same answers. Beginner matched Trained
AI in 40.1% of its decisions and Club in 57.7%. An average of ten games, as the
trend takes it, has a standard error of 1.5 points for either, so ten games
tell the two apart by 8 standard errors. None of the 6,688 flagged decisions
had every row of its table favouring the move made in the game, which puts that
share under 0.045% (by the rule of three, at 95%), and 27% had a row for it and
none against, so as the rule was registered the thresholds stand. Read so, it
could hardly trip: a move that leaves the hand closer to complete almost never
leaves more tiles that improve it as well, and not one of the 8,197
disagreements with a table, flagged or not, had every row for the move made.
The rule did not say how a row that ties should count. Counted with the move
made, 54% of the flags (± 1.7 points) have every row for it or tied, and 74% of
Club's; the least raise that brings that under half on the games used to
choose it, flagging only moves given under 0.2% rather than 5%, leaves 47%
(± 2.7) on the games held out, and Club's at 72%. The thresholds are unchanged
until the owner decides which reading holds. The table is more often silent:
for 60% of the flags, and 82% of Club's, it gives no reason for Trained AI's
choice, which the hand review calls a judgement, a trade of speed against
value or safety that its numbers do not capture.

**Speed.** Measured on 14 October 2026 in Chromium 155 on the desktop, through
the page's own worker on one thread, a position took a median 163 ms one at a
time and 157 ms in batches of eight, with identical weights; natively the same
batch nearly halves the cost. A game of 150 to 200 decisions takes about half
a minute there; phones were not measured. The review runs in the background
with its progress and an estimate of the time left, goes on while its screen
is closed, and stops at **Stop** or when the player leaves Play. A device that
takes over half a second a position holds after each hand and offers the next
hand or the rest of the game; any hand can also be reviewed alone from the
list of hands. These buttons replace one another as the review goes, so focus
is handed on rather than dropped: from **Stop** to **Continue the review** and
back, from a hand's own button to the list of hands once that hand is under
review, and to the flagged decisions when nothing is left to press. The
review needs the network: while it cannot be reached the
screen says so and waits.

## Verification

`npm run verify` in `web/` covers these behaviours: real-WASM discard previews
and Tile rendering, riichi waits, disabled-tile navigation, consecutive keyboard
turns, saved-state migration, ura-dora indicators with hints off, final-hand
review, the whole-game review with the shipped network (its batches, flags,
replay, retry, keyboard focus, in the flagged list too as later hands rank it
afresh, and trend, and its way in and out on a phone, by touch and by
keyboard),
history and export, and settlements that balance event by event.
Its browser checks run in Chromium, including narrow portrait and landscape
sizes; they are not physical-iPhone or Safari testing.
