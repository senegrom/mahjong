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

### The hand's review

**View table / my hand** on a hand's result opens the hand's review, **Your
hand, looked at again**, below the result, as **Review final hand** does on the
last hand's. On a phone, up to 760 pixels wide, the result is a sheet over the
table and the hand, so it steps aside as well, leaving its chip, **Show
result**, at the foot of the screen. The button goes once the review is open,
so focus goes on to the review's heading rather than dropping to the page,
however the button was pressed: the review is what the press brought, and on a
phone the table and the hand that the sheet uncovered are just above it. A
screen reader reads the heading out, in the region "your decisions this hand",
so the player hears where they are, also while Trained AI is still reviewing
the hand and the region is busy.

Where the result is not a sheet, at desktop width and held sideways on a phone
wider than 760 pixels, it covered nothing, and the review opens below it, often
below the foot of the screen. So there the heading is brought into view however
the button was pressed; otherwise a player who clicked would see the button go
and nothing else. Where the result is a sheet, only a press by keyboard moves
the page. A mouse's click or a finger's tap there asked to see the table and the
hand, which the sheet uncovers by stepping aside, and is shown no focus ring to
follow below them, while the scroll would take the top of the table off a small
screen. The page tells the two apart by the click itself. A click's detail is
its click count: at least one for a mouse button, a tap or a pen, in every
browser, and none for the click a browser makes for Enter or Space on a focused
button. A screen reader's press counts none in Chromium and may count one
elsewhere; either way focus moves, and a screen reader follows it.

The heading is brought into view, with its focus ring for a keyboard, by as
little movement as will show them: below the game's bar where that stays at the
top of the screen, clear of the chip on a phone, and a ring's width and a
little from the foot of the screen where no chip sits. Measured in Chromium,
the page moves this many pixels for each button; on a phone a mouse's click or
a finger's tap moves it none. The figures are the same with Club's review,
done as it opens, and with Trained AI's, still under way, except at 844x390,
which has a row for each. The 844x390 rows are pressed from the foot of the
page, where a fling down to the result's buttons ends; there the distance
depends on where the press starts (see below).

| Screen | Pressed by | View table / my hand | Review final hand |
| --- | --- | ---: | ---: |
| 1100x900 | any | 53 | 53 |
| 1366x768 | any | 53 | 53 |
| 1280x600 | any | 56 | 53 |
| 844x390, a wide phone held sideways, Club reviewing, from the foot | any | 263 | 315 |
| 844x390, Trained AI still reviewing, from the foot | any | 154 | 204 |
| 390x844 | keyboard | 2 | 343 |
| 375x667 | keyboard | 183 | 516 |
| 360x640 | keyboard | 124 | 451 |
| 320x568 | keyboard | 171 | 566 |
| 568x320, a small phone held sideways | keyboard | 574 | 847 |

On a phone that scroll costs what lies above the heading. For **View table / my
hand** at 390x844 everything stays in view; at 375x667, 360x640 and 320x568 the
top of the table, its round, wall count and indicators, goes under the game's
bar, and at 568x320, where the result is still a sheet, most of the hand goes
too. On the final hand the standings lie between the hand and the review, so
**Review final hand** moves the page further. At 390x844 the table goes off the
top of the screen while the hand, the standings and the heading stay in view.
On the smaller phones the hand goes as well: at 375x667 all but its last 23
pixels, which show below the game's bar; at 360x640 what is left of it sits
under the bar; and at 320x568 all of it, off the top of the screen. At 568x320
the hand and the standings' heading both go off the top.

At 844x390 the two advisers differ. With Club's review, done as it opens, the
browser's own scroll anchoring moves the page as the review opens and leaves
the heading under the game's bar; the page then brings it out just below the
bar, 72 pixels from the top of the screen. With Trained AI's review under way,
a shorter review that says how far it has got, the page moves less, and the
heading ends lower, 181 to 207 pixels from the top, in view below the bar.
Where it starts matters too. Pressed where the button has only just come into
view, **View table / my hand** moves the page 72 pixels and the heading lands
358 pixels from the top. From there **Review final hand** moves it 358 pixels
with Club, or 177 with Trained AI. With Trained AI the figures also vary by
a few dozen pixels from press to press, as the review's progress line changes
length while it downloads or saves.

The way back to the result is the control just before the review, one
**Shift+Tab** away: on a phone the chip, which hands focus back to the result
as when it first appeared, and where the result is not a sheet, its own
buttons, which stay where they were.

A review the engine cannot build opens nothing. The result stays as it was, on
a phone too, and so does the pressed button, with focus, and the result says
why beside its buttons, where the player pressed one: "The review could not be
built. Your match is unchanged." When the result's **Save this hand** or
**Save game so far**, or on the last hand's result **Save final hand** or
**Save whole game**, cannot save its file, the result says so the same way,
from the same line: "The hand could not be saved to a file." or "The game
could not be saved to a file.", each followed by "Your current match is
unchanged." On a phone the last hand's result offers **Review whole game** as
well, and when the finished match cannot be replayed for that review it says
so from the same line: "The game could not be replayed for review:", why, and
"Your saved match is unchanged." The line keeps the words of the last press
that failed until that press succeeds. The page's notice line would not do: it
sits at the top of the page, above the screen at desktop width, and under the
sheet on a phone, or at best above it, dimmed by its shade, at the far end of
the screen from the buttons. The result says it from a status line of its own
after its buttons, which waits there, empty and out of sight, from the moment
the result appears, since a screen reader hears a status line only once it
knows it. Where the result is not a sheet, the line is brought into view
should it fall below the foot of the screen; on a phone it stays at the foot
of the sheet with the buttons, whose row spans the sheet from edge to edge, so
that nothing scrolling beneath it shows beside them. The standings and the
settings' **Export game** say their own failures the same way, beside their
own buttons (below); the notice line now says only what the trained network
picker could not do.

The review's own status lines, that Trained AI is reviewing the hand and how
far it has got, and that Club reviews in its place while the network cannot be
reached, come into the page with the review. They wait in it, empty and out of
sight, until the review has been drawn and a tenth of a second more, before
their words arrive, so that a screen reader can hear them after the heading it
reads out as focus arrives.

### The standings as the game ends

**Next hand** on the last hand brings the final standings in place of another
deal, and its button gives way to the caption **Final hand**, so focus goes on
to the standings' heading, which says where the player finished ("You finished
4th"), in the region "final standings", rather than dropping to the page. Where
the result is not a sheet, the standings appear just above it. On a phone the
result is a sheet over them, and focus must not be hidden under it, so when the
keyboard pressed the button the sheet steps aside for the standings, leaving
its chip, as it does for the reviews; the chip, after the standings' own
buttons, brings the result back with focus. A tap leaves the result up over the
standings, as it was: focus is on their heading beneath the sheet, where a
screen reader reads it out and no ring is drawn.

Only a press by keyboard moves the page to the standings' heading, at every
size, to bring it into view with its ring, clear of the game's bar and, on a
phone, of the chip. A mouse's click or a finger's tap moves focus without
asking for a scroll, so the page is left as the browser leaves it when the
standings come in above the result, as it was before focus moved there:
measured in Chromium at nine sizes, from where the result left the page and
from the foot of the page, it ends exactly where it did before. With the page
at its foot at 844x390, where a fling down to the result's buttons ends, the
browser's own scroll anchoring moves the page 233 pixels as the standings come
in; the Final hand caption that took the pressed button's place stays in view,
and the standings' heading lies above the screen, where a screen reader still
reads it out. A scroll to the heading there would take the caption the player
pressed off the screen.

Measured in Chromium, a press by keyboard moves the page 30 pixels at 390x844,
203 at 375x667, 138 at 360x640, 202 at 320x568 and 586 at 568x320. At desktop
width and at 844x390 the heading is already in view where the result left the
page, and the page does not move; from the foot of the page at 844x390 the
keyboard's press brings the heading in below the bar, 255 pixels above where a
click leaves the page.

Agent watch does the same as a watched game ends. There **Next hand** on the
last hand is pressed on the result or among the watch's controls, whose button
plays the followed agent's choices and deals the hands. Either brings the
standings, the result's button giving way to its **Final hand** caption and
the controls' becoming one that cannot be pressed, and either used to leave
focus on the page. Focus now goes on to the standings' heading ("Followed
agent finished 2nd"), and only a press by keyboard moves the page to it, the
result stepping aside on a phone as in play, with its chip after the
standings' buttons. Measured in Chromium on a game of Club agents from seed
23, a press by keyboard on the result's button moves the page 387 pixels at
390x844, 720 at 320x568 and 568 at 568x320, and 304 pixels up at 844x390,
where the browser's scroll anchoring had left the heading above the screen;
at 1100x900 the heading is already in view. On the controls' button, at the
top of the watch, it moves the page 14 pixels at 1100x900, 387 at 390x844,
720 at 320x568 and 462 at 844x390. A click or a tap asks for no scroll, so
the page ends where it did before focus moved, at each of these sizes: at
844x390 the browser's own scroll anchoring moves it 267 pixels as the
standings come in after a press on the result's button, as it did. On a
phone the controls' button lies mostly under the result's sheet, where a
keyboard or a screen reader reaches it; at 390x844, once the page is scrolled
to bring it there, a finger can also tap it in the 78 pixels between the game's
bar and the sheet, and that tap is taken as a tap on the result's button: focus
goes to the heading, the page does not move and the sheet stays.

The standings' own **Save whole game** and **Review whole game** say a failure
where they were pressed as well, from a status line of their own after their
buttons, which waits there, empty and out of sight, from the moment the
standings appear: "The game could not be saved to a file. Your current match
is unchanged.", or "The game could not be replayed for review:", why, and
"Your saved match is unchanged." Nothing opens, and the pressed button keeps
focus. The notice line said it before, at the top of the page, which lies
above the screen wherever the standings' buttons are pressed: measured in
Chromium, 421 to 440 pixels above the top of the screen at 1100x900, 553 to 572
at 1366x768, 497 to 534 at 844x390, on a phone held upright 96 to 644 pixels
above it, and 619 to 738 at 568x320.
On a phone the result is a sheet over the standings. A finger or a mouse
reaches their buttons once the sheet has stepped aside for a review; a
keyboard or a screen reader reaches them beneath it, so a press there that
fails has the sheet step aside, each time, for the words and the button with
focus to be seen, and the chip brings the result back. The buttons and the
line are brought into view by as little movement as will, clear of the game's
bar and the chip: measured, the page does not move for any press at 1100x900,
1366x768 and 844x390, nor on a phone for a finger or a mouse; for a keyboard
beneath the sheet it moves 41 pixels at 390x844, 42 to 61 at 320x568, and at
568x320 21 for Save whole game and 41 for Review whole game, to bring the line
out from under the chip. Once the game is saved, or its review opens, from any
button, every line that said it could not be takes its words back.

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

When **Export game** cannot save its file, it says so beside its button, from
a status line of its own that waits in its panel, empty and out of sight,
before its words arrive: "The game could not be saved to a file. Your current
match is unchanged." The button keeps focus. On a phone, upright or held
sideways, the settings are a modal dialog, and the notice line that said it
before lies behind the dialog. Measured in Chromium, while the dialog was
open the accessibility tree ignored that line (`activeModalDialog`), so a
screen reader heard nothing until the dialog closed, and a sighted player saw
it only dimmed behind the backdrop, above the dialog at 390x844, or not at
all, above the screen at 320x568 and 844x390. The panel's own line is inside
the dialog, where it is heard and seen: should it fall below the dialog's
foot, the dialog's contents are brought up to show it, by 23 pixels at
320x568 and 5 at 844x390, and at 390x844 the dialog has the room. At desktop
width the settings are part of the page and the line is in view: under the
button at 1100x900, and beside it on the same row at 1280x600 and 1366x768.

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

**Which decisions are listed.** The list opens on the flagged decisions, and a
filter under its heading lists instead every decision where Trained AI chose
differently, or every decision, each of its three buttons saying how many it
holds: "Flagged (7)", "Trained AI chose differently (64)", "Every decision
(182)". The button pressed says so (`aria-pressed`) and keeps focus, and the
heading names the list. Where Trained AI chose differently, the decisions are
ranked as the flagged ones are, by the gap between the weights, those whose
moves could not be compared (an unscored quad) after the rest, and the
flagged ones among them say **Flagged** and keep the accent's edge; every
decision is listed in the order it was made. Each list leads with five, the
rest behind **Show all**. A status line, in the page from the start, says what
the list now shows ("Showing 64 decisions where Trained AI chose differently,
7 of them flagged, the widest departure first. The first 5 are listed."), and
goes quiet once focus leaves the filter, or once the hands reviewed since have
changed the counts it gave, so it never stays to say a count the list has
outgrown. A flagged decision keeps Trained AI's choice back
for its retry; any other says it ("Trained AI: discards the 9 circles"), or
that Trained AI chose the same, or that the move was the only one, and opens
with **Replay**: its position, the move and Trained AI's view of it, with no
retry. Keyboard focus on a listed decision stays with it in every list as the
review goes on, and the status line says where it went: further departures put
a decision lower in a ranked list, and in the list of every decision a hand
reviewed out of turn puts its decisions before those of the hands after it.

**Every hand, face up.** At every step of a replay the table is shown as it
stood, every player's hand face up all the time, with no control to hide or
show them: each opponent's concealed tiles in their seat, the tile just drawn
apart, above their sets (the called tile turned) and their discards in order
(the riichi tile turned, the tiles called away marked), the seat headed by its
wind, where the opponent sat and who played them, the dealer, the score and
riichi; the round, the honba and riichi sticks, the
dora indicators and the tiles left in the wall in the middle; then what
happened since the player's previous decision, and the player's own seat. The
seat whose move it was is edged in gold and says so in words: "To play" on the
player's own turn; when the player answers another's tile, the edge is on the
seat that let it go, which says "Discarded" or "Declared a kan", and the
player's seat says "To answer" without one. Each hand is a list named for its seat ("South's
tiles, Right (Club)", "East's tiles, yours"), the table is headed "The table at
your decision 5", and a status line says it as the step changes ("South
discarded the 5 bamboo, 45 tiles left in the wall. West has declared riichi."),
adding "Trained AI flagged your move here." at a flagged decision other than
the one opened, whose own step says so.
The tiles are the review's own: the replay the review is built from reads them
at each decision once the hand is over (`scanMatch`, through the engine's
`review_hands`), never from the match being played. The engine refuses them
while a hand is still being played, as it leaves such a hand out of a log.
`web/tests/game-review.test.js` rebuilds every hand at every decision of two
fixture games from the hand's mjai log, through calls, quads, riichi and wins,
and the engine's own test rebuilds them through Mortal's reader of the same
events. An opponent's tiles take a row of fourteen with the drawn tile's gap,
on a darker rack that sets them apart from the discards below, as large as
their seat has room for: up to the size of their discards on a wide screen,
and up to 24 pixels wide on a smaller one, larger than their discards there.
Where one row would draw them under 18 pixels wide, they take two rows of
seven instead, the drawn tile last, as the player's own hand does on a phone.
On a phone held upright, three seats side by side would leave each hand a few
tiles to a row, so there the opponents take the width of the screen, one
under another in the order they play after the player, their discards a size
larger than in play. Measured in Chromium on 19 October 2026 at each of the
139 decisions of the browser check's game, an opponent's tiles were 18 to 20
pixels wide on one row at 1100 by 900 and 20 to 22 at 1366 by 768, beside
discards of 27 and the player's own tiles of 52; 22 on one row at 390 by 844
and 24 on two at 320 by 568, one opponent under another; and 24 on two at
800 by 1280 and held sideways at 844 by 390, three abreast. No page scrolled
sideways. The table with the player's seat, the events since the previous
decision open between them, took 682 to 1,106 pixels of height at 1100 by
900, 687 to 1,111 at 1366 by 768, 651 to 1,012 at 800 by 1280, 656 to 1,235
at 390 by 844, 762 to 1,287 at 320 by 568 and 532 to 899 at 844 by 390.

**Replay and choose again.** A flagged decision opens at the position it was
made from: the table as it stood, every player's hand, and what happened since
the player's previous decision. **From the deal**, **Previous** and **Next** step
through the hand a decision of the player's at a time, saying whether each
earlier move matched Trained AI; the step's status says the move as well, so a
screen reader hears it with each step. A step that disables the button pressed
hands keyboard focus on rather than dropping it: at the deal to **Next**, at
the hand's last decision to **Previous**, and at the decision to the hand,
whose description asks for the choice, as play gives the hand focus when a
turn comes. **Choose again** does the same, since the answer it closes held the
button. At the flagged decision the player chooses
again, a tile from the hand or a move below it, before anything is revealed,
or asks for **Trained AI's answer** outright; until then the replay goes no
further than that decision, and once it is answered it goes on to the end of
the hand. An answer to another's discard
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

**Any decision, any hand.** A reviewed hand is replayed whole, from its deal,
by **Replay this hand** in the list of hands, which stays open for the player
while the hand is replayed, so **Back to the list** returns focus to the
hand's button. In any replay a list of the hand's decisions has a button for
each step, saying the move and whether Trained AI agreed ("Trained AI agreed",
"Trained AI differed", "Flagged", "Only move"); the step shown is the current
one (`aria-current`), and a step's button keeps focus as it shows its step.
The list starts open above the table where the screen has room for it, and
folded on a phone or in a replay opened to choose again; from then on it stays
as the player leaves it, open or folded, as the step changes and as later
hands are reviewed. So does the list of what happened since the previous
decision, also across a step where nothing did. The same filter lists
only the hand's flagged decisions, or those where Trained AI chose
differently, each button saying how many it holds, and says what it shows
from its own status line. A decision that was not flagged shows the move made
and Trained AI's view of it: the weights of the moves, and for a turn what
each of the two moves leaves. A flagged decision other than the one opened
offers **Choose again here**, which asks for the choice before the answer as
at the decision opened, or **Show Trained AI's answer**.

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
review code with Trained 0.1.810, the site's default network then, and these
thresholds; on the four games also reviewed in the page, the two gave the
same answers. Beginner matched Trained
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
value or safety that its numbers do not capture. All of this was measured
with 0.1.810; 0.1.1110, the default since 19 October, has not been measured
so.

**Speed.** Measured on 14 October 2026 in Chromium 155 on the desktop, through
the page's own worker on one thread, a position took a median 163 ms one at a
time and 157 ms in batches of eight, with identical weights; natively the same
batch nearly halves the cost. A game of 150 to 200 decisions takes about half
a minute there; phones were not measured. The review runs in the background
with its progress and an estimate of the time left, goes on while its screen
is closed, and stops at **Stop** or when the player leaves Play. A device that
takes over half a second a position holds after each hand and offers the next
hand or the rest of the game; any hand can also be reviewed alone from the
list of hands, which opens by itself once, when the review turns out that
slow. After that the list of hands, like the trend's table of games, stays as
the player leaves it, open or folded, as the review goes on and across a
replay. These buttons replace one another as the review goes, so focus
is handed on rather than dropped: from **Stop** to **Continue the review** and
back, from a hand's own button to the list of hands once that hand is under
review, and to the flagged decisions when nothing is left to press. The
review needs the network: while it cannot be reached the
screen says so and waits.

## Verification

`npm run verify` in `web/` covers these behaviours: real-WASM discard previews
and Tile rendering, riichi waits, disabled-tile navigation, consecutive keyboard
turns, saved-state migration, ura-dora indicators with hints off, final-hand
review, focus going on from the result's buttons to the hand's review and to
the final standings, pressed by keyboard, mouse and touch, and back to the
result, at desktop and phone sizes, the page moving to the review for any
press where the result is not a sheet and otherwise only for the keyboard, and
to the standings only for the keyboard, in play and in Agent watch, a review
that could not be built and a file that could not be saved said beside the
result's buttons, which keep focus, a whole game that could not be saved or
replayed for review said beside the standings' buttons, the result stepping
aside on a phone for a press beneath it, and a game that **Export game** could
not save said beside its button, inside the settings' dialog on a phone, the
sheet's buttons spanning it on a phone, the status lines that wait in the
page, empty, before their words arrive, the whole-game review with the
shipped network (its batches, flags,
replay, retry, keyboard focus, in the flagged list too as later hands rank it
afresh, and trend, every hand face up at every step of a replayed hand at
desktop, tablet and phone sizes, the filters of its list and of a hand's
decisions, by keyboard and pointer, its lists staying open or folded as the
player leaves them while the steps change and later hands are reviewed, by
touch and by keyboard, and its way in and out on a phone, by touch and by
keyboard),
history and export, and settlements that balance event by event.
Its browser checks run in Chromium, including narrow portrait and landscape
sizes; they are not physical-iPhone or Safari testing.
