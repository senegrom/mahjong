# Gameplay interface

The table UI keeps the rules engine and match state separate from presentation. Desktop uses a larger table surface with clearer opponent zones; compact phone layouts keep the round, opponent type and status visible while moving secondary controls into the settings sheet.

When **Hints and markings** is enabled, every tile in the human hand has a small number directly below the face showing how many copies of that tile are still unseen from public information. The count includes the player's own concealed copies and subtracts visible unclaimed discards, called sets and exposed dora indicators exactly once. Zero is highlighted as exhausted and one as scarce. Turning Hints off removes the counts together with the other learning aids.

Calls are presented as a staged discard followed by the legal response choices. A quad that may be robbed is staged as the kan it is, with Ron offered as robbing it, and a robbed quad's result and hand history say it was robbed. On phones, hand results appear in a large bottom sheet with the result value first, then the winning hand, yaku and score changes; the table can be revealed again without losing the result. The value is what the hand is worth, written from the engine's payments: a ron's total, a dealer's self-draw as what each other player pays ("4,000 all"), and any other self-draw as each non-dealer's share and the dealer's ("2,000 / 4,000"). Reduced-motion preferences suppress nonessential transitions.

The 7-by-2 phone hand layout is intentional: it preserves large, reliable touch targets instead of squeezing fourteen tiles into one row. The newly drawn tile remains slightly separated, including its hint count, and carries no border merely for being newly drawn.

On a phone held sideways, 500 pixels high or less, the game's bar stays at the top of the screen and leaves 264 to 374 pixels under it, so a call is laid out to show the whole hand and every call button there at once. While the hand is played, the player's choices are drawn in their seat, right under the hand, with the discards and called sets after them, so nothing lies between the hand and the buttons, and the hand keeps its place as the discards grow. The choices are drawn there rather than moved by CSS, so a screen reader and the keyboard go through them in the order they are seen, and a choice that has focus as the phone turns keeps it. The tiles stay 44 pixels wide on a phone under 640 pixels wide too. The discard on offer is staged in a card no taller than the buttons, its tile small, with Discard or Kan over who let it go, and the buttons follow it in the same rows, each the call's word, such as Chii, beside the call's tiles; the rest of the buttons' names ("Chii — 3–4–5 characters"), the card's other words ("discarded the 5 characters") and the prompt are still read out. The play area scrolls with the page there instead of sticking to the top of the screen, where it hid its own foot and the bar hid the hand, and an offered call brings the buttons into view with the hand above them, below the bar; from the foot of the page it scrolls back up to the hand, as far as keeps the buttons in view. Measured in Chromium at 568x320, 667x375, 740x360, 844x390 and 932x430, the whole hand and every button fit for a chii and Pass, two or three chii shapes and Pass, a pon, a chii and Pass, and a pon, a kan and Pass, late in a hand and beside a called set too; with more choices than fit, the buttons come first and the top of the hand goes under the bar. On the player's own turn, from 640 pixels wide, the whole hand is on the screen at the top of the page, as before. A finished hand and its result are laid out as before.

## The table at each size

How the table is laid out follows the window, its height as well as its width.

- **From 968 pixels wide and 900 high** the three opponents sit around the
  centre, Opposite above it and Left and Right beside it, their discards 27
  pixels wide.
- **Under 900 pixels high, or under 968 wide, as a tablet is,** from 761 pixels
  wide and over 500 high, the table is the dense one: the centre a bar across
  the top and the opponents in one row under it, Left, Opposite and Right,
  their discards 22 pixels wide. Where a seat is 300 pixels wide or more, its
  discards stand beside its name, score and called sets rather than under them,
  so the seat is only as tall as three rows of discards. Around its centre the
  table is 426 to 477 pixels high, and in a 1366 by 768 laptop's browser, 650
  pixels high, the hand lay wholly under the screen as the turn began; the
  dense table is 228. Measured in Chromium at 1366x650 and 1280x600, the whole
  hand is on the first screen, early in a hand (452 to 552 pixels down) and
  late in one against a riichi (484 to 584). Under 700 pixels high the page's
  rows also stand 8 pixels apart, not 14.
- **From 761 pixels wide the player's hand is one row.** Fourteen tiles of 52
  pixels need a window 889 wide, or 906 with a scrollbar. Under that the tile
  just drawn used to wrap onto a row of its own, so that the seat grew and
  shrank at every turn, on an upright tablet as in a narrow window. The seat
  is measured as a container, which counts a scrollbar and a zoomed page as
  they are, and the tiles are as wide as fourteen fit with three pixels to
  spare: 52 pixels from a seat 822 wide inside (a window of 892, or 909 with a
  scrollbar), 48 from 766, 44 from 710 and 40 under that. A browser without
  container queries steps by the window's width instead.
- **On a phone** each opponent has a third of the table's width, and in play
  their discards are as large as six to a row fit there, a riichi's tile, which
  lies a third wider, among them: 12.5 pixels wide at 320, 15 at 360, 16.5 at
  390, 17.5 at 412 and 18.5 at 430, where they were 14 on every phone and 11.5
  under 360. The seat is measured as a container and its width names the size,
  in steps; a browser without container queries keeps the old sizes, and a
  review's table its own. The rows have one pixel between their tiles there,
  and the ring around a dora is one pixel wide, so that it covers no neighbour.
  Larger ones would need the table laid out anew, since a third of a phone's
  width is all a seat has. The four modes' buttons show one word each under 620
  pixels wide, Play, Watch, Physical and Guided, in one row of 50 pixels where
  their whole names took 82; a screen reader still hears the whole names. The
  play area scrolls with the page: it used to stay six pixels from the top of
  the screen as the page moved, under the game's bar, which stays there too,
  and a play area taller than the screen held a call's buttons under its foot.
  While the result's chip shows, the page ends 56 pixels lower, so that its
  last line can be scrolled clear of the chip: the result tells the page
  that the chip shows, in play and in Agent watch. In the settings' dialog an
  opened section stays where it was pressed, rather than going to the foot of
  the list.

A window need not be a whole number of pixels wide: at a browser zoom of 125
per cent, 951 of the screen's pixels are 760.8. The bounds of a phone's width
and a phone's height held sideways end at 760.98 and 500.98 pixels, in the
styles and in what the page asks `matchMedia`, so that such a window is laid
out as the whole pixel under it; ended at 760 and 500, they left a window of
760.8 matched neither by them nor by 761 and 501, and the table, the seats and
the hand fell back to styles meant for no size. `media-bounds.test.js` holds
every such pair to that, and `ui-polish-check.mjs` a frame zoomed to 760.8 by
900 and to 1000 by 500.8.

Words in gold are written in a lighter gold than the borders and rings
(`--gold-text` in `app.css`): the borders' gold is 3.5 to 4.4 to 1 against the
table's panels, and small text needs 4.5. The honba and the riichi bets in the
table's centre are ivory, since the middle of the centre is the lightest felt
on the page, and a review's word for what a seat was doing has a backing of
its own. The other screens write their words in the same lighter gold: an
adviser's next choice, its estimate and its chosen row in Agent watch,
Physical agent play and a guided game, a guide's step over its prompt, a
palette's "Choose a tile", the chosen list and the adviser's move in a hand's
review, Trained AI's preference, and in the game's review the filter pressed,
Flagged on a line and in the step bar, and the AI tag in Trained AI's view.
In the borders' gold an adviser's, a guide's and a palette's were 3.6 to 4.4 to
1 on the light theme's panels, and Flagged on a line, under the gold wash of
the decision on show, 3.6; the reviews' others passed, on darker panels. In
the lighter gold they are 5.2 to 11.2 to 1, measured by pixels in both themes.
Words on the bare felt beside them are ivory at 0.9 or more, 4.7 to 1 on the
lightest felt there is. What went wrong with a Trained opponent, and why a
physical table cannot be analysed, are said in a red that reads on their
panel in either theme (`--accent-text`); the accent was 4.3 to 1 on the light
theme's pale panel. The first panel's buttons are filled; as plain controls
they were 2.3 to 1 in the light theme, and 1.3 under the pointer. A list
pressed, the hand review's or the game review's filter, is still told from
the others by its colour alone, as before, besides what a screen reader
hears (`aria-pressed`). Under a finger (`pointer: coarse`) the settings' and
the history's headings are 44 pixels high, and a yaku's name on a result 32.
`ui-polish-check.mjs` holds the layouts above, the panel's contrast, and
Flagged on a review's line and the review's two preferences in both themes;
`webapp-reliability-check.mjs` an analysed table's words in gold and the
refusal's red, and `guided-game-check.mjs` a guide's step.

## Tile faces

Classic, Matisse and Van Gogh have a picture for every tile. Unavailable saved
tile-face choices fall back to Classic without resetting other preferences.
The shared renderer also supports future incomplete sets: a tile without approved artwork shows its name
in text on the ivory face instead, so it plainly waits for its picture and never
borrows another set's. In the hand the number stands over its suit, or an
honour's first word over its second, in the words the tile is announced with.
On faces of 40px or less, as in discard rows, dora indicators and the physical
editor's entered tiles where its entry is wide (in a narrow one, on a phone,
they are 44px wide to be pressed), a number stands over its suit's letter as tile notation
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

- **Silver, broken into dashes:** discarding that tile leaves one shanten, one
  step from a ready hand. The dashes give it a shape of its own: silver and
  the green of the safety ring are equally light, and were one ring to a
  player who does not tell red from green.
- **Gold:** discarding that tile leaves tenpai, a ready hand (zero shanten).

These describe the resulting shape, not a promise of a yaku or permission to
declare riichi. Open hands also receive readiness markings. Only legal discards
can have these borders: pending calls, disabled tiles, hidden faces, melds and
completed-hand displays do not acquire them. The existing dora/ura-dora foil and
red border, safety and selection cues are retained; overlapping marks use the
existing striped ring so no cue hides another. The foil's sheen, and the white
dragon's reveal under it, are each a layer of their own while they move, so
that a frame of either paints nothing of the page around it again; a dora's
face is then put together apart from its neighbours, and where the screen
scales the page by a fraction (125%) it may sit a fraction of a pixel from
where a plain tile's would. The sheen holds still for
the tenth of a second or so in which a desktop's graphics card answers a batch
of a review, and goes on from where it was (see [WebGPU on
desktops](ONNX_RUNTIME.md#webgpu-on-desktops)).

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
each. A review's table adds one more at a decision that answers a discard: a
gold outline around the tile on offer, in the row of the seat that let it go
(see the whole-game review below).

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
the player has deliberately focused another control. The confirmation's
**Discard** and **Cancel** go as they are pressed, and the hand takes focus
from them too, with the keys on or off, rather than the page. With **Select
before discarding** on, only the tile a press selected is announced as
pressed; the arrow keys' marker, drawn the same way, is not. Browser shortcuts
and other focused controls remain unaffected: a key held down is cancelled in
the hand only where the hand's keys act on it (Enter, Space, the digits, r and
t), so a held arrow down, Page Down or Tab still scrolls the page or moves
focus on. Enter held down on a tile, a call or the confirmation's buttons
presses once, and so does a double click: its second press is let go where the
first has already acted, since the next decision is drawn where the last one
was (on a tile only while Select before discarding is off; with it on, a
second press on the selected tile is how it is thrown). The browser tells a
double click by its second press alone, so two deliberate presses at the same
spot inside the system's double-click time are one press too: throwing the
drawn tile turn after turn against Club, the second turn's press is let go and
must be made again. That is accepted as the price of one move a press.

The turn's prompt shows its sentence alone ("Your turn. Choose a tile to
discard."), with no line about the keys: the hand takes focus as the turn
comes, unless another control has it, and the tile guide gives the keys. A
screen reader hears the sentence from the prompt, a live region, as the turn
comes, and "Focus your hand to use arrows and Enter." as the hand takes focus,
from what the hand names as its description: a line of its own, out of sight,
which is no live region and repeats nothing of the prompt, so a turn is not
said twice. It holds the keys wherever they work: with **Keyboard shortcuts in
your hand** on, on the player's turn, and not on a touch screen.

While a Trained opponent waits for the network, the prompt says what it waits
for: the worker's own note while it loads the network, and otherwise where
the network stands (`offline.network`), "downloading the trained network
(108 MB) 37%" while its bytes arrive, the size being the played network's as
it is fetched, or "waiting for another window to download the trained
network". Nothing is said of a first visit's install, which the network does
not wait for, nor of the save's runtime, held at 99 while the opponents
already play. Only the words are in the prompt's live region. How far the download has got stands beside them, outside it, so
a screen reader hears the download once and not at every percent.

While that wait is the download, the page's or the worker's, or a turn
behind another window's, and never while the network only starts or
thinks, the turn's controls offer **Play Club meanwhile** under the prompt,
what it does in its title. Pressed, this match's Trained opponents are Club's
for the rest of the match, by the command **Continue with Club opponents**
records after a failure, and the download carries on. The button goes, and
focus goes on to the hand. The table shows Club, in its seats and in the
Opponents select, but the table the player chose stays their stored choice: a
stand-in is for one match. A stand-in never sticks, and never deals a long
wait behind the player's back. A new game goes back to the chosen table when
the network is at hand, which is when the page holds it (`offline.network`
'ready': stored, or bytes storage refused held for the opponents' worker) or
that worker has it loaded, and the stand-in ends. Where it is not at hand but
its download is under way, the page's (downloading, or waiting behind another
window) or the worker's by its note, the new game keeps Club, and the
stand-in holds. Where neither, the new game is the chosen table and the
stand-in ends: its first Trained move asks for the network, and the wait,
with **Play Club meanwhile**, comes again. The
stand-in ends when the player picks opponents themself, in the select or the
custom table, which is stored as any pick is. Nothing marks it in what is
saved: after a reload the page knows it by the two tables alone, the match
playing Club exactly where the stored choice says Trained, keeps both as they
are, and asks for the network again where it can be saved. In a browser that
cannot save it (`offline.supported` false) no worker runs after the reload
and nothing is fetching it, so the next new game is the chosen table, and
its first Trained move brings the wait. The stored choice is the table the
player's own record holds, seat by seat, as every page now writes it; a record
that holds none (one from before 810, which is read as Trained once) is no
choice to stand in for, and a saved Club match there stays the player's table,
as before. Club taken after a failure is not a stand-in either: it becomes the
stored table, as before.

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
the hand. The region is not marked busy meanwhile: a screen reader may keep
what a busy region says until it is busy no longer, and the line that says
the review is under way is in it. The heading is a second-level one, as the
result's own, the standings' and the review of the game's are. The review is a
part of the page beside the result, not a part of it, and on a phone the
result's sheet steps aside as the review opens, its heading out of the page's
outline with it: a third-level heading then followed the page's first, or the
standings', with no second between them.

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
| 844x390, a wide phone held sideways, Club reviewing, from the foot | any | 315 | 315 |
| 844x390, Trained AI still reviewing, from the foot | any | 139 | 140 |
| 390x844 | keyboard | 2 | 52 |
| 375x667 | keyboard | 183 | not measured |
| 360x640 | keyboard | 124 | not measured |
| 320x568 | keyboard | 171 | 52 |
| 568x320, a small phone held sideways | keyboard | 574 | 114 |

On a phone that scroll costs what lies above the heading. For **View table / my
hand** at 390x844 everything stays in view; at 375x667, 360x640 and 320x568 the
top of the table, its round, wall count and indicators, goes under the game's
bar, and at 568x320, where the result is still a sheet, most of the hand goes
too. On the final hand the standings lie between the hand and the review. On a
phone the page is already at them when **Review final hand** can be pressed,
the sheet having stepped aside for them as the game ended and its chip having
brought the result back (below), so the review's heading, right under them, is
a short move away; before, pressed from where the result had left the page, it
moved the page 343 to 847 pixels and took the hand off the screen. The figures
for the final hand on a phone are measured from the standings, by
`full-review-check.mjs`, which prints them, at the sizes it runs.

At 844x390 the two advisers differ. With Club's review, done as it opens, the
browser's own scroll anchoring moves the page as the review opens and leaves
the heading under the game's bar; the page then brings it out just below the
bar, 72 pixels from the top of the screen. With Trained AI's review under way,
a shorter review that says how far it has got, the page moves less, and the
heading ends lower, 229 to 248 pixels from the top, in view below the bar.
Where it starts matters too. Pressed where the button has only just come into
view, **View table / my hand** moves the page 124 pixels and the heading lands
358 pixels from the top. From there **Review final hand** moves it 358 pixels
with Club, or 112 with Trained AI. With Trained AI the figures also vary by
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
unchanged." Until the game is over every hand's result offers **Review game so
far**, and once it is over, on a phone, the last hand's result offers **Review
whole game** as well; when the match cannot be replayed for that review, the
result says so from the same line: "The game could not be replayed for
review:", why, and "Your saved match is unchanged." The line keeps the words
of the last press that failed until that press succeeds. The page's notice
line would not do: it sits at the top of the page, above the screen at desktop
width, and under the sheet on a phone, or at best above it, dimmed by its
shade, at the far end of the screen from the buttons. The result says it from
a status line of its own after its buttons, which waits there, empty and out
of sight, from the moment the result appears, since a screen reader hears a
status line only once it knows it. Where the result is not a sheet, the line
is brought into view should it fall below the foot of the screen; on a phone
it stays at the foot of the sheet with the buttons, whose row spans the sheet
from edge to edge, so that nothing scrolling beneath it shows beside them. The
standings and the settings' **Export game** say their own failures the same
way, beside their own buttons (below); the notice line now says only what the
trained network picker could not do.

The review's own status lines, that Trained AI is reviewing the hand, and that
Club reviews in its place while the network cannot be reached, come into the
page with the review. They wait in it, empty and out of sight, until the review
has been drawn and a tenth of a second more, before their words arrive, so that
a screen reader can hear them after the heading it reads out as focus arrives.
The first says only "Trained AI is reviewing your decisions…", once. How far
the review has got, and the download it may be waiting for ("3 of 14", "0 of 14
· downloading the trained network 37%"), is drawn beside it, on the same line,
and is no part of the status line: it changes at every decision and at every
percent, and a status line is read out whenever its words change.

Where Trained AI's review of the hand fails, the review says why in an alert,
with **Retry Trained review** and **Review with Club**. Either button goes from
the page with the failure it answers, so as it is pressed focus goes on to the
review's heading, where it came as the review opened, rather than dropping to
the page; the page moves to show the heading only for a keyboard's press.

With Trained AI as the adviser, each decision gives Trained AI's weight for its
own choice and, where the player chose otherwise, for the player's move; a
riichi gives its declaration weight and the discard's weight given riichi.
Nothing above the notes says what the weights are. The line that gives them
says it in its title, shown as a tooltip: "How strongly Trained AI prefers each
move, not a chance of winning", and, where the line has a riichi, that
declaring riichi and the discard given riichi are weighed as separate
decisions, not as one move. A tooltip shows only under a mouse, so the tile
guide in the settings says both, for a keyboard or a finger too, under
"Trained AI's percentages": "How strongly it prefers each move, not a chance
of winning. Declaring riichi and the discard given riichi are weighed
separately." Where one of the last hand's rules made Trained AI's move, whose
weight did not choose it, the line says "Trained preference: by rule", the
rule's line its title ("Last hand: every win keeps it first, so no riichi"),
which a screen reader hears in full; the player's move keeps its weight. The
tile guide says what the rules do, under "by rule": "In South 4, the last
hand, Trained AI skips riichi where every win keeps it first, and declines a
win that would leave it last."

A decision with one legal move, as every discard after a riichi is, was no
choice. Its line says **Only move** in place of an adviser's answer, whichever
adviser reviews and whether or not the answers are kept back, and it is left
out of the decisions matched, as the review of the game leaves it out of its
own figure, so that the two give a hand the same one. The count then says what
it counts ("1 of 4 decisions with a choice matched Club.", under **Every
decision (13)**); a hand without such a move reads as before ("13 of 14
decisions matched Trained AI 0.1.3000."). Trained AI's notes say which moves
were the only ones; Club's do not, so the page asks the engine how many moves
were open at each, while the hand is still there to ask.

The decisions are a list, which says so in so many words, since Safari does
not tell VoiceOver that a list drawn without markers is one. It scrolls in a
box of its own, at most 42% of the screen tall, and nothing in it takes focus,
so the box itself does, named "decisions", for a keyboard to scroll it: Safari
gives a scrolling box no focus by itself, which left a keyboard no way to the
list's lower lines.

The notes honour the setting that has reviews keep Trained AI's move back
(**Hide Trained AI's move in reviews** in Options, `reviewHideMove`, below), so
that this second way into the hand does not give away what the review of the
game asks the player to find. With it on, whichever adviser gives the notes,
each decision shows only the move made, edged alike whether the adviser agreed
or not, and **Show answers**, after the filter, adds what the adviser would
have done instead, with its preference, or Club's reason and numbers. Pressed,
the button reads **Hide answers** and keeps focus. How many decisions matched,
and the filter with its counts, stay: they say how far the player went wrong,
not what was better.

### The standings as the game ends

**Next hand** on the last hand brings the final standings in place of another
deal, and its button gives way to the caption **Final hand**, so focus goes on
to the standings' heading, which says where the player finished ("You finished
4th"), in the region "final standings", rather than dropping to the page. Where
the result is not a sheet, the standings appear just above it. On a phone the
result is a sheet over them. Left up, as it was after a tap, it showed nothing
of them: the game had ended with no place on the screen and no **Play again**,
and no scroll uncovered either. So there the sheet steps aside for the
standings however the button was pressed, leaving its chip, as it does for the
reviews; the chip, after the standings' own buttons, brings the result back
with focus. A finished match that the page is loaded on opens the same way on a
phone: the sheet stepped aside, its chip showing, and the standings at the top
of the screen, with focus on their heading. So does one that Play is drawn
again over, coming back from another mode: the sheet, out of sight, hands
focus to the standings rather than taking it, which would leave it on the
page.

Where the result is not a sheet, only a press by keyboard moves the page to the
standings' heading, to bring it into view with its ring, clear of the game's
bar. A mouse's click or a finger's tap there moves focus without asking for a
scroll, so the page is left as the browser leaves it when the standings come in
above the result, as it was before focus moved there: measured in Chromium at
1100x900, 1280x600, 1366x768 and 844x390, from where the result left the page
and from the foot of the page, it ends exactly where it did before. With the
page at its foot at 844x390, where a fling down to the result's buttons ends,
the browser's own scroll anchoring moves the page 233 pixels as the standings
come in; the Final hand caption that took the pressed button's place stays in
view, and the standings' heading lies above the screen, where a screen reader
still reads it out. A scroll to the heading there would take the caption the
player pressed off the screen. On a phone the page moves for every press, since
what the press brought was under the sheet: for a keyboard by as little as
brings the heading into view with its ring, clear of the bar and the chip, and
for a tap to the heading as near the top of the screen as the page scrolls,
below the bar, with the places and the standings' buttons under it, and no
ring.

Measured in Chromium, a press by keyboard moves the page 30 pixels at 390x844,
203 at 375x667, 138 at 360x640, 202 at 320x568 and 586 at 568x320. A tap moves
it further, to where the standings are whole on the screen: 284 pixels at
390x844, 517 at 320x568 and 728 at 568x320, their heading as near the top as
the page scrolls, which on a tall phone is not to the top, the standings being
the last of the page. At desktop
width and at 844x390 the heading is already in view where the result left the
page, and the page does not move; from the foot of the page at 844x390 the
keyboard's press brings the heading in below the bar, 255 pixels above where a
click leaves the page.

Agent watch hands focus on the same way as a watched game ends, and still as
play did before on a phone: its result steps aside for the standings only when
the keyboard pressed the button. There **Next hand** on the
last hand is pressed on the result or among the watch's controls, whose button
plays the followed agent's choices and deals the hands. Either brings the
standings, the result's button giving way to its **Final hand** caption and
the controls' becoming one that cannot be pressed, and either used to leave
focus on the page. Focus now goes on to the standings' heading ("Followed
agent finished 2nd"), and only a press by keyboard moves the page to it, the
result stepping aside for it on a phone, with its chip after the
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
On a phone the result is a sheet that has stepped aside for the standings as
the game ended, so a finger or a mouse reaches their buttons. Once its chip has
brought the result back over them, a keyboard or a screen reader still reaches
them beneath it, so a press there that fails has the sheet step aside, each
time, for the words and the button with focus to be seen, and the chip brings
the result back. The buttons and the
line are brought into view by as little movement as will, clear of the game's
bar and the chip: measured, the page does not move for any press at 1100x900,
1366x768 and 844x390; on a phone, where the standings are the last of the page
and the line comes in under the chip, it moves 34 to 42 pixels at 390x844
and 33 to 61 at 320x568 for any press, a finger's, a mouse's or a keyboard's beneath the
sheet, to bring the line out from under the chip. Once the game is saved, or its review opens, from any
button, every line that said it could not be takes its words back.

### Final-hand review and saves

Opening final standings does not start another hand, so it does not clear the
final hand's event history or an already open review. Review final hand and
Save final hand remain available below the standings, including after reload.

The whole game can be saved as one mjai log as well,
`riichi-game-<date>-<time>.mjai.jsonl`: Save game so far on the score screen
between hands, Save whole game on the final standings, where the log closes
with end_game, and Export game in the settings during play. On a phone the
final hand's result sheet, once its chip has brought it back, covers the
standings, so it offers Save whole game itself. The log opens at East 1 and holds every finished hand, numbered as the
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

**One screen.** **Review whole game** on the final standings, there again after
a finished match is reloaded, puts every decision the player made in every hand
to Trained AI, the final hand first and then the rest from the first deal, and
opens on that hand, at its first decision, on one screen headed "Whole game". A
step bar at the top goes from one decision to the next that its filter keeps,
and gives the move made there beside Trained AI's; under it is the decision,
with the list of hands and decisions in a column of its own on its left where
the screen is 1100 pixels wide and 600 high or more, and elsewhere folded
between the step bar and the decision ("Hands · Flagged (46)"). The decision is
the table as it stood, every hand face up by default, the player's own seat and
Trained AI's view. No list of cards stands between the player and a decision,
and no **Back to the list** between one decision and another; **Back to the
standings** is the screen's one way out. The hand review reviews one hand and
is gone at the next deal; this one is rebuilt for the whole game. On a phone
the final hand's result is a sheet that steps aside for the standings as the
game ends, and that its chip brings back over them, so the sheet offers the
review too, as it offers the game's file. The sheet steps aside whenever the
review opens, as it does for the hand's review, whichever button opened it: its
own, or the standings' own, which it covers but a keyboard or a screen reader
can still reach. **Back to the standings** then lands on the standings,
uncovered, with focus on their own **Review whole game**, and the result is a
tap away on its chip, which a keyboard reaches after the standings' buttons.
The chip goes as it brings the result back, so focus goes to the result, as
when it first appeared, rather than to the page. Up to 480 pixels wide the
sheet's buttons take their short labels, **Review hand**, **Save hand**,
**Review game** and **Save game**. Measured in Chromium, they then take two
rows over the result from 360 pixels up with Windows' fonts and from 390 with
one as wide as Verdana, and three on narrower screens.

**After every hand.** Until the game is over, every hand's result offers
**Review game so far** (**Review game** up to 480 pixels wide) after the hand's
own buttons, at every size, since nothing else offers it before the end. It
opens the same screen on the match so far, headed "Game so far", on the hand
just played, at its first decision, and Trained AI is asked about that hand
first, then about the rest from the first deal. The headline above the step bar
says how far the review has got, and once every hand is reviewed it says so,
the whole game's ("All hands reviewed: 75 decisions."), and is no longer shown.
**Back to the result** brings the table and the result back as they were left,
with focus on the button that opened the review; on a phone the sheet is up
again rather than stepping aside, the button in its row at its foot. Where the
player was, `{ hand, step }`, is kept for the match (App's `reviewPlace`): the
review opens there again, from the result or from the standings, and only where
it has been taken on to hands played since does it open on the newest of them
instead, the hand just played, or the final one. Nothing else moves it: not the
hands reviewed later, nor another filter. A new game, or a reload, lets it go
with the review. The review is built at the first press and taken on to each
later result, so a hand reviewed once is never put to the network again. **Next
hand** stops it, since the next deal's Trained opponents ask the same network
and go first; opened again, it goes on with the hands still waiting, the hand
just played first, and at the end the standings' **Review whole game** goes on
from it the same way. The game so far stays out of the trend, and says so: a
game joins it once the match is over and every hand is reviewed, which for a
game whose every hand was reviewed as it was played is the moment the last
**Next hand** ends it. Nothing of it is saved with the match, so after a reload
a press asks about each hand so far once more. **View table / my hand** stays,
and still opens the hand's own notes under the result; up to 480 pixels wide it
reads **View table**, so that the fifth button adds no fourth row of buttons
over the result on the narrowest phones. While those notes, or the last hand's
beside the standings, ask Trained AI a decision at a time, the review waits for
them before each batch it sends, and then goes on where it was: the hand under
review, the hands waiting and the player's word stay as they were, so nothing
need start it again, the review of the whole game included. The page's one
worker answers its requests in turn, so notes asked between the review's
batches would wait behind one at every decision: a hand of fifteen decisions
would take twenty seconds rather than two or three. Only the first note can
wait behind a batch, the one already sent, whichever of the two was opened
first, and its twenty seconds run from when the worker begins it, so the wait
slows it but cannot make it run out of time. [UI reliability](UI_RELIABILITY.md)
says how requests are timed, and how the review sizes its batches to the
device.

**How it is built.** Nothing is added to the saved match, and its format is
unchanged. The finished match is replayed from its saved commands, from the
first deal, on an engine of the review's own; the replay must end exactly where
the match did, or the review refuses to start and says why ("The game could
not be replayed for review: The replay did not end where the match did. Your
saved match is unchanged."). A replay that ends where the match did, but in a
hand still being played, is refused too ("A hand is still being played"),
though a review opens only at a result. While the match is under way the
review keeps the engine it scanned the match's hands on, and at each later
result scans on from the last one, not from the first deal again; a scan that
cannot go on gives way to one from the first deal, and a finished match keeps
none. At the end of each hand the review
asks the questions the hand review asks (`review-policy.js`), eight positions
a forward pass, fewer on a slow device: Mortal's planes for each decision as it
was made, then, where the move is a riichi, the network's or one decline-last
left, which tile it would discard. The questions Trained AI hesitates over are
asked again of the game with characters and circles renamed, in a forward pass
of their own, and judged by the mean of the two answers' logits, and the last
hand's rules are played over that ([Agent modes](AGENT_MODES.md)). A move
with one legal choice, such as the discard after riichi, is not asked and is
left out of every figure. There too, where its replay stands at the hand's
end, the review writes the table at each of the hand's decisions as a typed-in
position (the engine's `review_position`), which its reasoning about a
decision changes and asks about again (below): written when the reasoning
first wanted one, each hand took a replay of its own from the first deal,
while the page drew. A hand whose tables cannot be written is reviewed all the
same, and the reasoning says nothing of its decisions.

**What is flagged.** A decision is flagged when the player's move had under 5%
of Trained AI's weight while its own choice had 60% or more, so a network that
is not decisive flags nothing. Where one of the last hand's rules made Trained
AI's move, whose weight did not choose it, the decision is flagged only where
decline-last declined a win and the player took it: declining changed the
place in 65 of 2610's ruled deals, 64 of them for the better. Any other move
there is listed under Differed only, a riichi the dama rule threw undeclared
among them: that rule changed 126 deals, 38 of them for the worse. The
**Flagged** filter's title states the rule ("Your move under 5%, Trained AI's
choice 60% or more, or a last-hand win that leaves you last"), shown as a
tooltip and read as the button's description, so it is at hand without taking
a paragraph of the screen. A tooltip shows only under a mouse, so the tile
guide in the settings says it too, for a finger or a keyboard, under
"Flagged": "In a review, a decision where your move had under 5% and Trained
AI's choice 60% or more, or where you took a last-hand win that leaves you
last." Riichi is judged in two stages, as the weights are
shown everywhere: when the player and the network both declared, the tiles are
compared by the discard weights given riichi; otherwise the move itself is
compared, the declaration carrying its declaration weight. Flagged decisions
are not ranked: every list is in the order played, hand by hand, so none is
reshuffled under its reader as later hands are reviewed, and every flagged
decision departed far already, so a ranking mostly reordered like with like. A
flagged decision's line in the list has the accent's edge, and says **Flagged**
where the filter keeps more than the flagged decisions; the step bar says it
after both moves.

**The list and its filter.** The list gives every hand of the game in the order
played, each headed by its name, with its honba where the hand repeats one, so
no two hands share a name; what it changed for the player ("+3,900", "−8,000",
nothing where it changed nothing); and how it ended ("West wins on North's
discard"). A hand not reviewed yet says how far its review has got ("Reviewing,
5 of 15", "15 decisions, not reviewed") and offers **Review** ("Review: East 2"
to a screen reader), and a hand the player made no decision in says "No
decisions". Under each hand is a line for each decision the filter keeps, a
button to it: its number, then the player's move and Trained AI's in tiles, "12
[R] → [8s]", a tick where Trained AI agreed, "Only move", "1 on [7p] Pass →
Pon" for an answer to a discard, dots where Trained AI has not answered yet,
and **Flagged** at the end under Differed and All. Its name says the same in
words ("Decision 12. You: discards the red dragon. Trained AI: discards the 8
bamboo. Flagged."). A discard is its tile alone; a riichi and a quad declared
from the hand are the word and their tile, a chii the word and its run, and any
other move its word. A hand's name goes to its first decision the filter keeps,
or to its first decision where the filter keeps none there. The decision on
show is the current line (`aria-current`), edged in gold, and one the filter
leaves out is drawn in its place all the same, dim, until the player moves on.
The filter, under the trend at the top of the list, keeps the flagged
decisions, those where Trained AI chose differently, or every decision, each of
its three buttons saying how many it holds: "Flagged (46)", "Differed (59)",
"All (75)", the last counting the hands not reviewed yet too, whose tables and
moves are known before Trained AI answers; Differed's title says what its word
stands for ("Trained AI chose another move"). The button pressed says so
(`aria-pressed`) and keeps focus, and a status line, in the page from the
start, says what the filter now keeps ("59 decisions where Trained AI differed,
46 of them flagged.") and goes quiet once focus leaves the filter, or once the
hands reviewed since have changed the counts it gave, so it never stays to say
a count the list has outgrown. The filter is what Previous and Next step
through, and is kept as the player leaves it with the other preferences
(`reviewFilter`), in the next game and after a reload. On a screen 1100 pixels
wide and 600 high or more (`REVIEW_WIDE`) the list is a column of its own, 300
pixels wide, on the left of the decision, which comes first in the page, so
that Tab reaches Previous and Next before the list; the column stays on the
screen as the decision is scrolled, scrolls on its own, and is headed "Hands"
for a screen reader. Elsewhere it is folded under the step bar, saying its
filter ("Hands · Flagged (46)"), and open, its hands scroll in a box of their
own, at most 60% of the screen tall, so that the decision is never far below;
it starts folded and stays as the player leaves it, open or folded, across
steps, hands and the hands reviewed meanwhile. Either way the list keeps the
line of the decision on show in view by scrolling its own box, never the page,
which `scrollIntoView` would move as well: a step brings the new line into view
by as little as will, below the bars that stay over a folded list at the top of
the screen, and so does the list as it opens, and once more after the page
around it has finished easing, which with reduced motion takes a frame or two
after a change of the screen's size. Where hands reviewed meanwhile or another
filter draw the list again, what the player is looking at stays where it is, as
a browser's scroll anchoring keeps it: the control in the list that has focus,
else the line on show, else the first line in view, or the top of a list at its
top. Safari has no scroll anchoring, so the boxes have none, and the list does
it itself. Its lines are kept by their hand and number, so that lines of other
hands arriving as they are reviewed never take one away, or keyboard focus on
it. Where the screen changes across that size, a control of the list that had
focus has it again where the list is drawn now.

**Every hand, face up.** At every decision the table is shown as it stood,
every player's hand face up unless the player keeps the others' back (Hidden
answers, below), with no control on the table to hide or show them: each
opponent's concealed tiles in their seat, the tile just drawn apart, above
their sets (the called tile turned) and their discards in order (the riichi
tile turned, the tiles called away marked), the seat headed by its wind, where
the opponent sat and who played them, the dealer, the score and riichi; the
round, the honba and riichi sticks, the dora indicators and the tiles left in
the wall in the centre; then the player's own seat, and what happened since the
player's previous decision ("Since the previous decision (3)", "Since the deal
(2)"), folded until it is opened. On a phone held upright the player's seat
comes first, so that the player's tiles are the first thing under the step bar;
held sideways, and on every larger screen, it comes after the table, as in
play. It is drawn in one place or the other rather than moved by CSS, so that a
screen reader and the keyboard go through the decision in the order it is seen;
a control of the seat that has focus keeps it as the screen turns. The seat
whose move it was is edged in gold and says so in words: "To play" on the
player's own turn; when the player answers another's tile, the edge is on the
seat that let it go, which says "Discarded" or "Declared a kan", and the
player's seat says "To answer" without one. The tile on offer is outlined in
the same gold at the end of that seat's discards, in the gap the row leaves
between tiles, so nothing moves as it comes and goes, and its name adds "the
tile on offer"; a quad offered to be robbed holds every copy of its tile, so
there no row has it to outline. Each hand is a list named for its seat
("South's tiles, Right (Club)", "East's tiles, yours"). The tiles are the
review's own: the replay the review is built from reads them at each decision
once the hand is over (`scanMatch`, through the engine's `review_hands`), never
from the match being played. The engine refuses them while a hand is still
being played, as it leaves such a hand out of a log.
`web/tests/game-review.test.js` rebuilds every hand at every decision of two
fixture games from the hand's mjai log, through calls, quads, riichi and wins,
and the engine's own test rebuilds them through Mortal's reader of the same
events. An opponent's tiles take a row of fourteen with the drawn tile's gap,
on a darker rack that sets them apart from the discards below, as large as
their seat has room for, up to 24 pixels wide. Where one row would draw them
under 18 pixels wide, they take two rows of seven instead, the drawn tile last,
as the player's own hand does on a phone. From 761 pixels wide, at every height
over 500, the table is a dense one: the centre a bar across the top, the three
opponents in one row under it, Left, Opposite and Right, each seat as tall as
what it holds, and their discards 22 pixels wide. The player's own tiles keep
one row there too, as large as the decision has room for beside the list: 52
pixels wide where it is 860 pixels wide or more, 42 from 715, 40 from 685 and
38 below (`--mine-tile`, set by container queries on the stage), since the
hand's fourteen tiles, their gaps and the seat's padding take fourteen tiles
and 125 pixels, and a window 761 pixels wide leaves the decision 678 beside a
classic scrollbar. A wide screen's table, the opponents around the centre, left
most of the screen empty felt and pushed the player's seat below it. On a phone
held upright, three seats side by side would leave each hand a few tiles to a
row, so there the opponents take the width of the screen, one under another in
the order they play after the player, their discards a size larger than in
play. The table is drawn afresh at a change of hand; the rest of the decision
stays. Its seats show riichi but never the word for a call, which play shows
for a moment as it is made: the table is the one at the decision, its sets
show what was called, and a word on a timer would still claim a call a step
back and move the table as it went. The browser check goes through every step
of a hand at 1100 by 900, 1440 by 900, 1366 by 768, 800 by 1280, 390 by 844,
320 by 568 and 844 by 390: every tile is on the screen, an opponent's 18 pixels
wide or more on one row or on two of seven, the dense table's opponents in one
row under its centre and the player's tiles in one row, the player's seat first
only on a phone held upright, the step bar on the screen with the player's
tiles, scrolled to, clear of it, and no page scrolls sideways. Measured over
every step of that hand, in Chromium, in pixels:

| Screen | Opponents' tiles | Player's tiles | Table and player's seat, tall |
| --- | ---: | ---: | ---: |
| 1440x900 | 24 | 52 | 617 to 719 |
| 1366x768 | 24 | 52 | 617 to 719 |
| 1100x900, beside the list | 24 | 42 | 565 to 743 |
| 800x1280 | 24 | 42 | 600 to 743 |
| 390x844 | 22 | 39 to 41 | 714 to 996 |
| 320x568 | 24 | 29 to 31 | 811 to 1044 |
| 844x390 | 24 | 44 | 581 to 658 |

**Trained AI's move, on show.** Wherever Trained AI was asked, its move is on
show: in the decision's line in the list (above), in the step bar, over its tile
in the hand and in its view. The step bar gives both moves with Trained AI's
weights: "You [R] 2.0% · Trained AI [8s] 95.5% · Flagged", "You [3p] 87.0% ·
Trained AI agreed", "You [5s] · Only move", "On South's [7p]: You Pass 0.3% ·
Trained AI Pon 99.0%" ("On South's kan [7p]:" where the player could rob a
quad), "You [8p] · Trained AI …" in a hand not reviewed yet, and, where both
declared riichi, "You Riichi [3s] 2.0% · Trained AI Riichi [6m] 90.0%, given
riichi". Where one of the last hand's rules made Trained AI's move, whose
weight did not choose it, "by rule" stands in that move's weight's place, the
rule's line its title: "You Riichi [9p] 97.0% · Trained AI [9p] by rule", and
"You [9p] by rule · Trained AI agreed" where the player made the rule's move
too. Up to 480 pixels wide "AI" stands for "Trained AI". With hints on, a
tile of the line that is dora is marked so, as the same tile is in the list,
in the hand and in the view. A status line, in
the page from the start, says the step in one line as it changes ("East 4,
decision 12 of 15. Your turn, 36 tiles left. You: discards the red dragon,
2.0%. Trained AI: discards the 8 bamboo, 95.5%. Flagged."), the table in the
words the table itself uses for the wall and a riichi, and a rule's line in
full ("Trained AI: discards the 9 circles, by rule (last hand: every win keeps
it first, so no riichi)."). In the hand, "You" stands over the tile the player
threw and "AI" over Trained AI's, which is ringed in blue as Agent watch and
Guided play ring the agent's choice, as the tile guide's blue ring says ("You
· AI" where they agreed); a screen reader hears them after the tile's name
("your move", "Trained AI's choice"). Every tile keeps the line for them, and
none lifts, so nothing in the hand moves between steps. Trained AI's view, at
every decision put to it, lists every move by its weight, each as its tile or
a word and its tiles, the riichi discards under their declaration, tagged as
the hand is ("AI", "You"); a screen reader hears each in the engine's words
("discards the 8 bamboo, 95.5%, Trained AI's choice"). Where both declared
riichi it says once that the tiles are compared given riichi; where one of the
last hand's rules made Trained AI's move, agreed or not, it gives the rule's
line instead ("Last hand: every win keeps it first, so no riichi."), over
weights that stay the network's. It leads with the five heaviest moves and
those it tags; **Show 3 more moves** shows the rest and stays, as **Show fewer
moves**, so focus stays on it. The tile guide in the settings says what the
weights are, under "Trained AI's percentages": how strongly Trained AI prefers
each move, not a chance of winning, and that declaring riichi and the discard
given riichi are weighed separately; and, under "by rule", what the last
hand's rules do. For a turn, "What each move leaves", folded until it is
opened, sets out what each of the two moves leaves, a column each ("You",
"Trained AI") where the review measured it, counted as the hand review counts
it: the hand left, the unseen tiles that improve it, whether the tile could
deal into a declared riichi, and its worth as dora (the engine's
`review_measure`). A move that throws the tile of a column before it, a
riichi beside the plain discard of its tile, has no column, since its numbers
would be the same, and the table, needing two, is then not shown. It stays
open or folded as the player leaves it. Nothing is played on from any
decision, and the saved match is never touched.

**Why, and what would change its mind.** Where the player threw one tile and
Trained AI chose to throw another, and no last-hand rule made its move, whose
weights then did not choose it, its view ends with the review's reasoning
about the two (`review-insight.js`), which App keeps for the match beside the
review (`MatchInsight`). First the Why line: the engine's fact that separates
the two tiles, with its verdict. **Why** where Trained AI puts 95% or more on
the tiles that share the fact and, where the fact has a clean change, taking
it away halves its lead; **Partly** where a check fails; **Not explained**
where no fact favours its tile ("Partly: You had already thrown a 7
characters, so a wait on it would be furiten. Trained AI puts 93.0% on tiles
you had already thrown."); **Not checked** where the check could not be run,
the fact said all the same: Trained AI, asked the table again, answered it
otherwise than the review recorded, or could not be asked. Partly, Not
explained and Not checked are buttons whose title, a tooltip, is the check's
workings or why it was not run ("Asked again, Trained AI answered this table
differently, so the check was not run."), and which show them under the line
when pressed, for a keyboard, a finger or a screen reader. A line that fails
before it has its fact says why ("Not checked: The trained opponent did not
answer in time."), and a line that could not ask Trained AI asks again when
its decision is next shown. The line is worked out as the decision is shown,
in the page's engine and at most two forward passes, the second for its
tables Trained AI hesitates over, renamed, and drawn with what follows it
once it is had, so nothing moves under a pointer. Then **What would change
its mind?**, with about how long the search takes, from the tables it asks
first to the most it may ask, each 1.43 positions on average, those Trained
AI hesitates over asked twice, at the review's own time a position as it
stood when the decision came on show, so the words do not change under the
player ("What would change its mind? · about 8–14 s"; one figure where both
round alike, "about 1 s", and minutes from a minute, "about 4–5 min"). Where
the review's batches have moved to the graphics card or back since that time
was measured, the review has none until it measures one there, the page has
it tell its screen, and the button says no time meanwhile, for the decision
on show too: a time from the other runtime would be a hundred times out. It asks Trained AI about the table changed in one
thing the player could see, a tile of the hand, a discard, the dora
indicator, a riichi, the score or the round, only as a game can reach it, and
never the wall, the dealer or a call taken back; pairs of changes only where
no single change flips the choice. Pressed, it gives way at once to its
progress, counted against the most it may ask, so the bar never runs back
and fills early where no pair is needed ("Trying changes · 8 of 64"), and
**Cancel**, focus going to Cancel, and then to what it found, focus going to
its first line: "One change and it throws your west wind:",
then a line for each kind of change that does, a faster hand, a hand worth
more, what is safe, the shape of the hand and the table, each with how many of
its changes did of how many were tried ("A faster hand (7 of 7 tried): if
your white dragon had been a 2 circles (your hand a step closer to waiting
and one safe tile fewer): your west wind 99.6%, the 7 characters 0.2%."), and
what moved nothing ("Not moved by the score (±16,000) or the round."); or the
pair that does ("No single change does it. Two together: …"); or "Nothing
asked makes it throw your west wind." with the nearest change. The search
judges a changed table by the network's own answer, so one where a last-hand
rule would decide, a riichi any of whose tiles would keep the player first or
a win that would leave them last, is not judged at all, and its kind of
change is left out of what moved nothing. A status line
says the search began and, unless focus went to it, what it found. The button
and Cancel take each other's place, so Enter held down on either presses it
once, and the second press of a double click or tap, which lands on the one
that took the first one's place, is let go. The button is offered at every
such discard, on a phone too, but not where Trained AI
chose its tile by more than e^7 to one (a margin below −7, where nothing
small moved it in the prototype), not at a call or a decision to declare
riichi or not, not while Trained AI cannot be reached, and not where the Why
line found that Trained AI does not answer the decision's table as the review
recorded. A search finding that first says so ("Asked again, Trained AI
answered this table differently, so no change was tried."), and the Why line
is then Not checked too. A table is answered as the review recorded where each
move's weight is within a thousandth of the review's, the bound the graphics
card's answers are held to against WebAssembly's before the card is given a
review's batches ([WebGPU on desktops](ONNX_RUNTIME.md#webgpu-on-desktops)): the first hand of a visit is
reviewed on WebAssembly and asked again on the card once that serves, so a
tighter bound here told a card the page had let through that it answered
differently. A search asks 23 to 61 tables, those Trained AI
hesitates over twice (102 of the recorded searches' 237): 5 to 14 seconds at
desktop Chromium's 157 ms a position, 16 to 44 seconds at half a second. A
step to another decision, **Cancel**, closing the review, the next deal and
leaving Play stop a search under way, which is forgotten; what was found
stays with its decision alone, shown whenever it is again and never asked
again. While the reasoning asks, the
review of the rest of the game waits at its gate, as it waits for the hand's
notes, for the whole of a Why line or a search, until the worker has
answered its last pass. Each of its passes, as many tables as take about as
long as the review's batch, those it hesitates over counted twice and asked
again in a pass of their own, waits for any batch with the worker, and the
review sends a batch in the turn
it finds its gate open, so neither ever waits in the worker's queue for the
other. Being part of Trained AI's
view, all of it waits with the view while the settings keep Trained AI's move
back: nothing of it is drawn, said or asked until the answer is shown. It
explains Trained AI's choice, not good play, and changes only what the player
could see, never a hidden hand.

**Hidden answers.** Two settings make the review a quiz, both off unless the
player turns them on: **Hide Trained AI's move in reviews** and **Hide other
players' tiles in reviews**, after the other checkboxes in Options, and from
761 pixels wide in the review's header too, without "in reviews". Both places
change the same settings, stored with the other preferences (`reviewHideMove`,
`reviewHideTiles`). With the first on, a decision Trained AI answered, but for
an only move, keeps its answer back: the step bar reads "You [R] · Trained AI ?
**Show answer**", without the player's weight, which would say how Trained AI
rated it; its line in the list reads "12 [R] → ?", heard "Trained AI's move
hidden", as the step's line says; no tile is ringed or tagged "AI", and Trained
AI's view waits. The flag, the filter with its counts and the agreement figure
stay: they say how far the player went wrong, not what was better. The decision
is put to the player instead, as in play: every legal discard in the hand is a
tile to press, every other move a button under the hand in play's words
("Pass", "Pon — triplet", "Riichi — discard the 3 bamboo"), and the hand says
"Pick a move, or Show answer." A move picked answers the decision and is kept
as **Now**: in the step bar ("You [R] 2.0% · Now [8s] 95.5% · Trained AI [8s]
95.5%"), over its tile, in the step's line, in Trained AI's view and, where it
is neither move and the review measured it, in a column of "What each move
leaves" between "You" and "Trained AI". With the second on, the three
opponents' tiles lie face down in their racks at every decision, the same size,
the drawn tile's gap kept, named "South's tiles, Right (Club), face down" and
nothing more, until the decision is revealed; where only they are kept back the
step bar offers **Show tiles**, and **Show answer** shows both. The racks turn
over where they lie, and the table stays where it was on the screen: the step
bar keeps its height as its button goes, and where the question under the hand
lies above the table, on a phone held upright, the page moves by as much as it
took. The key s does what the step bar's button does, and while the hand is
live the digits pick its first nine tiles and 0 the one just drawn, as in play.
A reveal takes away the control that made it, the button, the moves under the
hand or the tile, so focus goes on to Next, or to Previous at the end; a key
leaves focus where it was, unless the control that had it goes. Only a
keyboard's press moves the page to show Next: on a screen 500 pixels high or
less Next is often far above, and a pointer's press leaves the hand it pressed
on the screen. Enter held down presses once, so it does not step on past the
answer it showed. A reveal is kept for the match (App's `reviewRevealed`, by
"hand:step"), through steps, closing and opening the review, and hands added
later; a new game or a reload asks again. A setting turned on keeps back only
what the player has not revealed, and turned off shows everything. Show tiles
in a hand not reviewed yet leaves its move to be answered once Trained AI has.
Nothing kept back holds Next.

**Stepping.** The step bar stays at the top of the screen, under the game's bar
on a phone, except on a screen 500 pixels high or less, where it would take a
third of it. **Previous** and **Next** go to the decision the filter keeps
before or after the one on show, across hands, from wherever the one on show
is, the filter's or not; they are named for the filter ("Next flagged
decision", "Next decision where Trained AI differed", "Next decision") and
disabled only at the two ends of the game so far. While the hand on show is
being reviewed, the flagged and the differing decisions hold nothing of it yet,
so Next waits for its answers rather than passing over them: it is said to be
disabled and drawn so, but keeps focus (`aria-disabled`), so that a player who
tabbed to it is still on it when the answers arrive, not moved to Previous; All
steps through it at once. Beside them is the place ("East 1 · decision 3 of 7",
"East 1 · 3 of 7" up to 480 pixels wide). With "Keyboard shortcuts in your
hand" on, and not on a touch screen, as play's keys are, ← and → step as
Previous and Next do, Shift+← and Shift+→ go to the hand before or after, at
its first decision the filter keeps, s shows what a decision keeps back, and
Esc goes back to the result or the standings. While the hand is live (Hidden
answers, above) the digits pick a tile of it, and in it ← and → move along the
tiles, as in play, rather than step. A key is left alone while a dialog is open
over the page, the settings' or the table's discards, with Ctrl, Alt or Meta
held, while text is composed, in a field, or once something else has handled
it; screen readers in browse mode keep their own arrows. Keys never move focus
themselves, unless the control that had it goes, as a tile of the live hand
does once a key answers the decision. Previous and Next keep focus, and a step
to either end hands it to the other, without moving the page; a line of the
list keeps it as the decision on show changes. A step that takes away the
control that had focus, such as the dim line of a decision the filter leaves
out, or **Show 3 more moves** at a decision with fewer, hands it to Next, or to
Previous at the end, never to the page. A control focused below the step bar is
brought into view under it, never beneath it, also when a step makes the bar
taller. The decision stays on the page across steps and hands, so what the
player opened or folded in it stays as they left it. The tile guide's Keys
entry gives the review's keys.

**Agreement and its trend.** The figure is the share of decisions with a choice
where the player's move was Trained AI's, counted inside the trend ("16 of 75
decisions with a choice"). It opens the list, folded to one line ("Matched
Trained AI 0.1.1110: 21% so far", without "so far" once the game is whole), the
rest inside, and stays open or folded as the player leaves it. When every hand
is reviewed, the game enters a trend kept in this browser only
(`riichi.review-trend.v1`, the last 50 games; a browser that keeps nothing
still reviews, without a trend). The figure is shown against the
decision-weighted average of up to ten earlier games, with a line of the last
twelve and a table of the same. The change is stated, not coloured: matching
Trained AI is not the same as playing well, since the network has habits of its
own. The review is judged by the version of Trained AI the page plays, which
the player can pick in the game settings, and the trend names it ("Matched
Trained AI 0.1.1110"). Another version judges differently, so each game is kept
with the version that judged it, a game judged again by another version is kept
beside its first entry, and the figure, the line and the table take only the
games of the version playing now, judged since the review averaged over the
suits; a note counts the others ("Not counted: 2 games judged otherwise."),
and another version's come back when it is picked again.

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
with 0.1.810; 0.1.1110, 0.1.1410, 0.1.2610 and 0.1.3000, the defaults since,
have not been measured so.

**Speed.** Measured on 14 October 2026 in Chromium 155 on the desktop, through
the page's own worker on one thread, a position took a median 163 ms one at a
time and 157 ms in batches of eight, with identical weights; natively the same
batch nearly halves the cost. A decision takes about 1.1 positions, none where
its move is forced and two where Trained AI hesitates, so a game of 150 to 200
decisions takes about half a minute there; phones were not measured. A device
the review measures at over 1.25 seconds a position is asked fewer at a time,
as many as take about ten seconds and at least one, so that on a device that
takes up to ten seconds a position a batch ends well within its deadline and
nothing waits long behind it ([UI reliability](UI_RELIABILITY.md)). The
positions asked again, renamed, go in a batch of their own, so a hesitant
group never counts as a slow device. The review runs in the
background with its progress and an estimate of the time left ("3 of 6
decisions · about 2 minutes left"), which counts every position a decision has
taken, goes on while its screen is closed, waits
before each batch while a hand's own notes or its own reasoning about a
decision ask Trained AI, which counts neither in the estimate nor against the
device, and stops at **Stop** or when the player leaves Play. A device that
takes over half a second a position holds after each hand
("Reviewed 1 of 3 hands; about 0.8 seconds a decision on this device.") and
offers **Review next hand** or **Review the rest**; any hand can also be
reviewed alone by **Review** in its header in the list, which opens by itself
once, when the review turns out that slow. A hold is made for a device measured
slow, so it is lifted where the review's batches move to the graphics card
meanwhile, or back from it, as on the passes of a search pressed during the
hold: the time measured says nothing of where they run now, the review has no
time a decision to offer the next hand by, and it goes on by itself with the
hands queued, holding again after the next only if it measures the device slow
there too. After that the list, like the trend,
stays as the player leaves it, open or folded, as the review goes on. These
buttons replace one another as the review goes, so focus is handed on rather
than dropped: from **Stop** to **Continue** and back, from a hand's **Review**
to the hand's name once that hand is under review, and to Next, or Previous at
the end, when nothing is left to press. Enter held down on **Stop**,
**Continue** or a hold's two buttons presses it once, and there the second
press of a double click or tap, which lands on the button that took the first
one's place, is let go. A batch stopped by **Stop**
still runs to its end in the worker, and **Continue** sends the next only once
it has. The review needs the network: while it
cannot be reached the screen says so and waits ("Waiting until Trained AI can
be reached.", as a review not yet asked for starts by itself once it can be;
once asked for, "Trained AI cannot be reached right now.", and **Continue**
comes back when it can).

## Verification

`npm run verify` in `web/` covers these behaviours: real-WASM discard previews
and Tile rendering, riichi waits, disabled-tile navigation, consecutive
keyboard turns, a double click and a held key or Enter each playing one move,
the confirmation's buttons handing focus to the hand, saved-state migration,
ura-dora indicators with hints off,
final-hand review, focus going on from the result's buttons to the hand's
review and to the final standings, pressed by keyboard, mouse and touch, and
back to the result, at desktop and phone sizes, the page moving to the review
for any press where the result is not a sheet and otherwise only for the
keyboard, and to the standings only for the keyboard where the result is not a
sheet (in Agent watch at every size), the sheet stepping aside for the
standings on a phone however the last Next hand was pressed, as a finished
match is loaded and as Play comes back to one from Agent watch, with the
places and Play again in view and focus on their heading, a first visit's wait
for the network's download naming its size and offering Club meanwhile, which
keeps Trained as the stored choice, a result or standings left behind by
another window's move disabled and saying to reload, with focus on that line
if one of their buttons had it,
a review that could not be built and a file that could not be saved said
beside the result's buttons, which keep focus, a whole game that could not be
saved or replayed for review said beside the standings' buttons, the result
stepping aside on a phone for a press beneath it, and a game that **Export
game** could not save said beside its button, inside the settings' dialog on a
phone, the sheet's buttons spanning it on a phone, the status lines that wait
in the page, empty, before their words arrive, the hand's notes keeping the
adviser's answers behind **Show answers** while Trained AI's move is hidden,
every decision edged alike until then, its button keeping focus, the game's
review with the shipped network (its batches, the final hand asked about first
and opened on, its flags hand by hand in the order played, Trained AI's move on
show in the list, the step bar, the hand and the view, Previous and Next across
hands by keyboard, focus handed on at either end and as a control goes, Next
keeping focus while it waits for a hand's answers, the keys and where they are
left alone, a hand entered by Shift and an arrow or by its name at its first
decision the filter keeps, and the trend, every hand face up at every step of a
hand at desktop, tablet and phone sizes, the step bar on the screen as the
screen changes size and at each step, the page staying where a scroll put it,
and focus never under it nor moving the page as it lands on the bar, the filter
that Next follows, a decision it leaves out kept in view, and the filter kept
across a reload, by keyboard and pointer, its lists staying open or folded as
the player leaves them while the steps change and later hands are reviewed, by
touch and by keyboard, the folded list showing the line on show as it opens and
bringing it out from under the bars at each step, the list a column of its own
on a wide screen that keeps the line on show in view and keeps still what the
player looks at, a line or a filter pressed or an earlier hand, as later hands
are reviewed, the page still, a line of the list keeping focus as the list
moves between its column and its fold, the player's tiles in one row from
tablet width, its two settings, the same in Options and, from tablet width, in
its header and kept across a reload, a decision keeping Trained AI's move and
the other players' tiles back until Show answer, s, a digit, a tile or a call
picked by keyboard answers it, the pick kept as Now and the racks turning over
where they lay, the table staying where it was on the screen at the top of a
desktop's page and on a phone held upright, focus going on to Next as the
control used goes, the page moved to show it only for a keyboard's press on a
low screen, a held Enter pressing once, a tile of the live hand keeping focus
as the phone turns upright, the live hand tabbed through clear of the bars on a
phone, and each reveal kept through steps and opening again and gone after a
reload and in a new game, and its way in and out on a phone, by touch and by
keyboard; its reasoning at a discard where Trained AI threw another tile, the
Why line and, above the margin floor, the button saying about how long, the
search pressed by keyboard, Enter held down, with its progress, Cancel and
focus handed on, a double press on Cancel let go,
what it found grouped by the kind of change, kept for its decision alone and
fitting a phone, Cancel and a step away stopping it as the review goes on,
and nothing of it drawn, said or asked while Trained AI's move is kept back,
its passes told apart from the review's batches; and the review of the game
so far from a hand's result, opened on the hand just played and asking about
it first, kept out of the trend, back to the result's button at desktop and
phone sizes and open again where it was left, stopped by Next hand and asking
nothing again at the next hand's result or at the standings), the review, of
the game so far or of the whole game, waiting while a hand's own notes or its
reasoning ask Trained AI and going on where it was, history and export, and
settlements that balance event by event. Its browser checks run in
Chromium, including narrow portrait and landscape sizes; they are not
physical-iPhone or Safari testing.
