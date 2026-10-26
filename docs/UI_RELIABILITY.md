# UI reliability and saved matches

The browser owns one `MatchSession` at a time. Disposing it aborts outstanding
inference and frees its WASM game. Every asynchronous continuation checks that
owner before applying an answer, reporting an error or changing the interface.
A failed neural opponent remains explicitly Trained until the player retries,
continues the same position with Club opponents, or starts a new match.

The page has one trained worker, which answers its requests in turn. At a
hand's result two things ask it: the hand's own notes (**View table / my
hand**, or **Review final hand** at the last hand), a decision at a time, and
Trained AI's review of the game, so far or whole, up to eight decisions a
request, which goes on in the background. While the notes ask, the review
waits before each batch it would send, then goes on exactly where it was, the
hands waiting and the player's word as they were, so nothing the player chose
is lost and nothing need start the review again. Without that wait every note
would queue behind a batch, and a hand's notes would take twenty seconds
rather than two or three. With it, only the first note can queue behind a
batch, the one already sent, whichever was opened first, and the notes after
it behind none. The next deal's Trained opponents go first another way:
**Next hand** stops the review, so only their first move can queue behind the
batch already sent. The notes say they have stopped however they stop:
answered, failed, out of time, closed, left for another mode, or turned to
Club; and a review stopped or closed while it waits stops waiting at once.

A request's deadline is its own, and runs from when the worker begins it,
which the worker says as it does: twenty seconds for a single decision or a
batch of up to four, forty for one of five to eight. The time a request waits
behind the ones ahead of it does not count, so a note or a move queued behind
a batch waits for the batch, however slow the device, and is timed only from
when it begins. A request that runs past its deadline ends the worker and
fails every request still waiting on it, those queued behind it too, and
whoever asked says so: the notes that the trained opponent did not answer in
time, the review the same as it stops, and a Trained opponent's table by
offering its recovery. A request the worker never begins, as when the worker
dies without an error, fails the same way when its backstop runs out: its own
deadline and those of the requests still ahead of it, counted from when it was
sent, or from when the network is ready if it was still loading, and again
from each answer the worker gives.

The review sizes its batches to the device, by the time a decision it has
measured: eight decisions up to 1.25 seconds a decision, and on a slower
device as many as take about ten seconds, at least one. On a device that takes
up to ten seconds a decision, a batch then takes at most half its deadline,
and whatever queues behind it waits about ten seconds at most. The first batch
can include loading the network, so it is not measured, and the first two are
eight whatever the device, which run out of time from five seconds a decision.
The review's reasoning about a decision asks in passes of the same size.

## Saved matches

After each successful player/AI action and each hand transition, the application
stores `riichi.match.v2` in local storage. The versioned record contains the
initial seed, original opponent mode, a validated sequence of legal commands,
and the final visible state. Neural action choices are recorded too. Reloading
replays these exact choices on the rules engine; it does not ask the network to
choose the past moves again. A pending opponent turn resumes after restoration.

Replay is deliberately fail-closed: malformed records, illegal actions and an
engine version producing a different state are rejected without overwriting the
saved record. The user can deliberately choose New game to replace it. Storage
failure leaves play available and displays a warning. A new match takes its
identity only inside its first real save, so missing storage, Web Locks or
`crypto.randomUUID` never stop unsaved play. This is local restoration,
not cloud sync or an offline-installation feature. Clearing website data removes
the saved match. A future change to engine rules may require a save migration.

Preferences (opponent strength, hints, discard confirmation, shortcuts, and the
game review's filter and its two hiding settings) are stored separately. A
first visit starts with Trained opponents and the Trained AI review adviser,
the review's flagged decisions and nothing hidden. Saved preferences are read
as they stand, so a player whose record holds Club keeps Club; only a
preference the record lacks takes the default. What the player revealed in a
review is kept for the match only, never saved. Cancelling an opponent change
leaves both the displayed selection and active game unchanged. New game
requests confirmation whenever an unfinished match has progress, including
between hands.

## Input and presentation

- Hand shortcuts run only with focus inside the hand. Browser modifiers, other
  controls and repeated destructive key events are excluded. The marker is
  cleared explicitly after a choice. Shortcuts can be disabled in Options.
- Select-before-discard is optional and defaults on for coarse pointers. A
  second tap or the labelled Discard button confirms the selected tile.
- Mobile play keeps the hand and actions together. All opponents' discard rows
  remain visible in a compact table, with a larger native-dialog inspection view.
- Dora types are global (including repeated indicators); held-dora han is a
  separate value. Results include applicable ura-dora, and review notes use the
  indicators from that historical position. Indicator keys identify slots,
  not tile names.
- The safe count counts held copies and is explicitly qualified as safety
  against declared riichi, not a guarantee against undeclared hands. Tile names
  expose selected, drawn, safe and dora states to assistive technology.
- A list the player opens or folds stays that way. The game's review keeps
  whether its lists are open in its own components (`bind:open`), never in an
  `open` set from other state: Svelte sets such an attribute again whenever
  anything drawn beside it changes, which folded a replayed hand's list of
  decisions under the player's press, at every step and as later hands were
  reviewed, and dropped keyboard focus on a step's button in it to the page.
  So the list of hands and decisions, folded above the decision on a narrow
  screen, the trend at its top and the trend's table of games, a decision's
  events since the previous decision and its table of what each move leaves,
  which start folded and are taken from the page by a step that has none, all
  stay as the player leaves them across steps, hands and the hands reviewed
  meanwhile.
- The review's list keeps the line of the decision on show in view by setting
  the scroll of its own box, the column it is on a wide screen or its box of
  hands where it is folded, never by `scrollIntoView`, which would move the
  page under the player too. A step and the list opening bring the line into
  the part of the box the player can see, below the bars stuck over a folded
  list, which the line's `scroll-margin-top` measures; the toggle that opens a
  folded list arrives after the browser has opened it, so the list goes by
  whether it was on the screen before rather than by its box. Where hands
  reviewed meanwhile or another filter draw the list again, the list keeps
  still what the player is looking at, as scroll anchoring would: the control
  in the box that has focus, else the line on show, else the first line in
  view. Keeping the line on show still whatever the player looked at moved
  everything they read above it by the lines arriving in between. Safari has
  no scroll anchoring, so the boxes turn Chromium's off
  (`overflow-anchor: none`) and behave alike in every browser.
- A screen reader announces a status line only once it already knows it. So
  these lines wait in the page, empty and out of sight, before their words
  arrive, rather than arriving with them:
  - the page's notice line, at the top of the page. It now says only that
    this browser is not keeping the settings, so the trained network picker
    cannot load another network. On a phone the picker sits in the settings'
    dialog, and while that is open this line is not heard (see below);
  - the warning below it that this browser cannot safely save shared matches;
  - the result's own line after its buttons, which says why the hand's review
    could not be built, why the hand or the game could not be saved from the
    result, and why the game could not be replayed for review: the game so
    far, from every hand's result at every size, and once the game is over the
    whole game, from the last hand's result on a phone. It speaks where the
    player pressed, since the top of the page is above the screen at desktop
    width and under the sheet on a phone;
  - in play, the final standings' own line after their buttons, which says
    why the whole game could not be saved, or replayed for review, from their
    **Save whole game** and **Review whole game**. It waits there from the
    moment the standings appear;
  - **Export game**'s own line in the settings, after its button, which says
    why the game could not be saved. On a phone, upright or held sideways,
    the settings are a modal dialog, and while it is open Chromium's
    accessibility tree ignores all that lies outside it (`activeModalDialog`),
    so a line behind it would be heard only once the dialog closed. This one
    is inside it;
  - the hand review's two lines, that Trained AI is reviewing and that Club
    reviews in its place, which wait in the review until it has been drawn,
    and a tenth of a second more;
  - the line that says which decisions the game's review now steps through,
    under the filter at the top of its list, which comes into the page with
    the filter's buttons, speaks only when one is pressed, and goes quiet
    again when focus leaves the buttons or the counts it gave change.

  Two lines still do not speak where the player pressed. The trained network
  picker's refusal, on the notice line, lies behind the settings' dialog on a
  phone, as **Export game**'s failure did. Agent watch's line that its game
  could not be saved comes into the page with its words, below the watch's
  controls, whichever of its save buttons was pressed: the controls', the
  result's or the standings'.

  Other status lines still come into the page with their words: the line that
  says the game is loading, the settings' line that new tile graphics are
  loading, a yaku's explanation on a hand's result, the game review's
  headline and its step bar's line, which says the step, the table and both
  moves in one line as the step changes, or that Trained AI's move is hidden,
  and again as the player reveals it, both with the review, and the status
  lines of Agent watch (its file note among them), the guided physical game
  and physical agent play.

## Verification

Run `npm run verify` in `web/` after `npm ci`. Its unit tests include pure
session tests and tests against the rebuilt WASM, including a complete-match
save/restore and the seed-369 duplicate-indicator regression. Its browser
tests use the production build under `/mahjong/`, without
application test hooks; deterministic saved positions and test-only worker
responses exercise recovery, inputs, hints, and small-screen layouts.

Set `CHROME_BIN` when Chrome/Chromium is not in a standard system location.
Screenshots and the structured browser report are saved to `web/test-results/`.
Pages publication needs three CI jobs to pass: the rules engine, the web app
and the browser worker parity check. Browser viewport emulation is not a
physical iPhone/Safari test.

## Saves across windows, legacy records and calls

Saved-match commands run inside a same-origin Web Lock. Each write compares the
exact revision originally read, so a stale window cannot take over merely by
acquiring the lock later. Other windows stop accepting moves when storage changes
and offer **Reload latest match**. Nothing is saved by `pagehide`; successful
commands were already saved in their transaction. Returning from the back/forward
cache reloads the current match. Browsers without usable storage/Web Locks may
play without shared persistence and show a warning rather than use a racy fallback.

Saves written between 5 and 10 September 2026 are no longer resumed: those
without a format or of formats 2 to 4, and the first ones, saved under
`riichi.match.v1`. Restoring one says when it was saved and that this version
cannot resume a match that old, and the page keeps the record. New game
replaces one under the current key. The first key is read only while there is
no current save, so that the page can say this, and nothing writes or deletes
it. A save of format 5 or later replays its recorded commands, and any rules or
replay divergence is still rejected without overwriting the original record.
`web/tests/review2-engine.test.js` checks the refusal on the real engine, and
`web/scripts/cleanup-check.mjs` checks what the page says and that it keeps
both records.

A submitted human call remains pending until every other eligible claimant has
answered, including competing ron claims. Pending calls replay exactly and can
recover to Club. The retained final hand and exported log preserve their original
player-to-seat mapping. Ties have a joint place and a distinct player identifier.
Meld presentation obtains the actual claimed tile from its authoritative event,
not from its position in the sorted sequence. Native Tab focus and arrow-key
markers stay synchronised, and Enter prioritises the focused tile. Tiles disabled
while deciding a call stay readable.

The regression suite includes real shared browser tabs, simultaneous writers,
shift-Tab/mouse input, a completed match, and late-hand, multiple-meld, long-call
and tied-standings layouts at narrow portrait and landscape viewport sizes.
