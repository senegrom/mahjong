# UI reliability and saved matches

The browser owns one `MatchSession` at a time. Disposing it aborts outstanding
inference and frees its WASM game. Every asynchronous continuation checks that
owner before applying an answer, reporting an error or changing the interface.
A failed neural opponent remains explicitly Trained until the player retries,
continues the same position with Club opponents, or starts a new match.

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

Preferences (opponent strength, hints, discard confirmation and shortcuts) are
stored separately. A first visit starts with Trained opponents and the Trained
AI review adviser. Saved preferences are read as they stand, so a player whose
record holds Club keeps Club; only a preference the record lacks takes the
default. Cancelling an opponent change leaves both the displayed
selection and active game unchanged. New game requests confirmation whenever
an unfinished match has progress, including between hands.

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
    result, and on a phone why the whole game could not be replayed for
    review. It speaks where the player pressed, since the top of the page is
    above the screen at desktop width and under the sheet on a phone;
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
  - the lines that say what a filter of the player's decisions shows, under
    the whole-game review's list and in a replayed hand's list of decisions,
    which come into the page with their buttons, speak only when one is
    pressed, and go quiet again when focus leaves the buttons or the counts
    they gave change.

  Two lines still do not speak where the player pressed. The trained network
  picker's refusal, on the notice line, lies behind the settings' dialog on a
  phone, as **Export game**'s failure did. Agent watch's line that its game
  could not be saved comes into the page with its words, below the watch's
  controls, whichever of its save buttons was pressed: the controls', the
  result's or the standings'.

  Other status lines still come into the page with their words: the line that
  says the game is loading, the settings' line that new tile graphics are
  loading, a yaku's explanation on a hand's result, the whole-game review's
  headline (with the review, and again on Back to the list), the decision
  replay's "Your decision N of M" and its line that says the table at each
  step, and the status lines of Agent watch (its file note among them), the
  guided physical game and physical agent play.

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
