# Gameplay interface

The table UI keeps the rules engine and match state separate from presentation. Desktop uses a larger table surface with clearer opponent zones; compact phone layouts keep the round, opponent type and status visible while moving secondary controls into the settings sheet.

When **Hints and markings** is enabled, every tile in the human hand has a small number directly below the face showing how many copies of that tile are still unseen from public information. The count includes the player's own concealed copies and subtracts visible unclaimed discards, called sets and exposed dora indicators exactly once. Zero is highlighted as exhausted and one as scarce. Turning Hints off removes the counts together with the other learning aids.

Calls are presented as a staged discard followed by the legal response choices. On phones, hand results appear in a large bottom sheet with the result value first, then the winning hand, yaku and score changes; the table can be revealed again without losing the result. Reduced-motion preferences suppress nonessential transitions.

The 7-by-2 phone hand layout is intentional: it preserves large, reliable touch targets instead of squeezing fourteen tiles into one row. The newly drawn tile remains slightly separated, including its hint count, and carries no border merely for being newly drawn.
