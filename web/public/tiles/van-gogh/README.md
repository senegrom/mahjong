# Van Gogh

**Current 9 disks:** The Potter’s Table C is active, replacing Nine Stars I. This corrects the misunderstood restoration: Carl explicitly chose the pottery painting. The approved SVG is copied byte-for-byte, and Nine Stars remains preserved in the design archive. 8 disks and all other tiles are unchanged. Earlier sections below describe the design history.

Choose **Options → Tile face → Van Gogh**. The twenty-five approved faces appear in hands, discards, melds, indicators, waits, agent modes, reviews and scoring. The preference is saved on this device, and the new faces are included in preloading and offline preparation.

The selected studies are A–B, E, G–H and L, plus **The Potter’s Table C** for 9 disks, **Four Oranges A** for 4 disks, **Irises at Dusk C** for 7 characters, **Lemon Terrace** for 6 characters, **The Red Vineyard A** for 5 characters, **Three Café Lanterns B** for 3 disks, **Green Still Life B** replacing J for 6 bamboo, **Copper Sunset** for 5 bamboo, **Moonlit Wind Chime A** for 9 bamboo, **Bamboo Raft** for 8 bamboo, **Seven with Irises C** for 7 bamboo, green Moonlit Four C for 4 bamboo, green Triple Shoots A for 3 bamboo, green Garden Rhythm B for 2 bamboo, **Almond Branches (Characters A)** for 3 of characters, **Night Cafe (Characters B)** for 2 of characters, **Cypress Fields (Characters C)** for 4 of characters, **Blazing Dawn (East A)** and **Wind Ribbons (North B)**. Almond Branches replaces the earlier Painted Letters D. White dragon uses the later, simpler L. East wind K is excluded. The other 9 identities awaiting Van Gogh artwork use their Classic faces. Hidden tiles keep the shared back. The white dragon retains its own pale dragon under the normal dora ring and foil sheen.

The [preview](preview.html) shows the actual exports and a mixed hand. The [manifest](manifest.json) records approval, source hashes and exact crop rectangles. [Design studies](../../../../docs/design/van-gogh/README.md) preserve the original boards and generation prompts; the [raft record](../../../../docs/design/van-gogh/eight-bamboo-raft.json) records eight bamboo, and the [green still-life record](../../../../docs/design/van-gogh/six-bamboo-green-still-life.json) records the six-bamboo replacement.

Earlier PNG sources are cropped losslessly. The green 2, 3 and 4 bamboo sources, Copper Sunset and Seven with Irises use documented quality-95 WebP game exports, without regeneration or repainting. The vineyard, raft and Green Still Life B use documented quality-80 WebP images embedded in their source SVGs. The self-contained SVGs use the shared 300 × 400 canvas, 26-unit rounded corners and 1% bleed. Reproduce exports from the repository root with Node.js and ImageMagick:

```sh
node web/scripts/export-van-gogh-tiles.mjs
```

The 2 and 4 of characters use exact copies of their full-canvas source PNGs. To export only those two while keeping every other face byte-for-byte unchanged:

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=2m,4m
```

## Garden Rhythm: green 2 bamboo

The approved revised green B is active for `2s` (`Sou2`). The two bamboo stalks, garden canal and bridge are unchanged in composition. The deployed 300 by 400 source is a high-quality WebP crop, with source checksum, original-board checksum and crop coordinates recorded in `docs/design/van-gogh/two-bamboo-green.json`. Earlier tile artwork is unchanged. Run `node web/scripts/export-van-gogh-tiles.mjs --only=2s` to regenerate this face. That addition brought the set to 15 painted faces and 19 Classic fallbacks.

## Triple Shoots: green 3 bamboo replacement

The approved greener **A — Triple Shoots** replaces the original C artwork for `3s` (`Sou3`), using the existing `approved/Sou3.png` and `approved/Sou3.svg` paths. Exactly three bamboo stalks form the tile identity. No other playable tile is changed. Original C remains in the first study sheet and Git history.

The source is a 300 × 400, quality-95 WebP export of the selected panel, not a lossless full-resolution original. The original board checksum, 439 × 673 crop coordinates and optimized-source checksum are recorded in `docs/design/van-gogh/three-bamboo-green.json`. Run `node web/scripts/export-van-gogh-tiles.mjs --only=3s` to reproduce this replacement. That replacement retained 15 painted faces and 19 Classic fallbacks.

## Moonlit Four: green 4 bamboo

The approved greener **C — Moonlit Four** is active for `4s` (`Sou4`). Four distinct bamboo stalks stand against the emerald and teal night sky. The committed source is a 300 × 400, quality-95 WebP game export of the selected right panel, not a lossless full-resolution original. The original board checksum and 438 × 671 crop coordinates are recorded in `docs/design/van-gogh/four-bamboo-green.json`. The original review board and full-resolution selected crop were also preserved in the downloadable approval archive.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=4s` to reproduce the playable PNG and SVG. That addition brought the set to 16 painted faces and 18 Classic fallbacks.

## Seven with Irises: 7 bamboo

On 29 September 2026 Carl selected **C — Seven with Irises** from the 7 bamboo study board and requested deployment. It is active for `7s` (`Sou7`). The six green stalks and central reddish-brown stalk form seven distinct bamboo motifs, with blue-purple irises, golden fields and a small cottage. This is the selected C artwork, not a repaint.

The study-sheet heading, gutters and caption were removed with the exact crop `[1043, 115, 479, 793]` from the 1536 × 1024 board. The entire cropped composition was resized to the shared 300 × 400 game canvas and saved as quality-95 WebP, following the newer bamboo source convention. This committed source is a web-optimized export, not the full-resolution lossless original. The original board, full-resolution crop and production-source hashes are recorded in `docs/design/van-gogh/seven-bamboo-irises.json`.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=7s` to reproduce the playable PNG, self-contained SVG, registration, manifest and preview. That addition brought the set to 17 painted faces and 17 Classic fallbacks; the subsequent raft addition preserves it unchanged.

## Bamboo Raft: approved 8 bamboo

Carl selected the raft idea for eight bamboo and approved the resulting standalone painting for deployment. Eight jointed poles are lashed together on a swirling blue river beneath a golden sun. The complete approved composition is used, without repainting, adding symbols or recolouring. The blue/gold palette is an explicitly approved artwork exception to the set's all-green convention for `8s`; Mahjong rules and All Green eligibility are unchanged.

The source `docs/design/van-gogh/studies/14-eight-bamboo-raft-approved.svg` and playable `approved/Sou8.svg` are byte-for-byte identical. They embed a 300 × 400 quality-80 WebP export of the full 1024 × 1536 portrait. The source is a game-sized export, not the full-resolution original PNG. Original, raster and SVG checksums and processing are recorded in `docs/design/van-gogh/eight-bamboo-raft.json`.

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=8s
```

This copies the approved SVG exactly and leaves all seventeen earlier faces untouched. The set now has **18 painted faces and 16 Classic fallbacks**. Focused regressions verify source/runtime identity, embedded raster integrity, registration and isolated export preservation.

## Copper Sunset — 5 bamboo

The approved **Copper Sunset** is active for `5s` / `Sou5`: four green bamboo stalks frame one tall reddish-brown central stalk. The earlier all-green concept is not used. The complete approved composition is retained, with no cropping, recolouring or repainting.

`studies/15-five-bamboo-copper-sunset.webp` is a 300×400, quality-95 WebP game export, following the recent bamboo convention. It is not a lossless full-resolution original. Source and original checksums, full-canvas coordinates and processing details are recorded in `docs/design/van-gogh/five-bamboo-copper.json`; the untouched original is preserved in `van-gogh-five-bamboo-copper-original.zip` supplied in chat.

The set now contains **19 painted faces and 15 Classic fallbacks**. All 18 previous painted faces and their source artwork are unchanged. Regenerate this addition alone with `node web/scripts/export-van-gogh-tiles.mjs --only=5s`.

## Green Still Life B: corrected green 6 bamboo

Carl approved the corrected B still life with **six distinct bamboo stalks**, a pale green two-handled pot, green bottle and two limes, and requested deployment. This replaces **J — Green Rhythm** for `6s` (`Sou6`); none of the earlier five-stalk candidates is used. The approved green colours, objects and complete composition are preserved without repainting or recolouring.

The full 1086 × 1448 approved portrait is resized proportionally to 300 × 400 with Lanczos and saved as quality-80 WebP, following the raft convention. The source `docs/design/van-gogh/studies/15-six-bamboo-green-still-life-b-approved.svg` and playable `approved/Sou6.svg` are identical, with the standard 1% bleed and rounded clip. The obsolete J runtime PNG is removed; its original artwork remains in the second study sheet and Git history. Original, raster and SVG checksums are recorded in `docs/design/van-gogh/six-bamboo-green-still-life.json`. The downloadable approval archive preserves the full-resolution original PNG; the committed SVG is an optimized game export, not that original.

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=6s
```

This copies the approved B SVG exactly and leaves the other eighteen faces unchanged, including Copper Sunset, the raft and Seven with Irises. The set remains at **19 painted faces and 15 Classic fallbacks**. Focused regressions cover the approved image hashes, replacement of J, registration, the 14-tile preview hand and isolated export preservation.

## Moonlit Bamboo Wind Chime — 9 bamboo

Carl selected the **first option, Wind Chime A**, for `9s` / `Sou9` and explicitly requested deployment. Nine hanging bamboo tubes are arranged in three groups of three against a swirling cobalt sky and golden moon. The support rail, copper-orange cords, foliage and village remain part of the exact approved painting. No repainting, recolouring or additional symbols were applied.

The source `docs/design/van-gogh/studies/16-nine-bamboo-wind-chime-a-approved.svg` and playable `web/public/tiles/van-gogh/approved/Sou9.svg` are byte-for-byte identical. They embed a proportional 300 × 400 quality-80 WebP export of the complete 1086 × 1448 portrait, with the shared rounded clipping and 1% bleed. This is a game-sized export, not the full-resolution original. Original, raster and SVG hashes are recorded in `docs/design/van-gogh/nine-bamboo-wind-chime.json`; the untouched original is preserved in `Van_Gogh_9_Bamboo_Wind_Chime_Approved.zip` supplied in chat.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=9s` to reproduce the approved SVG, registration, manifest and preview without rewriting any other tile artwork.

This addition completes **all nine bamboo identities**. The set now has **20 painted faces and 14 Classic fallbacks**. All nineteen previously approved faces are unchanged.

## Three Café Lanterns B — 3 disks

Carl selected **B — Three Café Lanterns**, the middle panel of the three-disk concept board, for `3p` / `Pin3` and explicitly approved deployment. Three large golden lanterns hang above the night café. No repainting, recolouring, additional lanterns or symbol substitutions were applied.

The caption and presentation gutters are excluded with the exact `[503, 127, 442, 860]` crop from the original 1448 × 1086 board. Its entire composition is resized to the shared 300 × 400 game canvas using Lanczos and encoded as quality-90 WebP. This is an optimized game export, not a full-resolution or lossless copy. The untouched board and full-resolution selected PNG remain in the approval archive `van-gogh-3-disks-lanterns-prepared.zip` supplied in chat.

Source `docs/design/van-gogh/studies/17-three-disks-cafe-lanterns-b-approved.svg` and runtime `web/public/tiles/van-gogh/approved/Pin3.svg` are byte-for-byte identical. The embedded image uses the existing 26-unit rounded clipping and 1% bleed. Original-board, full-resolution-crop, WebP and SVG checksums are recorded in `docs/design/van-gogh/three-disks-lanterns.json`.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=3p` to reproduce the game SVG, registration, manifest and preview without rewriting other artwork. This addition brings the set to **21 painted faces and 13 Classic fallbacks**. All twenty previously approved faces, including the complete bamboo suit, are unchanged.

## The Potter’s Table C — 9 disks replacement

Carl selected **C — The Potter’s Table** from the final artwork-only comparison and requested deployment. It replaces **I — Nine Stars** for `9p` / `Pin9`: nine blue-and-cream patterned plates in a three-by-three arrangement, with a wooden table, sunflower corners and swirling sky. This is the selected painting, not a regeneration.

Only the right-hand painting is cropped from the original composite at `[551, 56, 729, 1093]`; its inaccurate checklist and exterior margin are excluded. The complete selected crop is resized to a 300 × 400 quality-95 WebP and embedded in the standard SVG. This game export is not the lossless full-resolution original. Original, crop, embedded-raster and SVG checksums are recorded in `docs/design/van-gogh/nine-disks-potters-table-c.json`. The untouched original and lossless selected crop are preserved in the downloadable `van-gogh-nine-disks-potters-table-C-originals.zip` supplied in chat.

The original Nine Stars PNG and SVG are preserved byte-for-byte under `docs/design/van-gogh/superseded/nine-stars-Pin9.*`; the earlier study board is untouched. The stale Nine Stars PNG is removed from the playable directory after archiving. All other artwork is unchanged. This replacement retains **21 approved faces and 13 Classic fallbacks**. Regenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=9p`.

## The Red Vineyard A — 5 characters

Carl selected the **first option, The Red Vineyard**, for `5m` / `Man5` (`五萬`) and explicitly requested deployment. Copper-red vines and grape clusters form the characters over the golden vineyard landscape. The wheat and iris alternatives are not used. The entire approved composition and colours are retained without repainting or recolouring.

The complete 1086 × 1448 portrait is resized proportionally with Lanczos to a 300 × 400 quality-80 WebP, following the raft and green still-life export convention. The source `docs/design/van-gogh/studies/18-five-characters-vineyard-a-approved.svg` and game `web/public/tiles/van-gogh/approved/Man5.svg` are identical and use the standard rounded clip and 1% bleed. This is an optimized game export, not the full-resolution original PNG. Original, raster and SVG checksums are recorded in `docs/design/van-gogh/five-characters-vineyard.json`; the untouched original is preserved in `Van_Gogh_5_Characters_Vineyard_Approved.zip` supplied in chat.

Regenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=5m`. All 21 previously approved Van Gogh faces and all other sets remain unchanged. This addition brings the set to **22 painted faces and 12 Classic fallbacks**, with all nine bamboo faces preserved.

## Lemon Terrace — 6 characters (六萬)

Carl approved the latest red 六萬 calligraphy with lemons, cypresses, a lake and village for `6m` / `Man6`. This is the selected character painting, not the earlier sunflower study. No existing tile artwork is changed.

The complete 1295 × 1214 source is fitted to the shared 300 × 400 face using Lanczos and quality-80 WebP, embedded in a self-contained SVG with the usual rounded clip and 1% bleed. Fitting changes the aspect ratio; it is not a proportional resize. There is no selective cropping, repainting or colour edit. The unchanged full-resolution original is preserved in `Van_Gogh_6_Characters_Lemon_Terrace_Approved.zip` supplied in chat. See `docs/design/van-gogh/six-characters-lemon-terrace.json` for original, raster and SVG hashes.

The set now has **23 painted faces and 11 Classic fallbacks**. `node web/scripts/export-van-gogh-tiles.mjs --only=6m` copies the approved source to the game exactly without rewriting other faces. Regression tests cover hashes, registration, preloading, hidden tiles, the fourteen-tile preview hand and repeated selective export.

## Nine Stars — original 9 disks restored

The original nine luminous disks in a 3×3 arrangement are active again for `9p` / `Pin9`. Both the 380×471 PNG and its existing game SVG are byte-for-byte identical to the archived originals. The Potter’s Table C source, original provenance and archived runtime SVG are all retained. No other tile or tile coverage changes.

`docs/design/van-gogh/nine-stars-restored.json` records this decision and the hashes. Run `node web/scripts/export-van-gogh-tiles.mjs --only=9p` to reproduce the restored face without rewriting other artwork.

## Irises at Dusk C — 7 characters (七萬)

Carl selected the last of the three seven-characters designs: blue-violet irises form 七 above a golden 萬, against the painted sunset lake and village. This is the exact approved C painting, not the olive-grove or wheat alternatives. No repainting, recolouring, additional lettering or selective cropping is applied.

The full 1086 × 1448 portrait is resized proportionally to a 300 × 400 quality-80 WebP and embedded in the standard rounded SVG with 1% bleed. The source `docs/design/van-gogh/studies/20-seven-characters-irises-c-approved.svg` and runtime `web/public/tiles/van-gogh/approved/Man7.svg` are identical. Original, raster and SVG hashes are recorded in `docs/design/van-gogh/seven-characters-irises.json`. This committed SVG is a game export, not the full-resolution original; the untouched PNG is preserved in `Van_Gogh_7_Characters_Irises_Approved.zip` supplied in chat.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=7m` to reproduce this addition without rewriting another tile. The set now has **24 painted faces and 10 Classic fallbacks**. All 23 previous faces, including restored Nine Stars I, the complete bamboo suit, Vineyard and Lemon Terrace, remain unchanged. The new regressions verify exact artwork hashes, image geometry, registration, preloading, hidden faces, the fourteen-tile preview hand and repeated selective export.

## Four Oranges A — 4 disks

Carl selected **A**, the left panel with **four whole oranges on blue cloth**, for `4p` / `Pin4` and explicitly requested GitHub deployment. The blue-and-white jug, leafy branch, yellow wall, window and distant village are preserved. The bowl (B) and orange slices (C) are not used.

The source `docs/design/van-gogh/studies/20-four-disks-oranges-a-approved.svg` and playable `web/public/tiles/van-gogh/approved/Pin4.svg` are identical. They embed a quality-90 300 × 400 WebP of the complete approved crop `[25, 138, 442, 796]` from the 1448 × 1086 concept board, with the set's 26-unit corners and 1% bleed. The label and presentation gutters were excluded; nothing was repainted or recoloured. The complete tall crop is fitted to the common face, changing its aspect ratio. This is a game-sized export, not the full-resolution original.

Board, full-resolution crop, decoded crop pixels, raster and SVG checksums are recorded in `docs/design/van-gogh/four-disks-oranges.json`. The original board and lossless selected crop are preserved in `Van_Gogh_4_Disks_Oranges_A_Approved.zip`, supplied in chat.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=4p` to reproduce this tile, its registration, manifest and preview without rewriting any other tile artwork. The set now has **25 painted faces and 9 Classic fallbacks**. All 24 previously approved faces remain unchanged, including Irises at Dusk C for 7 characters and the restored Nine Stars I.
