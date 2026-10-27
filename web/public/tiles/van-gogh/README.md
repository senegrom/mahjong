# Van Gogh

**Current Red dragon:** **The Painter’s Vermilion Dragon** (`7z` / `Chun`) replaces Vermilion E, following Carl’s request to deploy the painting with the paintbrush and palette. The complete 1086 × 1448 painting is proportionally downsampled to a 300 × 400 quality-70 AVIF with zero stretch and zero extra bleed, using the shared 26-unit rounded corners. The previous E PNG and SVG are preserved in the design archive.

**Current White dragon:** **C — The Almond-Blossom Dragon** (`5z` / `Haku`) replaces Quiet Dragon L, following Carl's selection and request to deploy C. White almond blossoms and gnarled branches form the dragon against turquoise sky. The full 1086 × 1448 painting is proportionally downsampled to 300 × 400 with zero stretch and zero extra bleed, using the shared 26-unit rounded corners. The previous L PNG and SVG are preserved in the design archive.

**Current North wind:** Snowbound North B (`4z` / `Pei`) replaces Wind Ribbons, following Carl's winter selection. The snowy scene is cropped to exactly 3:4, not squeezed; the complete 北, moon, village and snowy path remain visible. The former North source and playable PNG/SVG are retained in the design archive. Earlier North sections below describe that superseded design. This replacement does not add a tile or change any other artwork.

**Current 9 disks:** The Potter’s Table C is active, replacing Nine Stars I. This corrects the misunderstood restoration: Carl explicitly chose the pottery painting. The approved SVG is kept byte-for-byte as its study, the playable face differs from it only in its fit (see [Proportions](#proportions-cropped-not-stretched)), and Nine Stars remains preserved in the design archive. That restoration left the other tiles unchanged. Eight disks now uses Lemons by the Sea. Earlier sections below describe the design history.

Choose **Options → Tile face → Van Gogh**. The 34 approved faces appear in hands, discards, melds, indicators, waits, agent modes, reviews and scoring. The preference is saved on this device, and the new faces are included in preloading and offline preparation.

**The Billowing Courtyard A** is active for South wind (2z / Nan). **The Red Sail C** is active for 1 character (1m / 一萬), completing the characters suit. All three numbered suits are complete. **Green Dragon at the Vineyard C** is active for green dragon (6z / Hatsu). **Lemons by the Sea** is active for 8 disks, completing all nine disks. The selected studies are A–B and G–H, plus **The Painter’s Vermilion Dragon** for red dragon, **The Almond-Blossom Dragon C** for white dragon, **Seven Apples at the Window** for 7 disks, **The Provençal Kitchen B** for 6 disks, **Starry Olive Grove** for 8 characters, **The Potter’s Table C** for 9 disks, **Four Oranges A** for 4 disks, **Irises at Dusk C** for 7 characters, **The Scarlet Loom C** for 6 characters, **The Red Vineyard A** for 5 characters, **Three Café Lanterns B** for 3 disks, **Green Still Life B** replacing J for 6 bamboo, **Copper Sunset** for 5 bamboo, **Moonlit Wind Chime A** for 9 bamboo, **Bamboo Raft** for 8 bamboo, **Seven with Irises C** for 7 bamboo, green Moonlit Four C for 4 bamboo, green Triple Shoots A for 3 bamboo, green Garden Rhythm B for 2 bamboo, **Almond Branches (Characters A)** for 3 of characters, **Night Cafe (Characters B)** for 2 of characters, **Cypress Fields (Characters C)** for 4 of characters, **Blazing Dawn (East A)** and **Snowbound North (Winter B)**. Almond Branches replaces the earlier Painted Letters D. White dragon uses The Almond-Blossom Dragon C. East wind K is excluded. **Cloudborne West** is active for West wind (3z / Shaa), completing all 34 identities; no tile requires an unpainted fallback. Hidden tiles keep the shared back. The white dragon retains its own almond-blossom artwork under the normal dora ring and foil sheen.

The [preview](preview.html) shows the actual exports and a mixed hand. The [manifest](manifest.json) records approval, source hashes and exact crop rectangles. [Design studies](../../../../docs/design/van-gogh/README.md) preserve the original boards and generation prompts; the [raft record](../../../../docs/design/van-gogh/eight-bamboo-raft.json) records eight bamboo, and the [green still-life record](../../../../docs/design/van-gogh/six-bamboo-green-still-life.json) records the six-bamboo replacement.

Earlier PNG sources are cropped losslessly. The green 2, 3 and 4 bamboo sources, Copper Sunset and Seven with Irises use documented quality-95 WebP game exports, without regeneration or repainting. The vineyard, raft and Green Still Life B use documented quality-80 WebP images embedded in their source SVGs. The self-contained SVGs use the shared 300 × 400 canvas, 26-unit rounded corners and 1% bleed. Reproduce exports from the repository root with Node.js and ImageMagick:

```sh
node web/scripts/export-van-gogh-tiles.mjs
```

The 2 and 4 of characters use exact copies of their full-canvas source PNGs. To export only those two while keeping every other face byte-for-byte unchanged:

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=2m,4m
```

## The Painter’s Vermilion Dragon: Red dragon

The selected paintbrush-and-palette painting is embedded in a self-contained SVG as a 300 × 400 quality-70 AVIF with 4:4:4 chroma. The playable path remains `approved/Chun.svg`. The complete 3:4 composition retains the brush, palette and vermilion dragon with zero stretch and zero extra bleed. The [approved study](../../../../docs/design/van-gogh/studies/31-red-dragon-painter-c-approved.svg) is a game-sized export of the selected original; the [approval and processing record](../../../../docs/design/van-gogh/red-dragon-painter-c.json) records its dimensions and checksums. Vermilion E is preserved as its exact previous [PNG](../../../../docs/design/van-gogh/superseded/red-dragon-vermilion-E-Chun.png) and [SVG](../../../../docs/design/van-gogh/superseded/red-dragon-vermilion-E-Chun.svg).

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=7z
```

This reproduces Red Dragon and its generated metadata. The other 33 painted faces are unchanged, including the selected Almond-Blossom White Dragon.

## The Almond-Blossom Dragon C: White dragon

The selected C painting is embedded in a self-contained SVG as a 300 × 400 quality-70 AVIF with 4:4:4 chroma. The playable path remains `approved/Haku.svg`. The [approval and processing record](../../../../docs/design/van-gogh/white-dragon-almond-blossom-c.json) records the original image, exact dimensions, source and raster checksums, and archived L exports. The source SVG is a game-sized export of the selected original.

```sh
node web/scripts/export-van-gogh-tiles.mjs --only=5z
```

This reproduces White Dragon and its generated metadata. The set retains all 34 painted faces.

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

Run `node web/scripts/export-van-gogh-tiles.mjs --only=4p` to reproduce this tile, its registration, manifest and preview without rewriting any other tile artwork. This addition brought the set to **25 painted faces**. The other nine tiles were Classic fallbacks then; the game now writes out their names instead, until they are painted. All 24 previously approved faces remain unchanged, including Irises at Dusk C for 7 characters and the restored Nine Stars I.

## Proportions: cropped, not stretched

Every face draws one picture in a 300 × 400 box. Until October 2026 that box was 3:4 whatever the painting's shape, so a painting that is not 3:4 was stretched to fill it, and eighteen faces were more than 2% out of shape. Nine of them were squeezed before they were embedded: the 300 × 400 exports for 2s, 3s, 4s, 7s, 8s, 3p, 4p, 9p and 6m were resized from crops of other shapes, which their provenance records. Three Café Lanterns, for example, showed its lanterns 46% wider than painted.

Each of these faces now has a fit in `web/scripts/export-van-gogh-tiles.mjs`, applied by `web/scripts/face-fit.mjs`: the face crops the painting instead of stretching it, keeping its proportions to within 2%, and centres the cut unless that would clip a counted object or character. Two faces need more stretch than that to keep their identifying content whole, and take the most any face may show, 4%: Cypress Fields (4m), whose cypress flame tip and the point of 萬's foot touch the top and bottom edges, and Three Café Lanterns (3p), whose top lantern's finial and bottom lantern's drop do the same. Only each SVG's picture box changes. Every embedded raster, PNG and approved study is unchanged, and the manifest records each tile's fit and box. The nine squeezed exports keep fewer source pixels across the cut, so they are a little softer than the PNG faces; sharper versions would need the full-resolution originals kept in the chat archives.

Earlier sections that call a playable SVG byte-for-byte identical to its study describe the set before this change. The studies are untouched, and each fitted face differs from its study only in its picture box.

## Starry Olive Grove — 8 characters (八萬)

Carl approved the olive-branch painting for `8m` / `Man8` and explicitly requested GitHub deployment. Two spreading olive branches form 八 above copper-red 萬 in a swirling blue-and-gold night landscape. This is the newly approved eight-characters painting, not the earlier olive seven study.

The entire 1086 × 1448 portrait is resized proportionally to a 300 × 400 quality-80 WebP using Lanczos, then embedded in the standard self-contained SVG with rounded clipping and 1% bleed. No repainting, recolouring or selective crop is applied. The source and runtime SVGs are identical; original, raster and SVG hashes are recorded in `docs/design/van-gogh/eight-characters-olive-grove.json`. The untouched full-resolution source PNG is preserved in `Van_Gogh_8_Characters_Starry_Olive_Grove_Approved.zip` supplied in chat. The game export is not the lossless original.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=8m` to copy this face exactly without rewriting any other artwork. This addition brings the set to **26 painted faces**; the other eight tiles are written out until they are painted, with characters **2–8** and all nine bamboo faces covered. All previous artwork, including Irises at Dusk C, Four Oranges A and The Potter’s Table C, remains unchanged.

## The Provençal Kitchen B — 6 disks

Carl selected **B, the middle kitchen painting**, as `6p` / `Pin6`: exactly six decorated plates in two columns of three on an ochre wall, with a blue shutter and the dresser still life below. The sunflower and fishing-float alternatives are not selected.

The source `docs/design/van-gogh/studies/22-six-disks-provencal-kitchen-b-approved.svg` embeds a 300 × 400 quality-90 WebP export of the complete 464 × 873 selected crop, squeezed to the shared face. The runtime `web/public/tiles/van-gogh/approved/Pin6.svg` differs from it only in its picture box, which crops the painting back to its own proportions, as the proportions section describes: the face keeps the six plates and the wall above them, and the cut falls on the jug, lemons and dresser below. No objects, colours or painted details were regenerated, and the shared rounded clipping is retained.

The untouched 1491 × 1055 board and lossless selected crop are supplied in `Van_Gogh_6_Disks_Kitchen_B_Approved.zip` in chat. `docs/design/van-gogh/six-disks-kitchen.json` records original, crop, production and SVG checksums and exact processing.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=6p` to reproduce this face and its registration without rewriting any other tile artwork. The gallery includes the kitchen; the existing fourteen-tile example hand is unchanged.

## Seven Apples at the Window — 7 disks

Carl approved the final corrected standalone painting: **four apples in the bowl and three on the cloth**, seven in total. The earlier six-apple and eight-apple attempts are not used.

The original 1024 × 1536 PNG is cropped at (62, 180) to **900 × 1200, exactly 3:4**. This removes the white border and excess setting above and below, while retaining all seven apples. Uniform Lanczos downsampling gives a 300 × 400 quality-90 WebP; the SVG adds the shared 26-unit corner clipping and 1% bleed. No stretching, regeneration, repainting or recolouring is applied. The new source and runtime SVGs are byte-identical, without needing an additional fit override.

The original and lossless crop are preserved in `Van_Gogh_7_Disks_Seven_Apples_Approved.zip` supplied in chat. The repository source SVG embeds the game-sized export, not the full-resolution original. `docs/design/van-gogh/seven-disks-apples.json` records the selection, framing, dimensions and checksums. Run `node web/scripts/export-van-gogh-tiles.mjs --only=7p` to reproduce the new face, registry, manifest and gallery.

This addition brings the set to **28 painted faces and 6 named fallbacks**. Every earlier painting, fit and manifest entry is preserved; the fourteen-tile example hand and gameplay are unchanged.

## Poppy Field — 9 characters, smaller 萬

Carl selected the middle poppy-field idea, requested a smaller characters sign to sit more organically among the flowers, and approved the resulting revised image as `9m` / `Man9` (九萬). The earlier large-sign version, wheat and harbour are not used. The approved floral nine and smaller golden 萬 are retained without repainting or recolouring.

The source SVG embeds the entire approved 1122 × 1402 portrait in a 300 × 400 AVIF (quality 50) game export. The existing face-fit helper restores the original proportions with zero stretch, clipping a small centered strip from each side to fill the 3:4 face. Source and runtime differ only in image placement. This is a lossy game-sized export; the untouched PNG is preserved in `Van_Gogh_9_Characters_Poppy_Field_Approved.zip` supplied in chat. Original, raster, source and runtime hashes and the fit are recorded in `docs/design/van-gogh/nine-characters-poppy-field.json`.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=9m` to reproduce this face without rewriting other artwork. The set now has **29 painted faces and 5 unpainted identities showing their names**. Characters 2–9 are covered; only 1 characters remains in that suit. The fourteen-tile hand retains the previous thirteen featured paintings.

## Lemons by the Sea — 8 disks

Carl approved the standalone painting of **eight lemons in a 3–2–3 arrangement** for `8p` / `Pin8`. It completes the disks suit. Seven Apples for 7 disks, The Potter’s Table for 9 disks and all other artwork remain unchanged.

The entire 1086 × 1448 original is exactly 3:4 and was proportionally downsampled to 300 × 400 with Lanczos, then encoded as quality-90 WebP. There is no repainting, recolouring or compositional crop; the game uses its standard 26-unit rounded corners and 1% bleed. This is an optimized game export, not a lossless full-resolution original. The original PNG is preserved in `van-gogh-eight-disks-lemons-original.zip` supplied in chat.

The source and runtime SVGs are identical. Original, raster and SVG checksums are recorded in `docs/design/van-gogh/eight-disks-lemons.json`. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=8p`. The set now has **30 painted faces and 4 unpainted identities showing their names**.

## Green Dragon at the Vineyard C — green dragon

Carl selected **C from the first green-dragon board**, the green dragon flying above a golden vineyard and village, as **6z / Hatsu**. The later mountain/pine dragon, vase, cypress and iris alternatives are not used. The explicitly approved colours are preserved, including the blue sky and golden vines; this selection is not converted to monochrome green.

The original 1448 × 1086 board is cropped at **(979, 122) to 450 × 600, exactly 3:4**. The crop removes captions, gutters and excess lower vineyard while keeping the dragon head, horns, claws, village and vineyard rows. Uniform Lanczos downsampling gives a 300 × 400 AVIF (quality 65, 4:4:4), wrapped with the shared 26-unit corner clipping and 1% bleed. There is no stretching, regeneration, repainting or recolouring. The source and runtime SVGs are byte-identical and need no additional fit override.

`docs/design/van-gogh/green-dragon-vineyard.json` records selection, crop and image checksums. The untouched board, full selected panel and lossless 3:4 crop are preserved in `Van_Gogh_Green_Dragon_Vineyard_C_Approved.zip`, supplied in chat. The repository SVG embeds the game-sized image, not the full-resolution original.

Run `node web/scripts/export-van-gogh-tiles.mjs --only=6z` to reproduce this face, registry, manifest and gallery without rewriting other artwork. This addition brings the set to **31 painted faces and 3 named fallbacks**. Every prior painting, proportion fit and manifest entry, the fourteen-tile sample hand and gameplay remain unchanged.


## The Red Sail C — 1 character

Carl selected the third boat painting, preserving the small 一萬 on its cream hull, and requested GitHub deployment. The complete 1086 × 1448 portrait is exactly 3:4; it is uniformly downsampled to a 300 × 400 quality-60 AVIF game export with no crop, stretching, repainting or recolouring. The source SVG embeds the approved game raster in the shared study box. The runtime changes only that image box to zero bleed, with no stretching and the normal 26-unit rounded-corner clip. The game export is not the full-resolution PNG. The untouched original is supplied in `Van_Gogh_1_Characters_Red_Sail_Approved.zip`; its hash and processing details are recorded in `docs/design/van-gogh/one-characters-red-sail.json`.

Regenerate only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=1m`. Regression tests pin the source and embedded-raster hashes, check the full-frame geometry, verify registration and hidden-tile privacy, and ensure selective exports are repeatable without changing other artwork. All 31 previous faces remain byte-for-byte unchanged.

## South wind — The Billowing Courtyard A

Carl selected **A — The Billowing Courtyard** as `2z` / `Nan`, South wind. The billowing white linen and its wooden frame are the approved scene; neither the boat B nor the garden gate C is used.

The original board and exact 393 × 640 A crop are preserved in `van-gogh-south-wind-A-prepared.zip` supplied in chat. This repository stores a 300 × 400 quality-70 AVIF game export inside its approved study SVG. The existing face-fit helper corrects the tall painting's proportions to within 2%, trimming surrounding sky and paving rather than squeezing the whole image into the tile. The runtime differs from the study only in image placement. No repainting or recolouring was applied.

Checksums, crop, encoding and fit are recorded in `south-wind-billowing-courtyard-a.json` in the design directory. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=2z`. All prior artwork, including 7 and 8 disks and The Potter’s Table 9 disks, is unchanged; the existing fourteen-tile preview hand is preserved. The set now has **33 painted faces**, with **West wind** the only remaining unpainted identity.

## Snowbound North B — winter replacement

Carl selected the middle winter panel and requested GitHub deployment. The stone marked 北 stands beside a snowy path, warm-lit village and swirling blue-and-gold sky. The cypress and harbour alternatives are not used.

The crop `[497, 126, 456, 608]` on the original 1448 × 1086 triptych is exactly 3:4. A small strip of upper sky and excess foreground are trimmed, not stretched. The full selected panel and untouched board remain in the separately supplied approval ZIP. Uniform LANCZOS downsampling produces a 300 × 400 quality-60 AVIF; there is no repainting, recolouring or glyph editing. The playable face uses zero stretch and zero extra bleed, apart from the usual rounded-corner mask.

Source, original, crop, raster, runtime and archive checksums are recorded in `docs/design/van-gogh/north-wind-winter.json`. Reproduce only this replacement with `node web/scripts/export-van-gogh-tiles.mjs --only=4z`. All other selected faces and the existing fourteen-tile preview hand remain unchanged.

## Cloudborne West — final tile

Carl approved the latest standalone `stormy_sunset_lighthouse_seascape.png` for West wind (`3z` / `Shaa`): a billowing cloud-shaped 西 above a lighthouse, sunset and breaking waves. The earlier re-uploaded JPEG and alternative concept sheets are not used.

The entire **1086 × 1448** approved PNG is exactly **3:4**. It is uniformly Lanczos-downsampled to **300 × 400**, encoded as quality-65, 4:4:4 AVIF, and embedded in the shared radius-26 SVG. There is **zero crop, zero stretch and zero bleed**; only the rounded corners are clipped. No image regeneration, repainting or recolouring occurs during deployment. Source and runtime SVGs embed identical raster bytes; the runtime changes only the image box to remove the shared study bleed.

`docs/design/van-gogh/west-wind-clouds.json` records the original and export checksums and processing. The untouched original and game export are preserved in `Van_Gogh_West_Wind_Clouds_Approved.zip`, supplied in chat; the repository source SVG embeds the game-sized image. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=3z`.

**All 34 tile identities now have approved Van Gogh artwork.** All 33 earlier paintings, their proportions and the fourteen-tile example hand remain unchanged.

## The Scarlet Loom C — Six Characters (6m / Man6)

Carl selected the loom option C and explicitly requested deployment, replacing Lemon Terrace. The full 1086 × 1448 portrait, including the loom and both spools, is uniformly reduced to 300 × 400 and encoded as quality-70 AVIF. There is no crop, stretch, repainting or recolouring. The normal rounded tile clipping is unchanged.

The original PNG checksum is `de7bf8b8be8d6a1a445f045e3bb73e152a2ea2bc767c8b37ca6beb010cac2f73`. The original remains with the approval conversation; the repository retains the optimized source SVG and the complete provenance in `docs/design/van-gogh/six-characters-scarlet-loom-c.json`. The former playable Lemon Terrace SVG is preserved byte-for-byte in `docs/design/van-gogh/superseded/six-characters-lemon-terrace-Man6.svg`; its original optimized source is unchanged.

Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=6m`. All 33 other Van Gogh faces, including the Almond-Blossom White Dragon and the Painter’s Red Dragon, remain unchanged. The set still contains all 34 identities.


## Three Bamboo in a Vase A — 3 bamboo (3s / Sou3)

Carl selected the left-hand vase painting and explicitly requested deployment. It replaces green Triple Shoots A, not the raft, wind chime or any other numbered tile. The crop at (22, 23), 576 × 768 pixels, removes the triptych gutters and A label, while preserving all three stalks and the complete vase. It is uniformly downsampled to 300 × 400 and encoded as quality-70 AVIF; the runtime fit has zero stretch and zero extra bleed. Standard rounded clipping remains unchanged.

The previous playable PNG and SVG are archived byte-for-byte under `docs/design/van-gogh/superseded/three-bamboo-triple-shoots-Sou3.*`; its original study and provenance remain unchanged. The original triptych and lossless crop are available with the approval conversation; this repository retains the optimized selected image, not a full-resolution PNG archive. See `docs/design/van-gogh/three-bamboo-vase-a.json` for crop coordinates, original and export checksums. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=3s`. All other 33 Van Gogh faces and all other sets remain unchanged.


## The Bamboo Duet B — 2 bamboo (2s / Sou2)

Carl selected the two green bamboo flutes on blue cloth and explicitly requested GitHub deployment. This replaces Garden Rhythm, not the new three-bamboo vase, any other numbered tile, or the footbridge and terracotta-wall alternatives. The complete 1086 × 1448 portrait is uniformly downsampled to 300 × 400 and encoded as quality-70 AVIF. There is no cropping, stretching, repainting or recolouring; the runtime fit has zero stretch and zero extra bleed. Both flutes remain fully visible. Standard rounded clipping is unchanged.

The former playable PNG and SVG are archived byte-for-byte under `docs/design/van-gogh/superseded/two-bamboo-garden-rhythm-Sou2.*`. Its original study and provenance remain unchanged. The full-resolution original PNG is retained with the approval conversation; this repository stores the optimized approved image and checksum provenance, not a full-resolution PNG archive. See `docs/design/van-gogh/two-bamboo-flutes-b.json`. Reproduce only this face with `node web/scripts/export-van-gogh-tiles.mjs --only=2s`. All 33 other Van Gogh faces and all other sets remain unchanged.
