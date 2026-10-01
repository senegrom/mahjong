# Van Gogh

Choose **Options → Tile face → Van Gogh**. The nineteen approved faces appear in hands, discards, melds, indicators, waits, agent modes, reviews and scoring. The preference is saved on this device, and the new faces are included in preloading and offline preparation.

The selected studies are A–B, E, G–J and L, plus **Bamboo Raft** for 8 bamboo, **Seven with Irises C** for 7 bamboo, green Moonlit Four C for 4 bamboo, green Triple Shoots A for 3 bamboo, green Garden Rhythm B for 2 bamboo, **Almond Branches (Characters A)** for 3 of characters, **Night Cafe (Characters B)** for 2 of characters, **Cypress Fields (Characters C)** for 4 of characters, **Blazing Dawn (East A)** and **Wind Ribbons (North B)**. Almond Branches replaces the earlier Painted Letters D. White dragon uses the later, simpler L. East wind K is excluded. The other 15 identities awaiting Van Gogh artwork use their Classic faces. Hidden tiles keep the shared back. The white dragon retains its own pale dragon under the normal dora ring and foil sheen.

The [preview](preview.html) shows the actual exports and a mixed hand. The [manifest](manifest.json) records approval, source hashes and exact crop rectangles. [Design studies](../../../../docs/design/van-gogh/README.md) preserve the original boards and generation prompts; the [raft record](../../../../docs/design/van-gogh/eight-bamboo-raft.json) records the latest addition.

Earlier PNG sources are cropped losslessly. The newer green bamboo sources, Copper Sunset and Seven with Irises use documented quality-95 WebP game exports, without regeneration or repainting. The raft uses a documented quality-80 WebP embedded in its source SVG. The self-contained SVGs use the shared 300 × 400 canvas, 26-unit rounded corners and 1% bleed. Reproduce exports from the repository root with Node.js and ImageMagick:

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
