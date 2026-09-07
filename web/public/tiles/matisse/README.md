# Matisse mahjong tile studies

A partial tile set inspired by Henri Matisse's cut-paper designs and the mass vestments for the Chapelle du Rosaire. The artwork was generated and refined with Carl in ChatGPT on 7 September 2026.

## Approved faces

| File | Tile | Selected treatment |
| --- | --- | --- |
| [Pin1.svg](approved/Pin1.svg) | 1 dot | Black disk, ivory rosette, yellow field: direction B |
| [Pin3.svg](approved/Pin3.svg) | 3 dots | Blue, red and green rosettes on ivory |
| [Pin5.svg](approved/Pin5.svg) | 5 dots | Five black and ivory rosettes on yellow |
| [Sou2.svg](approved/Sou2.svg) | 2 bamboo | Dance B: two sweeping bamboo forms in green on pale mint |
| [Sou8.svg](approved/Sou8.svg) | 8 bamboo | Two sweeping fans of four jointed fronds |
| [Ton.svg](approved/Ton.svg) | East wind | Ribbon lettering B, blue with a red stroke on pale yellow |
| [Chun.svg](approved/Chun.svg) | Red dragon | Red cut-paper 中 on pink: direction B |
| [Sou1.svg](approved/Sou1.svg) | 1 bamboo | Blue and green cut-paper bird on a bamboo perch |
| [Hatsu.svg](approved/Hatsu.svg) | Green dragon | Ivory 發 cut out of emerald green: style C |
| [Man7.svg](approved/Man7.svg) | 7 characters | Cut-out C: pink 萬 and oversized pale-yellow 七 on deep purple |
| [Haku.svg](approved/Haku.svg) | White dragon | Approved C/C1 blend: quiet ivory with a pearly silver dora reveal |

## In the game

Choose **Options → Tile face → Matisse**. All eleven approved faces appear throughout the game: hands, discards, melds, indicators, waits, reviews and scoring. The other 23 tile types use ivory placeholders showing only their names in black. Face-down tiles retain the shared back. The selection is saved on this device, and both sets are available offline after preparation completes.

White dragon uses matching quiet and [lit](approved/Haku-foil.svg) exports. When it is dora, the lit state appears beneath the moving shine. Both come directly from the approved board, with no redrawing of the motif.

## Next design directions

The 2 bamboo tile uses **Dance B** from `08-two-bamboo-approved.png`. Preserve the all-green appearance of tiles eligible for All Green: multiple shades of green are welcome, with neutral ivory for the tile substrate. The approved source pixels are kept exactly; no new colours are introduced when exporting.

The 7 characters tile uses **Cut-out C** from `06-character-compositions.png`. Carl also likes **Dance B** and suggested using its freer lettering on **deep purple** for other character tiles. Vary the scale and placement of the number and 萬 instead of keeping the number above the suit character.

East B is the approved wind. Keep the flowing black calligraphy of East A available for another tile. White dragon uses the approved abstract blend of C and C1 in `07-white-dragon-approved.png`; the earlier detailed dragon was rejected.

## Preview and exports

Open [preview.html](preview.html) locally, or visit `/tiles/matisse/preview.html` when running the web project. It is a standalone page with embedded artwork and a CSS-pixel hand preview at 390 or 844 pixels wide. The sample hand is for visual comparison only.

Every face has a lossless PNG crop at its native resolution and a self-contained SVG wrapper on the game's 300 × 400 canvas. The SVGs contain raster artwork; they are not vector redrawings. The symbols keep their original proportions, with ivory padding where required. Design-study shading is retained. These exports do not yet constitute the full tile set.

[manifest.json](manifest.json) records tile IDs, approval status, source paths, SHA-256 hashes and exact crop rectangles. The original boards are in [docs/design/matisse/studies](../../../../docs/design/matisse/studies). They preserve the alternative East styles A and C as references; only B is the approved East face. The earlier regular-grid bamboo in the five-dot board is superseded by the fan design.

To reproduce the exports, install Node.js and ImageMagick (`convert`), then run from the repository root:

```sh
node web/scripts/export-matisse-tiles.mjs
```

The script performs lossless rectangular extraction, canvas wrapping, text placeholder generation and preview generation. The eleven approved designs are taken directly from the selected source pixels. It also generates the approved tile list used by the game. If a tile's approval status changes, the script removes its obsolete export and placeholder paths.
