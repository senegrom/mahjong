# Matisse mahjong tile studies

A partial tile set inspired by Henri Matisse's cut-paper designs and the mass vestments for the Chapelle du Rosaire. The artwork was generated and refined with Carl in ChatGPT on 7 September 2026.

## Approved faces

| File | Tile | Selected treatment |
| --- | --- | --- |
| [Pin1.svg](approved/Pin1.svg) | 1 dot | Black disk, ivory rosette, yellow field: direction B |
| [Pin3.svg](approved/Pin3.svg) | 3 dots | Blue, red and green rosettes on ivory |
| [Pin5.svg](approved/Pin5.svg) | 5 dots | Five black and ivory rosettes on yellow |
| [Sou8.svg](approved/Sou8.svg) | 8 bamboo | Two sweeping fans of four jointed fronds |
| [Ton.svg](approved/Ton.svg) | East wind | Ribbon lettering B, blue with a red stroke on pale yellow |
| [Chun.svg](approved/Chun.svg) | Red dragon | Red cut-paper 中 on pink: direction B |
| [Sou1.svg](approved/Sou1.svg) | 1 bamboo | Blue and green cut-paper bird on a bamboo perch |
| [Hatsu.svg](approved/Hatsu.svg) | Green dragon | Ivory 發 cut out of emerald green: style C |

## Character concept awaiting redesign

[Man7.svg](concepts/Man7.svg) preserves the initial flowing black 七 above red 萬 as a reference. Carl found this arrangement too restrained and requested more expressive lettering and varied placement, without requiring the number at the top. This design remains a concept awaiting redesign.

The bird and green dragon were subsequently approved, bringing the set to eight approved faces and one remaining concept.

## Preview and exports

Open [preview.html](preview.html) locally, or visit `/tiles/matisse/preview.html` when running the web project. It is a standalone page with embedded artwork, an exact CSS-pixel hand preview at 390 or 844 pixels wide, and a toggle to include the character concept. The default hand includes only approved faces. The sample hand is for visual comparison only.

Every face has a lossless PNG crop at its native resolution and a self-contained SVG wrapper on the game's 300 × 400 canvas. The SVGs contain raster artwork; they are not vector redrawings. The symbols keep their original proportions, with ivory padding where required. Design-study shading is retained. These exports do not yet constitute the full tile set.

[manifest.json](manifest.json) records tile IDs, approval status, source paths, SHA-256 hashes and exact crop rectangles. The original boards are in [docs/design/matisse/studies](../../../../docs/design/matisse/studies). They preserve the alternative East styles A and C as references; only B is the approved East face. The earlier regular-grid bamboo in the five-dot board is superseded by the fan design.

To reproduce the exports, install Node.js and ImageMagick (`convert`), then run from the repository root:

```sh
node web/scripts/export-matisse-tiles.mjs
```

The script performs only lossless rectangular extraction, canvas wrapping and preview generation. The eight approved designs are taken directly from the selected source pixels. If a tile's approval status changes, the script removes its obsolete export paths.
