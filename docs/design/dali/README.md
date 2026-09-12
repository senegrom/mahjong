# Dalí mahjong tile set

A surrealist tile face set for the mahjong game. Eight faces are approved: the original six studies, Carl's selected three-disks alternative B, **Time coming apart**, and five-bamboo alternative A, **The Soft Grove**.

## Approved first studies

| File | Tile | Direction |
| --- | --- | --- |
| `Pin1.svg` | 1 dot | Soft Time — an impossibly soft pocket watch melting over a desert shelf |
| `Pin3.svg` | 3 disks | Time coming apart — B: jade and ruby disks above a blue clock dissolving into suspended fragments; its shadow remains whole |
| `Pin5.svg` | 5 dots | Levitation — five polished, slightly softened disks floating over the desert |
| `Sou1.svg` | 1 bamboo | Stilt Bird — one elongated surreal crane-like bird with impossibly long legs |
| `Sou2.svg` | 2 bamboo | Elastic Growth — two sinuous segmented bamboo stalks; all visible colour remains green |
| `Sou5.svg` | 5 bamboo | The Soft Grove — A: four jade stems and one ruby stem soften into droplets and pools, retaining the five-pip arrangement |
| `Man8.svg` | 8 characters | Impossible Stone — ivory 八 above a sculptural vermilion 萬 |
| `Chun.svg` | Red dragon | Molten Ruby — translucent ruby-red 中 stretched into an uncanny molten form |

The other 26 faces use `web/public/tiles/dali/placeholders/placeholder.svg` until they receive approved artwork.

## Presentation

The set follows the existing tile system's 300 × 400 face, 26-unit rounded clipping and 1% bleed. Each SVG embeds the actual approved PNG artwork. The first six faces are lossless crops of [the approved board](studies/01-first-six-approved.png), excluding its labels and surround. These replace the simplified SVG interpretations used by the initial deployment.

Three disks uses the exact [selected B image](studies/three-disks-b-approved.png), preserving the complete 1086 × 1448 source and its original PNG bytes. Alternatives A (the river) and C (the butterflies) were not selected. Keep the chosen artwork when exporting; do not regenerate or redraw it.

Five bamboo uses the exact [selected A image](studies/five-bamboo-a-approved.png), preserving the complete 1086 × 1448 source and its original PNG bytes. Its four green stems and central ruby stem remain clearly countable. Alternatives B (floating joints) and C (the hand shadow) were not selected.

Present three alternatives for each new tile before selection.

Choose **Options → Tile face → Dalí**. The preference is persisted with the existing settings mechanism. All 34 Dali tile identities are included in the preload inventory: eight approved faces plus one shared placeholder for the remaining identities. The option is part of the Svelte settings control, and the tile's hint rings follow the same rounded outline as its artwork.

## Reproducing the exports

Run `node web/scripts/export-dali-tiles.mjs` from the repository root with Node.js and ImageMagick installed. It records source paths, exact crop rectangles and SHA-256 hashes in `web/public/tiles/dali/manifest.json`, and produces the approved tile list and a preview at `/tiles/dali/preview.html`.
