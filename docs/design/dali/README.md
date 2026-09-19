# Dalí mahjong tile set

A surrealist tile face set for the mahjong game. Twelve faces are approved, drawn from the original studies and Carl's later selections. The latest are five-characters A, **Theatre of the Impossible**, and six-characters B, **The Runaway Shadow**.

## Approved first studies

| File | Tile | Direction |
| --- | --- | --- |
| `Pin1.svg` | 1 dot | Soft Time — an impossibly soft pocket watch melting over a desert shelf |
| `Pin3.svg` | 3 disks | Time coming apart — B: jade and ruby disks above a blue clock dissolving into suspended fragments; its shadow remains whole |
| `Pin5.svg` | 5 dots | Levitation — five polished, slightly softened disks floating over the desert |
| `Sou1.svg` | 1 bamboo | Stilt Bird — one elongated surreal crane-like bird with impossibly long legs |
| `Sou2.svg` | 2 bamboo | Elastic Growth — two sinuous segmented bamboo stalks; all visible colour remains green |
| `Sou5.svg` | 5 bamboo | The Soft Grove — A: four jade stems and one ruby stem soften into droplets and pools, retaining the five-pip arrangement |
| `Man5.svg` | 5 characters | Theatre of the Impossible — A: ivory 五 theatre drapes around a moonlit sea, above a complete ruby 萬 |
| `Man6.svg` | 6 characters | The Runaway Shadow — B: indigo 六 peeling from a curled desert plane, above ruby 萬 |
| `Man7.svg` | 7 characters | The Sleeping Seven — A: softened ivory 七 on a golden crutch, above a ruby 萬 |
| `Man8.svg` | 8 characters | The Window in Reality — B: star-filled 八 openings through a floating ivory membrane above a ruby 萬 |
| `Man9.svg` | 9 characters | Sapphire Suspension — C: ivory 九 with sapphire joints above a complete ruby 萬 suspended over sunset water |
| `Chun.svg` | Red dragon | Molten Ruby — translucent ruby-red 中 stretched into an uncanny molten form |

The other 22 faces use `web/public/tiles/dali/placeholders/placeholder.svg` until they receive approved artwork.

## Presentation

The set follows the existing tile system's 300 × 400 face, 26-unit rounded clipping and 1% bleed. Each SVG embeds the actual approved PNG artwork. The five remaining faces from the first studies are lossless crops of [the approved board](studies/01-first-six-approved.png), excluding its labels and surround. The board also preserves the earlier eight-character design that Carl replaced with B.

Three disks uses the exact [selected B image](studies/three-disks-b-approved.png), preserving the complete 1086 × 1448 source and its original PNG bytes. Alternatives A (the river) and C (the butterflies) were not selected. Keep the chosen artwork when exporting; do not regenerate or redraw it.

Five bamboo uses the exact [selected A image](studies/five-bamboo-a-approved.png), preserving the complete 1086 × 1448 source and its original PNG bytes. Its four green stems and central ruby stem remain clearly countable. Alternatives B (floating joints) and C (the hand shadow) were not selected.

Five characters adapts [the original A theatre study](studies/six-characters-a-theatre-study.png) from 六 to 五 at Carl's request. The [resulting five-character image](studies/five-characters-a-approved.png) preserves the moonlit opening, river, stage, ivory drapery and complete ruby 萬. Six characters uses the exact [selected B image](studies/six-characters-b-approved.png), without redrawing or cropping. Both exports preserve the complete 1086 × 1448 source PNG bytes.

Seven characters uses the exact [selected A image](studies/seven-characters-a-approved.png). Eight characters uses the exact [selected B recreation](studies/eight-characters-b-approved.png), with the starry opening changed from 七 to 八. Both preserve their complete 1086 × 1448 sources and original PNG bytes.

Nine characters uses the exact [approved repaired C image](studies/nine-characters-c-approved.png). It restores the missing left side of the central red character, with its strokes checked against the Classic tile. It retains the ivory/sapphire 九 and sunset-water composition, preserving the complete 1086 × 1448 source and its original PNG bytes.

Present three alternatives for each new tile before selection.

Choose **Options → Tile face → Dalí**. The preference is persisted with the existing settings mechanism. All 34 Dali tile identities are included in the preload inventory: twelve approved faces plus one shared placeholder for the remaining identities. The option is part of the Svelte settings control, and the tile's hint rings follow the same rounded outline as its artwork.

## Reproducing the exports

Run `node web/scripts/export-dali-tiles.mjs` from the repository root with Node.js and ImageMagick installed. It records source paths, exact crop rectangles and SHA-256 hashes in `web/public/tiles/dali/manifest.json`, and produces the approved tile list and a preview at `/tiles/dali/preview.html`.
