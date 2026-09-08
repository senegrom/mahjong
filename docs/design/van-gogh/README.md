# Van Gogh

First concepts for a new mahjong tile set, developed on 8 September 2026 after reviewing the existing Matisse artwork and its export conventions.

![Van Gogh — six first tile studies](studies/01-van-gogh-concepts.png)

## Six examples

| Study | Tile | Direction |
| --- | --- | --- |
| A — Sunflower | 1 disk (`1p`, `Pin1`) | One large golden sunflower head on turquoise and cobalt. The flower is the disk: no stem, vase or additional flowers. |
| B — Five Stars | 5 disks (`5p`, `Pin5`) | Five round yellow stars in a clear quincunx on a swirling ultramarine field. Dark gaps separate the disks. |
| C — Mistral | 3 bamboo (`3s`, `Sou3`) | Three jointed stalks in a one-above-two arrangement, with forceful green brushwork and pointed leaves. The entire face stays within the green family. |
| D — Painted Letters | 3 characters (`3m`, `Man3`) | Three golden strokes form 三 above vermilion 萬 on deep blue. Paint gives the characters their form. |
| E — Vermilion | Red dragon (`7z`, `Chun`) | A large red 中 on ochre, with two open counters and a strong central vertical. |
| F — Cloud Dragon | White dragon (`5z`, `Haku`) | A pale curling dragon in ivory cloud strokes, surrounded by expressive cobalt. |

These are **concepts for review**, not approved production faces or a complete playable set. They are not registered in the game's tile-face selector. The sheet contains six examples; it does not supply the remaining 28 tile types or a white-dragon foil state.

## Design language

Use Van Gogh's rhythmic oil strokes, vivid warm/cool contrasts and strongly drawn contours to build the actual tile symbols. Keep motifs large enough to survive the game's small display sizes. Vary the palette and composition across the set while maintaining a shared painted language.

The Matisse set establishes the practical constraints: full-face colour, a 3:4 portrait face, no painted rim or physical bevel, legible tile identities and countable suit motifs. Keep those constraints. Retain traditional glyphs for characters and winds and the bird convention for 1 bamboo. The bamboo tiles eligible for All Green (`2s`, `3s`, `4s`, `6s`, `8s`) and green dragon (`6z`) should use green hues throughout their artwork, following the existing set's design convention.

## First review

- **A and B are the strongest anchors.** The sunflower is a distinctive single disk; the five stars have an immediately readable count. Explore round sunflower heads and star disks across the disk suit without adding background stars that could confuse the count.
- **C has the right structure.** Keep the three stalks separate. In a final face, simplify the busy surrounding leaves and preserve unmistakable green highlights.
- **D and E read clearly at study scale.** Keep the open spaces inside the glyphs when refining the brushwork. Check every traditional character against the tile identity before approving a production face.
- **F needs the most simplification for play.** Reduce whiskers and fine strokes, broaden the light central area and make the dragon quieter. A later silver dora state should use exactly the same composition and crop as the base face.

Further ideas: a single bird above a wheat-coloured field for 1 bamboo; cypress-like motion in visibly segmented bamboo; large wind glyphs on distinct dawn, daylight, sunset and night palettes; and a green 發 formed from layered malachite and mint strokes. These are proposed directions, not additional generated examples.

## Source and next production step

The [concept sheet](studies/01-van-gogh-concepts.png) is the unmodified image-generation output. It contains captions and gutters outside the faces and is not a production atlas. The [manifest](manifest.json) records the study identities, source dimensions and hash. The [generation prompt](prompt.md) preserves the brief used with ChatGPT's built-in image generation tool.

Once a direction is selected, refine it as a separate flat 3:4 source face. Preserve the selected pixels in later exports and follow the existing 300 × 400 canvas, 26-unit corner radius and 1% bleed conventions. Check the individual art at the same small and rotated sizes used for hands, discards and melds before registering any faces for play. Continue the set through all 34 tile types, with a matching white-dragon foil state.

Reference: [Matisse set and export conventions](../../../web/public/tiles/matisse/README.md).
