# Van Gogh

First concepts for a new mahjong tile set, developed on 8 September 2026 after reviewing the existing Matisse artwork and its export conventions.

Carl approved all except K and requested deployment. **Van Gogh is a selectable set with fifteen distinct faces**: A–C, E, G–J and L, plus Almond Branches (Characters A), Night Café (Characters B), Cypress Fields (Characters C), Blazing Dawn (East A) and Wind Ribbons (North B). Almond Branches replaces D for 3 of characters; Night Café and Cypress Fields are the approved adaptations for 2 and 4 of characters. The later L is the active white dragon; F remains an earlier alternative. East wind K is excluded. The remaining 19 tile identities use Classic artwork. See the [playable exports and preview](../../../web/public/tiles/van-gogh/README.md).

![Van Gogh — six first tile studies](studies/01-van-gogh-concepts.png)

## First six examples

| Study | Tile | Direction |
| --- | --- | --- |
| A — Sunflower | 1 disk (`1p`, `Pin1`) | One large golden sunflower head on turquoise and cobalt. The flower is the disk: no stem, vase or additional flowers. |
| B — Five Stars | 5 disks (`5p`, `Pin5`) | Five round yellow stars in a clear quincunx on a swirling ultramarine field. Dark gaps separate the disks. |
| C — Mistral | 3 bamboo (`3s`, `Sou3`) | Three jointed stalks in a one-above-two arrangement, with forceful green brushwork and pointed leaves. The entire face stays within the green family. |
| D — Painted Letters | 3 characters (`3m`, `Man3`) | Three golden strokes form 三 above vermilion 萬 on deep blue. Paint gives the characters their form. |
| E — Vermilion | Red dragon (`7z`, `Chun`) | A large red 中 on ochre, with two open counters and a strong central vertical. |
| F — Cloud Dragon | White dragon (`5z`, `Haku`) | A pale curling dragon in ivory cloud strokes, surrounded by expressive cobalt. |

## Second studies

![Van Gogh — second tile studies](studies/02-van-gogh-concepts.png)

| Study | Tile | Direction |
| --- | --- | --- |
| G — Wheatfield Bird | 1 bamboo (`1s`, `Sou1`) | A large cobalt-and-copper bird on green bamboo against a golden field. |
| H — Twin Suns | 2 disks (`2p`, `Pin2`) | Two distinct sunflower heads on a flowing teal and cobalt field. |
| I — Nine Stars | 9 disks (`9p`, `Pin9`) | Nine luminous round star disks in a clear three-by-three arrangement. |
| J — Green Rhythm | 6 bamboo (`6s`, `Sou6`) | Six separate jointed stalk motifs, in two columns of three, with green hues throughout. |
| K — East at Dawn | East wind (`1z`, `Ton`) | Large cobalt 東 against peach, coral and golden brushstrokes. |
| L — Quiet Dragon | White dragon (`5z`, `Haku`) | A simpler pale dragon with a broad curling body and fewer details within an ivory field. Alternative to F. |

The second sheet was generated after Carl asked to continue and retry image generation. **G and H are the strongest new directions.** I keeps nine clearly separated disks; J keeps six clearly separated bamboo motifs. L removes much of F's filigree and gives the white dragon a simpler silhouette. K should receive a final glyph-structure check, and H's flower petals need more clearance at the edges, when these concepts become individual faces.

The first two sheets contain 12 studies covering 11 distinct tile types, including two white-dragon alternatives. Nine of those faces remain active after the Almond Branches replacement for 3 of characters. The normal dora foil shines over L's own painted dragon without substituting another set's dragon.

## Selected East wind

![East wind alternatives](studies/03-east-wind-alternatives.png)

On 12 September 2026, Carl selected **A — Blazing Dawn** for East and requested deployment. The playable `Ton` (`1z`) is cropped directly from the left-hand A, preserving its golden sunlight and wheat character against the orange sun and cobalt sky. The [prompt record](prompt-03.md) preserves the original generation and correction briefs. B and C remain alternatives; K remains rejected.

## Selected North wind

![North wind — Wind Ribbons](studies/04-north-wind-ribbons.png)

Carl selected A for East and commissioned North in B's Wind Ribbons style: ivory and turquoise 北 on violet and ultramarine. The corrected North design extends the right upright into the upper third, with its diagonal joining below the top. Both halves remain separated by a dark gap, with broad curls below.

Carl approved deployment, then supplied the [original shared image](https://chatgpt.com/s/m_6aa570cca934819188878ae3ac92bde4) to recover the exact artwork after its temporary working copy was lost. The original 1086 × 1448 PNG is now committed in this repository. The playable `Pei` (`4z`, North B) uses its entire canvas without repainting. The [source and prompt record](prompt-04.md) preserves the image hash and correction brief.

## Selected 3 of characters

![Three of characters — new directions](studies/06-three-characters-new-directions.png)

On 13 September 2026, Carl selected **A — Almond Branches** for 3 of characters and requested deployment. The playable `Man3` (`3m`, Characters A) is cropped directly from the approved left panel at `[25, 93, 523, 782]`. Flowering branches form 三萬 against an aquamarine field. The [complete prompt](prompt-06.md) and original sheet are preserved; the source image SHA-256 is `a0d0c325a5f63f6121555e099afcbba65cfaeb2941aee3962572b371282bfdd7`.

Carl also requested recreations of **B — Night Café** for 2 of characters (`二萬`) and **C — Cypress Fields** for 4 of characters (`四萬`), and approved their deployment. Both full-canvas adaptations are preserved as separate source images and now have playable exports: [Night Café 2](studies/07-two-characters-night-cafe.png) and [Cypress Fields 4](studies/08-four-characters-cypress-fields.png). The [deployment record](characters-two-four.md) describes the exact-source checks. The earlier Painted Letters D remains preserved in the first study sheet.

## Design language

Use Van Gogh's rhythmic oil strokes, vivid warm/cool contrasts and strongly drawn contours to build the actual tile symbols. Keep motifs large enough to survive the game's small display sizes. Vary the palette and composition across the set while maintaining a shared painted language.

The Matisse set establishes the practical constraints: full-face colour, a 3:4 portrait face, no painted rim or physical bevel, legible tile identities and countable suit motifs. Keep those constraints. Retain traditional glyphs for characters and winds and the bird convention for 1 bamboo. The bamboo tiles eligible for All Green (`2s`, `3s`, `4s`, `6s`, `8s`) and green dragon (`6z`) should use green hues throughout their artwork, following the existing set's design convention.

## First review

- **A and B are the strongest anchors.** The sunflower is a distinctive single disk; the five stars have an immediately readable count. Explore round sunflower heads and star disks across the disk suit without adding background stars that could confuse the count.
- **C has the right structure.** Keep the three stalks separate. In a final face, simplify the busy surrounding leaves and preserve unmistakable green highlights.
- **D and E read clearly at study scale.** Keep the open spaces inside the glyphs when refining the brushwork. Check every traditional character against the tile identity before approving a production face.
- **F needs the most simplification for play.** Reduce whiskers and fine strokes, broaden the light central area and make the dragon quieter. A later silver dora state should use exactly the same composition and crop as the base face.

The second sheet explores the bird, further bamboo rhythms and the dawn wind palette. Further ideas include the remaining wind glyphs on distinct daylight, sunset and night palettes, and a green 發 formed from layered malachite and mint strokes. These remaining directions are not yet generated examples.

## Source and next production step

Both the [first concept sheet](studies/01-van-gogh-concepts.png) and [second concept sheet](studies/02-van-gogh-concepts.png) are unmodified image-generation outputs. They contain captions and gutters outside the faces and are not production atlases. The [manifest](manifest.json) records the study identities, source dimensions and hashes. The [first prompt](prompt.md) and [second prompt](prompt-02.md) preserve the briefs used with ChatGPT's built-in image generation tool.

The deployed faces preserve the exact approved source pixels in lossless rectangular crops or full-canvas PNG copies. Captions and presentation gutters are excluded. Their SVG wrappers follow the existing 300 × 400 canvas, 26-unit corner radius and 1% bleed conventions. The [export script](../../../web/scripts/export-van-gogh-tiles.mjs) reproduces all fifteen faces and the preview. Use `--only=2m,4m` to export the two new character faces without re-encoding any existing artwork. Continue the remaining artwork through all 34 tile types as further designs are selected.

Reference: [Matisse set and export conventions](../../../web/public/tiles/matisse/README.md).

## Garden Rhythm: green 2 bamboo

The approved revised green B is active for `2s` (`Sou2`). The two bamboo stalks, garden canal and bridge are unchanged in composition. The deployed 300 by 400 source is a high-quality WebP crop, with source checksum, original-board checksum and crop coordinates recorded in `docs/design/van-gogh/two-bamboo-green.json`. Earlier tile artwork is unchanged. Run `node web/scripts/export-van-gogh-tiles.mjs --only=2s` to regenerate this face. The set now has 15 painted faces and 19 Classic fallbacks.
