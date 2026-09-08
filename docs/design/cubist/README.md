# Cubist mahjong tile set

**Set name:** Cubist  
**Set ID:** `cubist`  
**Stage:** initial concepts for review  
**Created:** 8 September 2026

![Cubist — Colour Planes: six tile concepts](studies/01-colour-planes.png)

## Colour Planes

The first direction uses angular overlapping planes, displaced contours, dark line fragments and restrained painted-paper texture. Ochre, parchment, oxblood, dusty cobalt and forest green give the collection its palette. Each face is a small Cubist composition with a strong primary symbol.

These six examples establish the direction before the rest of the 34 faces are designed. They are candidates, with no approved production exports or game integration yet.

| Candidate | Tile ID | Tile | Concept |
| --- | --- | --- | --- |
| A | `1p` | One disk | One large off-centre coin with faceted ochre, blue and red planes inside a closed circular contour. |
| B | `5p` | Five disks | Five separate disks in a quincunx; the centre is larger, and each disk has its own fractured interior. |
| C | `1s` | One bamboo | A single angular bird with an ochre beak, blue body, rust-red wing and broad tail against forest green. |
| D | `2s` | Two bamboo | Two staggered architectural stalks, with the entire face in shades of green. |
| E | `8m` | Eight characters | Two open strokes form 八 above a red angular 萬, interlocking with charcoal and blue planes. |
| F | `7z` | Red dragon | An oversized red 中 with an asymmetric enclosure, two open counters and a continuous central stem. |

**Suggested anchors:** C and F establish the character of the set; B establishes how countable disk tiles can share it.

## Continuity with the Matisse set

The current Matisse artwork and design notes were inspected on `main` at `17a6cae320eb302da6adcd8bd8a899a103fbd278`. The reference faces included `Pin5`, `Sou1`, `Sou2`, `Man8` and `Chun`.

Carry forward the existing project's full-bleed faces, large readable motifs, accurate suit counts and Chinese glyphs. Production faces should use the shared portrait 3:4 presentation without a painted rim, bevel or shadow. For All Green, keep every part of `2s`, `3s`, `4s`, `6s`, `8s` and `6z` within the green palette.

The study sheet is a presentation image, not a spritesheet or production asset. Labels and gutters sit outside the artwork. For selected candidates, prepare individual faces, check complete motifs and glyphs, and inspect their readability in the existing small and rotated tile views before export. In particular, the two-bamboo composition currently extends to the lower edge; decide the final crop when preparing its standalone face.

## Source

- [Original concept sheet](studies/01-colour-planes.png), 1254 × 1254 PNG.
- [Generation prompt](prompts/01-colour-planes.txt).
- Created with ChatGPT's built-in image generation tool. This is new artwork informed by the repository's design constraints; the existing tile images were inspected as references and were not edited.
- Visual review: six labelled candidates; one disk in A, five separate disks in B, one bird in C, two stalks in D, recognisable 八 and 萬 in E, and a continuous 中 stem with two openings in F. Small-size gameplay readability remains to be checked on individual approved faces.
