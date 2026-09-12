# Van Gogh

Choose **Options → Tile face → Van Gogh**. The twelve approved faces appear in hands, discards, melds, indicators, waits, agent modes, reviews and scoring. The preference is saved on this device, and the new faces are included in preloading and offline preparation.

The selected studies are A–E, G–J and L, plus **Blazing Dawn (East A)** and **Wind Ribbons (North B)**. White dragon uses the later, simpler L. East wind K is excluded. The other 22 identities awaiting Van Gogh artwork use their Classic faces. Hidden tiles keep the shared back. The white dragon retains its own pale dragon under the normal dora ring and foil sheen.

The [preview](preview.html) shows the actual exports and a mixed hand. The [manifest](manifest.json) records approval, source hashes and exact crop rectangles. [Design studies](../../../../docs/design/van-gogh/README.md) preserve both original boards and generation prompts.

The source pixels are cropped losslessly, with no regeneration, repainting or upscaling. The self-contained SVGs use the shared 300 × 400 canvas, 26-unit rounded corners and 1% bleed. Reproduce exports from the repository root with Node.js and ImageMagick:

```sh
node web/scripts/export-van-gogh-tiles.mjs
```
