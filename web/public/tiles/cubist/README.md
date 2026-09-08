# Cubist tiles

Five faces from study **02 — Further Studies** are approved for play: A three disks, B five bamboo, C nine characters, D East wind and E white dragon. Carl excluded F, the green dragon. The first study remains unapproved.

Choose **Options → Tile face → Cubist**. The other 29 identities use Classic artwork. Hidden tiles retain the shared back. The Cubist white dragon retains its approved empty centre under the standard dora ring and foil sheen.

The [preview](preview.html) shows the five exports and a compact mixed hand. The [manifest](manifest.json) records approval, exact crop rectangles, source hash and PNG hashes. The PNGs preserve the selected pixels at native resolution. SVGs wrap those same PNGs on the shared 300 × 400 canvas with 26-unit corners and 1% bleed.

To reproduce from the repository root:

```sh
node web/scripts/export-cubist-tiles.mjs
```

The original sheet and prompt are in `docs/design/cubist`. Extraction is mechanical and lossless; no approved artwork is regenerated or redrawn.
