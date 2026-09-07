# Classic white-dragon highlight

Carl requested a more stylised version of the Classic set's highlight dragon on 7 September 2026. The revised head keeps the left-facing three-quarter pose and stern expression, with broader planes, fewer scales and simpler horns and mane.

`white-dragon-stylized.png` is the generated source. The game uses a 192 × 256 lossless WebP export at `web/src/assets/white-dragon.webp`. Its existing dora mask and foil animation reveal the drawing; it is not printed permanently on the tile.

To reproduce the export from the repository root:

```sh
convert docs/design/classic/white-dragon-stylized.png -resize 192x256 -strip -define webp:lossless=true web/src/assets/white-dragon.webp
```
