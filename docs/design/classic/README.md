# Classic white-dragon highlight

Carl requested a much simpler, Chinese-calligraphy-inspired Classic dragon on 7 September 2026. The selected drawing reduces the head to three brush gestures: a forked horn, a hooked head and neck, and a small balancing sweep. It has no eye, scales or whiskers. It is a pictorial dragon mark, not a Chinese character.

`white-dragon-calligraphy.png` is the selected generated source. The game uses a 192 × 256 lossless WebP export at `web/src/assets/white-dragon.webp`, preserving the drawing and its brush texture. The Classic overlay uses 40% opacity with the existing dora mask and foil animation, so the black master appears as a silver impression only during the shine. The Matisse dragon uses its separately approved quiet and lit faces at full opacity.

`white-dragon-stylized.png` preserves the earlier sculptural study; it is no longer used in the game.

To reproduce the export from the repository root:

```sh
convert docs/design/classic/white-dragon-calligraphy.png -resize 192x256 -strip -define webp:lossless=true web/src/assets/white-dragon.webp
```
