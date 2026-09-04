# Riichi app icon

`mahjong-approved.webp` is a high-quality, 512 x 512 web-sized copy of the
approved red-dragon tile artwork. The PNG and ICO exports in `web/public`
use that artwork, not the former SVG icon.

To regenerate the committed exports from the repository root:

```sh
python -m pip install Pillow==11.3.0
python web/scripts/export-icons.py
node web/scripts/check-icons.mjs
```

Pillow is an optional asset-authoring dependency, not a web-build dependency.
The exporter checks the approved source checksum and decodes each output.
The PNGs are opaque RGB. The ICO includes 16, 32 and 48 pixel images, the
Apple Home Screen icon is 180 pixels, and manifest icons are 192 and 512
pixels. The separate maskable variant keeps the full artwork inside the
central safe circle.

The page explicitly links the Apple icon and the manifest. Vite substitutes
`%BASE_URL%` so the links also work at the `/mahjong/` GitHub Pages path.
The manifest's application ID is `/mahjong/`, distinct from the other apps
on this shared origin. CI checks both the source assets and `web/dist`.
