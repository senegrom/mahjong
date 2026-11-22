"""The picture an artist's tile face ships.

An approved face is kept as a lossless crop of its study, approved/<Name>.png,
which is not published. The game draws a face at most 60 CSS pixels wide, and a
crop of up to 1086 by 1448 pixels cost about 2 MB a face to send. So the face
ships that crop scaled to the box it is drawn in, a pixel to each of the face's
300 by 400 units, as WebP: the same picture in about a fortieth of the bytes.

    python face_raster.py <crop.png> <width> <height>

writes the WebP to standard output; <width> and <height> are the box's size in
whole pixels (face-fit.mjs). Needs Pillow built with WebP.
"""
import io
import sys

from PIL import Image

# Lanczos keeps the paintings' edges, and at this quality no face shows a
# block or a ring against its crop at four times the size it is played at.
QUALITY = 90
METHOD = 6


def face_raster(crop_path, width, height):
    """The crop at `crop_path` as WebP bytes, `width` by `height` pixels."""
    with Image.open(crop_path) as crop:
        mode = 'RGBA' if 'A' in crop.getbands() else 'RGB'
        picture = crop.convert(mode).resize((width, height), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    picture.save(out, 'WEBP', quality=QUALITY, method=METHOD)
    return out.getvalue()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.stdout.buffer.write(face_raster(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
