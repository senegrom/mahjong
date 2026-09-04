#!/usr/bin/env python3
"""Export the approved Riichi artwork as browser and Home Screen icons.

Optional development tool: python -m pip install Pillow==11.3.0
Generated files are committed; normal web builds do not need Python/Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from PIL import Image

WEB = Path(__file__).resolve().parents[1]
SOURCE = WEB / 'branding' / 'mahjong-approved.webp'
SOURCE_SHA256 = '230db3e42a712ff928b86359e77166ef5cd55a1a6eb20d6f799c288ef392a745'
PNG_SIZES = {
    'favicon-16x16.png': 16,
    'favicon-32x32.png': 32,
    'apple-touch-icon.png': 180,
    'icons/mahjong-192.png': 192,
    'icons/mahjong-512.png': 512,
}


def export_icons(output: Path) -> None:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Approved artwork checksum mismatch; refusing to export.')
    with Image.open(SOURCE) as image:
        image.load()
        if image.size != (512, 512):
            raise ValueError('Expected a 512 x 512 approved master.')
        source = image.convert('RGB')
    output.mkdir(parents=True, exist_ok=True)
    for filename, size in PNG_SIZES.items():
        target = output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        source.resize((size, size), Image.Resampling.LANCZOS).save(target, 'PNG', optimize=True)

    # Keep the whole approved artwork inside the maskable icon's safe circle.
    maskable = Image.new('RGB', (512, 512), source.getpixel((0, 0)))
    maskable.paste(source.resize((280, 280), Image.Resampling.LANCZOS), (116, 116))
    maskable.save(output / 'icons/mahjong-512-maskable.png', 'PNG', optimize=True)
    source.save(output / 'favicon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])

    for filename, size in {**PNG_SIZES, 'icons/mahjong-512-maskable.png': 512}.items():
        target = output / filename
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            image.load()
            if image.size != (size, size) or image.mode != 'RGB':
                raise ValueError(f'Invalid opaque PNG: {filename}')
        print(f'Validated {filename}: {size} x {size}')
    with Image.open(output / 'favicon.ico') as icon:
        if icon.ico.sizes() != {(16, 16), (32, 32), (48, 48)}:
            raise ValueError('Incorrect ICO resolutions.')
        for size in icon.ico.sizes():
            icon.ico.getimage(size).load()
    print('Validated favicon.ico: 16, 32 and 48 pixels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WEB / 'public')
    export_icons(parser.parse_args().output)
