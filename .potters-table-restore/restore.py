#!/usr/bin/env python3
"""Restore the approved Potter's Table C without uploading or changing artwork."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess

ROOT = Path.cwd()
OUT = Path('web/public/tiles/van-gogh')
DOCS = Path('docs/design/van-gogh')
SVG_SHA = '7bfed3b794e8a09dc61f2020662d9c70572d883f05194267df07f535778591b1'
PNG_SHA = '61eb9c2563a8039eb755315b1f5ef670a472fcff2e1facaef4b0e05e1af0267d'
OLD_SVG_SHA = '34f0fd31056df47c064383ca79b67d17aa9bd5a62a7aae1d3f9188701e98a89a'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    check(text.count(old) == 1, f'Patch anchor changed: {old[:100]}')
    return text.replace(old, new, 1)


record = json.loads((DOCS / 'nine-disks-potters-table-c.json').read_text())
prior = json.loads((OUT / 'manifest.json').read_text())
old_entry = next(t for t in prior['tiles'] if t['tile'] == '9p')
check(old_entry['candidate'] == 'I', '9 disks changed since approval; review before replacing')
source = Path(record['source'])
check(source == DOCS / 'studies/17-nine-disks-potters-table-c-approved.svg', 'Wrong selected source')
art = source.read_bytes()
check(sha(art) == SVG_SHA == record['svgSha256'], 'Approved Potter source checksum mismatch')
check((DOCS / 'superseded/potters-table-Pin9.svg').read_bytes() == art, 'Potter archive mismatch')
match = re.search(rb'href="data:image/webp;base64,([A-Za-z0-9+/=]+)"', art)
check(match is not None, 'Missing embedded artwork')
check(sha(base64.b64decode(match[1], validate=True)) == record['rasterSha256'], 'Raster mismatch')
old_png = (OUT / 'approved/Pin9.png').read_bytes()
old_svg = (OUT / 'approved/Pin9.svg').read_bytes()
check(sha(old_png) == PNG_SHA and sha(old_svg) == OLD_SVG_SHA, 'Nine Stars runtime has changed')
check(Path(record['supersedes']['archivedPng']).read_bytes() == old_png, 'Nine Stars PNG not preserved')
check(Path(record['supersedes']['archivedSvg']).read_bytes() == old_svg, 'Nine Stars SVG not preserved')
tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
mutable = {OUT / 'approved/Pin9.svg', OUT / 'approved/Pin9.png'}
preserved = {Path(p): sha(Path(p).read_bytes()) for p in tracked if p and Path(p).suffix.lower() in {'.png', '.svg', '.webp', '.jpg', '.jpeg', '.gif'} and Path(p).is_file() and Path(p) not in mutable}

exporter = Path('web/scripts/export-van-gogh-tiles.mjs')
text = exporter.read_text()
replacements = [
    ("['I', '9p', 'Nine disks', 'nineStarsOriginal', [0, 0, 380, 471]]", "['Disks C', '9p', 'Nine disks', 'pottersTable', [0, 0, 300, 400]]"),
    ("{ candidate: 'Disks C', tile: '9p', activeCandidate: 'I', source: sources.pottersTable, archivedSvg: 'docs/design/van-gogh/superseded/potters-table-Pin9.svg' }", "{ candidate: 'I', tile: '9p', activeCandidate: 'Disks C', source: sources.second, archivedPng: 'docs/design/van-gogh/superseded/nine-stars-Pin9.png', archivedSvg: 'docs/design/van-gogh/superseded/nine-stars-Pin9.svg' }"),
    ('Carl subsequently rejected the eight-star adaptation and requested the original nine-star face as 9 disks. Nine Stars I is now restored byte-for-byte and The Potter’s Table C is archived; 8 disks is unchanged.', 'The intervening Nine Stars restoration was a misunderstanding. Carl explicitly clarified that The Potter’s Table C must replace Nine Stars as 9 disks. The approved Potter SVG is restored byte-for-byte; Nine Stars remains archived. The eight-star adaptation is not deployed and 8 disks is unchanged.'),
    ('Nine Stars I (9p), restored from the original archive', 'The Potter’s Table C (9p), restored after the clarified selection'),
    ('Nine Stars I for 9 disks', 'The Potter’s Table C for 9 disks'),
    ('Nine Stars I uses its exact original 380 × 471 PNG and original SVG wrapper, with no resizing or repainting. The Potter’s Table C remains preserved in the design archive.', 'The Potter’s Table C uses its existing approved 300 × 400 quality-95 WebP game export, copied byte-for-byte without repainting or recompression. Nine Stars I remains preserved in the design archive.'),
]
for old, new in replacements:
    text = once(text, old, new)
changes = {exporter: text}
tile_tests = Path('web/tests/tile-faces.test.js')
changes[tile_tests] = once(tile_tests.read_text(), "'G', 'H', 'I', 'Six B (green)'", "'G', 'H', 'Disks C', 'Six B (green)'")
old_note = '**Current 9 disks:** Nine Stars I has been restored from the exact original PNG/SVG. The Potter’s Table C is preserved in the design archive. The eight-star adaptation is not deployed; 8 disks is unchanged. Earlier sections below describe the design history.'
new_note = '**Current 9 disks:** The Potter’s Table C is active, replacing Nine Stars I. This corrects the misunderstood restoration: Carl explicitly chose the pottery painting. The approved SVG is copied byte-for-byte, and Nine Stars remains preserved in the design archive. 8 disks and all other tiles are unchanged. Earlier sections below describe the design history.'
for doc in (DOCS / 'README.md', OUT / 'README.md'):
    updated = once(doc.read_text(), old_note, new_note)
    if doc == DOCS / 'README.md':
        updated = once(updated, 'and the restored Nine Stars I for 9 disks.', 'and The Potter’s Table C for 9 disks.')
    else:
        updated = once(updated, 'The selected studies are A–B, E, G–I and L, plus', 'The selected studies are A–B, E, G–H and L, plus **The Potter’s Table C** for 9 disks,')
    changes[doc] = updated

selection_path = DOCS / 'nine-disks-potters-table-reselected.json'
check(not selection_path.exists(), 'Correction record already exists')
selection = {
    'tile': '9p', 'candidate': 'Disks C', 'title': 'The Potter’s Table',
    'source': str(source), 'runtime': record['runtime'], 'svgSha256': SVG_SHA,
    'rasterSha256': record['rasterSha256'],
    'approval': 'Carl clarified: no, i want the potters table. replace 9 stars with potters table; then explicitly requested GitHub deployment.',
    'corrects': 'PR #95 restored Nine Stars after misunderstanding the requested nine. This record supersedes that active selection, not the preserved history.',
    'originalProvenance': 'docs/design/van-gogh/nine-disks-potters-table-c.json',
    'processing': 'Copy the existing approved Potter’s Table C SVG exactly. No generation, crop, recolouring, resizing or recompression in this restoration.',
    'nineStarsArchives': [record['supersedes']['archivedPng'], record['supersedes']['archivedSvg']],
    'eightDisksChanged': False,
}
# All input and patch anchors have been checked before any writes.
for file, content in changes.items():
    file.write_text(content, encoding='utf-8')
selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
# Remove only the stale runtime PNG after proving both archived originals are intact.
(OUT / 'approved/Pin9.png').unlink()
command = ['node', str(exporter), '--only=9p']
subprocess.run(command, check=True)
check((OUT / 'approved/Pin9.svg').read_bytes() == art, 'Runtime differs from approved C')
for file, expected in preserved.items():
    check(file.exists() and sha(file.read_bytes()) == expected, f'Unrelated artwork changed: {file}')
after = json.loads((OUT / 'manifest.json').read_text())
check([t for t in after['tiles'] if t['tile'] != '9p'] == [t for t in prior['tiles'] if t['tile'] != '9p'], 'Other tile entries changed')
check(after['remaining'] == prior['remaining'] and len(after['tiles']) == len(prior['tiles']), 'Coverage changed')
check(after['sources'] == prior['sources'], 'Source records changed')
preview = (OUT / 'preview.html').read_text()
rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', preview)
check(rack is not None and rack[1].count('<img ') == 14, 'Preview hand size changed')
exports = [OUT / 'approved/Pin9.svg', OUT / 'manifest.json', OUT / 'preview.html', Path('web/src/lib/van-gogh-faces.js')]
first = {p: p.read_bytes() for p in exports}
subprocess.run(command, check=True)
check(all(p.read_bytes() == b for p, b in first.items()), 'Re-export is not deterministic')
print(f'PASS: Potter C restored exactly; both Nine Stars archives verified; {len(preserved)} other image/source files and {len(after["tiles"])-1} other tile entries unchanged; 8 disks unchanged; deterministic export; 14-tile preview.')
