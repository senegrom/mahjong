#!/usr/bin/env python3
"""Install the two approved faces, but only after validating their original PNGs.

Run at the repository root:
  python3 web/scripts/install-van-gogh-one-eight.py --archive van-gogh-one-eight-originals.zip
No network requests, image generation, resizing, or changes to existing art.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import zipfile

CONTRACT = 'docs/design/van-gogh/one-eight-approved.json'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_source(data: bytes, entry: dict) -> None:
    if len(data) != entry['sourceBytes'] or digest(data) != entry['sourceSha256']:
        raise ValueError(f"The approved original does not match: {entry['source']}")
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Expected a PNG source')
    dimensions = struct.unpack('>II', data[16:24])
    if dimensions != (entry['sourceWidth'], entry['sourceHeight']):
        raise ValueError('PNG dimensions do not match the approval')
    c = entry['crop']
    if not (0 <= c['x'] < dimensions[0] and 0 <= c['y'] < dimensions[1]
            and 0 < c['width'] <= dimensions[0] - c['x']
            and 0 < c['height'] <= dimensions[1] - c['y']):
        raise ValueError('The approved crop is outside the image')


def read_sources(root: Path, entries: list[dict], archive: Path | None) -> dict[str, bytes]:
    """Validate all bytes before writing. Never use ZipFile.extractall."""
    if archive is None:
        result = {e['source']: (root / e['source']).read_bytes() for e in entries}
    else:
        expected = {e['source']: e for e in entries}
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if len(infos) != len(entries) or {i.filename for i in infos} != set(expected):
                raise ValueError('Archive must contain exactly the two approved source paths')
            result = {}
            for info in infos:
                if info.file_size != expected[info.filename]['sourceBytes']:
                    raise ValueError('Unexpected ZIP entry size')
                result[info.filename] = z.read(info)
    for entry in entries:
        checked_source(result[entry['source']], entry)
        destination = root / entry['source']
        if destination.is_symlink():
            raise ValueError('Refusing to write through a symlink')
        if destination.exists() and destination.read_bytes() != result[entry['source']]:
            raise ValueError(f'Refusing to overwrite other artwork: {destination}')
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source destination escapes the repository')
    return result


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'Exporter or tests have changed; review before applying: {old!r}')
    return text.replace(old, new, 1)


def prepare_changes(root: Path, entries: list[dict]) -> dict[str, str]:
    script_path = 'web/scripts/export-van-gogh-tiles.mjs'
    test_path = 'web/tests/tile-faces.test.js'
    script = (root / script_path).read_text()
    if "'Sunlit Fields A'" in script or "'Starry Night B'" in script:
        raise ValueError('One/eight integration is already present; use the standard exporter')
    added_sources = '\n'.join(f"  {e['sourceId']}: '{e['source']}'," for e in entries)
    added_definitions = []
    for e in entries:
        c = e['crop']
        label = 'One character' if e['tile'] == '1m' else 'Eight characters'
        added_definitions.append(
            f"  ['{e['candidate']}', '{e['tile']}', '{label}', '{e['sourceId']}', "
            f"[{c['x']}, {c['y']}, {c['width']}, {c['height']}]],")
    script = once(script, '\n};\nconst definitions = [',
                  '\n' + added_sources + '\n};\nconst definitions = [')
    script = once(script, '\n];\nconst onlyArgument =',
                  '\n' + '\n'.join(added_definitions) + '\n];\nconst onlyArgument =')
    script = once(script, "new Set(['nightCafe', 'cypressFields'])",
                  "new Set(['nightCafe', 'cypressFields', 'starryEight'])")
    script = once(script, 'The later L is the active white dragon;',
                  'The user also approved Sunlit Fields A for 1m and Starry Night B for 8m. The later L is the active white dragon;')
    script = once(script, 'Characters C (4m), East A and North B included;',
                  'Characters C (4m), Sunlit Fields A (1m), Starry Night B (8m), East A and North B included;')
    script = once(script,
                  "const handTiles = ['2m', '3m', '4m', ...approved.filter(tile => !['2m', '3m', '4m'].includes(tile)).slice(0, 9), '2z', '6z'];",
                  "const featuredCharacters = ['1m', '2m', '3m', '4m', '8m'];\nconst handTiles = [...featuredCharacters, ...approved.filter(tile => !featuredCharacters.includes(tile)).slice(0, 7), '2z', '6z'];")
    script = once(script, 'including Night Cafe for 2 of characters,',
                  'including Sunlit Fields for 1 of characters, Starry Night for 8, Night Cafe for 2,')
    tests = (root / test_path).read_text()
    tests = once(tests, "'4z', '2m', '4m'];", "'4z', '2m', '4m', '1m', '8m'];")
    tests = once(tests, "'Characters B', 'Characters C']);",
                 "'Characters B', 'Characters C', 'Sunlit Fields A', 'Starry Night B']);")
    tests = once(tests, "url.startsWith('tiles/van-gogh/')).length, 14",
                 "url.startsWith('tiles/van-gogh/')).length, 16")
    tests = once(tests, 'assert.equal(set.remaining.length, 20);',
                 'assert.equal(set.remaining.length, 18);')
    tests = once(tests, "for (const tile of ['1z', '4z', '2m', '4m'])",
                 "for (const tile of ['1z', '4z', '2m', '4m', '1m', '8m'])")
    cases = json.dumps([{k: e[k] for k in ('tile', 'source', 'sourceSha256', 'crop', 'fullCanvas')} for e in entries])
    tests += """

test('Van Gogh one and eight use the approved integrated-scenery originals', () => {
  const set = JSON.parse(readFileSync(new URL('tiles/van-gogh/manifest.json', publicRoot), 'utf8'));
  const cases = CASES;
  for (const expected of cases) {
    const entry = set.tiles.find(tile => tile.tile === expected.tile);
    assert.ok(entry);
    assert.equal(entry.source, expected.source);
    assert.deepEqual(entry.crop, expected.crop);
    const original = readFileSync(new URL(`../../${entry.source}`, import.meta.url));
    assert.equal(createHash('sha256').update(original).digest('hex'), expected.sourceSha256);
    if (expected.fullCanvas) assert.deepEqual(readFileSync(new URL(`tiles/van-gogh/${entry.png}`, publicRoot)), original);
    assert.ok(TILE_IMAGE_URLS.includes(tileImage(expected.tile, 'van-gogh')));
    assert.ok(!set.remaining.includes(expected.tile));
  }
});
""".replace('CASES', cases)
    changes = {script_path: script, test_path: tests}
    for rel in ('docs/design/van-gogh/README.md', 'web/public/tiles/van-gogh/README.md'):
        p = root / rel
        if p.exists():
            text = p.read_text()
            for old, new in [('fourteen distinct faces', 'sixteen distinct faces'),
                             ('fourteen approved faces', 'sixteen approved faces'),
                             ('remaining 20 tile identities', 'remaining 18 tile identities'),
                             ('other 20 identities', 'other 18 identities'),
                             ('all fourteen faces', 'all sixteen faces')]:
                text = text.replace(old, new)
            text += '\n## Sunlit Fields 1 and Starry Night 8\n\nThe revised Sunlit Fields A is active for `1m`, with the single blue cloud band forming the numeral. The approved Starry Night companion is active for `8m`, with the two diverging luminous sky forms creating the numeral. The exact source PNGs and crop coordinates are recorded in `docs/design/van-gogh/one-eight-approved.json`; no earlier artwork has been repainted or replaced.\n'
            changes[rel] = text
    return changes


def install(root: Path, archive: Path | None) -> None:
    contract_path = root / CONTRACT
    contract = json.loads(contract_path.read_text())
    entries = contract['tiles']
    sources = read_sources(root, entries, archive)
    changes = prepare_changes(root, entries)  # Validate all patch anchors before writes.
    approved = root / 'web/public/tiles/van-gogh/approved'
    before = {p: digest(p.read_bytes()) for p in approved.iterdir() if p.is_file()}
    for rel, data in sources.items():
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
    for rel, text in changes.items():
        (root / rel).write_text(text, encoding='utf-8')
    command = ['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=1m,8m']
    subprocess.run(command, cwd=root, check=True)
    for p, expected in before.items():
        if digest(p.read_bytes()) != expected:
            raise ValueError(f'Existing artwork changed: {p}')
    out = root / 'web/public/tiles/van-gogh'
    manifest = json.loads((out / 'manifest.json').read_text())
    if len(manifest['tiles']) != 16 or len(manifest['remaining']) != 18:
        raise ValueError('Unexpected approved/fallback tile count')
    for expected in entries:
        entry = next(e for e in manifest['tiles'] if e['tile'] == expected['tile'])
        if entry['source'] != expected['source'] or entry['crop'] != expected['crop']:
            raise ValueError('Wrong source or crop used')
        png = (out / entry['png']).read_bytes()
        if expected['fullCanvas'] and png != sources[expected['source']]:
            raise ValueError('Full-canvas artwork changed')
        if digest(png) != entry['pngSha256']:
            raise ValueError('Production PNG checksum mismatch')
    html = (out / 'preview.html').read_text()
    rack = re.search(r'<div class="rack" id="rack">(.*?)</div>', html)
    if rack is None or rack.group(1).count('<img ') != 14:
        raise ValueError('Preview must contain a 14-tile sample hand')
    exported = {p: digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}
    subprocess.run(command, cwd=root, check=True)
    if any(digest(p.read_bytes()) != h for p, h in exported.items()):
        raise ValueError('Re-export was not deterministic')
    contract['state'] = 'integrated-awaiting-production-ci'
    contract['sourceUploadVerified'] = True
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + '\n')
    print('Installed 1m and 8m from the exact approved originals; all existing art unchanged.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        install(args.root.resolve(), args.archive.resolve() if args.archive else None)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f'Not installed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
