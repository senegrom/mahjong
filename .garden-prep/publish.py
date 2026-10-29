import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

BASE = os.environ['BASE_SHA']
TARGET = os.environ['TARGET_BRANCH']
def sha(data):
    return hashlib.sha256(data).hexdigest()
def run(*args, **kwargs):
    return subprocess.check_output(args, **kwargs).decode().strip()

before = json.loads(Path('.garden-prep/before.json').read_text())
record = json.loads(Path('docs/design/van-gogh/four-bamboo-sunlit-garden-b.json').read_text())
runtime = record['runtime']
removed = 'web/public/tiles/van-gogh/approved/Sou4.png'
outputs = [runtime, 'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js']
first = {p: Path(p).read_bytes() for p in outputs}
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=4s'], check=True)
assert all(Path(p).read_bytes() == b for p, b in first.items()), 'Nondeterministic export'
current = json.loads(Path(outputs[1]).read_text())
assert [t for t in current['tiles'] if t['tile'] != '4s'] == [t for t in before['manifest']['tiles'] if t['tile'] != '4s'], 'Another manifest entry changed'
assert len(current['tiles']) == 34 and not current['remaining']
for path, expected in before['artwork'].items():
    if path not in (runtime, removed):
        assert sha(Path(path).read_bytes()) == expected, f'Unrelated artwork changed: {path}'
assert sha(Path(record['archivedSvg']).read_bytes()) == before['artwork'][runtime]
assert sha(Path(record['archivedPng']).read_bytes()) == before['artwork'][removed]
assert not Path(removed).exists()
assert sha(Path(runtime).read_bytes()) == '574e208e0e6dbb4a247b1d58fc09a2e9a610bf43d64d9d7a2948d1f5c2efe841'
allowed = [
    'docs/design/van-gogh/README.md',
    'docs/design/van-gogh/four-bamboo-sunlit-garden-b.json',
    record['source'], record['archivedSvg'], record['archivedPng'],
    'web/public/tiles/van-gogh/README.md', runtime, removed,
    'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html',
    'web/scripts/export-van-gogh-tiles.mjs', 'web/src/lib/van-gogh-faces.js',
    'web/tests/tile-faces.test.js', 'web/tests/tile-proportions.test.js', 'web/tests/van-gogh-four-garden.test.js',
]
tracked = set(run('git', 'diff', '--name-only', BASE).splitlines())
untracked = set(run('git', 'ls-files', '--others', '--exclude-standard').splitlines())
for p in tracked | untracked:
    assert p in allowed or p.startswith('.garden-prep/') or p == '.github/workflows/prepare-garden-art.yml', f'Unexpected change: {p}'
subprocess.run(['git', 'config', 'user.name', 'senegrom'], check=True)
subprocess.run(['git', 'config', 'user.email', '6349874+senegrom@users.noreply.github.com'], check=True)
with tempfile.TemporaryDirectory() as temporary:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
    subprocess.run(['git', 'read-tree', BASE], env=env, check=True)
    subprocess.run(['git', 'add', '-A', '--', *allowed], env=env, check=True)
    changed = set(run('git', 'diff', '--cached', '--name-only', '--no-renames', BASE, env=env).splitlines())
    assert changed == set(allowed), f'Unexpected final diff: {changed.symmetric_difference(allowed)}'
    tree = run('git', 'write-tree', env=env)
    message = 'Deploy Van Gogh Four Bamboo — Sunlit Garden B\n\nInstall the selected middle sunlit garden painting as 4s / Sou4. Preserve the documented 3:4 crop with four stalks, stone wall and terracotta pot, without stretching, repainting or recolouring. Archive Moonlit Four PNG and SVG byte-for-byte. All other 33 Van Gogh faces and unrelated artwork are unchanged.\n\nFocused tile tests, scoped ESLint, checksum verification, deterministic export and preservation checks passed. No preparation files, workflow changes, dependency changes or gameplay changes in this commit. Publish through existing full CI and Pages gates.'
    commit = run('git', 'commit-tree', tree, '-p', BASE, '-m', message, env=env)
subprocess.run(['git', 'push', 'origin', f'{commit}:refs/heads/{TARGET}'], check=True)
print('PRESERVATION PASS: all 33 other Van Gogh entries and all unrelated artwork unchanged')
print('CLEAN_ARTWORK_COMMIT=' + commit)
print('RUNTIME_SHA256=' + record['runtimeSvgSha256'])
print('CHANGED_FILES=' + str(len(allowed)))
