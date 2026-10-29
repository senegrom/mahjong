import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

BASE = os.environ['BASE_SHA']
TARGET = os.environ['TARGET_BRANCH']
sha = lambda data: hashlib.sha256(data).hexdigest()
run = lambda *args, **kwargs: subprocess.check_output(args, **kwargs).decode().strip()
before = json.loads(Path('.poppy-prep/before.json').read_text())
record = json.loads(Path('docs/design/van-gogh/nine-characters-poppy-hillside-b.json').read_text())
runtime = record['runtime']
outputs = [runtime, 'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html', 'web/src/lib/van-gogh-faces.js']
first = {p: Path(p).read_bytes() for p in outputs}
subprocess.run(['node', 'web/scripts/export-van-gogh-tiles.mjs', '--only=9m'], check=True)
assert all(Path(p).read_bytes() == b for p, b in first.items()), 'Nondeterministic export'
current = json.loads(Path(outputs[1]).read_text())
assert [t for t in current['tiles'] if t['tile'] != '9m'] == [t for t in before['manifest']['tiles'] if t['tile'] != '9m'], 'Another manifest entry changed'
assert len(current['tiles']) == 34 and not current['remaining']
for p, expected in before['artwork'].items():
    if p != runtime:
        assert sha(Path(p).read_bytes()) == expected, f'Unrelated artwork changed: {p}'
assert sha(Path(record['archivedSvg']).read_bytes()) == before['artwork'][runtime]
assert sha(Path(runtime).read_bytes()) == '2f1303facc7868b37827b9feace26618adca6c69fd2947758b7dc92bb5416c79'
allowed = [
    'docs/design/van-gogh/README.md',
    'docs/design/van-gogh/nine-characters-poppy-hillside-b.json',
    record['source'], record['archivedSvg'],
    'web/public/tiles/van-gogh/README.md', runtime,
    'web/public/tiles/van-gogh/manifest.json', 'web/public/tiles/van-gogh/preview.html',
    'web/scripts/export-van-gogh-tiles.mjs', 'web/src/lib/van-gogh-faces.js',
    'web/tests/tile-faces.test.js', 'web/tests/tile-proportions.test.js', 'web/tests/van-gogh-nine-characters.test.js',
]
tracked = set(run('git', 'diff', '--name-only', BASE).splitlines())
untracked = set(run('git', 'ls-files', '--others', '--exclude-standard').splitlines())
for p in tracked | untracked:
    assert p in allowed or p.startswith('.poppy-prep/') or p == '.github/workflows/prepare-poppy-art.yml', f'Unexpected change: {p}'
subprocess.run(['git', 'diff', '--check'], check=True)
subprocess.run(['git', 'config', 'user.name', 'senegrom'], check=True)
subprocess.run(['git', 'config', 'user.email', '6349874+senegrom@users.noreply.github.com'], check=True)
with tempfile.TemporaryDirectory() as temporary:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
    subprocess.run(['git', 'read-tree', BASE], env=env, check=True)
    subprocess.run(['git', 'add', '-A', '--', *allowed], env=env, check=True)
    changed = set(run('git', 'diff', '--cached', '--name-only', '--no-renames', BASE, env=env).splitlines())
    assert changed == set(allowed), f'Unexpected final diff: {changed.symmetric_difference(allowed)}'
    tree = run('git', 'write-tree', env=env)
    message = 'Deploy Van Gogh Nine Characters — Poppy Hillside B\n\nInstall Carl\'s exact selected middle painting as 9m / Man9. Preserve the entire 3:4 portrait with zero stretch and zero extra bleed. Archive the previous smaller-Wan Poppy Field SVG unchanged and retain its source and provenance.\n\nFocused tile tests, scoped ESLint, checksum verification, deterministic export and preservation checks passed. All 33 other Van Gogh faces and unrelated artwork are unchanged. No preparation files, workflows, dependency, security-policy or gameplay changes. Publish through existing full CI and Pages gates.'
    commit = run('git', 'commit-tree', tree, '-p', BASE, '-m', message, env=env)
subprocess.run(['git', 'push', 'origin', f'{commit}:refs/heads/{TARGET}'], check=True)
print('PRESERVATION PASS: all 33 other Van Gogh entries and all unrelated artwork unchanged')
print('CLEAN_ARTWORK_COMMIT=' + commit)
print('RUNTIME_SHA256=' + record['runtimeSvgSha256'])
print('CHANGED_FILES=' + str(len(allowed)))
