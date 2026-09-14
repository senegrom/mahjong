"""Record and verify the exact staged runtime tree, without network or publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

RUNTIME_FILES = (
    'web/runtime/ort-wasm-simd-threaded.wasm',
    'web/runtime/ort-wasm-simd-threaded.stock.mjs',
    'web/runtime/ort-wasm-simd-threaded.mjs',
)


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True).strip()


def changed(*args: str) -> set[str]:
    data = subprocess.check_output(['git', 'diff', '--name-only', '-z', *args])
    return {name.decode('utf-8') for name in data.split(b'\0') if name}


def snapshot(receipt: Path, expected_head: str) -> dict[str, object]:
    if git('rev-parse', 'HEAD') != expected_head:
        raise RuntimeError('Checkout is not the requested source commit')
    if changed('--cached'):
        raise RuntimeError('Refusing a pre-populated staging area')
    paths = changed()
    if not paths.issubset(RUNTIME_FILES):
        raise RuntimeError('Unexpected tracked changes outside runtime outputs')
    for path in RUNTIME_FILES:
        subprocess.run(['git', 'ls-files', '--error-unmatch', '--', path],
                       check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'add', '--', *RUNTIME_FILES], check=True)
    if changed():
        raise RuntimeError('Working tree changed while recording the snapshot')
    result: dict[str, object] = {
        'source_commit': expected_head,
        'tested_tree': git('write-tree'),
        'changed_paths': sorted(changed('--cached')),
    }
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def verify(receipt: Path, expected_head: str) -> dict[str, object]:
    result = json.loads(receipt.read_text(encoding='utf-8'))
    if result.get('source_commit') != expected_head or git('rev-parse', 'HEAD') != expected_head:
        raise RuntimeError('Source commit changed after snapshot')
    if changed():
        raise RuntimeError('Tracked files changed during or after testing')
    paths = changed('--cached')
    if not paths.issubset(RUNTIME_FILES) or sorted(paths) != result.get('changed_paths'):
        raise RuntimeError('Staged paths differ from the tested snapshot')
    if git('write-tree') != result.get('tested_tree'):
        raise RuntimeError('Staged contents differ from the tested snapshot')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('snapshot', 'verify'))
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    result = (snapshot if args.operation == 'snapshot' else verify)(
        args.receipt, args.expected_head)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
