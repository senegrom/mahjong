"""Offline Git regressions: publication may not alter a tree after testing."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import runtime_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.original = Path.cwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        os.chdir(self.repo)
        self.addCleanup(os.chdir, self.original)
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'CI test')
        self.run_git('config', 'user.email', 'ci@example.invalid')
        for name in (*publication.RUNTIME_FILES, 'source.txt'):
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original\n')
        self.run_git('add', '.')
        self.run_git('commit', '-qm', 'base')
        self.head = publication.git('rev-parse', 'HEAD')
        self.receipt = self.repo / 'receipt.json'

    def run_git(self, *args):
        subprocess.run(['git', *args], check=True, stdout=subprocess.DEVNULL)

    def record(self):
        Path(publication.RUNTIME_FILES[0]).write_bytes(b'\x00new runtime\xff')
        return publication.snapshot(self.receipt, self.head)

    def test_unchanged_tree_has_no_publication(self):
        result = publication.snapshot(self.receipt, self.head)
        self.assertEqual(result['changed_paths'], [])
        self.assertEqual(publication.verify(self.receipt, self.head), result)

    def test_verified_commit_preserves_exact_tree_and_parent(self):
        result = self.record()
        publication.verify(self.receipt, self.head)
        self.run_git('commit', '-qm', 'runtime')
        self.assertEqual(publication.git('rev-parse', 'HEAD^{tree}'), result['tested_tree'])
        self.assertEqual(publication.git('rev-parse', 'HEAD^'), self.head)

    def test_rejects_unrelated_changes_before_test(self):
        Path('source.txt').write_text('unexpected')
        with self.assertRaises(RuntimeError):
            self.record()

    def test_rejects_mutation_after_test(self):
        self.record()
        Path(publication.RUNTIME_FILES[0]).write_text('not tested')
        with self.assertRaises(RuntimeError):
            publication.verify(self.receipt, self.head)

    def test_rejects_restaged_mutation_after_test(self):
        self.record()
        Path(publication.RUNTIME_FILES[0]).write_text('not tested')
        self.run_git('add', publication.RUNTIME_FILES[0])
        with self.assertRaises(RuntimeError):
            publication.verify(self.receipt, self.head)

    def test_rejects_head_movement(self):
        self.record()
        self.run_git('commit', '-qm', 'moved')
        with self.assertRaises(RuntimeError):
            publication.verify(self.receipt, self.head)

    def test_rejects_unrelated_staged_changes(self):
        self.record()
        Path('source.txt').write_text('unexpected')
        self.run_git('add', 'source.txt')
        with self.assertRaises(RuntimeError):
            publication.verify(self.receipt, self.head)


if __name__ == '__main__':
    unittest.main()
