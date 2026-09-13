"""Old search backups must be rejected without changing training compatibility."""
import unittest
from unittest.mock import patch

import riichi_py

from neural.training_safety import require_search_engine, require_training_engine


class SearchNativeVersionTests(unittest.TestCase):
    def test_rebuilt_engine_supports_both_training_and_corrected_search(self):
        require_training_engine()
        require_search_engine()
        self.assertEqual(riichi_py.SEARCH_API_VERSION, 1)

    def test_old_or_unknown_search_api_is_refused_without_breaking_training(self):
        for version in (0, 2, None):
            with self.subTest(version=version), patch.object(riichi_py, 'SEARCH_API_VERSION', version):
                require_training_engine()
                with self.assertRaisesRegex(RuntimeError, 'Rebuild and reinstall'):
                    require_search_engine()

    def test_missing_search_version_is_also_refused(self):
        with patch.object(riichi_py, 'SEARCH_API_VERSION', 1):
            del riichi_py.SEARCH_API_VERSION
            with self.assertRaisesRegex(RuntimeError, 'Rebuild and reinstall'):
                require_search_engine()


if __name__ == '__main__':
    unittest.main()
