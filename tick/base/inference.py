# License: BSD 3 clause

import unittest
from contextlib import contextmanager
import warnings


class InferenceTest(unittest.TestCase):
    @contextmanager
    def assertWarnsRegex(self, expected_warning, expected_regex):
        """Reimplement assertWarnsRegex method because Python 3.5 ones is buggy
        """
        with warnings.catch_warnings(record=True) as w:
            yield
            self.assertGreater(len(w), 0, "No warning have been raised")
            self.assertLess(
                len(w), 2, "Several warnings have been raised. "
                "Expected 1")
            self.assertTrue(
                issubclass(w[0].category, expected_warning),
                'Expected %s got %s' % (expected_warning, w[0].category))
            self.assertRegex(
                str(w[0].message), expected_regex,
                "Warnings regex do not match")
