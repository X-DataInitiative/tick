# License: BSD 3 clause

import errno
import os
import platform
import unittest

from tick.base.build.base import throw_out_of_range, \
    throw_system_error, throw_invalid_argument, throw_domain_error, \
    throw_runtime_error, throw_string


class Test(unittest.TestCase):
    def test_throw_out_of_range(self):
        """...Test C++ out of range errors are correctly caught
        """
        with self.assertRaisesRegex(IndexError, "out_of_range"):
            throw_out_of_range()

    def test_throw_system_error(self):
        """...Test C++ system errors are correctly caught
        """
        ## Windows returns "permission denied" rather than "Permission denied"
        if platform.system() == 'Windows':
            with self.assertRaisesRegex(RuntimeError,
                                        os.strerror(errno.EACCES).lower()):
                throw_system_error()
        else:
            with self.assertRaisesRegex(RuntimeError,
                                        os.strerror(errno.EACCES)):
                throw_system_error()

    def test_throw_invalid_argument(self):
        """...Test C++ invalid argument errors are correctly caught
        """
        with self.assertRaisesRegex(ValueError, "invalid_argument"):
            throw_invalid_argument()

    def test_throw_domain_error(self):
        """...Test C++ domain errors are correctly caught
        """
        with self.assertRaisesRegex(ValueError, "domain_error"):
            throw_domain_error()

    def test_throw_runtime_error(self):
        """...Test C++ runtime errors are correctly caught
        """
        with self.assertRaisesRegex(RuntimeError, "runtime_error"):
            throw_runtime_error()

    def test_throw_string(self):
        """...Test C++ throw str errors are correctly caught
        """
        with self.assertRaisesRegex(RuntimeError, "string"):
            throw_string()
