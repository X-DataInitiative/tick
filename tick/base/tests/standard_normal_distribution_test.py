# License: BSD 3 clause

# -*- coding: utf8 -*-
import unittest

from tick.base.build.base import standard_normal_cdf, \
    standard_normal_inv_cdf

from scipy.stats import norm
import numpy as np
from numpy.random import normal, uniform


class Test(unittest.TestCase):
    def setUp(self):
        self.size = 10

    def test_standard_normal_cdf(self):
        """...Test normal cumulative distribution function
        """
        tested_sample = normal(size=self.size)
        actual = np.array([standard_normal_cdf(s) for s in tested_sample])
        expected = norm.cdf(tested_sample)

        np.testing.assert_almost_equal(actual, expected, decimal=7)

    def test_standard_normal_inv_cdf(self):
        """...Test inverse of normal cumulative distribution function
        """
        tested_sample = uniform(size=self.size)
        actual = np.array([standard_normal_inv_cdf(s) for s in tested_sample])
        expected = norm.ppf(tested_sample)
        np.testing.assert_almost_equal(actual, expected, decimal=7)

        actual_array = np.empty(self.size)
        standard_normal_inv_cdf(tested_sample, actual_array)
        np.testing.assert_almost_equal(actual_array, expected, decimal=7)
