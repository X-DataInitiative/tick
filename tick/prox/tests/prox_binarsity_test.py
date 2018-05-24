# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.prox import ProxBinarsity
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxBinarsity(self):
        """...Test of ProxBinarsity
        """
        coeffs = self.coeffs.copy()
        l_binarsity = 0.5
        t = 1.7
        out = np.array([
            0., 0., 0., -0.72681389, -0.72681389, 0.4845426, 0.4845426,
            0.4845426, 0., 0.
        ])
        blocks_start = [0, 3, 8]
        blocks_length = [3, 5, 2]
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length)

        val = 0
        for j, d_j in enumerate(blocks_length):
            start = blocks_start[j]
            val += np.abs(coeffs[start + 1:start + d_j] -
                          coeffs[start:start + d_j - 1]).sum()
        val *= l_binarsity

        self.assertAlmostEqual(prox.value(coeffs), val, delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)

    def test_ProxBinarsity_n_blocks(self):
        """...Test parameter n_blocks
        """
        l_binarsity = 0.5
        blocks_start = [0, 3, 8, 34]
        blocks_length = [3, 5, 2, 23]
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length)
        self.assertEqual(prox.n_blocks, 4)

    def test_ProxBinarsity_penalty_parameters_setting(self):
        """...Test ProxBinarsity parameters setting
        """
        l_binarsity = 0.5

        blocks_start = [0, 3, 8]
        blocks_length = [3, 5]
        msg = '^``blocks_start`` and ``blocks_length`` must have the same size$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length)

        blocks_start = [0, 3, 8]
        blocks_length = [4, 5, 1]
        msg = '^blocks must not overlap$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length)

        # test not overlapping but not penalizing all coefficients either
        blocks_start = [0, 3, 8]
        blocks_length = [2, 2, 2]
        ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                      blocks_length=blocks_length)

        blocks_start = [0, 8, 3]
        blocks_length = [1, 1, 1]
        msg = '^``block_start`` must be sorted$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length)

        blocks_start = [0, 3, 8]
        blocks_length = [-3, 5, 2]
        msg = '^all blocks must be of positive size$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length)

        blocks_start = [0, -3, 8]
        blocks_length = [3, 5, 2]
        msg = '^all blocks must have positive starting indices$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length)

        blocks_start = [0, 3, 8]
        blocks_length = [3, 5, 2]
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length)

        msg = '^blocks_length and blocks_start must have the same size$'
        with self.assertRaisesRegex(ValueError, msg):
            prox.blocks_length = np.array([3, 5, 2, 3], dtype=np.uint64)

        blocks_start = [0, 4, 8]
        blocks_length = [3, 3, 10]
        msg = '^last block is not within the range \[0, end-start\)$'
        with self.assertRaisesRegex(ValueError, msg):
            ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                          blocks_length=blocks_length, range=(0, 17))

    def test_prox_binarsity_range(self):
        """...Test ProxBinarsity deals with range correctly
        """
        np.random.seed(2093)
        coeffs = np.random.randn(10)
        # put a very high strength that will push the penalized coefficients
        # to zero
        l_binarsity = 1e2
        blocks_start = [1]
        blocks_length = [3]
        prox_range = (3, 7)
        prox = ProxBinarsity(strength=l_binarsity, blocks_start=blocks_start,
                             blocks_length=blocks_length, range=prox_range)
        start_penalized_coeff = prox_range[0] + blocks_start[0]
        end_penalized_coeff = prox_range[0] + blocks_start[0] + blocks_length[0]
        self.assertTrue(
            all(
                prox.call(coeffs)[start_penalized_coeff:end_penalized_coeff] ==
                0))


if __name__ == '__main__':
    unittest.main()
