# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes.inference import HawkesBasisKernels


class Test(unittest.TestCase):
    def test_em_basis_kernels(self):
        """...Test fit method of HawkesBasisKernels
        """
        ticks = [[
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ], [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19])
        ]]

        n_basis = 2
        n_nodes = len(ticks[0])

        kernel_support = 4
        kernel_dt = .1
        kernel_size = int(np.ceil(kernel_support / kernel_dt))

        C = 5e-2

        mu = np.zeros(n_nodes) + .2
        auvd = np.zeros((n_nodes, n_nodes, n_basis)) + .4
        auvd[1, :, :] += .2
        gdm = np.zeros((n_basis, kernel_size))
        for i in range(int(kernel_size)):
            gdm[0, i] = 0.1 * 0.29 * np.exp(-0.29 * i * kernel_dt)
        for i in range(int(kernel_size)):
            gdm[1, i] = 0.8 * 1 * np.exp(-1 * i * kernel_dt)

        em = HawkesBasisKernels(kernel_support=kernel_support,
                                kernel_size=kernel_size, n_basis=n_basis, C=C,
                                n_threads=2, max_iter=5, ode_max_iter=100)

        em.fit(ticks, baseline_start=mu, amplitudes_start=auvd,
               basis_kernels_start=gdm)

        np.testing.assert_array_almost_equal(em.baseline, [0.153022, 0.179124],
                                             decimal=4)

        np.testing.assert_array_almost_equal(
            em.amplitudes,
            [[[1.21125e-05, 1.744123e-03], [2.267314e-05, 3.287014e-03]], [[
                1.48773260e-05, 2.06898364e-03
            ], [6.60131078e-06, 7.28397551e-04]]], decimal=4)

        basis_kernels = np.array([[
            0.0001699, 0.00031211, 0.00043944, 0.0005521, 0.00066688,
            0.00078411, 0.0009040, 0.00101736, 0.001112, 0.00119935,
            0.00129047, 0.00135828, 0.0014302, 0.00146572, 0.00149012,
            0.00150987, 0.00152401, 0.00153267, 0.0015464, 0.00156525,
            0.00157363, 0.00156589, 0.00156298, 0.00155548, 0.0015339,
            0.00149196, 0.0014178, 0.0013323, 0.00125075, 0.00117292,
            0.0010985, 0.00100652, 0.00091741, 0.00082029, 0.00071975,
            0.00062118, 0.0005242, 0.0004228, 0.00029559, 0.00015301
        ], [
            0.0036240, 0.0066125, 0.00929557, 0.01163643, 0.01404666,
            0.01653209, 0.0190978, 0.0215138, 0.02351321, 0.02535836,
            0.02730293, 0.02874743, 0.0302991, 0.03096259, 0.03135936,
            0.0316486, 0.03182045, 0.03187917, 0.0320631, 0.03237183,
            0.03243794, 0.03216082, 0.03200454, 0.0317527, 0.0311904,
            0.03017171, 0.02846015, 0.02650652, 0.02465883, 0.02291335,
            0.0212655, 0.01931795, 0.01745896, 0.01547088, 0.01346942,
            0.01154156, 0.0096817, 0.00779345, 0.00543011, 0.00279355
        ]])

        np.testing.assert_array_almost_equal(em.basis_kernels, basis_kernels,
                                             decimal=3)


if __name__ == "__main__":
    unittest.main()
