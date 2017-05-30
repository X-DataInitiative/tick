import unittest
import numpy as np
import itertools
from tick.simulation import SimuSCCS


class Test(unittest.TestCase):

    def test_censoring(self):
        array_list = [np.ones((2, 3)) for i in range(3)]
        expected = [np.zeros((2, 3)) for i in range(3)]
        for i in range(1, 3):
            expected[i][:i] += 1
        censoring = np.arange(3)

        output = SimuSCCS._censor_array_list(array_list, censoring)

        for i in range(3):
            np.testing.assert_equal(output[i], expected[i])

    def test_filter_non_positive_samples(self):
        features = [np.ones((2, 3)) * i for i in range(10)]
        labels = [np.zeros((2, 1)) for i in range(10)]
        censoring = np.full((10, 1), 2)

        expected_idx = np.sort(np.random.choice(np.arange(10), 5, False))
        for i in expected_idx:
            labels[i][i % 2] = 1
        expect_feat = [features[i] for i in expected_idx]
        expect_lab = [labels[i] for i in expected_idx]
        expect_cens = censoring[expected_idx]

        out_feat, out_lab, out_cens, out_idx = SimuSCCS\
            ._filter_non_positive_samples(features, labels, censoring)

        np.testing.assert_array_equal(expect_cens, out_cens)
        for i in range(len(expect_cens)):
            np.testing.assert_array_equal(expect_feat[i], out_feat[i])
            np.testing.assert_array_equal(expect_lab[i], out_lab[i])
            self.assertGreater(expect_lab[i].sum(), 0)

    def test_simulation(self):
        def run_tests(n_samples, n_features, sparse, exposure_type,
                      distribution, first_tick_only, censoring):
            n_intervals = 5
            n_lags = 2
            sim = SimuSCCS(n_samples, n_intervals, n_features, n_lags, None,
                           sparse, exposure_type, distribution, first_tick_only,
                           censoring, seed=42, verbose=False)
            X, y, c, coeffs = sim.simulate()
            self.assertEqual(len(X), n_samples)
            self.assertEqual(len(y), n_samples)
            self.assertEqual(X[0].shape, (n_intervals, n_features))
            self.assertEqual(y[0].shape, (n_intervals,))
            self.assertEqual(c.shape, (n_samples,))
            self.assertEqual(coeffs.shape, (n_features * (n_lags + 1),))

        n_samples = [1, 100]
        n_features = [1, 3]
        exposure_type = ["infinite", "short"]
        distribution = ["multinomial", "poisson"]
        first_tick_only = [True, False]
        censoring = [True, False]
        sparse = [True, False]
        for n, n_feat, e, d, f, c in itertools.product(n_samples,
                                                       n_features,
                                                       exposure_type,
                                                       distribution,
                                                       first_tick_only,
                                                       censoring):
            if e == "short":
                for s in sparse:
                    run_tests(n, n_feat, s, e, d, f, c)
            else:
                run_tests(n, n_feat, True, e, d, f, c)


if __name__ == '__main__':
    unittest.main()
