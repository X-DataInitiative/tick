# License: BSD 3 clause

import unittest
import numpy as np
import itertools
from tick.survival import SimuSCCS


class Test(unittest.TestCase):
    def test_censoring(self):
        array_list = [np.ones((2, 3)) for _ in range(3)]
        expected = [np.zeros((2, 3)) for _ in range(3)]
        for i in range(1, 3):
            expected[i][:i] += 1
        censoring = np.arange(3)

        output = SimuSCCS._censor_array_list(array_list, censoring)

        for i in range(3):
            np.testing.assert_equal(output[i], expected[i])

    def test_filter_non_positive_samples(self):
        features = [np.ones((2, 3)) * i for i in range(10)]
        labels = [np.zeros((2, 1)) for _ in range(10)]
        censoring = np.full((10, 1), 2)

        expected_idx = np.sort(np.random.choice(np.arange(10), 5, False))
        for i in expected_idx:
            labels[i][i % 2] = 1
        expect_feat = [features[i] for i in expected_idx]
        expect_feat_c = expect_feat
        expect_lab = [labels[i] for i in expected_idx]
        expect_cens = censoring[expected_idx]

        out_feat, out_feat_c, out_lab, out_cens, out_idx = SimuSCCS\
            ._filter_non_positive_samples(features, features, labels, censoring)

        np.testing.assert_array_equal(expect_cens, out_cens)
        for i in range(len(expect_cens)):
            np.testing.assert_array_equal(expect_feat[i], out_feat[i])
            np.testing.assert_array_equal(expect_feat_c[i], out_feat_c[i])
            np.testing.assert_array_equal(expect_lab[i], out_lab[i])
            self.assertGreater(expect_lab[i].sum(), 0)

    def test_simulated_features(self):
        n_features = 3
        n_lags = np.repeat(2, n_features)
        sim = SimuSCCS(100, 10, n_features, n_lags, None, 'multiple_exposures',
                       verbose=False)
        feat, n_samples = sim.simulate_features(100)
        self.assertEqual(100, len(feat))
        print(np.sum([1 for f in feat if f.sum() <= 0]))

    def test_simulation(self):
        def run_tests(n_cases, n_features, sparse, exposure_type, distribution,
                      time_drift):
            n_intervals = 5
            n_lags = np.repeat(2, n_features).astype('uint64')
            sim = SimuSCCS(n_cases, n_intervals, n_features, n_lags,
                           time_drift, exposure_type, distribution, sparse,
                           verbose=False)
            X, X_c, y, c, coeffs = sim.simulate()
            self.assertEqual(len(X), n_cases)
            self.assertEqual(len(y), n_cases)
            self.assertEqual(X[0].shape, (n_intervals, n_features))
            self.assertEqual(y[0].shape, (n_intervals,))
            self.assertEqual(c.shape, (n_cases,))
            [
                self.assertEqual(co.shape, (int(n_lags[i] + 1),))
                for i, co in enumerate(coeffs)
            ]
            self.assertEqual(np.sum([1 for f in X if f.sum() <= 0]), 0)
            self.assertEqual(np.sum([1 for f in X_c if f.sum() <= 0]), 0)

        n_features = [1, 3]
        exposure_type = ["single_exposure", "multiple_exposures"]
        distribution = ["multinomial", "poisson"]
        sparse = [True, False]
        time_drift = [None, lambda x: np.log(np.sin(x * 2) + 5)]
        for n_feat, e, d, td in itertools.product(n_features, exposure_type,
                                                  distribution, time_drift):
            if e == "multiple_exposures":
                for s in sparse:
                    run_tests(10, n_feat, s, e, d, td)
            else:
                run_tests(10, n_feat, True, e, d, td)


if __name__ == '__main__':
    unittest.main()
