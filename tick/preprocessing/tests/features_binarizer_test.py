# License: BSD 3 clause

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr

from tick.preprocessing.features_binarizer import FeaturesBinarizer


class Test(unittest.TestCase):
    def setUp(self):

        self.features = np.array(
            [[0.00902084, 0.54159776, 0.,
              'z'], [0.46599565, -0.71875887, 0.,
                     2.], [0.52091721, -0.83803094, 1.,
                           2.], [0.47315496, 0.0730993, 1.,
                                 1.], [0.08180209, -1.11447889, 0., 0.],
             [0.45011727, -0.57931684, 0.,
              0.], [2.04347947, -0.10127498, 1.,
                    20.], [-0.98909384, 1.36281079, 0.,
                           0.], [-0.30637613, -0.19147753, 1.,
                                 1.], [0.27110903, 0.44583304, 0., 0.]])
        self.columns = [
            'c:continuous', 'a:continuous', 'd:discrete', 'b:discrete'
        ]
        self.df_features = pd.DataFrame(self.features, columns=self.columns)

        self.default_expected_intervals = np.array(
            [[0, 3, 0, 4], [2, 0, 0, 2], [3, 0, 1, 2], [2, 2, 1, 1],
             [1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 3], [0, 3, 0, 0],
             [0, 1, 1, 1], [1, 2, 0, 0]])

    def test_column_type_detection(self):
        """...Test column type detection
        """
        expected_column_types = [
            "continuous", "continuous", "discrete", "discrete"
        ]

        for i, expected_type in enumerate(expected_column_types):
            # auto detection and default continuous_threshold
            features_i = self.features[:, i]
            detected_type = FeaturesBinarizer._detect_feature_type(
                features_i, detect_column_type="auto",
                continuous_threshold="auto")
            self.assertEqual(expected_type, detected_type,
                             "failed for column %i" % i)

            # auto detection and continuous_threshold=7
            detected_type = FeaturesBinarizer._detect_feature_type(
                features_i, detect_column_type="auto", continuous_threshold=7)
            self.assertEqual(expected_type, detected_type,
                             "failed for column %i" % i)

            # column name detection
            detected_type = FeaturesBinarizer._detect_feature_type(
                features_i, detect_column_type="column_names",
                feature_name=self.columns[i])
            self.assertEqual(expected_type, detected_type,
                             "failed for column %i" % i)

        expected_column_types_with_threshold_equal_2 = \
            ["continuous", "continuous", "discrete", "discrete"]

        for i, expected_type in enumerate(
                expected_column_types_with_threshold_equal_2):
            # auto detection and continuous_threshold=2
            features_i = self.features[:, i]
            detected_type = FeaturesBinarizer._detect_feature_type(
                features_i, detect_column_type="auto", continuous_threshold=2)
            self.assertEqual(expected_type, detected_type,
                             "failed for column %i" % i)

    def test_quantile_boundaries_detection(self):
        """...Test boundaries detection for method `quantile`
        """
        n_cuts = 3
        binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                      detect_column_type="column_names",
                                      remove_first=False)
        # only for the two continuous features
        boundaries_0 = binarizer._get_boundaries(self.columns[0],
                                                 self.features[:, 0], fit=True)
        np.testing.assert_array_almost_equal(
            boundaries_0,
            np.array([-np.inf, 0.009021, 0.271109, 0.473155, np.inf]))

        boundaries_1 = binarizer._get_boundaries(self.columns[1],
                                                 self.features[:, 1], fit=True)
        np.testing.assert_array_almost_equal(
            boundaries_1,
            np.array([-np.inf, -0.718759, -0.191478, 0.445833, np.inf]))

    def test_linspace_boundaries_detection(self):
        """...Test boundaries detection for method `linspace`
        """
        n_cuts = 3
        binarizer = FeaturesBinarizer(method='linspace', n_cuts=n_cuts,
                                      detect_column_type="column_names",
                                      remove_first=False)
        # only for the two continuous features
        boundaries_0 = binarizer._get_boundaries(self.columns[0],
                                                 self.features[:, 0], fit=True)
        np.testing.assert_array_almost_equal(
            boundaries_0,
            np.array([-np.inf, -0.230951, 0.527193, 1.285336, np.inf]))

        boundaries_1 = binarizer._get_boundaries(self.columns[1],
                                                 self.features[:, 1], fit=True)
        np.testing.assert_array_almost_equal(
            boundaries_1,
            np.array([-np.inf, -0.495156, 0.124166, 0.743488, np.inf]))

    def test_assign_interval(self):
        """...Test interval assignment
        """
        n_cuts = 3
        binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                      detect_column_type="column_names",
                                      remove_first=False)

        for i, expected_interval in enumerate(
                self.default_expected_intervals.T):
            feature_name = self.columns[i]
            features_i = self.features[:, i]
            interval = binarizer._assign_interval(feature_name, features_i,
                                                  fit=True)
            np.testing.assert_array_equal(expected_interval, interval)

    def test_binarizer_fit(self):
        """...Test binarizer fit
        """
        n_cuts = 3
        enc = OneHotEncoder(sparse=True)
        expected_binarization = enc.fit_transform(
            self.default_expected_intervals)

        binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                      detect_column_type="auto",
                                      remove_first=False)
        # for pandas dataframe
        binarizer.fit(self.df_features)
        binarized_df = binarizer.transform(self.df_features)
        self.assertEqual(binarized_df.__class__, csr.csr_matrix)

        np.testing.assert_array_equal(expected_binarization.toarray(),
                                      binarized_df.toarray())
        # for numpy array
        binarizer.fit(self.features)
        binarized_array = binarizer.transform(self.features)
        self.assertEqual(binarized_array.__class__, csr.csr_matrix)

        np.testing.assert_array_equal(expected_binarization.toarray(),
                                      binarized_array.toarray())

        # test fit_transform
        binarized_array = binarizer.fit_transform(self.features)
        self.assertEqual(binarized_array.__class__, csr.csr_matrix)

        np.testing.assert_array_equal(expected_binarization.toarray(),
                                      binarized_array.toarray())

    def test_binarizer_remove_first(self):
        """...Test binarizer fit when remove_first=True
        """
        n_cuts = 3
        one_hot_encoder = OneHotEncoder(sparse=True)
        expected_binarization = one_hot_encoder.fit_transform(
            self.default_expected_intervals)

        binarizer = FeaturesBinarizer(method='quantile', n_cuts=n_cuts,
                                      detect_column_type="auto",
                                      remove_first=True)

        binarizer.fit(self.features)
        binarized_array = binarizer.transform(self.features)
        self.assertEqual(binarized_array.__class__, csr.csr_matrix)

        expected_binarization_without_first = \
            np.delete(expected_binarization.toarray(), [0, 4, 8, 10], 1)

        np.testing.assert_array_equal(expected_binarization_without_first,
                                      binarized_array.toarray())

        return


if __name__ == "__main__":
    unittest.main()
