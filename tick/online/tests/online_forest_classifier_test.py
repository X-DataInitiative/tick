# License: BSD 3 clause

import unittest
import numpy as np
from tick.base.inference import InferenceTest
from sklearn.datasets import make_moons, make_classification, make_circles

from tick.online import OnlineForestClassifier


# Test

class Test(InferenceTest):

    def test_online_forest_n_features_differs(self):
        n_samples = 1000
        n_classes = 2
        n_trees = 20

        X, y = make_classification(n_samples=n_samples, n_features=10,
                                   n_redundant=0,
                                   n_informative=2, random_state=1,
                                   n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)

        of = OnlineForestClassifier(n_classes=2, n_trees=n_trees, seed=123,
                                    step=1.,
                                    use_aggregation=True)

        of.fit(X, y)

        X, y = make_classification(n_samples=n_samples, n_features=10,
                                   n_redundant=0,
                                   n_informative=2, random_state=1,
                                   n_clusters_per_class=1)

        of.fit(X, y)

        X, y = make_classification(n_samples=n_samples, n_features=3,
                                   n_redundant=0,
                                   n_informative=2, random_state=1,
                                   n_clusters_per_class=1)

    def test_online_forest_n_classes_differs(self):
        pass


if __name__ == "__main__":
    unittest.main()
