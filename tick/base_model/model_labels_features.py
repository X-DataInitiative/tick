# License: BSD 3 clause

import numpy as np

from . import Model
from tick.preprocessing.utils import safe_array


class ModelLabelsFeatures(Model):
    """An abstract base class for a model for which data consists of a
    features matrix and a labels vector, namely for (one-class
    supervised learning)

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features) (read-only)
        The features matrix

    labels : `numpy.ndarray`, shape=(n_samples,)  (read-only)
        The labels vector

    n_samples : `int`  (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    _attrinfos = {
        "features": {
            "writable": False
        },
        "labels": {
            "writable": False
        },
        "n_samples": {
            "writable": False
        },
        "n_features": {
            "writable": False
        }
    }

    # fit_intercept should be in a model_generalized_linear, not here
    def __init__(self):
        Model.__init__(self)
        self.features = None
        self.labels = None
        self.n_features = None
        self.n_samples = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> Model:
        """Set the data into the model object

        Parameters
        ----------
        features : {`numpy.ndarray`, 'scipy.sparse.csr_matrix'}, shape=(n_samples, n_features)
            The features matrix, either dense or sparse

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : a child of `ModelLabelsFeatures`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below
        return Model.fit(self, features, labels)

    def _set_data(self, features, labels):
        n_samples, n_features = features.shape
        if n_samples != labels.shape[0]:
            raise ValueError(("Features has %i samples while labels "
                              "have %i" % (n_samples, labels.shape[0])))

        features = safe_array(features)
        labels = safe_array(labels)

        self._set("features", features)
        self._set("labels", labels)
        self._set("n_features", n_features)
        self._set("n_samples", n_samples)

    @property
    def _epoch_size(self):
        # This gives the typical size of an epoch when using a
        # stochastic optimization algorithm
        return self.n_samples

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_samples

    def _as_dict(self):
        dd = Model._as_dict(self)
        del dd["labels"]
        del dd["features"]
        return dd
