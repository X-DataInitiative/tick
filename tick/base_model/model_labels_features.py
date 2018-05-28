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

    dtype : `{'float64', 'float32'}`
        Type of the data arrays used.

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
        self.dtype = features.dtype
        n_samples, n_features = features.shape
        if n_samples != labels.shape[0]:
            raise ValueError(("Features has %i samples while labels "
                              "have %i" % (n_samples, labels.shape[0])))

        features = safe_array(features, dtype=self.dtype)
        labels = safe_array(labels, dtype=self.dtype)

        self._set("features", features)
        self._set("labels", labels)
        self._set("n_features", n_features)
        self._set("n_samples", n_samples)

    def astype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        new_model = tick.base.dtype_to_cpp_type.copy_with(
            self,
            ["_model", "features", "labels"]  # ignore on deepcopy
        )
        new_dtype = tick.base.dtype_to_cpp_type.extract_dtype(
            dtype_or_object_with_dtype)
        if self.features is not None:
            new_model._set('features', self.features.astype(new_dtype))
        if self.labels is not None:
            new_model._set('labels', self.labels.astype(new_dtype))
        new_model._set('_model',
                       new_model._build_cpp_model(dtype_or_object_with_dtype))
        return new_model

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
