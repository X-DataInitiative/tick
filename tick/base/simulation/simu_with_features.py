# License: BSD 3 clause

import numpy as np
from warnings import warn
from os import linesep

from . import Simu
from tick.simulation import features_normal_cov_toeplitz, \
    features_normal_cov_uniform

# TODO: features simulation isn't launch each time we call simulate
# TODO: there's a problem if we give other coeffs with another size or
# if we change features. Maybe these should be readonly...


class SimuWithFeatures(Simu):
    """Abstract class for the simulation of a model with a features
    matrix.

    Parameters
    ----------
    intercept : `float`, default=`None`
        The intercept. If None, then no intercept is used

    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If None, it is simulated

    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    features_type : `str`, default="cov_toeplitz"
        The type of features matrix to simulate

        * If ``"cov_toeplitz"`` : a Gaussian distribution with
          Toeplitz correlation matrix

        * If ``"cov_uniform"`` : a Gaussian distribution with
          correlation matrix given by O.5 * (U + U.T), where U is
          uniform on [0, 1] and diagonal filled with ones.

    cov_corr : `float`, default=0.5
        Correlation to use in the Toeplitz correlation matrix

    features_scaling : `str`, default="none"
        The way the features matrix is scaled after simulation

        * If ``"standard"`` : the columns are centered and
          normalized

        * If ``"min-max"`` : remove the minimum and divide by
          max-min

        * If ``"norm"`` : the columns are normalized but not centered

        * If ``"none"`` : nothing is done to the features

    seed : `int`
        The seed of the random number generator

    verbose : `bool`
        If True, print things

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the generated arrays.
        Used in the case features is None
    """

    _attrinfos = {
        "_features_type": {
            "writable": False
        },
        "_features_scaling": {
            "writable": False
        }
    }

    def __init__(self, intercept: float = None, features: np.ndarray = None,
                 n_samples: int = 200, n_features: int = 30,
                 features_type: str = "cov_toeplitz", cov_corr: float = 0.5,
                 features_scaling: str = "none", seed: int = None,
                 verbose: bool = True, dtype="float64"):

        Simu.__init__(self, seed, verbose)
        self.intercept = intercept
        self.features = features
        self.n_samples = n_samples
        self.n_features = n_features
        self.features_type = features_type
        self.cov_corr = cov_corr
        self.features_scaling = features_scaling
        self.features = None
        self.dtype = dtype

        if features is not None:
            if n_features != features.shape[1]:
                raise ValueError("``n_features`` does not match size of"
                                 "``features``")
            if n_samples != features.shape[0]:
                raise ValueError("``n_samples`` does not match size of"
                                 "``features``")
            features_type = 'given'

            self.features = features
            n_samples, n_features = features.shape
            self.n_samples = n_samples
            self.n_features = n_features
            self.features_type = features_type
            self.dtype = self.features.dtype

        # TODO: check and correct also n_samples, n_features and cov_corr and features_scaling

    def _scale_features(self, features: np.ndarray):
        features_scaling = self.features_scaling
        if features_scaling == "standard":
            features -= features.mean(axis=0)
            features /= features.std(axis=0)
        elif features_scaling == "min-max":
            raise NotImplementedError()
        elif features_scaling == "norm":
            raise NotImplementedError()
        return features

    @property
    def features_type(self):
        return self._features_type

    @features_type.setter
    def features_type(self, val):
        if val not in ["given", "cov_toeplitz", "cov_uniform"]:
            warn(linesep + "features_type was not understood, using" +
                 " cov_toeplitz instead.")
            val = "cov_toeplitz"
        self._set("_features_type", val)

    @property
    def features_scaling(self):
        return self._features_scaling

    @features_scaling.setter
    def features_scaling(self, val):
        if val not in ["standard", "min-max", "norm", "none"]:
            warn(linesep + "features_scaling was not understood, " +
                 "using ``'none'`` instead.")
            val = "none"
        self._set("_features_scaling", val)

    def simulate(self):
        """Launch the simulation of data.
        """
        self._start_simulation()
        features_type = self.features_type
        if features_type != "given":
            n_samples = self.n_samples
            n_features = self.n_features
            if features_type == "cov_uniform":
                features = features_normal_cov_uniform(n_samples, n_features,
                                                       dtype=self.dtype)
            else:
                cov_corr = self.cov_corr
                features = features_normal_cov_toeplitz(
                    n_samples, n_features, cov_corr, dtype=self.dtype)
        else:
            features = self.features

        features = self._scale_features(features)
        self.features = features

        # Launch the overloaded simulation
        result = self._simulate()

        self._end_simulation()
        # self._set("data", result)
        return result

    def _as_dict(self):
        dd = Simu._as_dict(self)
        dd.pop("features", None)
        dd.pop("labels", None)
        return dd
