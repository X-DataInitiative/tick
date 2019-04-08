# License: BSD 3 clause

import numpy as np

from tick.base.simulation import SimuWithFeatures


class SimuLogReg(SimuWithFeatures):
    """Simulation of a Logistic regression model

    Parameters
    ----------
    weights : `numpy.ndarray`, shape=(n_features,)
        The array of weights of the model

    intercept : `float`, default=`None`
        The intercept. If None, then no intercept is used

    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If `None`, it is simulated

    n_samples : `int`, default=200
        Number of samples

    features_type : `str`, default="cov_toeplitz"
        The type of features matrix to simulate

        * If ``"cov_toeplitz"`` : a Gaussian distribution with
          Toeplitz correlation matrix

        * If ``"cov_uniform"`` : a Gaussian distribution with
          correlation matrix given by .5 * (U + U.T), where U is
          uniform on [0, 1] and diagonal filled with ones.

    cov_corr : `float`, default=.5
        Correlation to use in the Toeplitz correlation matrix

    features_scaling : `str`, default="none"
        The way the features matrix is scaled after simulation

        * If ``"standard"`` : the columns are centered and
          normalized

        * If ``"min-max"`` : remove the minimum and divide by
          max-min

        * If ``"norm"`` : the columns are normalized but not centered

        * If ``"none"`` : nothing is done to the features

    seed : `int`, default=None
        The seed of the random number generator. If `None` it is not
        seeded

    verbose : `bool`, default=True
        If `True`, print things

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the generated arrays.
        Used in the case features is None

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated (or given) features matrix

    labels : `numpy.ndarray`, shape=(n_samples,)
        The simulated labels

    time_start : `str`
        Start date of the simulation

    time_elapsed : `int`
        Duration of the simulation, in seconds

    time_end : `str`
        End date of the simulation

    """

    _attrinfos = {"labels": {"writable": False}}

    def __init__(self, weights: np.ndarray, intercept: float = None,
                 features: np.ndarray = None, n_samples: int = 200,
                 features_type: str = "cov_toeplitz", cov_corr: float = 0.5,
                 features_scaling: str = "none", seed: int = None,
                 verbose: bool = True, dtype="float64"):

        n_features = weights.shape[0]
        SimuWithFeatures.__init__(self, intercept, features, n_samples,
                                  n_features, features_type, cov_corr,
                                  features_scaling, seed, verbose, dtype=dtype)
        self.weights = weights
        self._set("labels", None)

    def simulate(self):
        """
        Launch simulation of the data

        Returns
        -------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector
        """
        return SimuWithFeatures.simulate(self)

    @staticmethod
    def sigmoid(t):
        idx = t > 0
        out = np.empty(t.size, dtype=np.float)
        out[idx] = 1. / (1 + np.exp(-t[idx]))
        exp_t = np.exp(t[~idx])
        out[~idx] = exp_t / (1. + exp_t)
        return out

    def _simulate(self):
        # The features matrix already exists, and is created by the
        # super class
        features = self.features
        n_samples, n_features = features.shape
        u = features.dot(self.weights)
        # Add the intercept if necessary
        if self.intercept is not None:
            u += self.intercept
        p = np.empty(n_samples)
        p[:] = SimuLogReg.sigmoid(u)
        labels = np.empty(n_samples, dtype=self.dtype)
        labels[:] = np.random.binomial(1, p, size=n_samples)
        labels[labels == 0] = -1
        self._set("labels", labels)
        return features, labels
