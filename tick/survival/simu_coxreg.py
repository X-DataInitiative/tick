# License: BSD 3 clause


import numpy as np
from tick.base.simulation import SimuWithFeatures


# TODO: something better to tune the censoring level than this censoring factor


class SimuCoxReg(SimuWithFeatures):
    """Simulation of a Cox regression for proportional hazards

    Parameters
    ----------
    coeffs : `numpy.ndarray`, shape=(n_coeffs,)
        The array of coefficients of the model

    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If None, it is simulated

    n_samples : `int`, default=200
        Number of samples

    times_distribution : `str`, default="weibull"
        The distrubution of times. Only ``"weibull"``
        is implemented for now

    scale : `float`, default=1.0
        Scaling parameter to use in the distribution of times

    shape : `float`, default=1.0
        Shape parameter to use in the distribution of times

    censoring_factor : `float`, default=2.0
        Level of censoring. Increasing censoring_factor lead
        to less censored times and conversely.

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

    seed : `int`, default=None
        The seed of the random number generator. If `None` it is not
        seeded

    verbose : `bool`, default=True
        If True, print things

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated (or given) features matrix

    times : `numpy.ndarray`, shape=(n_samples,)
        Simulated times

    censoring : `numpy.ndarray`, shape=(n_samples,)
        Simulated censoring indicator, where ``censoring[i] == 1``
        indicates that the time of the i-th individual is a failure
        time, and where ``censoring[i] == 0`` means that the time of
        the i-th individual is a censoring time

    time_start : `str`
        Start date of the simulation

    time_elapsed : `int`
        Duration of the simulation, in seconds

    time_end : `str`
        End date of the simulation

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "_times_distribution": {
            "writable": False
        }
    }

    def __init__(self, coeffs: np.ndarray,
                 features: np.ndarray = None, n_samples: int = 200,
                 times_distribution: str = "weibull",
                 shape: float = 1., scale: float = 1.,
                 censoring_factor: float = 2.,
                 features_type: str = "cov_toeplitz",
                 cov_corr: float = 0.5, features_scaling: str = "none",
                 seed: int = None, verbose: bool = True):

        n_features = coeffs.shape[0]
        # intercept=None in this model
        SimuWithFeatures.__init__(self, None, features, n_samples,
                                  n_features, features_type, cov_corr,
                                  features_scaling, seed, verbose)
        self.coeffs = coeffs
        self.times_distribution = times_distribution
        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor
        self.times_distribution = times_distribution
        self.features = None
        self.times = None
        self.censoring = None

    # TODO: properties for times_dist, shape, scale, censoring factor

    def simulate(self):
        """Launch simulation of the data

        Returns
        -------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The simulated (or given) features matrix

        times : `numpy.ndarray`, shape=(n_samples,)
            Simulated times

        censoring : `numpy.ndarray`, shape=(n_samples,)
            Simulated censoring indicator, where ``censoring[i] == 1``
            indicates that the time of the i-th individual is a failure
            time, and where ``censoring[i] == 0`` means that the time of
            the i-th individual is a censoring time
        """
        return SimuWithFeatures.simulate(self)

    @property
    def times_distribution(self):
        return self._times_distribution

    @times_distribution.setter
    def times_distribution(self, val):
        if val != "weibull":
            raise ValueError("``times_distribution`` was not "
                             "understood, try using 'weibull' instead")
        self._set("_times_distribution", val)

    def _simulate(self):
        # The features matrix already exists, and is created by the
        # super class
        features = self.features
        n_samples, n_features = features.shape
        u = features.dot(self.coeffs)
        # Simulation of true times
        E = np.random.exponential(scale=1., size=n_samples)
        E *= np.exp(-u)
        scale = self.scale
        shape = self.shape
        if self.times_distribution == "weibull":
            T = 1. / scale * E ** (1. / shape)
        else:
            # There is not point in this test, but let's do it like that
            # since we're likely to implement other distributions
            T = 1. / scale * E ** (1. / shape)
        m = T.mean()
        # Simulation of the censoring
        c = self.censoring_factor
        C = np.random.exponential(scale=c * m, size=n_samples)
        # Observed time
        self._set("times", np.minimum(T, C))
        # Censoring indicator: 1 if it is a time of failure, 0 if it's
        #   censoring. It is as int8 and not bool as we might need to
        #   construct a memory access on it later
        censoring = (T <= C).astype(np.ushort)
        self._set("censoring", censoring)
        return self.features, self.times, self.censoring

    def _as_dict(self):
        dd = SimuWithFeatures._as_dict(self)
        dd.pop("features", None)
        dd.pop("times", None)
        dd.pop("censoring", None)
        return dd
