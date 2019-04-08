# License: BSD 3 clause

import numpy as np

from tick.base_model import Model, ModelFirstOrder
from tick.preprocessing.utils import safe_array
from .build.survival import ModelCoxRegPartialLikDouble \
    as _ModelCoxRegPartialLik_d
from .build.survival import ModelCoxRegPartialLikFloat \
    as _ModelCoxRegPartialLik_f

dtype_class_mapper = {
    np.dtype('float32'): _ModelCoxRegPartialLik_f,
    np.dtype('float64'): _ModelCoxRegPartialLik_d
}


class ModelCoxRegPartialLik(ModelFirstOrder):
    """Partial likelihood of the Cox regression model (proportional
    hazards).
    This class gives first order information (gradient and loss) for
    this model.

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features), (read-only)
        The features matrix

    times : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Obverved times

    censoring : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Boolean indicator of censoring of each sample.
        ``True`` means true failure, namely non-censored time

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_failures : `int` (read-only)
        Number of true failure times

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    censoring_rate : `float`
        The censoring_rate (percentage of ???)

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "features": {
            "writable": False
        },
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "n_samples": {
            "writable": False
        },
        "n_features": {
            "writable": False
        },
        "n_failures": {
            "writable": False
        },
        "censoring_rate": {
            "writable": False
        }
    }

    def __init__(self):
        ModelFirstOrder.__init__(self)
        self.features = None
        self.times = None
        self.censoring = None
        self.n_samples = None
        self.n_features = None
        self.n_failures = None
        self.censoring_rate = None
        self._model = None

    def fit(self, features: np.ndarray, times: np.array,
            censoring: np.array) -> Model:
        """Set the data into the model object

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `ModelCoxRegPartialLik`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below
        return Model.fit(self, features, times, censoring)

    def _set_data(self, features: np.ndarray, times: np.array,
                  censoring: np.array):  #

        if self.dtype is None:
            self.dtype = features.dtype
            if self.dtype != times.dtype:
                raise ValueError("Features and labels differ in data types")

        n_samples, n_features = features.shape
        if n_samples != times.shape[0]:
            raise ValueError(("Features has %i samples while times "
                              "have %i" % (n_samples, times.shape[0])))
        if n_samples != censoring.shape[0]:
            raise ValueError(("Features has %i samples while censoring "
                              "have %i" % (n_samples, censoring.shape[0])))

        features = safe_array(features, dtype=self.dtype)
        times = safe_array(times, dtype=self.dtype)
        censoring = safe_array(censoring, np.ushort)

        self._set("features", features)
        self._set("times", times)
        self._set("censoring", censoring)
        self._set("n_samples", n_samples)
        self._set("n_features", n_features)
        self._set(
            "_model", dtype_class_mapper[self.dtype](self.features, self.times,
                                                     self.censoring))

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_n_coeffs(self, *args, **kwargs):
        return self.n_features

    @property
    def _epoch_size(self):
        return self.n_failures

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_failures

    def _as_dict(self):
        dd = ModelFirstOrder._as_dict(self)
        del dd["features"]
        del dd["times"]
        del dd["censoring"]
        return dd
