

from .base import ModelSecondOrder, ModelSelfConcordant, LOSS_AND_GRAD
from .build.model import LogLikelihoodExpNoBeta, ExpFixedBetaContrast
import numpy as np

import multiprocessing


# TODO: this class can't be used with a stochastic solver... Make loss
# readonly and instantiate the C++ class only once and for all for the
# lifetime of the python class


class ModelHawkesFixedBeta(ModelSecondOrder, ModelSelfConcordant):
    """
    Model for Hawkes processes with exponential kernels.
    The user must specify the kernels (the decay coefficient of the
    exponentials). Both log-likelihood and L2 goodness-of-fit are
    available.

    Parameters
    ----------
    decays : `numpy.ndarray`, shape=(n_nodes, n_nodes)
        A (n_nodes, n_nodes) matrix containing the decays of
        interactions between all pairs of nodes (read-only)

    contrast : `str`, either "mle" or "L2", default="mle"
        The goodness-of-fit (read-only)

        * if ``"mle"``: maximum-likelihood is used
        * if ``"L2"``: least-squares is used

    approx : `int`, default=0 (read-only)
        Level of approximation used for computing exponential functions

        * if 0: no approximation
        * if 1: a fast approximated exponential function is used

    n_threads : `int`, default=-1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of physical cores available on
          the CPU
        * otherwise the desired number of threads

    Attributes
    ----------
    n_nodes : `int` (read-only)
        Number of components, or dimension of the Hawkes model

    data : `list` of `numpy.ndarray`
        blabla
    """
    # In Hawkes case, getting value and grad at the same time need only one pas over the data
    pass_per_operation = {k: v for d in [ModelSecondOrder.pass_per_operation, {LOSS_AND_GRAD: 1}] for k, v in d.items()}

    _attrinfos = {
        "decays": {
            "writable": False
        },
        "contrast": {
            "writable": False
        },
        "n_nodes": {
            "writable": False
        },
        "approx": {
            "writable": False
        },
        "n_threads": {
            "writable": False
        },
        "data": {
            "writable": False
        },
    }

    def __init__(self, decays: np.ndarray, contrast: str = "mle",
                 approx: int = 0, n_threads: int = -1):
        ModelSecondOrder.__init__(self)
        ModelSelfConcordant.__init__(self)

        if contrast not in ["mle", "L2"]:
            raise ValueError("contrast be either 'mle' or 'L2'")

        if n_threads <= 0:
            n_threads = multiprocessing.cpu_count()

        self._set("decays", decays)
        self._set("contrast", contrast)
        self._set("approx", approx)
        self._set("n_threads", n_threads)
        self._set("data", None)

    def _get_n_coeffs(self):
        return self._model.n_params

    # TODO: implement _set_data and not fit
    def fit(self, data):
        """Set the data into the model object

        Parameters
        ----------
        data : `list` of `numpy.ndarray`
            blabla

        """
        ModelSecondOrder.fit(self)
        ModelSelfConcordant.fit(self)

        n_nodes = len(data)
        self._set("n_nodes", n_nodes)
        self._set("data", data)

        if self.contrast == 'mle':
            model = LogLikelihoodExpNoBeta(len(data),
                                                 self.decays)
            model.Set(data)
        else:
            model = \
                ExpFixedBetaContrast(data,
                                     np.zeros((n_nodes, n_nodes)) + self.decays,
                                     self.n_threads,
                                     self.approx)
        self._set("_model", model)
        return self

    def _set_data(self, *args):
        pass

    def _loss(self, coeffs: np.ndarray) -> float:
        # Ultimately the methods will have the same name
        # So if should be avoided
        if self.contrast == "mle":
            return -1 * self._model.GetValue(coeffs)
        else:
            return self._model.compute_value(coeffs)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> np.ndarray:
        # Ultimately the methods will have the same name
        # So if should be avoided
        if self.contrast == "mle":
            self._model.GetValueGrad(coeffs, out)
            out *= -1
        else:
            self._model.compute_grad(coeffs, out)
        return out

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray):
        if self.contrast == "mle":
            value = self._model.GetValueGrad(coeffs, out)
            value *= -1
            out *= -1
        else:
            value = self._model.compute_value_grad(coeffs, out)
        return value

    def _hessian_norm(self, coeffs: np.ndarray,
                      point: np.ndarray) -> float:
        if self.contrast == "mle":
            return -self._model.GetHessianNorm(coeffs, point)
        raise NotImplementedError("_hessian_norm not implemented for "
                                  "L2 contrast")

    def _get_sc_constant(self) -> float:
        return 2.0
