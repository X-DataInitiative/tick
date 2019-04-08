# License: BSD 3 clause

import numpy as np

from tick.base_model import ModelSecondOrder, ModelSelfConcordant, \
    LOSS_AND_GRAD
from tick.hawkes.model.build.hawkes_model import (ModelHawkesExpKernLogLik as
                                                  _ModelHawkesExpKernLogLik)
from .base import ModelHawkes


class ModelHawkesExpKernLogLik(ModelHawkes, ModelSecondOrder,
                               ModelSelfConcordant):
    """Hawkes process model exponential kernels with fixed and given decay.
    It is modeled with (opposite) log likelihood loss:

    .. math::
        \\sum_{i=1}^{D} \\left(
            \\int_0^T \\lambda_i(t) dt
            - \\int_0^T \\log \\lambda_i(t) dN_i(t)
        \\right)

    where :math:`\\lambda_i` is the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^D
        \\sum_{t_k^j < t} \\phi_{ij}(t - t_k^j)

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels
    * :math:`t_k^j` are the timestamps of all events of node :math:`j`

    and with an exponential parametrisation of the kernels

    .. math::
        \phi_{ij}(t) = \\alpha^{ij} \\beta^{ij}
                       \exp (- \\beta^{ij} t) 1_{t > 0}

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Matrix :math:`B = (\\beta_{ij})_{ij} \in \mathbb{R}^{D \\times D}` by the
      parameter `decays`. This parameter is given to the model

    Parameters
    ----------
    decay : `float`
        The decay coefficient of the exponential kernels.
        All kernels share this decay.

    n_threads : `int`, default=1
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of threads available on
          the CPU
        * otherwise the desired number of threads

    Attributes
    ----------
    n_nodes : `int` (read-only)
        Number of components, or dimension of the Hawkes model

    data : `list` of `numpy.array` (read-only)
        The events given to the model through `fit` method.
        Note that data given through `incremental_fit` is not stored
    """
    # In Hawkes case, getting value and grad at the same time need only
    # one pas over the data
    pass_per_operation = \
        {k: v for d in [ModelSecondOrder.pass_per_operation,
                        {LOSS_AND_GRAD: 1}] for k, v in d.items()}

    _attrinfos = {
        "decay": {
            "cpp_setter": "set_decay"
        },
    }

    def __init__(self, decay: float, n_threads: int = 1):
        ModelSecondOrder.__init__(self)
        ModelSelfConcordant.__init__(self)
        # Calling "ModelHawkes.__init__" is necessary so that
        ## dtype is correctly set
        ModelHawkes.__init__(self, n_threads=1, approx=0)
        self.decay = decay
        self._model = _ModelHawkesExpKernLogLik(decay, n_threads)

    def fit(self, events, end_times=None):
        """Set the corresponding realization(s) of the process.

        Parameters
        ----------
        events : `list` of `list` of `np.ndarray`
            List of Hawkes processes realizations.
            Each realization of the Hawkes process is a list of n_node for
            each component of the Hawkes. Namely `events[i][j]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component j of realization i.
            If only one realization is given, it will be wrapped into a list

        end_times : `np.ndarray` or `float`, default = None
            List of end time of all hawkes processes that will be given to the
            model. If None, it will be set to each realization's latest time.
            If only one realization is provided, then a float can be given.
        """
        ModelHawkes.fit(self, events, end_times=end_times)
        ModelSecondOrder.fit(self, events)
        return ModelSelfConcordant.fit(self, events)

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray):
        value = self._model.loss_and_grad(coeffs, out)
        return value

    def _hessian_norm(self, coeffs: np.ndarray, point: np.ndarray) -> float:
        return self._model.hessian_norm(coeffs, point)

    def _get_sc_constant(self) -> float:
        return 2.0

    @property
    def decays(self):
        return self.decay

    @property
    def _epoch_size(self):
        # This gives the typical size of an epoch when using a
        # stochastic optimization algorithm
        return self.n_jumps

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_jumps
