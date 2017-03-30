import numpy as np

from tick.optim.model.base import ModelHawkes, LOSS_AND_GRAD
from .build.model import ModelHawkesFixedSumExpKernLeastSqList as \
    _ModelHawkesFixedSumExpKernLeastSqList


class ModelHawkesFixedSumExpKernLeastSq(ModelHawkes):
    """Hawkes process learner for sum-exponential kernels with fixed and given
    It is modeled with least square loss:

    .. math::
        \\sum_{i=1}^{D} \\left(
            \\int_0^T \\lambda_i(t)^2 dt
            - 2 \\int_0^T \\lambda_i(t) dN_i(t)
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

    and with an sum-exponential parametrisation of the kernels

    .. math::
        \phi_{ij}(t) = \sum_{u=1}^{U} \\alpha^u_{ij} \\beta^u
                       \exp (- \\beta^u t) 1_{t > 0}

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Integer :math:`U` by the attribute `n_decays`
    * Vector :math:`\\beta \in \mathbb{R}^{U}` by the
      parameter `decays`. This parameter is given to the model

    Parameters
    ----------
    decays : `numpy.ndarray`, shape=(n_decays, )
        An array giving the different decays of the exponentials kernels.

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

    data : `list` of `numpy.array` (read-only)
        The events given to the model through `fit` method.
        Note that data given through `incremental_fit` is not stored
    """
    # In Hawkes case, getting value and grad at the same time need only
    # one pas over the data
    pass_per_operation = \
        {k: v for d in [ModelHawkes.pass_per_operation,
                        {LOSS_AND_GRAD: 2}] for k, v in d.items()}

    _attrinfos = {
        "decays": {
            "writable": True,
            "cpp_setter": "set_decays"
        }
    }

    def __init__(self, decays: np.ndarray, approx: int = 0, n_threads: int = 1):
        ModelHawkes.__init__(self, approx=approx, n_threads=n_threads)
        self._end_times = None

        if isinstance(decays, list):
            decays = np.array(decays, dtype=float)
        elif decays.dtype != float:
            decays = decays.astype(float)
        self.decays = decays.copy()

        self._model = _ModelHawkesFixedSumExpKernLeastSqList(
            self.decays, self.n_threads, self.approx
        )

    @property
    def n_decays(self):
        return self._model.get_n_decays()

    @property
    def _epoch_size(self):
        # This gives the typical size of an epoch when using a
        # stochastic optimization algorithm
        return self.n_nodes

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_nodes
