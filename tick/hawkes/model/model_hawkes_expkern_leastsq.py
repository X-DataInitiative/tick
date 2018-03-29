# License: BSD 3 clause

import numpy as np

from tick.base_model import LOSS_AND_GRAD
from tick.hawkes.model.build.hawkes_model import (ModelHawkesExpKernLeastSq as
                                                  _ModelHawkesExpKernLeastSq)
from .base import ModelHawkes


class ModelHawkesExpKernLeastSq(ModelHawkes):
    """Hawkes process model exponential kernels with fixed and given decays.
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
    decays : `float` or `numpy.ndarray`, shape=(n_nodes, n_nodes)
        Either a `float` giving the decay of all exponential kernels or
        a (n_nodes, n_nodes) `numpy.ndarray` giving the decays of
        the exponential kernels for all pairs of nodes.

    approx : `int`, default=0 (read-only)
        Level of approximation used for computing exponential functions

        * if 0: no approximation
        * if 1: a fast approximated exponential function is used

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
        {k: v for d in [ModelHawkes.pass_per_operation,
                        {LOSS_AND_GRAD: 2}] for k, v in d.items()}

    _attrinfos = {"decays": {"writable": True, "cpp_setter": "set_decays"}}

    def __init__(self, decays: np.ndarray, approx: int = 0,
                 n_threads: int = 1):
        ModelHawkes.__init__(self, approx=approx, n_threads=n_threads)
        self.decays = decays

        if isinstance(decays, (int, float)):
            decays = np.array([[decays]], dtype=float)
        elif isinstance(decays, list):
            decays = np.array(decays)
        elif decays.dtype != float:
            decays = decays.astype(float)

        self._model = _ModelHawkesExpKernLeastSq(decays.copy(), self.n_threads,
                                                 self.approx)

    def _set_data(self, events: list):
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
        """
        ModelHawkes._set_data(self, events)

        # self.n_nodes will have correct value once data has been given to model
        decays = self.decays
        if isinstance(decays, (int, float)):
            decays_matrix = np.zeros((self.n_nodes, self.n_nodes)) + decays
            self._model.set_decays(decays_matrix)

    def incremental_fit(self, events, end_time=None):
        """Incrementally fit model with data by adding one Hawkes realization.

        Parameters
        ----------
        events : `list` of `np.ndarray`
            The events of each component of the realization. Namely
            `events[j]` contains a one-dimensional `np.ndarray` of
            the events' timestamps of component j

        end_time : `float`, default=None
            End time of the realization.
            If None, it will be set to realization's latest time.

        Notes
        -----
        Data is not stored, so this might be useful if the list of all
        realizations does not fit in memory
        """
        ModelHawkes.incremental_fit(self, events, end_time=end_time)

        # self.n_nodes will have correct value once data has been given to model
        decays = self.decays
        if isinstance(decays, (int, float)):
            decays_matrix = np.zeros((self.n_nodes, self.n_nodes)) + decays
            self._model.set_decays(decays_matrix)

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
