# License: BSD 3 clause

import sys
from warnings import warn

import numpy as np

from tick.base_model import LOSS_AND_GRAD
from tick.hawkes.model.build.hawkes_model import (
    ModelHawkesSumExpKernLeastSq as _ModelHawkesSumExpKernLeastSq)
from .base import ModelHawkes


class ModelHawkesSumExpKernLeastSq(ModelHawkes):
    """Hawkes process model for sum-exponential kernels with fixed and
    given decays.
    It is modeled with least square loss:

    .. math::
        \\sum_{i=1}^{D} \\left(
            \\int_0^T \\lambda_i(t)^2 dt
            - 2 \\int_0^T \\lambda_i(t) dN_i(t)
        \\right)

    where :math:`\\lambda_i` is the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i(t) + \\sum_{j=1}^D
        \\sum_{t_k^j < t} \\phi_{ij}(t - t_k^j)

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i(t)` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels
    * :math:`t_k^j` are the timestamps of all events of node :math:`j`

    and with a sum-exponential parametrisation of the kernels

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
        
    n_baselines : `int`, default=1
        In this model baseline is supposed to be either constant or piecewise 
        constant. If `n_baseline > 1` then piecewise constant setting is 
        enabled. In this case :math:`\\mu_i(t)` is piecewise constant on 
        intervals of size `period_length / n_baselines` and periodic.
        
    period_length : `float`, default=None
        In piecewise constant setting this denotes the period of the 
        piecewise constant baseline function.

    approx : `int`, default=0 (read-only)
        Level of approximation used for computing exponential functions

        * if 0: no approximation
        * if 1: a fast approximated exponential function is used

    n_threads : `int`, default=-1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of threads available on
          the CPU
        * otherwise the desired number of threads

    Attributes
    ----------
    n_nodes : `int` (read-only)
        Number of components, or dimension of the Hawkes model

    n_decays : `int` (read-only)
        Number of decays used in the sum-exponential kernel
        
    baseline_intervals : `np.ndarray`, shape=(n_baselines)
        Start time of each interval on which baseline is piecewise constant.

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
        },
        "n_baselines": {
            "writable": True,
            "cpp_setter": "set_n_baselines"
        },
        "_period_length": {
            "writable": False,
        },
    }

    def __init__(self, decays: np.ndarray, n_baselines=1, period_length=None,
                 approx: int = 0, n_threads: int = 1):
        ModelHawkes.__init__(self, approx=approx, n_threads=n_threads)
        self._end_times = None

        if n_baselines <= 0:
            raise ValueError('n_baselines must be positive')
        if n_baselines > 1 and period_length is None:
            raise ValueError('period_length must be given if multiple '
                             'baselines are used')
        if period_length is not None and n_baselines == 1:
            warn('period_length has no effect when using a constant baseline')

        if isinstance(decays, list):
            decays = np.array(decays, dtype=float)
        elif decays.dtype != float:
            decays = decays.astype(float)
        self.decays = decays.copy()
        self.n_baselines = n_baselines
        self.period_length = period_length

        self._model = _ModelHawkesSumExpKernLeastSq(
            self.decays, self.n_baselines, self.cast_period_length(),
            self.n_threads, self.approx)

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

    @property
    def period_length(self):
        return self._period_length

    @period_length.setter
    def period_length(self, val):
        self._set("_period_length", val)
        if hasattr(self, '_model') and self._model is not None:
            self._model.set_period_length(self.cast_period_length())

    def cast_period_length(self):
        if self.period_length is None:
            return sys.float_info.max
        else:
            return self.period_length

    @property
    def baseline_intervals(self):
        return np.arange(self.n_baselines) * (
            self._model.get_period_length() / self.n_baselines)
