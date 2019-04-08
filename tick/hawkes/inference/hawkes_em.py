# License: BSD 3 clause

import numpy as np

from tick.hawkes.inference.base import LearnerHawkesNoParam
from tick.hawkes.inference.build.hawkes_inference import (HawkesEM as
                                                          _HawkesEM)
from tick.solver.base.utils import relative_distance


class HawkesEM(LearnerHawkesNoParam):
    """This class is used for performing non parametric estimation of
    multi-dimensional Hawkes processes based on expectation
    maximization algorithm.

    Hawkes processes are point processes defined by the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i = \\mu_i + \\sum_{j=1}^D \\int \\phi_{ij} dN_j

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels

    Parameters
    ----------
    kernel_support : `float`, default=`None`
        The support size common to all the kernels. Might be `None` if
        `kernel_discretization` is set

    kernel_size : `int`, default=10
        Number of discretizations of the kernel

    kernel_discretization : `np.ndarray`, default=None
        Explicit discretization of the kernel. If set, it will override
        `kernel_support` and `kernel_size` values.

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    n_threads : `int`, default=1
        Number of threads used for parallel computation.

        * if `int <= 0`: the number of physical cores available on the CPU
        * otherwise the desired number of threads

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes of the estimated Hawkes process

    n_realizations : `int`
        Number of given realizations

    kernel : `np.array` shape=(n_nodes, n_nodes, kernel_size)
        The estimated kernels

    baseline : `np.array` shape=(n_nodes)
        The estimated baseline

    References
    ----------
    Lewis, E., & Mohler, G. (2011).
    A nonparametric EM algorithm for multiscale Hawkes processes.
    `preprint, 1-16`_.

    The n-dimensional extension of this algorithm can be found in the latex
    documentation.

    .. _preprint, 1-16: http://paleo.sscnet.ucla.edu/Lewis-Molher-EM_Preprint.pdf
    """

    def __init__(self, kernel_support=None, kernel_size=10,
                 kernel_discretization=None, tol=1e-5, max_iter=100,
                 print_every=10, record_every=10, verbose=False, n_threads=1):

        LearnerHawkesNoParam.__init__(
            self, n_threads=n_threads, verbose=verbose, tol=tol,
            max_iter=max_iter, print_every=print_every,
            record_every=record_every)

        if kernel_discretization is not None:
            self._learner = _HawkesEM(kernel_discretization, n_threads)
        elif kernel_support is not None:
            self._learner = _HawkesEM(kernel_support, kernel_size, n_threads)
        else:
            raise ValueError('Either kernel support or kernel discretization '
                             'must be provided')

        self.baseline = None
        self.kernel = None

        self.history.print_order = ["n_iter", "rel_baseline", "rel_kernel"]

    def fit(self, events, end_times=None, baseline_start=None,
            kernel_start=None):
        """Fit the model according to the given training data.

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

        baseline_start : `None` or `np.ndarray`, shape=(n_nodes), default=None
            Used to force start values for baseline parameter
            If `None` starts with uniform 1 values

        kernel_start : `None` or `np.ndarray`, shape=(n_nodes, n_nodes, kernel_size), default=None
            Used to force start values for kernel parameter
            If `None` starts with random values
        """
        LearnerHawkesNoParam.fit(self, events, end_times=end_times)
        self.solve(baseline_start=baseline_start, kernel_start=kernel_start)
        return self

    def _solve(self, baseline_start=None, kernel_start=None):
        """
        Performs nonparametric estimation and stores the result in the
        attributes `kernel` and `baseline`

        Parameters
        ----------
        baseline_start : `None` or `np.ndarray`, shape=(n_nodes), default=None
            Used to force start values for mu parameter
            If `None` starts with uniform 1 values

        kernel_start : `None` or `np.ndarray', shape=(n_nodes, n_nodes, kernel_size), default=None
            Used to force start values for kernel parameter
            If `None` starts with random values
        """
        if kernel_start is None:
            self.kernel = 0.1 * np.random.uniform(
                size=(self.n_nodes, self.n_nodes, self.kernel_size))
        else:
            if kernel_start.shape != (self.n_nodes, self.n_nodes,
                                      self.kernel_size):
                raise ValueError(
                    'kernel_start has shape {} but should have '
                    'shape {}'.format(
                        kernel_start.shape,
                        (self.n_nodes, self.n_nodes, self.kernel_size)))
            self.kernel = kernel_start.copy()

        if baseline_start is None:
            self.baseline = np.zeros(self.n_nodes) + 1
        else:
            self.baseline = baseline_start.copy()

        for i in range(self.max_iter):
            if self._should_record_iter(i):
                prev_baseline = self.baseline.copy()
                prev_kernel = self.kernel.copy()

            self._learner.solve(self.baseline, self._flat_kernels)

            if self._should_record_iter(i):
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_kernel = relative_distance(self.kernel, prev_kernel)

                converged = max(rel_baseline, rel_kernel) <= self.tol
                force_print = (i + 1 == self.max_iter) or converged
                self._handle_history(i, rel_baseline=rel_baseline,
                                     rel_kernel=rel_kernel, force=force_print)

                if converged:
                    break

    def get_kernel_supports(self):
        """Computes kernel support. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernels` API

        Returns
        -------
        output : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the support of
            kernel i, j
        """
        return np.zeros((self.n_nodes, self.n_nodes)) + self.kernel_support

    def get_kernel_values(self, i, j, abscissa_array):
        """Computes value of the specified kernel on given time values. This
        makes our learner compliant with `tick.plot.plot_hawkes_kernels` API

        Parameters
        ----------
        i : `int`
            First index of the kernel

        j : `int`
            Second index of the kernel

        abscissa_array : `np.ndarray`, shape=(n_points, )
            1d array containing all the times at which this kernel will
            computes it value

        Returns
        -------
        output : `np.ndarray`, shape=(n_points, )
            1d array containing the values of the specified kernels at the
            given times.
        """
        indices_in_support = (abscissa_array > 0) & \
                             (abscissa_array < self.kernel_support)
        index = np.searchsorted(self.kernel_discretization,
                                abscissa_array[indices_in_support]) - 1

        kernel_values = np.empty_like(abscissa_array)
        kernel_values[np.invert(indices_in_support)] = 0
        kernel_values[indices_in_support] = self.kernel[i, j, index]
        return kernel_values

    def get_kernel_norms(self):
        """Computes kernel norms. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernel_norms` API

        Returns
        -------
        norms : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the norm of
            kernel i, j
        """
        return self._learner.get_kernel_norms(self._flat_kernels)

    def objective(self, coeffs, loss: float = None):
        raise NotImplementedError()

    def score(self, events=None, end_times=None, baseline=None, kernel=None):
        """Compute score metric
        Score metric is log likelihood (the higher the better)

        Parameters
        ----------
        events : `list` of `list` of `np.ndarray`, default = None
            List of Hawkes processes realizations used to measure score.
            Each realization of the Hawkes process is a list of n_node for
            each component of the Hawkes. Namely `events[i][j]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component j of realization i.
            If only one realization is given, it will be wrapped into a list
            If None, events given while fitting model will be used

        end_times : `np.ndarray` or `float`, default = None
            List of end time of all hawkes processes used to measure score.
            If None, it will be set to each realization's latest time.
            If only one realization is provided, then a float can be given.

        baseline : `np.ndarray`, shape=(n_nodes, ), default = None
            Baseline vector for which the score is measured
            If `None` baseline obtained during fitting is used

        kernel : `None` or `np.ndarray`, shape=(n_nodes, n_nodes, kernel_size), default=None
            Used to force start values for kernel parameter
            If `None` kernel obtained during fitting is used

        Returns
        -------
        likelihood : `double`
            Computed log likelihood value
        """
        if events is None and not self._fitted:
            raise ValueError('You must either call `fit` before `score` or '
                             'provide events')

        if events is None and end_times is None:
            learner = self
        else:
            learner = HawkesEM(**self.get_params())
            learner._set('_end_times', end_times)
            learner._set_data(events)

        n_nodes = learner.n_nodes
        kernel_size = learner.kernel_size

        if baseline is None:
            baseline = self.baseline

        if kernel is None:
            kernel = self.kernel

        flat_kernels = kernel.reshape((n_nodes, n_nodes * kernel_size))

        return learner._learner.loglikelihood(baseline, flat_kernels)

    def get_params(self):
        return {
            'kernel_support': self.kernel_support,
            'kernel_size': self.kernel_size,
            'kernel_discretization': self.kernel_discretization,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'print_every': self.print_every,
            'record_every': self.record_every,
            'verbose': self.verbose,
            'n_threads': self.n_threads
        }

    @property
    def _flat_kernels(self):
        return self.kernel.reshape((self.n_nodes,
                                    self.n_nodes * self.kernel_size))

    @property
    def kernel_support(self):
        return self._learner.get_kernel_support()

    @kernel_support.setter
    def kernel_support(self, val):
        self._learner.set_kernel_support(val)

    @property
    def kernel_size(self):
        return self._learner.get_kernel_size()

    @kernel_size.setter
    def kernel_size(self, val):
        self._learner.set_kernel_size(val)

    @property
    def kernel_dt(self):
        return self._learner.get_kernel_fixed_dt()

    @kernel_dt.setter
    def kernel_dt(self, val):
        self._learner.set_kernel_dt(val)

    @property
    def kernel_discretization(self):
        return self._learner.get_kernel_discretization()
