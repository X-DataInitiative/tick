# License: BSD 3 clause

import numpy as np

from tick.hawkes.inference.base import LearnerHawkesNoParam
from tick.hawkes.inference.build.hawkes_inference import (HawkesBasisKernels as
                                                          _HawkesBasisKernels)
from tick.solver.base.utils import relative_distance


class HawkesBasisKernels(LearnerHawkesNoParam):
    """This class is used for performing non parametric estimation of
    multi-dimensional Hawkes processes based on expectation maximization
    algorithm and the hypothesis that kernels are linear
    combinations of some basis kernels.

    Hawkes processes are point processes defined by the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i = \\mu_i + \\sum_{j=1}^D \\int \\phi_{ij} dN_j

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels

    The basis kernel hypothesis translates to:

    .. math::
        \\phi_{ij}(t) = \\sum_{u}^U a_{ij}^u g^u(t)

    where

    * :math:`U` is the number of basis kernels.
    * :math:`g^u` is a basis kernel
    * :math:`a_{ij}^u` is the amplitude of basis kernel :math:`u` in kernel
      :math:`\phi_{ij}`

    Finally we also suppose that basis kernels :math:`g^u` are piecewise
    constant on a given support and number of intervals.

    Parameters
    ----------
    kernel_support : `float`
        The support size common to all the kernels.

    n_basis : `int`, default=`None`
        Number of non parametric basis kernels to be used.
        If `None` or 0, it will be set to `n_nodes`

    kernel_size : `int`, default=10
        Number of discretizations of the kernel

    C : `float`, default=1e-1
        The penalization parameter. It penalizes both the amplitudes
        squared values and the basis kernels smoothness through the
        integral of their squared derivative.

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

    Other Parameters
    ----------------
    ode_max_iter : `int`, default=100
        Maximum number of loop for inner ODE (ordinary differential equation)
        algorithm.

    ode_tol : `float`, default=1e-5
        Tolerance of loop for inner inner ODE (ordinary differential equation)
        algorithm.

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes of the estimated Hawkes process

    n_realizations : `int`
        Number of given realizations`

    baseline : `np.array` shape=(n_nodes)
        The estimated baseline

    amplitudes : `np.array` shape=(n_nodes, n_nodes, n_basis)
        Amplitudes of all basis kernels for all kernels.

    basis_kernels : `np.array` shape=(n_basis, kernel_size)
        Estimated basis kernels

    kernel_dt : `float`
        Kernel discretization step. It is equal to
        `kernel_support` / `kernel_size`

    kernel_discretization : `np.ndarray`, shape=(kernel_size + 1, )
        Kernel discretizations points, denotes the interval on which basis
        kernels are piecewise constant.

    References
    ----------
    Zhou, K., Zha, H. and Song, L., 2013, June. Learning Triggering Kernels for
    Multi-dimensional Hawkes Processes. In `ICML (3) (pp. 1301-1309)`_.

    Some rewriting notes for implementing the algorithm can be found in the
    doc/tex directory.

    .. _ICML (3) (pp. 1301-1309): http://jmlr.org/proceedings/papers/v28/zhou13.html
    """

    _attrinfos = {
        'baseline': {
            'writable': False
        },
        'amplitudes': {
            'writable': False
        },
        'basis_kernels': {
            'writable': False
        },
        '_amplitudes_2d': {
            'writable': False
        },
    }

    def __init__(self, kernel_support, n_basis=None, kernel_size=10, tol=1e-5,
                 C=1e-1, max_iter=100, verbose=False, print_every=10,
                 record_every=10, n_threads=1, ode_max_iter=100, ode_tol=1e-5):

        LearnerHawkesNoParam.__init__(self, max_iter=max_iter, verbose=verbose,
                                      tol=tol, print_every=print_every,
                                      record_every=record_every,
                                      n_threads=n_threads)

        self.ode_max_iter = ode_max_iter
        self.ode_tol = ode_tol

        alpha = 1. / C
        if n_basis is None:
            n_basis = 0

        self._learner = _HawkesBasisKernels(kernel_support, kernel_size,
                                            n_basis, alpha, n_threads)
        self._amplitudes_2d = None

        self.history.print_order = [
            "n_iter", "rel_baseline", "rel_amplitudes", "rel_basis_kernels"
        ]

    def fit(self, events, end_times=None, baseline_start=None,
            amplitudes_start=None, basis_kernels_start=None):
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

        baseline_start : `None` or `np.ndarray`, shape=(n_nodes)
            Used to force start values for baseline attribute
            If `None` starts with uniform 1 values

        amplitudes_start : `None` or `np.ndarray`, shape=(n_nodes,n_nodes,D)
            Used to force start values for amplitude parameter
            If `None` starts with random values uniformly sampled between
            0.5 and 0.9

        basis_kernels_start : `None` or `np.darray`, shape=(D,kernel_size)
            Used to force start values for the basis kernels
            If `None` starts with random values uniformly sampled between
            0 and 0.1
        """
        LearnerHawkesNoParam.fit(self, events, end_times=end_times)
        self.solve(baseline_start=baseline_start,
                   amplitudes_start=amplitudes_start,
                   basis_kernels_start=basis_kernels_start)
        return self

    def _solve(self, baseline_start=None, amplitudes_start=None,
               basis_kernels_start=None):
        """Perform nonparametric estimation

        Parameters
        ----------
        baseline_start : `None` or `np.ndarray`, shape=(n_nodes)
            Used to force start values for baseline attribute
            If `None` starts with uniform 1 values

        amplitudes_start : `None` or `np.ndarray', shape=(n_nodes,n_nodes,D)
            Used to force start values for amplitude parameter
            If `None` starts with random values uniformly sampled between
            0.5 and 0.9

        basis_kernels_start : `None` or `p.andarray, shape=(D,kernel_size)
            Used to force start values for the basis kernels
            If `None` starts with random values uniformly sampled between
            0 and 0.1
        """
        if baseline_start is None:
            self._set("baseline", np.zeros(self.n_nodes) + 1)
        else:
            self._set("baseline", baseline_start.copy())

        if amplitudes_start is None:
            self._set(
                "amplitudes",
                np.random.uniform(
                    0.5, 0.9, size=(self.n_nodes, self.n_nodes, self.n_basis)))
        else:
            self._set("amplitudes", amplitudes_start.copy())

        if basis_kernels_start is None:
            self._set(
                "basis_kernels",
                0.1 * np.random.uniform(size=(self.n_basis, self.kernel_size)))
        else:
            self._set("basis_kernels", basis_kernels_start.copy())

        self._set(
            '_amplitudes_2d',
            self.amplitudes.reshape((self.n_nodes,
                                     self.n_nodes * self.n_basis)))

        for i in range(self.max_iter):
            if self._should_record_iter(i):
                prev_baseline = self.baseline.copy()
                prev_amplitudes = self.amplitudes.copy()
                prev_basis_kernels = self.basis_kernels.copy()

            rel_ode = self._learner.solve(self.baseline, self.basis_kernels,
                                          self._amplitudes_2d,
                                          self.ode_max_iter, self.ode_tol)

            if self._should_record_iter(i):
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_amplitudes = relative_distance(self.amplitudes,
                                                   prev_amplitudes)
                rel_basis_kernels = relative_distance(self.basis_kernels,
                                                      prev_basis_kernels)

                converged = max(rel_baseline, rel_amplitudes,
                                rel_basis_kernels) <= self.tol
                force_print = (i + 1 == self.max_iter) or converged

                self._handle_history(i + 1, rel_baseline=rel_baseline,
                                     rel_amplitudes=rel_amplitudes,
                                     rel_basis_kernels=rel_basis_kernels,
                                     rel_ode=rel_ode, force=force_print)

                if converged:
                    break

    def get_kernel_supports(self):
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

        kernels_ij_sum = np.zeros(self.kernel_size)
        for d in range(self.n_basis):
            kernels_ij_sum += self.amplitudes[i, j, d] * self.basis_kernels[d]

        kernel_values[indices_in_support] = kernels_ij_sum[index]
        return kernel_values

    def objective(self, coeffs, loss: float = None):
        raise NotImplementedError()

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
    def n_basis(self):
        return self._learner.get_n_basis()

    @n_basis.setter
    def n_basis(self, val):
        if val is None:
            val = 0
        self._learner.set_n_basis(val)

    @property
    def C(self):
        return 1. / self._learner.get_alpha()

    @C.setter
    def C(self, val):
        self._learner.set_alpha(1. / val)

    @property
    def kernel_discretization(self):
        return self._learner.get_kernel_discretization()

    @property
    def kernel_dt(self):
        return self._learner.get_kernel_dt()

    @kernel_dt.setter
    def kernel_dt(self, val):
        self._learner.set_kernel_dt(val)
