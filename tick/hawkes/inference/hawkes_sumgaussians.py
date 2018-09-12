# License: BSD 3 clause

import math
import numpy as np
from scipy.stats import norm

from tick.hawkes.inference.base import LearnerHawkesNoParam
from tick.hawkes.inference.build.hawkes_inference import (HawkesSumGaussians as
                                                          _HawkesSumGaussians)
from tick.solver.base.utils import relative_distance


class HawkesSumGaussians(LearnerHawkesNoParam):
    """A class that implements parametric inference for Hawkes processes
    with parametrisation of the kernels as sum of Gaussian basis functions
    and a mix of Lasso and group-lasso regularization

    Hawkes processes are point processes defined by the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i(t) = \\mu_i + \\sum_{j=1}^D
        \\sum_{t_k^j < t} \\phi_{ij}(t - t_k^j)

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels
    * :math:`t_k^j` are the timestamps of all events of node :math:`j`

    and with an parametrisation of the kernels as sum of Gaussian basis 
    functions

    .. math::
        \phi_{ij}(t) = \sum_{m=1}^M \\alpha^{ij}_m f (t - t_m), \\quad
        f(t) = (2 \\pi \\sigma^2)^{-1} \exp(- t^2 / (2 \\sigma^2))

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Vector :math:`\mu \in \mathbb{R}^{D}` by the attribute
      `baseline`
    * Vector :math:`(t_m) \in \mathbb{R}^{M}` by the variable
      `means_gaussians`
    * Number :math:`\\sigma` by the variable `std_gaussian`
    * Tensor 
      :math:`A = (\\alpha^{ij}_m)_{ijm} \in \mathbb{R}^{D \\times D \\times M}`
      by the attribute `amplitudes`

    Parameters
    ----------
    max_mean_gaussian : `float`
        The mean of the last Gaussian basis function. This can be considered
        a proxy of the kernel support.

    n_gaussians : `int`
        The number of Gaussian basis functions used to approximate each kernel.

    step_size : `float`
        The step-size used in the optimization for the EM algorithm.

    C : `float`, default=1e3
        Level of penalization

    lasso_grouplasso_ratio : `float`, default=0.5
        Ratio of Lasso-Nuclear regularization mixing parameter with
        0 <= ratio <= 1.

        * For ratio = 0 this is Group-Lasso regularization
        * For ratio = 1 this is lasso (L1) regularization
        * For 0 < ratio < 1, the regularization is a linear combination
          of Lasso and Group-Lasso.

    max_iter : `int`, default=50
        Maximum number of iterations of the solving algorithm

    tol : `float`, default=1e-5
        The tolerance of the solving algorithm (iterations stop when the
        stopping criterion is below it). If not reached it does ``max_iter``
        iterations

    n_threads : `int`, default=1
        Number of threads used for parallel computation.

    verbose : `bool`, default=False
        If `True`, we verbose things

        * if `int <= 0`: the number of physical cores available on the CPU
        * otherwise the desired number of threads

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Other Parameters
    ----------------
    approx : `int`, default=0 (read-only)
        Level of approximation used for computing exponential functions

        * if 0: no approximation
        * if 1: a fast approximated exponential function is used

    em_max_iter : `int`, default=30
        Maximum number of loop for inner em algorithm.

    em_tol : `float`, default=None
        Tolerance of loop for inner em algorithm. If relative difference of
        baseline and adjacency goes bellow this tolerance, em inner loop
        will stop.
        If None, it will be set given a heuristic which look at last

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes / components in the Hawkes model
        
    baseline : `np.array`, shape=(n_nodes,)
        Inferred baseline of each component's intensity

    amplitudes : `np.ndarray`, shape=(n_nodes, n_nodes, n_gaussians)
        Inferred adjacency matrix
        
    means_gaussians : `np.array`, shape=(n_gaussians,)
        The means of the Gaussian basis functions.

    std_gaussian : `float`
        The standard deviation of each Gaussian basis function.

    References
    ----------
    Xu, Farajtabar, and Zha (2016, June) in ICML,
    `Learning Granger Causality for Hawkes Processes`_.
    
    .. _Learning Granger Causality for Hawkes Processes: http://jmlr.org/proceedings/papers/v48/xuc16.pdf
    """

    _attrinfos = {
        "_learner": {
            "writable": False
        },
        "_model": {
            "writable": False
        },
        "n_gaussians": {
            "cpp_setter": "set_n_gaussians"
        },
        "em_max_iter": {
            "cpp_setter": "set_em_max_iters"
        },
        "max_mean_gaussian": {
            "cpp_setter": "set_max_mean_gaussian"
        },
        "step_size": {
            "cpp_setter": "set_step_size"
        },
        "baseline": {
            "writable": False
        },
        "amplitudes": {
            "writable": False
        },
        "approx": {
            "writable": False
        }
    }

    def __init__(self, max_mean_gaussian, n_gaussians=5, step_size=1e-7, C=1e3,
                 lasso_grouplasso_ratio=0.5, max_iter=50, tol=1e-5,
                 n_threads=1, verbose=False, print_every=10, record_every=10,
                 approx=0, em_max_iter=30, em_tol=None):

        LearnerHawkesNoParam.__init__(
            self, verbose=verbose, max_iter=max_iter, print_every=print_every,
            tol=tol, n_threads=n_threads, record_every=record_every)
        self.baseline = None
        self.amplitudes = None

        self.n_gaussians = n_gaussians
        self.max_mean_gaussian = max_mean_gaussian
        self.step_size = step_size

        strength_lasso = lasso_grouplasso_ratio / C
        strength_grouplasso = (1. - lasso_grouplasso_ratio) / C

        self.em_max_iter = em_max_iter
        self.em_tol = em_tol

        self._learner = _HawkesSumGaussians(
            n_gaussians, max_mean_gaussian, step_size, strength_lasso,
            strength_grouplasso, em_max_iter, n_threads, approx)

        self.verbose = verbose

        self.history.print_order += ["rel_baseline", "rel_amplitudes"]

    def fit(self, events, end_times=None, baseline_start=None,
            amplitudes_start=None):
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
            Set initial value of baseline parameter
            If `None` starts with uniform 1 values

        amplitudes_start : `None` or `np.ndarray`, shape=(n_nodes, n_nodes, n_gaussians)
            Set initial value of amplitudes parameter
            If `None` starts with random values uniformly sampled between 0.5
            and 0.9`
        """
        LearnerHawkesNoParam.fit(self, events, end_times=end_times)
        self.solve(baseline_start=baseline_start,
                   amplitudes_start=amplitudes_start)
        return self

    def _solve(self, baseline_start=None, amplitudes_start=None):
        """Perform one iteration of the algorithm

        Parameters
        ----------
        baseline_start : `None` or `np.ndarray`, shape=(n_nodes)
            Set initial value of baseline parameter
            If `None` starts with uniform 1 values

        amplitudes_start : `None` or `np.ndarray', shape=(n_nodes, n_nodes, n_gaussians)
            Set initial value of adjacency parameter
            If `None` starts with random values uniformly sampled between 0.5
            and 0.9
        """

        if baseline_start is None:
            baseline_start = np.ones(self.n_nodes)

        self._set('baseline', baseline_start.copy())

        if amplitudes_start is None:
            amplitudes_start = np.random.uniform(
                0.5, 0.9, (self.n_nodes, self.n_nodes, self.n_gaussians))
        else:
            if amplitudes_start.shape != (self.n_nodes, self.n_nodes,
                                          self.n_gaussians):
                raise ValueError(
                    'amplitudes_start has shape {} but should have '
                    'shape {}'.format(
                        amplitudes_start.shape,
                        (self.n_nodes, self.n_nodes, self.n_gaussians)))

        self._set('amplitudes', amplitudes_start.copy())

        _amplitudes_2d = self.amplitudes.reshape(
            (self.n_nodes, self.n_nodes * self.n_gaussians))

        max_relative_distance = 1e-1
        for i in range(self.max_iter):
            if self._should_record_iter(i):
                prev_baseline = self.baseline.copy()
                prev_amplitudes = self.amplitudes.copy()

                inner_prev_baseline = self.baseline.copy()
                inner_prev_amplitudes = self.amplitudes.copy()

            self._learner.solve(self.baseline, _amplitudes_2d)

            if self._should_record_iter(i):
                inner_rel_baseline = relative_distance(self.baseline,
                                                       inner_prev_baseline)
                inner_rel_adjacency = relative_distance(
                    self.amplitudes, inner_prev_amplitudes)

                if self.em_tol is None:
                    inner_tol = max_relative_distance * 1e-2
                else:
                    inner_tol = self.em_tol

                if max(inner_rel_baseline, inner_rel_adjacency) < inner_tol:
                    break

                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_amplitudes = relative_distance(self.amplitudes,
                                                   prev_amplitudes)

                max_relative_distance = max(rel_baseline, rel_amplitudes)
                # We perform at least 5 iterations as at start we sometimes
                # reach a low tolerance if inner_tol is too low
                converged = max_relative_distance <= self.tol and i > 5
                force_print = (i + 1 == self.max_iter) or converged

                self._handle_history(i + 1, rel_baseline=rel_baseline,
                                     rel_amplitudes=rel_amplitudes,
                                     force=force_print)

                if converged:
                    break

    @property
    def C(self):
        return 1 / (self.strength_grouplasso + self.strength_lasso)

    @C.setter
    def C(self, val):
        if val < 0 or val is None:
            raise ValueError("`C` must be positive, got %s" % str(val))
        else:
            ratio = self.lasso_grouplasso_ratio
            self.strength_lasso = ratio / val
            self.strength_grouplasso = (1 - ratio) / val

    @property
    def lasso_grouplasso_ratio(self):
        ratio = self.strength_lasso / self.strength_grouplasso
        return ratio / (1. + ratio)

    @lasso_grouplasso_ratio.setter
    def lasso_grouplasso_ratio(self, val):
        if val < 0 or val > 1:
            raise ValueError("`lasso_grouplasso_ratio` must be between 0 "
                             "and 1, got %s" % str(val))
        else:
            C = self.C
            self.strength_lasso = val / C
            self.strength_grouplasso = (1 - val) / C

    @property
    def strength_lasso(self):
        return self._learner.get_strength_lasso()

    @strength_lasso.setter
    def strength_lasso(self, val):
        self._learner.set_strength_lasso(val)

    @property
    def strength_grouplasso(self):
        return self._learner.get_strength_grouplasso()

    @strength_grouplasso.setter
    def strength_grouplasso(self, val):
        self._learner.set_strength_grouplasso(val)

    @property
    def means_gaussians(self):
        return np.arange(self.n_gaussians) * self.max_mean_gaussian / \
               self.n_gaussians

    @property
    def std_gaussian(self):
        return self.max_mean_gaussian / (self.n_gaussians * math.pi)

    def get_kernel_supports(self):
        """Computes kernel support. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernels` API

        Returns
        -------
        output : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the support of
            kernel i, j
        """
        return np.zeros((self.n_nodes, self.n_nodes)) + self.n_gaussians

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
        x = np.zeros_like(abscissa_array)
        for m in range(self.n_gaussians):
            x += self.amplitudes[i, j, m] * \
                 norm.pdf((abscissa_array - self.means_gaussians[m]) \
                          / self.std_gaussian) / self.std_gaussian
        return x

    def get_kernel_norms(self):
        """Computes kernel norms. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernel_norms` API

        Returns
        -------
        norms : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the norm of
            kernel i, j
        """
        return np.einsum('ijk->ij', self.amplitudes)

    def objective(self, coeffs, loss: float = None):
        raise NotImplementedError()
