# License: BSD 3 clause

import numpy as np

from tick.hawkes import ModelHawkesExpKernLogLik, SimuHawkesExpKernels
from tick.hawkes.inference.base import LearnerHawkesNoParam
from tick.hawkes.inference.build.hawkes_inference import (HawkesADM4 as
                                                          _HawkesADM4)
from tick.prox import ProxNuclear
from tick.prox.prox_l1 import ProxL1
from tick.solver.base.utils import relative_distance


class HawkesADM4(LearnerHawkesNoParam):
    """A class that implements parametric inference for Hawkes processes
    with an exponential parametrisation of the kernels and a mix of Lasso
    and nuclear regularization

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

    and with an exponential parametrisation of the kernels

    .. math::
        \phi_{ij}(t) = \\alpha^{ij} \\beta \exp (- \\beta t) 1_{t > 0}

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Vector :math:`\mu \in \mathbb{R}^{D}` by the attribute
      `baseline`
    * Matrix :math:`A = (\\alpha^{ij})_{ij} \in \mathbb{R}^{D \\times D}`
      by the attribute `adjacency`
    * Number :math:`\\beta \in \mathbb{R}` by the parameter `decay`. This
      parameter is given to the model

    Parameters
    ----------
    decay : `float`
        The decay used in the exponential kernel

    C : `float`, default=1e3
        Level of penalization

    lasso_nuclear_ratio : `float`, default=0.5
        Ratio of Lasso-Nuclear regularization mixing parameter with
        0 <= ratio <= 1.

        * For ratio = 0 this is nuclear regularization
        * For ratio = 1 this is lasso (L1) regularization
        * For 0 < ratio < 1, the regularization is a linear combination
          of Lasso and nuclear.

    max_iter : `int`, default=50
        Maximum number of iterations of the solving algorithm

    tol : `float`, default=1e-5
        The tolerance of the solving algorithm (iterations stop when the
        stopping criterion is below it). If not reached it does ``max_iter``
        iterations

    verbose : `bool`, default=False
        If `True`, we verbose things

    n_threads : `int`, default=1
        Number of threads used for parallel computation.

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
    rho : `float`, default=0.1
        Positive parameter of the augmented Lagrangian. Called penalty
        parameter, the higher it is, the more strict will be the
        penalization.

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
        relative difference obtained in the main loop.

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes / components in the Hawkes model

    baseline : `np.array`, shape=(n_nodes,)
        Inferred baseline of each component's intensity

    adjacency : `np.ndarray`, shape=(n_nodes, n_nodes)
        Inferred adjacency matrix

    References
    ----------
    Zhou, K., Zha, H., & Song, L. (2013, May).
    Learning Social Infectivity in Sparse Low-rank Networks Using
    Multi-dimensional Hawkes Processes. In `AISTATS (Vol. 31, pp. 641-649)
    <http://www.jmlr.org/proceedings/papers/v31/zhou13a.pdf>`_.
    """

    _attrinfos = {
        "_learner": {
            "writable": False
        },
        "_model": {
            "writable": False
        },
        "decay": {
            "cpp_setter": "set_decay"
        },
        "rho": {
            "cpp_setter": "set_rho"
        },
        "_C": {
            "writable": False
        },
        "baseline": {
            "writable": False
        },
        "adjacency": {
            "writable": False
        },
        "_prox_l1": {
            "writable": False
        },
        "_prox_nuclear": {
            "writable": False
        },
        "_lasso_nuclear_ratio": {
            "writable": False
        },
        "approx": {
            "writable": False
        }
    }

    def __init__(self, decay, C=1e3, lasso_nuclear_ratio=0.5, max_iter=50,
                 tol=1e-5, n_threads=1, verbose=False, print_every=10,
                 record_every=10, rho=.1, approx=0, em_max_iter=30,
                 em_tol=None):

        LearnerHawkesNoParam.__init__(
            self, verbose=verbose, max_iter=max_iter, print_every=print_every,
            tol=tol, n_threads=n_threads, record_every=record_every)
        self.baseline = None
        self.adjacency = None
        self._C = 0
        self._lasso_nuclear_ratio = 0

        self.decay = decay
        self.rho = rho

        self._prox_l1 = ProxL1(1.)
        self._prox_nuclear = ProxNuclear(1.)

        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.verbose = verbose

        self.em_max_iter = em_max_iter
        self.em_tol = em_tol

        self._learner = _HawkesADM4(decay, rho, n_threads, approx)

        # TODO add approx to model
        self._model = ModelHawkesExpKernLogLik(self.decay,
                                               n_threads=self.n_threads)

        self.history.print_order += ["rel_baseline", "rel_adjacency"]

    def fit(self, events, end_times=None, baseline_start=None,
            adjacency_start=None):
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

        adjacency_start : `None` or `np.ndarray`, shape=(n_nodes, n_nodes)
            Set initial value of adjacency parameter
            If `None` starts with random values uniformly sampled between 0.5
            and 0.9`
        """
        LearnerHawkesNoParam.fit(self, events, end_times=end_times)
        self.solve(baseline_start=baseline_start,
                   adjacency_start=adjacency_start)
        return self

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
        LearnerHawkesNoParam._set_data(self, events)

        events, end_times = self._clean_events_and_endtimes(events)

        self._model.fit(events, end_times=end_times)
        self._prox_nuclear.n_rows = self.n_nodes

    def _solve(self, baseline_start=None, adjacency_start=None):
        """Perform one iteration of the algorithm

        Parameters
        ----------
        baseline_start : `None` or `np.ndarray`, shape=(n_nodes)
            Set initial value of baseline parameter
            If `None` starts with uniform 1 values

        adjacency_start : `None` or `np.ndarray', shape=(n_nodes, n_nodes)
            Set initial value of adjacency parameter
            If `None` starts with random values uniformly sampled between 0.5
            and 0.9
        """

        if baseline_start is None:
            baseline_start = np.ones(self.n_nodes)

        self._set('baseline', baseline_start.copy())

        if adjacency_start is None:
            adjacency_start = np.random.uniform(0.5, 0.9,
                                                (self.n_nodes, self.n_nodes))
        self._set('adjacency', adjacency_start.copy())

        z1 = np.zeros_like(self.adjacency)
        z2 = np.zeros_like(self.adjacency)
        u1 = np.zeros_like(self.adjacency)
        u2 = np.zeros_like(self.adjacency)

        if self.rho <= 0:
            raise ValueError("The parameter rho equals {}, while it should "
                             "be strictly positive.".format(self.rho))

        max_relative_distance = 1e-1
        for i in range(self.max_iter):

            if self._should_record_iter(i):
                prev_objective = self.objective(self.coeffs)
                prev_baseline = self.baseline.copy()
                prev_adjacency = self.adjacency.copy()

            for _ in range(self.em_max_iter):
                inner_prev_baseline = self.baseline.copy()
                inner_prev_adjacency = self.adjacency.copy()
                self._learner.solve(self.baseline, self.adjacency, z1, z2, u1,
                                    u2)
                inner_rel_baseline = relative_distance(self.baseline,
                                                       inner_prev_baseline)
                inner_rel_adjacency = relative_distance(
                    self.adjacency, inner_prev_adjacency)

                if self.em_tol is None:
                    inner_tol = max_relative_distance * 1e-2
                else:
                    inner_tol = self.em_tol

                if max(inner_rel_baseline, inner_rel_adjacency) < inner_tol:
                    break

            z1 = self._prox_nuclear.call(np.ravel(self.adjacency + u1),
                                         step=1. / self.rho) \
                .reshape(self.n_nodes, self.n_nodes)
            z2 = self._prox_l1.call(np.ravel(self.adjacency + u2),
                                    step=1. / self.rho) \
                .reshape(self.n_nodes, self.n_nodes)

            u1 += self.adjacency - z1
            u2 += self.adjacency - z2

            if self._should_record_iter(i):
                objective = self.objective(self.coeffs)

                rel_obj = abs(objective - prev_objective) / abs(prev_objective)
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_adjacency = relative_distance(self.adjacency,
                                                  prev_adjacency)

                max_relative_distance = max(rel_baseline, rel_adjacency)
                # We perform at least 5 iterations as at start we sometimes
                # reach a low tolerance if inner_tol is too low
                converged = max_relative_distance <= self.tol and i > 5
                force_print = (i + 1 == self.max_iter) or converged

                self._handle_history(i + 1, obj=objective, rel_obj=rel_obj,
                                     rel_baseline=rel_baseline,
                                     rel_adjacency=rel_adjacency,
                                     force=force_print)

                if converged:
                    break

    def objective(self, coeffs, loss: float = None):
        """Compute the objective minimized by the learner at `coeffs`

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The objective is computed at this point

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given `coeffs`

        Notes
        -----
        Because of the auxiliary variables, the expression of the
        truly optimized objective is a bit modified. Hence this objective
        value might not reach its exact minimum especially for high
        penalization levels.
        """
        if loss is None:
            loss = self._model.loss(coeffs)

        return loss + \
               self._prox_l1.value(self.adjacency.ravel()) + \
               self._prox_nuclear.value(self.adjacency.ravel())

    @property
    def coeffs(self):
        return np.hstack((self.baseline, self.adjacency.ravel()))

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        if val < 0 or val is None:
            raise ValueError("`C` must be positive, got %s" % str(val))
        else:
            self._set("_C", val)
            self._prox_l1.strength = self.strength_lasso
            self._prox_nuclear.strength = self.strength_nuclear

    @property
    def lasso_nuclear_ratio(self):
        return self._lasso_nuclear_ratio

    @lasso_nuclear_ratio.setter
    def lasso_nuclear_ratio(self, val):
        if val < 0 or val > 1:
            raise ValueError("`lasso_nuclear_ratio` must be between 0 and 1, "
                             "got %s" % str(val))
        else:
            self._set("_lasso_nuclear_ratio", val)
            self._prox_l1.strength = self.strength_lasso
            self._prox_nuclear.strength = self.strength_nuclear

    @property
    def strength_lasso(self):
        return self.lasso_nuclear_ratio / self.C

    @property
    def strength_nuclear(self):
        return (1 - self.lasso_nuclear_ratio) / self.C

    def _corresponding_simu(self):
        """Create simulation object corresponding to the obtained coefficients
        """
        return SimuHawkesExpKernels(adjacency=self.adjacency,
                                    decays=self.decay, baseline=self.baseline)

    def get_kernel_supports(self):
        """Computes kernel support. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernels` API

        Returns
        -------
        output : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the support of
            kernel i, j
        """
        corresponding_simu = self._corresponding_simu()
        get_support = np.vectorize(lambda kernel: kernel.get_plot_support())
        return get_support(corresponding_simu.kernels)

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
        corresponding_simu = self._corresponding_simu()
        return corresponding_simu.kernels[i, j].get_values(abscissa_array)

    def get_kernel_norms(self):
        """Computes kernel norms. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernel_norms` API

        Returns
        -------
        norms : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the norm of
            kernel i, j
        """
        corresponding_simu = self._corresponding_simu()
        get_norm = np.vectorize(lambda kernel: kernel.get_norm())
        return get_norm(corresponding_simu.kernels)

    def score(self, events=None, end_times=None, baseline=None,
              adjacency=None):
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

        adjacency : `np.ndarray`, shape=(n_nodes, n_nodes), default = None
            Adjacency matrix for which the score is measured
            If `None` adjacency obtained during fitting is used

        Returns
        -------
        likelihood : `double`
            Computed log likelihood value
        """
        if events is None and not self._fitted:
            raise ValueError('You must either call `fit` before `score` or '
                             'provide events')

        if baseline is not None or adjacency is not None:
            if baseline is None:
                baseline = self.baseline
            if adjacency is None:
                adjacency = self.adjacency
            coeffs = np.hstack((baseline, adjacency.ravel()))
        else:
            coeffs = self.coeffs

        if events is None and end_times is None:
            model = self._model
        else:
            model = ModelHawkesExpKernLogLik(self.decay,
                                             n_threads=self.n_threads)
            model.fit(events, end_times)

        return -model.loss(coeffs)
