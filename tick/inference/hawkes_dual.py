# License: BSD 3 clause

import numpy as np

from tick.inference.base import LearnerHawkesNoParam
from tick.optim.model import ModelHawkesFixedExpKernLogLik
from tick.optim.prox.prox_l1 import ProxL1
from tick.optim.prox.prox_nuclear import ProxNuclear
from tick.optim.solver.base.utils import relative_distance
from tick.simulation import SimuHawkesExpKernels
from .build.inference import HawkesSDCALoglikKern as _HawkesSDCALoglikKern


class HawkesDual(LearnerHawkesNoParam):
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

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes / components in the Hawkes model

    baseline : `np.array`, shape=(n_nodes,)
        Inferred baseline of each component's intensity

    adjacency : `np.ndarray`, shape=(n_nodes, n_nodes)
        Inferred adjacency matrix
    """

    _attrinfos = {
        "_learner": {"writable": False},
        "_model": {"writable": False},
        "decay": {
            "cpp_setter": "set_decay"
        },
        "_C": {"writable": False},
        "baseline": {"writable": False},
        "adjacency": {"writable": False},
        "approx": {"writable": False}
    }

    def __init__(self, decay, l_l2sq, max_iter=50, tol=1e-5, n_threads=1,
                 verbose=False, print_every=10, record_every=10):

        LearnerHawkesNoParam.__init__(self, verbose=verbose, max_iter=max_iter,
                                      print_every=print_every, tol=tol,
                                      n_threads=n_threads,
                                      record_every=record_every)
        self.decay = decay
        self.l_l2sq = l_l2sq
        self.verbose = verbose

        self._learner = _HawkesSDCALoglikKern(decay, l_l2sq, n_threads, tol)

        # TODO add approx to model
        self._model = ModelHawkesFixedExpKernLogLik(self.decay,
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

        objective = self.objective(self.coeffs)
        for i in range(self.max_iter + 1):
            prev_objective = objective
            prev_baseline = self.baseline.copy()
            prev_adjacency = self.adjacency.copy()

            self._learner.solve()

            objective = self.objective(self.coeffs)

            rel_obj = abs(objective - prev_objective) / abs(prev_objective)
            rel_baseline = relative_distance(self.baseline, prev_baseline)
            rel_adjacency = relative_distance(self.adjacency, prev_adjacency)

            # We perform at least 5 iterations as at start we sometimes reach a
            # low tolerance if inner_tol is too low
            converged = rel_obj <= self.tol
            force_print = (i == self.max_iter) or converged

            self._handle_history(i, obj=objective, rel_obj=rel_obj,
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
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(coeffs) ** 2

        return loss + prox_l2_value

    @property
    def coeffs(self):
        return self._learner.get_iterate()

    # @property
    # def l_l2sq(self):
    #     return self._learner.get_l_l2sq()
    #
    # @l_l2sq.setter
    # def l_l2sq(self, val):
    #     if val < 0 or val is None:
    #         raise ValueError("`l_l2sq` must be positive, got %s" % str(val))
    #     else:
    #         self._learner.set_l_l2sq(val)

    def _corresponding_simu(self):
        """Create simulation object corresponding to the obtained coefficients
        """
        return SimuHawkesExpKernels(adjacency=self.adjacency,
                                    decays=self.decay,
                                    baseline=self.baseline)

    @property
    def baseline(self):
        if not self._fitted:
            raise ValueError('You must fit data before getting estimated '
                             'baseline')
        else:
            return self.coeffs[:self.n_nodes]

    @property
    def n_nodes(self):
        return self._model.n_nodes

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
