from warnings import warn

import numpy as np

from tick.base import actual_kwargs
from tick.inference.base import LearnerOptim
from tick.optim.model.base import ModelLipschitz
from tick.optim.prox import ProxElasticNet, ProxL1, ProxL2Sq, ProxPositive
from tick.optim.solver import AGD, GD, SGD, SVRG, BFGS
from tick.simulation import SimuHawkes


class LearnerHawkesParametric(LearnerOptim):
    """Base Hawkes process learner for given kernels, with many choices of
    penalization and solvers.

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

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Vector :math:`\mu \in \mathbb{R}^{n\_nodes}` by the attribute
      `baseline`

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : 'l1', 'l2', 'elasticnet', default='l2'
        The penalization to use. Default is ridge penalization.

    solver : 'gd', 'agd', 'bfgs', 'svrg', default='agd'
        The name of the solver to use

    step : `float`, default=None
        Initial step size used for learning. Used in 'gd', 'agd', 'sgd'
        and 'svrg' solvers

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

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.

        * For ratio = 0 this is ridge (L2 squared) regularization.
        * For ratio = 1 this is lasso (L1) regularization.
        * For 0 < ratio < 1, the regularization is a linear combination
          of L1 and L2.

        Used in 'elasticnet' penalty

    random_state : int seed, or None (default)
        The seed that will be used by stochastic solvers. If `None`, a random
        seed will be used (based on timestamp and other physical metrics).
        Used in 'sgd', and 'svrg' solvers

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes / components in the Hawkes model

    baseline : `np.array`, shape=(n_nodes,)
        Inferred baseline of each component's intensity

    coeffs : `np.array`, shape=(n_nodes * n_nodes + n_nodes, )
        Raw coefficients of the model.
    """

    _attrinfos = {
        "n_nodes": {"writable": False},
        "coeffs": {"writable": False},
    }

    _solvers = {
        "gd": GD,
        "agd": AGD,
        "sgd": SGD,
        "svrg": SVRG,
        "bfgs": BFGS,
    }

    _penalties = {
        "none": ProxPositive,
        "l1": ProxL1,
        "l2": ProxL2Sq,
        "elasticnet": ProxElasticNet,
    }

    @actual_kwargs
    def __init__(self, penalty="l2", C=1e3, solver="agd", step=None,
                 tol=1e-5, max_iter=100, verbose=False, print_every=10,
                 record_every=10, elastic_net_ratio=0.95, random_state=None):
        self.coeffs = None

        extra_prox_kwarg = {"positive": True}

        LearnerOptim.__init__(self, penalty=penalty, C=C,
                              solver=solver, step=step, tol=tol,
                              max_iter=max_iter, verbose=verbose,
                              warm_start=False, print_every=print_every,
                              record_every=record_every,
                              elastic_net_ratio=elastic_net_ratio,
                              random_state=random_state,
                              extra_prox_kwarg=extra_prox_kwarg)

    def fit(self, events: list, start=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        events : `list` of `np.array`
            The events of each component of the Hawkes. Namely
            `events[j]` contains a one-dimensional `numpy.array` of
            the events' timestamps of component j

        start : `np.array` or `float`, default=None
            If `np.array`, the initial `coeffs` coefficients passed to the
            solver, ie. the optimization algorithm.
            If a `float` is given, the initial point will be the vector
            filled with this float.
            If `None` it will be automatically chosen.

        Returns
        -------
        output : `LearnerHawkesParametric`
            The current instance of the Learner
        """
        solver_obj = self._solver_obj
        model_obj = self._model_obj
        prox_obj = self._prox_obj

        # Pass the data to the model
        model_obj.fit(events)

        if self.step is None and self.solver in self._solvers_with_step:

            if self.solver in self._solvers_with_linesearch:
                self._solver_obj.linesearch = True
            elif self.solver == "svrg":
                if isinstance(self._model_obj, ModelLipschitz):
                    self.step = 1. / self._model_obj.get_lip_max()
                else:
                    warn("SVRG step needs to be tuned manually", RuntimeWarning)
                    self.step = 1.
            elif self.solver == "sgd":
                warn("SGD step needs to be tuned manually", RuntimeWarning)
                self.step = 1.

        # Determine the range of the prox
        # User cannot specify a custom range if he is using learners
        self._set_prox_range(model_obj, prox_obj)

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        if isinstance(start, (int, float)):
            start = start * np.ones(model_obj.n_coeffs)

        if isinstance(start, np.ndarray):
            if start.shape != (model_obj.n_coeffs,):
                raise ValueError("'start' array has wrong shape %s instead of "
                                 "(%i, )" % (str(start.shape),
                                             model_obj.n_coeffs))
            coeffs_start = start.copy()
        else:
            coeffs_start = np.ones(model_obj.n_coeffs)

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start)

        # Get the learned coefficients
        self._set("coeffs", coeffs)
        self._set("_fitted", True)

        return self

    def _set_prox_range(self, model_obj, prox_obj):
        prox_obj.range = (0, model_obj.n_coeffs)

    @property
    def baseline(self):
        if not self._fitted:
            raise ValueError('You must fit data before getting estimated '
                             'baseline')
        else:
            return self.coeffs[:self.n_nodes]

    @property
    def n_nodes(self):
        return self._model_obj.n_nodes

    def _corresponding_simu(self):
        """Create simulation object corresponding to the obtained coefficients
        """
        return SimuHawkes()

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
