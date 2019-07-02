# License: BSD 3 clause

from warnings import warn

import numpy as np

from tick.base import actual_kwargs
from tick.base.learner import LearnerOptim
from tick.base_model import ModelLipschitz
from tick.hawkes import SimuHawkes
from tick.plot import plot_point_process
from tick.prox import ProxElasticNet, ProxL1, ProxL2Sq, ProxPositive


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

    events : `list` of `np.ndarray`
        Events given to learner during fitting
    """

    _attrinfos = {
        "n_nodes": {
            "writable": False
        },
        "coeffs": {
            "writable": False
        },
    }

    _solvers = {
        "gd": 'GD',
        "agd": 'AGD',
        "sgd": 'SGD',
        "svrg": 'SVRG',
        "bfgs": 'BFGS',
    }

    _penalties = {
        "none": ProxPositive,
        "l1": ProxL1,
        "l2": ProxL2Sq,
        "elasticnet": ProxElasticNet,
    }

    @actual_kwargs
    def __init__(self, penalty="l2", C=1e3, solver="agd", step=None, tol=1e-5,
                 max_iter=100, verbose=False, print_every=10, record_every=10,
                 elastic_net_ratio=0.95, random_state=None):
        self.coeffs = None
        self.events = None

        extra_prox_kwarg = {"positive": True}

        LearnerOptim.__init__(
            self, penalty=penalty, C=C, solver=solver, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, warm_start=False,
            print_every=print_every, record_every=record_every,
            elastic_net_ratio=elastic_net_ratio, random_state=random_state,
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
        self._set('events', events)

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
                    warn("SVRG step needs to be tuned manually",
                         RuntimeWarning)
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
                raise ValueError(
                    "'start' array has wrong shape %s instead of "
                    "(%i, )" % (str(start.shape), model_obj.n_coeffs))
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

    def score(self, events=None, end_times=None, coeffs=None):
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

        coeffs : `np.ndarray`
            Coefficients at which the score is measured

        Returns
        -------
        likelihood : `double`
            Computed log likelihood value
        """
        if events is None and not self._fitted:
            raise ValueError('You must either call `fit` before `score` or '
                             'provide events')

        if coeffs is None:
            coeffs = self.coeffs

        if events is None and end_times is None:
            model = self._model_obj
        else:
            model = self._construct_model_obj()
            model.fit(events, end_times)

        return -model.loss(coeffs)

    def estimated_intensity(self, events, intensity_track_step, end_time=None):
        """Value of intensity for a given realization with the fitted parameters

        Parameters
        ----------
        events : `list` of `np.ndarray`, default = None
            One Hawkes processes realization, a list of n_node for
            each component of the Hawkes. Namely `events[i]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component i.

        intensity_track_step : `float`, default = None
            How often the intensity should be computed

        end_time : `float`, default = None
            End time of hawkes process.
            If None, it will be set to realization's latest time.

        Returns
        -------
        tracked_intensity : `list` of `np.array`
            intensity values for all components

        intensity_tracked_times : `np.array`
            Times at wich intensity has been recorded
        """
        if end_time is None:
            end_time = max(map(max, events))

        simu = self._corresponding_simu()
        if intensity_track_step is not None:
            simu.track_intensity(intensity_track_step)

        simu.set_timestamps(events, end_time)
        return simu.tracked_intensity, simu.intensity_tracked_times

    def plot_estimated_intensity(self, events, n_points=10000, plot_nodes=None,
                                 t_min=None, t_max=None,
                                 intensity_track_step=None, max_jumps=None,
                                 show=True, ax=None):
        """Plot value of intensity for a given realization with the fitted
        parameters

        events : `list` of `np.ndarray`, default = None
            One Hawkes processes realization, a list of n_node for
            each component of the Hawkes. Namely `events[i]` contains a
            one-dimensional `numpy.array` of the events' timestamps of
            component i.

        n_points : `int`, default=10000
            Number of points used for intensity plot.

        plot_nodes : `list` of `int`, default=`None`
            List of nodes that will be plotted. If `None`, all nodes are
            considered

        t_min : `float`, default=`None`
            If not `None`, time at which plot will start

        t_max : `float`, default=`None`
            If not `None`, time at which plot will stop

        intensity_track_step : `float`, default=`None`
            Defines how often intensity will be computed. If this is too low,
            computations might be long. By default, a value will be
            extrapolated from (t_max - t_min) / n_points.

        max_jumps : `int`, default=`None`
            If not `None`, maximum of jumps per coordinate that will be plotted.
            This is useful when plotting big point processes to ensure a only
            readable part of them will be plotted

        show : `bool`, default=`True`
            if `True`, show the plot. Otherwise an explicit call to the show
            function is necessary. Useful when superposing several plots.

        ax : `list` of `matplotlib.axes`, default=None
            If not None, the figure will be plot on this axis and show will be
            set to False.
        """

        simu = self._corresponding_simu()
        end_time = max(map(max, events))

        if t_max is not None:
            end_time = max(end_time, t_max)

        if intensity_track_step is None:
            display_start_time = 0
            if t_min is not None:
                display_start_time = t_min
            display_end_time = end_time
            if t_min is not None:
                display_start_time = t_min

            intensity_track_step = (display_end_time - display_start_time) \
                                   / n_points

        simu.track_intensity(intensity_track_step)
        simu.set_timestamps(events, end_time)

        plot_point_process(simu, plot_intensity=True, n_points=n_points,
                           plot_nodes=plot_nodes, t_min=t_min, t_max=t_max,
                           max_jumps=max_jumps, show=show, ax=ax)
