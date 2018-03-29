# License: BSD 3 clause

import numpy as np

from tick.base import actual_kwargs
from tick.hawkes import (ModelHawkesExpKernLogLik, ModelHawkesExpKernLeastSq,
                         SimuHawkesExpKernels)
from tick.hawkes.inference.base import LearnerHawkesParametric
from tick.prox import ProxElasticNet, ProxL1, ProxL2Sq, ProxNuclear, \
    ProxPositive


class HawkesExpKern(LearnerHawkesParametric):
    """
    Hawkes process learner for exponential kernels with fixed and given decays,
    with many choices of penalization and solvers.

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
        \phi_{ij}(t) = \\alpha^{ij} \\beta^{ij}
                       \exp (- \\beta^{ij} t) 1_{t > 0}

    In our implementation we denote:

    * Integer :math:`D` by the attribute `n_nodes`
    * Vector :math:`\mu \in \mathbb{R}^{D}` by the attribute
      `baseline`
    * Matrix :math:`A = (\\alpha^{ij})_{ij} \in \mathbb{R}^{D \\times D}`
      by the attribute `adjacency`
    * Matrix :math:`B = (\\beta_{ij})_{ij} \in \mathbb{R}^{D \\times D}` by the
      parameter `decays`. This parameter is given to the model

    Parameters
    ----------
    decays : `float` or `np.ndarray`, shape=(n_nodes, n_nodes)
        The decays used in the exponential kernels. If a `float` is given,
        the initial point will be the matrix filled with this float.

    gofit : {'least-squares', 'likelihood'}, default='least-squares'
        Goodness-of-fit used for model's fitting
    
    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'nuclear', 'none'}, default='l2'
        The penalization to use. Default is ridge penalization.
        If nuclear is chosen, it is applied on the adjacency matrix.

    solver : {'gd', 'agd', 'bfgs', 'svrg'}, default='agd'
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

    adjacency : `np.ndarray`, shape=(n_nodes, n_nodes)
        Inferred adjacency matrix

    coeffs : `np.array`, shape=(n_nodes * n_nodes + n_nodes, )
        Raw coefficients of the model. Row stack of `self.baseline` and
        `self.adjacency`
    """

    _attrinfos = {
        "gofit": {
            "writable": False
        },
        "decays": {
            "writable": False
        },
    }

    _penalties = {
        "none": ProxPositive,
        "l1": ProxL1,
        "l2": ProxL2Sq,
        "elasticnet": ProxElasticNet,
        "nuclear": ProxNuclear
    }

    @actual_kwargs
    def __init__(self, decays, gofit="least-squares", penalty="l2", C=1e3,
                 solver="agd", step=None, tol=1e-5, max_iter=100,
                 verbose=False, print_every=10, record_every=10,
                 elastic_net_ratio=0.95, random_state=None):

        self._actual_kwargs = \
            HawkesExpKern.__init__.actual_kwargs

        self._set_gofit(gofit)
        self.decays = decays

        LearnerHawkesParametric.__init__(
            self, penalty=penalty, C=C, solver=solver, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, print_every=print_every,
            record_every=record_every, elastic_net_ratio=elastic_net_ratio,
            random_state=random_state)

        if penalty == "nuclear" and solver in self._solvers_stochastic:
            raise ValueError("Penalty 'nuclear' cannot be used with "
                             "stochastic solver '%s'" % solver)

    def _construct_model_obj(self):
        if self.gofit == "least-squares":
            model = ModelHawkesExpKernLeastSq(self.decays)
        elif self.gofit == "likelihood":
            # decays must be constant
            if isinstance(self.decays, np.ndarray):
                if self.decays.min() == self.decays.max():
                    decays = self.decays.min()
            if not isinstance(self.decays, (int, float)):
                raise NotImplementedError("With 'likelihood' goodness of fit, "
                                          "you must provide a constant decay "
                                          "for all kernels")

            model = ModelHawkesExpKernLogLik(self.decays)
        return model

    def _set_gofit(self, val):
        if val not in ["least-squares", "likelihood"]:
            raise ValueError(
                "Parameter gofit (goodness of fit) must be either "
                "'least-squares' or 'likelihood'")
        self.gofit = val

    def _set_prox_range(self, model_obj, prox_obj):
        if self.penalty == "nuclear":
            prox_obj.range = (self.n_nodes, model_obj.n_coeffs)
            prox_obj.n_rows = self.n_nodes
        else:
            LearnerHawkesParametric._set_prox_range(self, model_obj, prox_obj)

    @property
    def adjacency(self):
        if not self._fitted:
            raise ValueError('You must fit data before getting estimated '
                             'adjacency')
        else:
            return self.coeffs[self.n_nodes:].reshape((self.n_nodes,
                                                       self.n_nodes))

    def _corresponding_simu(self):
        return SimuHawkesExpKernels(adjacency=self.adjacency,
                                    decays=self.decays, baseline=self.baseline)

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
        if baseline is not None or adjacency is not None:
            if baseline is None:
                baseline = self.baseline
            if adjacency is None:
                adjacency = self.adjacency
            coeffs = np.hstack((baseline, adjacency.ravel()))
        else:
            coeffs = None

        return LearnerHawkesParametric.score(
            self, events=events, end_times=end_times, coeffs=coeffs)
