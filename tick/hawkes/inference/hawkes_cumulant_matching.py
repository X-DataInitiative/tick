from itertools import product

import numpy as np
import scipy
from scipy.linalg import qr, sqrtm, norm

from tick.base import Base
from tick.hawkes.inference.base import LearnerHawkesNoParam
from tick.hawkes.inference.build.hawkes_inference import (HawkesCumulant as
                                                          _HawkesCumulant)

# Tensorflow is not a project requirement but is needed for this class
try:
    import tensorflow as tf
except ImportError:
    tf = None
    pass


class HawkesCumulantMatching(LearnerHawkesNoParam):
    """This class is used for performing non parametric estimation of
    multi-dimensional Hawkes processes based cumulant matching.

    It does not make any assumptions on the kernel shape and recovers
    the kernel norms only.

    This class relies on `Tensorflow`_ to perform the matching of the
    cumulants. If `Tensorflow`_ is not installed, it will not work.

    Parameters
    ----------
    integration_support : `float`
        Controls the maximal lag (positive or negative) upon which we
        integrate the cumulant densities (covariance and skewness),
        this amounts to neglect border effects. In practice, this is a
        good approximation if the support of the kernels is smaller than
        integration support and if the spectral norm of the kernel norms
        is sufficiently distant from the critical value, namely 1.
        It denoted by :math:`H` in the paper.

    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='none'
        The penalization to use. By default no penalization is used.
        Penalty is only applied to adjacency matrix.

    solver : {'momentum', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'gd'}, default='adam'
        Name of tensorflow solver that will be used.

    step : `float`, default=1e-2
        Initial step size used for learning. Also known as learning rate.

    tol : `float`, default=1e-8
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). If not reached the solver does ``max_iter``
        iterations

    max_iter : `int`, default=1000
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=100
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

    solver_kwargs : `dict`, default=`None`
        Extra arguments that will be passed to tensorflow solver

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes / components in the Hawkes model

    baseline : `np.array`, shape=(n_nodes,)
        Inferred baseline of each component's intensity

    adjacency : `np.ndarray`, shape=(n_nodes, n_nodes)
        Inferred adjacency matrix

    mean_intensity : list of `np.array` shape=(n_nodes,)
        Estimated mean intensities, named :math:`\\widehat{L}` in the paper

    covariance : list of `np.array` shape=(n_nodes,n_nodes)
         Estimated integrated covariance, named :math:`\\widehat{C}`
         in the paper

    skewness : list of `np.array` shape=(n_nodes,n_nodes)
        Estimated integrated skewness (sliced), named :math:`\\widehat{K^c}`
        in the paper

    R : `np.array` shape=(n_nodes,n_nodes)
        Estimated weight, linked to the integrals of Hawkes kernels.
        Use to derive adjacency and baseline

    Other Parameters
    ----------------
    cs_ratio : `float`, default=`None`
        Covariance-skewness ratio. The higher it is, themore covariance
        has an impact the result which leads to more symmetric
        adjacency matrices. If None, a default value is computed based
        on the norm of the estimated covariance and skewness cumulants.

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    References
    ----------
    Achab, M., Bacry, E., Gaïffas, S., Mastromatteo, I., & Muzy, J. F.
    (2017, July). Uncovering causality from multivariate Hawkes integrated
    cumulants.
    `In International Conference on Machine Learning (pp. 1-10)`_.

    .. _In International Conference on Machine Learning (pp. 1-10): http://proceedings.mlr.press/v70/achab17a.html
    .. _Tensorflow: https://www.tensorflow.org
    """
    _attrinfos = {
        '_cumulant_computer': {
            'writable': False
        },
        '_solver': {
            'writable': False
        },
        '_elastic_net_ratio': {
            'writable': False
        },
        '_tf_feed_dict': {},
        '_tf_graph': {},
        '_events_of_cumulants': {
            'writable': False
        }
    }

    def __init__(self, integration_support, C=1e3, penalty='none',
                 solver='adam', step=1e-2, tol=1e-8, max_iter=1000,
                 verbose=False, print_every=100, record_every=10,
                 solver_kwargs=None, cs_ratio=None, elastic_net_ratio=0.95):
        try:
            import tensorflow
        except ImportError:
            raise ImportError('`tensorflow` >= 1.4.0 must be available to use '
                              'HawkesCumulantMatching')

        self._tf_graph = tf.Graph()

        LearnerHawkesNoParam.__init__(
            self, tol=tol, verbose=verbose, max_iter=max_iter,
            print_every=print_every, record_every=record_every)

        self._elastic_net_ratio = None
        self.C = C
        self.penalty = penalty
        self.elastic_net_ratio = elastic_net_ratio
        self.step = step
        self.cs_ratio = cs_ratio
        self.solver_kwargs = solver_kwargs
        if self.solver_kwargs is None:
            self.solver_kwargs = {}

        self._cumulant_computer = _HawkesCumulantComputer(
            integration_support=integration_support)
        self._learner = self._cumulant_computer._learner
        self._solver = solver
        self._tf_feed_dict = None
        self._events_of_cumulants = None

        self.history.print_order = ["n_iter", "objective", "rel_obj"]

    def compute_cumulants(self, force=False):
        """Compute estimated mean intensity, covariance and sliced skewness

        Parameters
        ----------
        force : `bool`
            If `True` previously computed cumulants are not reused
        """
        self._cumulant_computer.compute_cumulants(verbose=self.verbose,
                                                  force=force)

    @property
    def mean_intensity(self):
        if not self._cumulant_computer.cumulants_ready:
            self.compute_cumulants()

        return self._cumulant_computer.L

    @property
    def covariance(self):
        if not self._cumulant_computer.cumulants_ready:
            self.compute_cumulants()

        return self._cumulant_computer.C

    @property
    def skewness(self):
        if not self._cumulant_computer.cumulants_ready:
            self.compute_cumulants()

        return self._cumulant_computer.K_c

    def objective(self, adjacency=None, R=None):
        """Compute objective value for a given adjacency or variable R

        Parameters
        ----------
        adjacency : `np.ndarray`, shape=(n_nodes, n_nodes), default=None
            Adjacency matrix at which we compute objective.
            If `None`, objective will be computed at `R`

        R : `np.ndarray`, shape=(n_nodes, n_nodes), default=None
            R variable at which objective is computed. Superseded by
            adjacency if adjacency is not `None`

        Returns
        -------
        Value of objective function
        """
        cost = self._tf_objective_graph()
        L, C, K_c = self._tf_placeholders()

        if adjacency is not None:
            R = scipy.linalg.inv(np.eye(self.n_nodes) - adjacency)

        with self._tf_graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._tf_model_coeffs.assign(R))

                return sess.run(
                    cost, feed_dict={
                        L: self.mean_intensity,
                        C: self.covariance,
                        K_c: self.skewness
                    })

    @property
    def _tf_model_coeffs(self):
        """Tensorflow variable of interest, used to perform minimization of
        objective function
        """
        with self._tf_graph.as_default():
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                return tf.get_variable("R", [self.n_nodes, self.n_nodes],
                                       dtype=tf.float64)

    @property
    def adjacency(self):
        return np.eye(self.n_nodes) - scipy.linalg.inv(self.solution)

    @property
    def baseline(self):
        return scipy.linalg.inv(self.solution).dot(self.mean_intensity)

    def _tf_placeholders(self):
        """Tensorflow placeholders to manage cumulants data
        """
        d = self.n_nodes
        if self._tf_feed_dict is None:
            with self._tf_graph.as_default():
                L = tf.placeholder(tf.float64, d, name='L')
                C = tf.placeholder(tf.float64, (d, d), name='C')
                K_c = tf.placeholder(tf.float64, (d, d), name='K_c')
                self._tf_feed_dict = L, C, K_c

        return self._tf_feed_dict

    def _tf_objective_graph(self):
        """Objective fonction written as a tensorflow graph
        """
        d = self.n_nodes

        if self.cs_ratio is None:
            cs_ratio = self.approximate_optimal_cs_ratio()
        else:
            cs_ratio = self.cs_ratio

        with self._tf_graph.as_default():
            L, C, K_c = self._tf_placeholders()
            R = self._tf_model_coeffs
            I = tf.constant(np.eye(d), dtype=tf.float64)

            # Construct model
            variable_covariance = \
                tf.matmul(R, tf.matmul(tf.diag(L), R, transpose_b=True))

            variable_skewness = \
                tf.matmul(C, tf.square(R), transpose_b=True) \
                + 2.0 * tf.matmul(R, R * C, transpose_b=True) \
                - 2.0 * tf.matmul(R, tf.matmul(
                    tf.diag(L), tf.square(R), transpose_b=True))

            covariance_divergence = tf.reduce_mean(
                tf.squared_difference(variable_covariance, C))

            skewness_divergence = tf.reduce_mean(
                tf.squared_difference(variable_skewness, K_c))

            cost = (1 - cs_ratio) * skewness_divergence
            cost += cs_ratio * covariance_divergence

            # Add potential regularization
            cost = tf.cast(cost, tf.float64)
            if self.strength_lasso > 0:
                reg_l1 = tf.contrib.layers.l1_regularizer(self.strength_lasso)
                cost += reg_l1((I - tf.matrix_inverse(R)))
            if self.strength_ridge > 0:
                reg_l2 = tf.contrib.layers.l2_regularizer(self.strength_ridge)
                cost += reg_l2((I - tf.matrix_inverse(R)))

            return cost

    def fit(self, events, end_times=None, adjacency_start=None, R_start=None):
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

        adjacency_start : `str` or `np.ndarray, shape=(n_nodes + n_nodes * n_nodes,), default=`None`
            Initial guess for the adjacency matrix. Will be used as
            starting point in optimization.
            If `None` and `R_start` is also `None`, a default starting point
            is estimated from the estimated cumulants
            If `"random"`, a starting point is estimated from estimated
            cumulants with a bit a randomness

        R_start : `np.ndarray`, shape=(n_nodes, n_nodes), default=None
            R variable at which we start optimization. Superseded by
            adjacency_start if adjacency_start is not `None`
        """
        LearnerHawkesNoParam.fit(self, events, end_times=end_times)
        self.solve(adjacency_start=adjacency_start, R_start=R_start)

    def _solve(self, adjacency_start=None, R_start=None):
        """Launch optimization algorithm

        Parameters
        ----------
        adjacency_start : `str` or `np.ndarray, shape=(n_nodes + n_nodes * n_nodes,), default=`None`
            Initial guess for the adjacency matrix. Will be used as 
            starting point in optimization.
            If `None`, a default starting point is estimated from the 
            estimated cumulants
            If `"random"`, as with `None`, a starting point is estimated from
            estimated cumulants with a bit a randomness

        max_iter : `int`
            The number of training epochs.

        step : `float`
            The learning rate used by the optimizer.

        solver : {'adam', 'momentum', 'adagrad', 'rmsprop', 'adadelta', 'gd'}, default='adam'
            Solver used to minimize the loss. As the loss is not convex, it
            cannot be optimized with `tick.optim.solver` solvers
        """
        self.compute_cumulants()

        if adjacency_start is None and R_start is not None:
            start_point = R_start
        elif adjacency_start is None or adjacency_start == 'random':
            random = adjacency_start == 'random'
            start_point = self.starting_point(random=random)
        else:
            start_point = scipy.linalg.inv(
                np.eye(self.n_nodes) - adjacency_start)

        cost = self._tf_objective_graph()
        L, C, K_c = self._tf_placeholders()

        # Launch the graph
        with self._tf_graph.as_default():
            with tf.variable_scope("solver", reuse=tf.AUTO_REUSE):
                tf_solver = self.tf_solver(self.step, **self.solver_kwargs)
                optimization = tf_solver.minimize(cost)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._tf_model_coeffs.assign(start_point))
                # Training cycle
                for n_iter in range(self.max_iter):
                    if self._should_record_iter(n_iter):
                        # We don't use self.objective here as it would be very
                        # slow
                        prev_obj = sess.run(
                            cost, feed_dict={
                                L: self.mean_intensity,
                                C: self.covariance,
                                K_c: self.skewness
                            })

                    sess.run(
                        optimization, feed_dict={
                            L: self.mean_intensity,
                            C: self.covariance,
                            K_c: self.skewness
                        })

                    if self._should_record_iter(n_iter):
                        obj = sess.run(
                            cost, feed_dict={
                                L: self.mean_intensity,
                                C: self.covariance,
                                K_c: self.skewness
                            })
                        rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                        prev_obj = obj
                        converged = rel_obj < self.tol

                        force = converged or n_iter + 1 == self.max_iter
                        self._handle_history(n_iter + 1, objective=obj,
                                             rel_obj=rel_obj, force=force)

                        if converged:
                            break

                self._set('solution', sess.run(self._tf_model_coeffs))

    def approximate_optimal_cs_ratio(self):
        """Heuristic to set covariance skewness ratio close to its
        optimal value
        """
        norm_sq_C = norm(self.covariance) ** 2
        norm_sq_K_c = norm(self.skewness) ** 2
        return norm_sq_K_c / (norm_sq_K_c + norm_sq_C)

    def starting_point(self, random=False):
        """Heuristic to find a starting point candidate

        Parameters
        ----------
        random : `bool`
            Use a random orthogonal matrix instead of identity

        Returns
        -------
        startint_point : `np.ndarray`, shape=(n_nodes, n_nodes)
            A starting point candidate
        """
        sqrt_C = sqrtm(self.covariance)
        sqrt_L = np.sqrt(self.mean_intensity)
        if random:
            random_matrix = np.random.rand(self.n_nodes, self.n_nodes)
            M, _ = qr(random_matrix)
        else:
            M = np.eye(self.n_nodes)
        initial = np.dot(np.dot(sqrt_C, M), np.diag(1. / sqrt_L))
        return initial

    def get_kernel_values(self, i, j, abscissa_array):
        raise ValueError('Hawkes cumulant cannot estimate kernel values')

    def get_kernel_norms(self):
        """Computes kernel norms. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernel_norms` API

        Returns
        -------
        norms : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the norm of
            kernel i, j
        """
        return self.adjacency

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        available_solvers = [
            'momentum', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'gd'
        ]
        if val.lower() not in available_solvers:
            raise ValueError('solver must be one of {}, recieved {}'.format(
                available_solvers, val))

        self._set('_solver', val)

    @property
    def tf_solver(self):
        if self.solver.lower() == 'momentum':
            return tf.train.MomentumOptimizer
        elif self.solver.lower() == 'adam':
            return tf.train.AdamOptimizer
        elif self.solver.lower() == 'adagrad':
            return tf.train.AdagradOptimizer
        elif self.solver.lower() == 'rmsprop':
            return tf.train.RMSPropOptimizer
        elif self.solver.lower() == 'adadelta':
            return tf.train.AdadeltaOptimizer
        elif self.solver.lower() == 'gd':
            return tf.train.GradientDescentOptimizer

    @property
    def elastic_net_ratio(self):
        return self._elastic_net_ratio

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, val):
        if val < 0 or val > 1:
            raise ValueError("`elastic_net_ratio` must be between 0 and 1, "
                             "got %s" % str(val))
        else:
            self._set("_elastic_net_ratio", val)

    @property
    def strength_lasso(self):
        if self.penalty == 'elasticnet':
            return self.elastic_net_ratio / self.C
        elif self.penalty == 'l1':
            return 1. / self.C
        else:
            return 0.

    @property
    def strength_ridge(self):
        if self.penalty == 'elasticnet':
            return (1 - self.elastic_net_ratio) / self.C
        elif self.penalty == 'l2':
            return 1. / self.C
        return 0.


class _HawkesCumulantComputer(Base):
    """Private class to compute Hawkes cumulants
    """

    _cpp_obj_name = '_learner'
    _attrinfos = {
        'integration_support': {
            'cpp_setter': 'set_integration_support'
        },
        '_learner': {},
        'L': {},
        'C': {},
        'K_c': {},
        '_L_day': {},
        '_J': {},
        '_events_of_cumulants': {},
    }

    def __init__(self, integration_support=100.):
        Base.__init__(self)
        self.integration_support = integration_support
        self._learner = _HawkesCumulant(self.integration_support)

        self.L = None
        self.C = None
        self.K_c = None

        self._L_day = None
        self._J = None
        self._events_of_cumulants = None

    def compute_cumulants(self, verbose=False, force=False):
        """Compute estimated mean intensity, covariance and sliced skewness

        Parameters
        ----------
        verbose : `bool`
            If `True`, a message will be printed when previously computed
            cumulants are reused

        force : `bool`
            If `True` cumulants will always be recomputed
        """
        if len(self.realizations) == 0:
            raise RuntimeError('Cannot compute cumulants if no realization '
                               'has been provided')

        # Check if cumulants have already been computed
        if self.cumulants_ready and not force:
            if verbose:
                print('Use previouly computed cumulants')
            return

        # Remember for which realizations cumulants have been computed
        self._events_of_cumulants = self.realizations
        self.compute_L()
        self.compute_C_and_J()
        self.K_c = self.compute_E_c()

        self._learner.set_are_cumulants_ready(True)

    @staticmethod
    def _same_realizations(events_1, events_2):
        if len(events_1) != len(events_2):
            return False

        for r in range(len(events_1)):
            if len(events_1[r]) != len(events_2[r]):
                return False

            # Fast check that both arrays share the same pointer
            for i in range(len(events_1[r])):
                if events_1[r][i].__array_interface__ != \
                        events_2[r][i].__array_interface__:
                    return False
        return True

    def compute_L(self):
        self._L_day = np.zeros((self.n_realizations, self.n_nodes))

        for day, realization in enumerate(self.realizations):
            for i in range(self.n_nodes):
                process = realization[i]
                self._L_day[day][i] = len(process) / self.end_times[day]

        self.L = np.mean(self._L_day, axis=0)

    def compute_C_and_J(self):
        self.C = np.zeros((self.n_nodes, self.n_nodes))
        self._J = np.zeros((self.n_realizations, self.n_nodes, self.n_nodes))

        d = self.n_nodes
        for day in range(len(self.realizations)):
            C = np.zeros((d, d))
            J = np.zeros((d, d))
            for i, j in product(range(d), repeat=2):
                res = self._learner.compute_A_and_I_ij(day, i, j,
                                                       self._L_day[day][j])
                C[i, j] = res[0]
                J[i, j] = res[1]
            # we keep the symmetric part to remove edge effects
            C[:] = 0.5 * (C + C.T)
            J[:] = 0.5 * (J + J.T)
            self.C += C / self.n_realizations
            self._J[day] = J.copy()

    def compute_E_c(self):
        E_c = np.zeros((self.n_nodes, self.n_nodes, 2))

        d = self.n_nodes

        for day in range(len(self.realizations)):
            for i in range(d):
                for j in range(d):
                    E_c[i, j, 0] += self._learner.compute_E_ijk(
                        day, i, j, j, self._L_day[day][i], self._L_day[day][j],
                        self._J[day][i, j])

                    E_c[i, j, 1] += self._learner.compute_E_ijk(
                        day, j, j, i, self._L_day[day][j], self._L_day[day][j],
                        self._J[day][j, j])

        E_c /= self.n_realizations

        K_c = np.zeros((self.n_nodes, self.n_nodes))
        K_c += 2 * E_c[:, :, 0]
        K_c += E_c[:, :, 1]
        K_c /= 3.

        return K_c

    @property
    def n_nodes(self):
        return self._learner.get_n_nodes()

    @property
    def n_realizations(self):
        return len(self._learner.get_n_jumps_per_realization())

    @property
    def realizations(self):
        return self._learner.get_timestamps_list()

    @property
    def end_times(self):
        return self._learner.get_end_times()

    @property
    def cumulants_ready(self):
        events_didnt_change = self._events_of_cumulants is not None and \
                              _HawkesCumulantComputer._same_realizations(
                                  self._events_of_cumulants, self.realizations)

        return self._learner.get_are_cumulants_ready() and events_didnt_change
