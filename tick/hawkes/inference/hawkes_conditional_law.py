# License: BSD 3 clause

import itertools
import sys
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

from tick.base import Base, ThreadPool
from tick.hawkes.inference.build.hawkes_inference import (PointProcessCondLaw)


# noinspection PyPep8Naming
class HawkesConditionalLaw(Base):
    """This class is used for performing non parametric estimation of
    multi-dimensional marked Hawkes processes based on conditional laws.

    Marked Hawkes processes are point processes defined by the intensity:

    .. math::
        \\forall i \\in [1 \\dots D], \\quad
        \\lambda_i = \\mu_i + \\sum_{j=1}^D \\int \\phi_{ij} * f_{ij}(v_j) dN_j

    where

    * :math:`D` is the number of nodes
    * :math:`\mu_i` are the baseline intensities
    * :math:`\phi_{ij}` are the kernels
    * :math:`v_j` are the marks (considered iid) of the process :math:`N_j`
    * :math:`f_{ij}` the mark functions supposed to be piece-wise constant
      on intervals :math:`I^j(l)`

    The estimation is made from empirical computations of

    .. math::
        \\lim_{\\epsilon \\rightarrow 0}
        E [ (N_i[t + lag + \\delta + \\epsilon] -
             \Lambda[t + lag + \\epsilon]) | N_j[t]=1
        \quad \& \quad
        v_j(t) \in I^j(l) ]

    For all the possible values of :math:`i`, :math:`i` and :math:`l`.
    The :math:`lag` is sampled on a uniform grid defined by
    :math:`\\delta`: :math:`lag = n * \\delta`.

    Estimation can be performed using several realizations.

    Parameters
    ----------
    claw_method : {'lin', 'log'}, default='lin'
        Specifies the way the conditional laws are sampled. It can be either:

        * 'lin' : sampling is linear on [0, max_lag] using sampling period
          delta_lag
        * 'log' : sampling is semi-log. It uses linear sampling on [0, min_lag]
          with sampling period delta_lag and log sampling on [min_lag, max_lag]
          using :math:`\\exp(\\delta)` sampling period.

    delta_lag : `float`, default=0.1
        See claw_methods

    min_lag : `float`, default=1e-4
        See claw_methods

    max_lag : `float`, default=40
        See claw_methods

    quad_method : {'gauss', 'lin', 'log'}, default=gauss
        Sampling used for quadrature

        * 'gauss' for gaussian quadrature
        * 'lin' for linear quadrature
        * 'log' for log quadrature

    min_support : `float`, default=1e-4
        Start value of kernel estimation. It is used for 'log' quadrature
        method only, otherwise it is set to 0.

    max_support : `float`, default=40
        End value of kernel estimation

    n_quad : `int` : default=50
        The number of quadrature points between [min_support, max_support]
        used for solving the system.
        Be aware that the complexity increase as this number squared.

    n_threads : `int`, default=1
        Number of threads used for parallel computation.

        * if `int <= 0`: the number of physical cores available on the CPU
        * otherwise the desired number of threads

    Other Parameters
    ----------------
    delayed_component : list of `int`, shape=(n_nodes, ), default=None
        list of node indices corresponding to node that should be delayed
        (to avoid simultaneous jumps of different components which can be a
        problem in the estimation)

    delay : `float`
        The delayed used for `delayed_component`. Selected components are
        all delayed with the same value

    marked_components : `dict`
        A dictionary that indicates which component is considered as marked
        and what are the corresponding intervals ``I_j(l)``

    Attributes
    ----------
    n_nodes : `int`
        Number of nodes of the estimated Hawkes process

    n_realizations : `int`
        Number of given realizations

    baseline : np.ndarray, shape=(n_nodes,)
        Estimation of the baseline

    kernels_norms : np.ndarray, shape=(n_nodes, n_nodes)
        L1 norm matrix of the kernel norms

    kernels : list of list
        Kernel's estimation on the quadrature points

    mean_intensity : list of `float`
        The estimated mean intensity

    symmetries1d : list of 2-tuple
        List of component index pairs for imposing symmetries on the mean
        intensity (e.g, ``[(0,1),(2,3)]`` means that the mean intensity of
        the components 0 and 1 must be the same and the mean intensity of the
        components 2 and 3 also
        Can be set using can be set using the `set_model` method.

    symmetries2d : list of 2-tuple of 2-tuple
        List of kernel coordinates pairs to impose symmetries on the kernel
        matrix (e.g., ``[[(0,0),(1,1)],[(1,0),(0,1)]]`` for a bidiagonal
        kernel in dimension 2)
        Can be set using can be set using the `set_model` method.

    mark_functions : list of 2-tuple
        The mark functions as a list (lexical order on i,j and l, see below)

    References
    ----------
    Bacry, E., & Muzy, J. F. (2014).
    Second order statistics characterization of Hawkes processes and
    non-parametric estimation. `arXiv preprint arXiv:1401.0903`_.

    .. _arXiv preprint arXiv:1401.0903: https://arxiv.org/pdf/1401.0903.pdf
    """

    _attrinfos = {
        '_hawkes_object': {},
        '_lags': {},
        '_lock': {
            'writable': False
        },
        '_phi_ijl': {},
        '_norm_ijl': {},
        '_ijl2index': {},
        '_index2ijl': {},
        '_n_index': {},
        '_mark_probabilities': {},
        '_mark_probabilities_N': {},
        '_mark_min': {},
        '_mark_max': {},
        '_lam_N': {},
        '_lam_T': {},
        '_claw': {},
        '_claw1': {},
        '_claw_X': {},
        '_n_events': {},
        '_int_claw': {},
        '_IG': {},
        '_IG2': {},
        '_quad_x': {},
        '_quad_w': {}
    }

    def __init__(self, delta_lag=.1, min_lag=1e-4, max_lag=40, n_quad=50,
                 max_support=40, min_support=1e-4, quad_method='gauss',
                 marked_components=None, delayed_component=None, delay=0.00001,
                 model=None, n_threads=1, claw_method='lin'):

        Base.__init__(self)

        # Init the claw sampling parameters
        self.delta_lag = delta_lag
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.claw_method = claw_method

        # Init quadrature method
        self.quad_method = quad_method
        self.n_quad = n_quad
        self.min_support = min_support
        self.max_support = max_support

        # Init marked components
        if marked_components is None:
            marked_components = dict()
        self.marked_components = marked_components

        # Init attributes
        self.n_realizations = 0
        self._lags = None

        self._compute_lags()

        self.symmetries1d = []
        self.symmetries2d = []
        self.delayed_component = np.array(delayed_component)
        self.delay = delay

        # _claw : list of 2-tuple
        # Represents the conditional laws written above (lexical order on i,
        # j and l, see below). Each conditional law is represented by a
        # pair (x, c) where x are the abscissa
        self._claw = None
        # _claw1 : list of list
        # Represents the conditional laws written above without conditioning by
        # the mark (so a i,j list)
        self._claw1 = None
        self._lock = None

        # quad_x : `np.ndarray`, shape=(n_quad, )
        # The abscissa of the quadrature points used for the Fredholm system
        self._quad_x = None

        # quad_w : `np.ndarray`, shape=(n_quad, )
        # The weights the quadrature points used for the Fredholm system
        self._quad_w = None

        self._phi_ijl, self._norm_ijl = None, None
        self.kernels, self.kernels_norms, self.baseline = None, None, None
        self.mark_functions = None

        if n_threads == -1:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        self.n_threads = n_threads
        if model:
            self.set_model(model)

    def fit(self, events: list, T=None):
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

        T : `double`, default=None
            The duration (in physical time) of the realization. If it is None then
            T is considered to be the time of the last event (of any component).

        Returns
        -------
        output : `HawkesConditionalLaw`
            The current instance of the Learner
        """
        if not isinstance(events[0][0], np.ndarray):
            events = [events]

        for timestamps in events:
            self.incremental_fit(timestamps, compute=False, T=T)
        self.compute()

        return self

    def set_model(self, symmetries1d=list(), symmetries2d=list(),
                  delayed_component=None):
        """Set the model to be used.

        Parameters
        ----------
        symmetries1d : list of 2-tuple
            List of component index pairs for imposing symmetries on the mean
            intensity (e.g, ``[(0,1),(2,3)]`` means that the mean intensity of
            the components 0 and 1 must be the same and the mean intensity of
            the components 2 and 3 also.
            Can be set using can be set using the `set_model` method.

        symmetries2d : list of 2-tuple of 2-tuple
            List of kernel coordinates pairs to impose symmetries on the kernel
            matrix (e.g., ``[[(0,0),(1,1)],[(1,0),(0,1)]]`` for a bidiagonal
            kernel in dimension 2)
            Can be set using can be set using the `set_model` method.

        delayed_component : list of `int`, shape=(N, ), default=`None`
            list of node indices corresponding to node that should be delayed
            (to avoid simultaneous jumps of different components which can be a
            problem in the estimation)

        If no model is specified then default values for these fields are used

        Notes
        -----
        We set the symmetries, the kernel names and delayed components for
        first realization only
        """
        self.symmetries1d = symmetries1d
        self.symmetries2d = symmetries2d
        self.delayed_component = np.array(delayed_component)

    def _init_basics(self, realization):
        """Init the dimension
        """
        self.n_nodes = len(realization)

        return realization

    def _init_marked_components(self):
        """Init marked components

        This builds the field self.marked_components so that it is set to
           [component1_mark_intervals, ..., componentN_mark_intervals]
        where each componentj_mark_intervals is of the form
           [[min1, max1], [min2, max2], ..., [mink, maxk]]
        It describes the intervals the function f^ij are constants on.
        """
        marked_components = self.marked_components
        self.marked_components = []
        for i in range(0, self.n_nodes):
            self.marked_components.append([])
            if i in marked_components:
                self.marked_components[i].append(
                    [-sys.float_info.max, marked_components[i][0]])
                for j in range(0, len(marked_components[i]) - 1):
                    self.marked_components[i].append(
                        marked_components[i][j:j + 2])
                self.marked_components[i].append(
                    [marked_components[i][-1], sys.float_info.max])
            else:
                self.marked_components[i].append(
                    [-sys.float_info.max, sys.float_info.max])

    def _init_index(self):
        """Init for indexing

        Given i,j,l --> index and vice versa (i and j are components of the
        Hawkes and l is the marked interval index of the component j)
        """
        self._ijl2index = []
        self._index2ijl = []
        index = 0
        for i in range(0, self.n_nodes):
            self._ijl2index.append([])
            for j in range(0, self.n_nodes):
                self._ijl2index[i].append([])
                for l in range(0, len(self.marked_components[j])):
                    self._ijl2index[i][j].append(index)
                    self._index2ijl.append((i, j, l))
                    index += 1
        self._n_index = len(self._index2ijl)

    def _init_mark_stats(self):
        """We initialize the mark probabilities and min-max of the marks
        """
        # Proba for the mark
        self._mark_probabilities = []
        # In order to compute the probability we need to store the number of
        # events
        self._mark_probabilities_N = []
        self._mark_min = [sys.float_info.max] * self.n_nodes
        self._mark_max = [sys.float_info.min] * self.n_nodes
        for i in range(0, self.n_nodes):
            self._mark_probabilities_N.append(
                [0] * len(self.marked_components[i]))
            self._mark_probabilities.append(
                [0] * len(self.marked_components[i]))

    def _init_lambdas(self):
        """Init the lambda's
        """
        self.mean_intensity = [0] * self.n_nodes
        self._lam_N = [0] * self.n_nodes
        self._lam_T = [0] * self.n_nodes
        # Used to store the number of events of each component that
        # have been used to perform estimation on all the lags
        # versus the number of events that could not be used for all the lags
        # Warning : we don't take care of marks for this computation
        # normally we should do this computation independantly for each mark
        self._n_events = np.zeros((2, self.n_nodes))

    def _init_claws(self):
        """Init the claw storage
        """
        self._claw = [0] * len(self._index2ijl)

    def _index_to_lexical(self, index):
        """Convert index to lexical order (i,j,l)

        Parameters
        ----------
        index : `int`

        Returns
        -------
        i : `int`
            First node of the Hawkes

        j : `int`
            Second node of the Hawkes

        l : `int`
            Marked interval index of the component j

        Examples
        --------
        >>> from tick.hawkes import HawkesConditionalLaw
        >>> import numpy as np
        >>> learner = HawkesConditionalLaw()
        >>> learner.incremental_fit([np.array([2.1, 3, 4]),
        ...                          np.array([2., 2.01, 8])],
        ...                         compute=False)
        >>> learner._index_to_lexical(2)
        (1, 0, 0)
        """
        return self._index2ijl[index]

    def _lexical_to_index(self, i, j, l):
        """Convert lexical order (i,j,l) to index

        Parameters
        ----------
        i : `int`
            First node of the Hawkes

        j : `int`
            Second node of the Hawkes

        l : `int`
            Marked interval index of the component j

        Returns
        -------
        index : `int`

        Examples
        --------
        >>> from tick.hawkes import HawkesConditionalLaw
        >>> import numpy as np
        >>> learner = HawkesConditionalLaw()
        >>> learner.incremental_fit([np.array([2.1, 3, 4]),
        ...                          np.array([2., 2.01, 8])],
        ...                         compute=False)
        >>> learner._lexical_to_index(1, 0, 0)
        2
        """
        return self._ijl2index[i][j][l]

    def incremental_fit(self, realization, T=None, compute=True):
        """Allows to add some more realizations before estimation is performed.

        It updates the conditional laws (stored in `self._claw` and
        `self._claw1`) and of the mean intensity (in `self._mean_intensity`).

        Parameters
        ----------
        realization : list of `np.narrays` or list of 2-tuple of `np.arrays`

            * list of `np.narrays`, shape=(N) , representing the arrival times
              of each component
            * list of pairs (t,m) np.arrays representing the arrival times of
              each component (x) and the cumulative marks signal (m)

        T : `double`, default=None
            The duration (in physical time) of the realization. If it is -1 then
            T is considered to be the time of the last event (of any component).

        compute : `bool`, default=`False`
            Computes kernel estimation. If set to `False`, you will have to
            manually call `compute` method afterwards.
            This is useful to add multiple realizations and compute only once
            all conditional laws have been updated.
        """
        # If first realization we perform some init
        if self.n_realizations == 0:
            realization = self._init_basics(realization)
            self._init_marked_components()
            self._init_index()
            self._init_mark_stats()
            self._init_lambdas()
            self._init_claws()

        else:
            if compute and self._has_been_computed_once():
                warnings.warn(("compute() method was already called, "
                               "computed kernels will be updated."))

        # We perform some checks
        if self.n_nodes != len(realization):
            msg = 'Bad dimension for realization, should be %d instead of %d' \
                  % (self.n_nodes, len(realization))
            raise ValueError(msg)

        # Realization normalization
        if not isinstance(realization[0], (list, tuple)):
            realization = [(r, np.arange(len(r), dtype=np.double) + 1)
                           for r in realization]

        # Do we need to delay the realization ?
        if self.delayed_component:
            old_realization = realization
            realization = []
            for i in range(0, self.n_nodes):
                if any(self.delayed_component == i):
                    if len(old_realization[i][0]) == 0:
                        realization.append(old_realization[i])
                    else:
                        realization.append((old_realization[i][0] + self.delay,
                                            old_realization[i][1]))
                else:
                    realization.append(old_realization[i])

        # We compute last event time
        last_event_time = -1
        for i in range(0, self.n_nodes):
            if len(realization[i][0]) > 0:
                last_event_time = max(realization[i][0][-1], last_event_time)

        # If realization empty --> return
        if last_event_time < 0:
            warnings.warn(
                "An empty realization was passed. No computation was performed."
            )
            return

        # We set T if needed
        if T is None:
            T = last_event_time
        elif T < last_event_time:
            raise ValueError("Argument T (%g) specified is too small, "
                             "you should use default value or a value "
                             "greater or equal to %g." % (T, last_event_time))

        # We update the mark probabilities and min-max
        for i in range(0, self.n_nodes):
            if len(realization[i][0]) == 0:
                continue
            # We have to take into account the first mark
            der = np.hstack([realization[i][1][0], np.diff(realization[i][1])])
            total = 0
            self._mark_min[i] = min(self._mark_min[i], np.min(der))
            self._mark_max[i] = max(self._mark_max[i], np.max(der))

            for l, interval in enumerate(self.marked_components[i]):
                self._mark_probabilities_N[i][l] += \
                    np.sum((der >= interval[0]) & (der < interval[1]))
                total += self._mark_probabilities_N[i][l]

            for l, interval in enumerate(self.marked_components[i]):
                self._mark_probabilities[i][l] = \
                    self._mark_probabilities_N[i][l] / total
            der[:] = 1

        # We update the Lambda
        for i in range(0, self.n_nodes):
            if len(realization[i][0]) <= 0:
                continue
            self._lam_N[i] += len(realization[i][0])
            self._lam_T[i] += T
            self.mean_intensity[i] = self._lam_N[i] / self._lam_T[i]

        # We update the _n_events of component i
        # Warning : we don't take care of marks for this computation
        # normally we should do this computation independantly for each mark
        for i in range(0, self.n_nodes):
            good = np.sum(realization[i][0] <= T - self._lags[-1])
            bad = len(realization[i][0]) - good
            self._n_events[0, i] += good
            self._n_events[1, i] += bad

        # We might want to use threads, since this is the time consuming part
        with_multi_processing = self.n_threads > 1
        if with_multi_processing:
            pool = ThreadPool(with_lock=True, max_threads=self.n_threads)
            self._set('_lock', pool.lock)

        for index, (i, j, l) in enumerate(self._index2ijl):
            if with_multi_processing:
                pool.add_work(self._PointProcessCondLaw, realization, index, i,
                              j, l, T)
            else:
                self._PointProcessCondLaw(realization, index, i, j, l, T)

        if with_multi_processing:
            pool.start()

        # Here we compute the G^ij (not conditioned to l)
        # It is recomputed each time
        self._claw1 = []
        for i in range(0, self.n_nodes):
            self._claw1.append([])
            for j in range(0, self.n_nodes):
                index = self._ijl2index[i][j][0]
                self._claw1[i].append(np.copy(self._claw[index]))
                self._claw1[i][j] *= self._mark_probabilities[j][0]
                for l in range(1, len(self._ijl2index[i][j])):
                    index = self._ijl2index[i][j][l]
                    self._claw1[i][j] += self._claw[index] * \
                                         self._mark_probabilities[j][l]

        self.n_realizations += 1

        # Deal with symmetrization
        for (i, j) in self.symmetries1d:
            t = (self.mean_intensity[i] + self.mean_intensity[j]) / 2
            self.mean_intensity[i] = t
            self.mean_intensity[j] = t

            t = (self._mark_min[i] + self._mark_min[j]) / 2
            self._mark_min[i] = t
            self._mark_min[j] = t

            t = (self._mark_max[i] + self._mark_max[j]) / 2
            self._mark_max[i] = t
            self._mark_max[j] = t

            if self.marked_components[i] != self.marked_components[j]:
                continue
            for l in range(0, len(self.marked_components[i])):
                t = (self._mark_probabilities_N[i][l] +
                     self._mark_probabilities_N[j][l]) / 2
                self._mark_probabilities_N[i][l] = t
                self._mark_probabilities_N[j][l] = t

                t = (self._mark_probabilities[i][l] +
                     self._mark_probabilities[j][l]) / 2
                self._mark_probabilities[i][l] = t
                self._mark_probabilities[j][l] = t

        for ((i1, j1), (i2, j2)) in self.symmetries2d:
            t = (self._claw1[i1][j1] + self._claw1[i2][j2]) / 2
            self._claw1[i1][j1] = t
            self._claw1[i2][j2] = t
            if self.marked_components[j1] != self.marked_components[j2]:
                continue
            for l in range(0, len(self.marked_components[j1])):
                index1 = self._ijl2index[i1][j1][l]
                index2 = self._ijl2index[i2][j2][l]
                t = (self._claw[index1] + self._claw[index2]) / 2
                self._claw[index1] = t
                self._claw[index2] = t

        # We can remove the thread lock (lock disallows pickling)
        self._set('_lock', None)

        if compute:
            self.compute()

    def _PointProcessCondLaw(self, realization, index, i, j, l, T):

        claw_X = np.zeros(len(self._lags) - 1)
        claw_Y = np.zeros(len(self._lags) - 1)

        lambda_i = len(realization[i][0]) / T

        PointProcessCondLaw(
            realization[i][0], realization[j][0], realization[j][1],
            self._lags, self.marked_components[j][l][0],
            self.marked_components[j][l][1], T, lambda_i, claw_X, claw_Y)

        self._claw_X = claw_X

        # TODO: this lock acquire is very expensive here
        if self.n_threads > 1:
            self._lock.acquire()

        # Update claw
        if self.n_realizations == 0:
            self._claw[index] = claw_Y
        else:
            self._claw[index] *= self.n_realizations
            self._claw[index] += claw_Y
            self._claw[index] /= self.n_realizations + 1

        # Unlock
        if self.n_threads > 1:
            self._lock.release()

    def _compute_lags(self):
        """Computes the lags at which the claw will be computed
        """
        claw_method = self.claw_method

        # computes the claw either on a uniform grid (lin) or a semi log
        # uniform grid (log)
        if claw_method == "log":
            y1 = np.arange(0., self.min_lag, self.min_lag * self.delta_lag)
            y2 = np.exp(
                np.arange(
                    np.log(self.min_lag), np.log(self.max_lag),
                    self.delta_lag))
            self._lags = np.append(y1, y2)

        if claw_method == "lin":
            self._lags = np.arange(0., self.max_lag, self.delta_lag)

    def _compute_ints_claw(self):
        """Computes the claw and its integrals at the difference of
        quadrature points using a linear interpolation
        """

        self._int_claw = [0] * self._n_index
        # Builds a linear interpolation of the claws at the difference of
        # quadrature (only positive abscissa are kept)
        for index in range(self._n_index):
            xe = self._claw_X
            ye = self._claw[index]
            xs2 = np.array(
                [(a - b)
                 for (a, b) in itertools.product(self._quad_x, repeat=2)])
            xs2 = np.append(xe, xs2)
            xs2 = np.append(self._quad_x, xs2)
            xs2 = np.array(np.lib.arraysetops.unique(xs2))
            xs2 = np.array(np.core.fromnumeric.sort(xs2))
            xs2 = xs2[xs2 >= 0.]
            ys2 = np.zeros(len(xs2))
            j = 0
            for i in range(1, len(xe)):
                while j < len(xs2) and xs2[j] < xe[i]:
                    ys2[j] = (ye[i - 1]) + ((ye[i]) - (ye[i - 1])) * (
                        xs2[j] - xe[i - 1]) / (xe[i] - xe[i - 1])
                    j += 1
            sc = (xs2, ys2)
            self._int_claw[index] = sc

        # Computes the integrals of the claws (IG) and the integrals of x
        # times the claws from 0 to the abscissa we have just computed
        self._IG = []
        self._IG2 = []
        for i in range(self._n_index):
            xc = self._int_claw[i][0]
            yc = self._int_claw[i][1]

            iyc_IG = np.append(
                np.array(0.), np.cumsum(np.diff(xc) * (yc[:-1] + yc[1:]) / 2.))
            self._IG += [(xc, iyc_IG)]

            iyc_IG2 = np.append(
                np.array(0.),
                np.cumsum((yc[:-1] + yc[1:]) / 2. * np.diff(xc) * xc[:-1] +
                          np.diff(xc) * np.diff(xc) / 3. * np.diff(yc) +
                          np.diff(xc) * np.diff(xc) / 2. * yc[:-1]))
            self._IG2 += [(xc, iyc_IG2)]

    @staticmethod
    def _lin0(sig, t):
        """Find closest value of a signal, zero value border
        """
        x, y = sig
        if t >= x[-1]:
            return 0
        index = np.searchsorted(x, t)
        if index == len(y) - 1:
            return y[index]
        elif np.abs(x[index] - t) < np.abs(x[index + 1] - t):
            return y[index]
        else:
            return y[index + 1]

    @staticmethod
    def _linc(sig, t):
        """Find closest value of a signal, continuous border
        """
        x, y = sig
        if t >= x[-1]:
            return y[-1]
        index = np.searchsorted(x, t)
        if np.abs(x[index] - t) < np.abs(x[index + 1] - t):
            return y[index]
        else:
            return y[index + 1]

    def _G(self, i, j, l, t):
        """Returns the value of a claw at a point
        Used to fill V and M with 'gauss' method
        """
        if t < 0:
            warnings.warn("G(): should not be called for t < 0")
        index = self._ijl2index[i][j][l]
        return HawkesConditionalLaw._lin0(self._int_claw[index], t)

    def _DIG(self, i, j, l, t1, t2):
        """Returns the integral of a claw between t1 and t2
        """
        if t1 >= t2:
            warnings.warn("t2>t1 wrong in DIG")
        index = self._ijl2index[i][j][l]
        return HawkesConditionalLaw._linc(self._IG[index], t2) - \
               HawkesConditionalLaw._linc(self._IG[index], t1)

    def _DIG2(self, i, j, l, t1, t2):
        """Returns the integral of x times a claw between t1 and t2
        """
        if t1 >= t2:
            warnings.warn("t2>t1 wrong in DIG2")
        index = self._ijl2index[i][j][l]
        return HawkesConditionalLaw._linc(self._IG2[index], t2) - \
               HawkesConditionalLaw._linc(self._IG2[index], t1)

    def compute(self):
        """Computes kernel estimation by solving a Fredholm system.
        """

        # We raise an exception if a claw component had no input to be computed
        if any(self._n_events[0, :] == 0):
            k = np.where(self._n_events[0, :] == 0)[0]
            msg = "Cannot run estimation : not enough events for components {}" \
                .format(k)
            raise ValueError(msg)

        # Here we compute the quadrature points and the corresponding weights
        # self.quad_x and self.quad_w
        if self.quad_method in {'gauss', 'gauss-'}:
            self._quad_x, self._quad_w = leggauss(self.n_quad)
            self._quad_x = self.max_support * (self._quad_x + 1) / 2
            self._quad_w *= self.max_support / 2

        elif self.quad_method == 'log':
            logstep = (np.log(self.max_support) - np.log(
                self.min_support) + 1.) / \
                      self.n_quad
            x1 = np.arange(0., self.min_support, self.min_support * logstep)
            x2 = np.exp(
                np.arange(
                    np.log(self.min_support), np.log(self.max_support),
                    logstep))
            self._quad_x = np.append(x1, x2)
            self._quad_w = self._quad_x[1:] - self._quad_x[:-1]
            self._quad_w = np.append(self._quad_w, self._quad_w[-1])
            self.n_quad = len(self._quad_x)
            self._quad_x = np.array(self._quad_x)
            self._quad_w = np.array(self._quad_w)

        elif self.quad_method == 'lin':
            x1 = np.arange(0., self.max_support,
                           self.max_support / self.n_quad)
            self._quad_x = x1
            self._quad_w = self._quad_x[1:] - self._quad_x[:-1]
            self._quad_w = np.append(self._quad_w, self._quad_w[-1])
            self.n_quad = len(self._quad_x)
            self._quad_x = np.array(self._quad_x)
            self._quad_w = np.array(self._quad_w)

        # Computes the claw and its integrals at the difference of
        # quadrature points using a linear interpolation
        self._compute_ints_claw()

        # For each i we write and solve the system V =  M PHI
        index_first = 0
        self._phi_ijl = []
        self._norm_ijl = []
        self.kernels = []
        self.kernels_norms = np.zeros((self.n_nodes, self.n_nodes))

        for i in range(0, self.n_nodes):

            # We must compute the last valid index which corresponds to i
            index_last = index_first
            for index_last in range(index_first, self._n_index):
                (i1, j1, l1) = self._index2ijl[index_last]
                if i1 != i:
                    index_last -= 1
                    break

            # Number of indices corresponding to i
            n_index = index_last - index_first + 1

            # Compute V and M
            V = self._compute_V(i, n_index, self.n_quad, index_first,
                                index_last)
            M = self._compute_M(n_index, self.n_quad, index_first, index_last,
                                self.quad_method)
            # Then we solve it
            res = solve(M, V)

            self._estimate_kernels_and_norms(i, index_first, index_last, res,
                                             self.n_quad, self.quad_method)

            index_first = index_last + 1

        self._estimate_baseline()
        self._estimate_mark_functions()

    def _compute_V(self, i, n_index, n_quad, index_first, index_last):
        V = np.zeros((n_index * n_quad, 1))
        for index in range(index_first, index_last + 1):
            (x, j, l) = self._index2ijl[index]
            for n in range(0, n_quad):
                index_i_quad = (index - index_first) * n_quad + n
                V[index_i_quad] = self._G(i, j, l, self._quad_x[n])
        return V

    def _compute_M(self, n_index, n_quad, index_first, index_last, method):
        M = np.zeros((n_index * n_quad, n_index * n_quad))
        for index in range(index_first, index_last + 1):
            (x, j, l) = self._index2ijl[index]
            for index1 in range(index_first, index_last + 1):
                (i1, j1, l1) = self._index2ijl[index1]
                fact = self.mean_intensity[j1] / self.mean_intensity[j]
                for n in range(0, n_quad):
                    for n1 in range(0, n_quad):
                        if method == 'gauss' or method == 'gauss-':
                            self._fill_M_for_gauss(M, method, n_quad,
                                                   index_first, index, index1,
                                                   j, l, j1, l1, fact, n, n1)

                        elif method == 'log' or method == 'lin':
                            self._fill_M_for_log_lin(
                                M, method, n_quad, index_first, index, index1,
                                j, l, j1, l1, fact, n, n1)
        return M

    def _fill_M_for_gauss(self, M, method, n_quad, index_first, index, index1,
                          j, l, j1, l1, fact, n, n1):
        def x_value(n_lower, n_greater, j_lower, j_greater, l_greater):
            return self._mark_probabilities[j1][l1] * self._quad_w[n1] * \
                   self._G(j_lower, j_greater, l_greater,
                           self._quad_x[n_greater] - self._quad_x[n_lower])

        if n > n1:
            x = x_value(n1, n, j1, j, l)

        elif n < n1:
            x = fact * x_value(n, n1, j, j1, l1)

        else:
            if method == 'gauss-':
                x = 0
            else:
                x1 = x_value(n1, n, j1, j, l)
                x2 = fact * x_value(n, n1, j, j1, l1)
                x = (x1 + x2) / 2

        if method == 'gauss-':
            row = (index - index_first) * n_quad + n
            col = (index1 - index_first) * n_quad + n
            M[row, col] -= x

        if l == l1 and j == j1 and n == n1:
            x += 1

        row = (index - index_first) * n_quad + n
        col = (index1 - index_first) * n_quad + n1
        M[row, col] += x

    def _fill_M_for_log_lin(self, M, method, n_quad, index_first, index,
                            index1, j, l, j1, l1, fact, n, n1):

        mark_probability = self._mark_probabilities[j1][l1]

        ratio_dig = lambda n_q: ((self._quad_x[n] - self._quad_x[n_q]) / self._quad_w[n_q])

        ratio_dig2 = lambda n_q: 1. / self._quad_w[n_q]

        dig_arg_greater = lambda n_q: (j1, j, l, self._quad_x[n] - self._quad_x[n_q] - self._quad_w[n_q], self._quad_x[n] - self._quad_x[n_q])

        dig_arg_lower = lambda n_q: (j, j1, l1, self._quad_x[n_q] - self._quad_x[n], self._quad_x[n_q] - self._quad_x[n] + self._quad_w[n_q])

        x = 0
        if n > n1:
            x += mark_probability * self._DIG(*dig_arg_greater(n1))

            if n1 < n_quad - 1:
                x -= ratio_dig(n1) * mark_probability * \
                     self._DIG(*dig_arg_greater(n1))

                x += ratio_dig2(n1) * mark_probability * \
                     self._DIG2(*dig_arg_greater(n1))

            if n1 > 0:
                x += ratio_dig(n1 - 1) * mark_probability * \
                     self._DIG(*dig_arg_greater(n1 - 1))

                x -= ratio_dig2(n1 - 1) * mark_probability * \
                     self._DIG2(*dig_arg_greater(n1 - 1))

        elif n < n1:

            x += fact * mark_probability * self._DIG(*dig_arg_lower(n1))

            if n1 < n_quad - 1:
                x -= fact * ratio_dig(n1) * mark_probability * \
                     self._DIG(*dig_arg_lower(n1))

                x -= fact * ratio_dig2(n1) * mark_probability * \
                     self._DIG2(*dig_arg_lower(n1))

            if n1 > 0:
                x += fact * ratio_dig(n1 - 1) * mark_probability * \
                     self._DIG(*dig_arg_lower(n1 - 1))

                x += fact * ratio_dig2(n1 - 1) * mark_probability * \
                     self._DIG2(*dig_arg_lower(n1 - 1))

        elif n == n1:
            x += fact * self._mark_probabilities[j1][l1] * \
                 self._DIG(*dig_arg_lower(n1))

            if n1 < n_quad - 1:
                x -= fact * ratio_dig(n1) * mark_probability * \
                     self._DIG(*dig_arg_lower(n1))

                x -= fact * ratio_dig2(n1) * mark_probability * \
                     self._DIG2(*dig_arg_lower(n1))

            if n1 > 0:
                x += ratio_dig(n1 - 1) * mark_probability * \
                     self._DIG(*dig_arg_greater(n1 - 1))

                x -= ratio_dig2(n1 - 1) * mark_probability * \
                     self._DIG2(*dig_arg_greater(n1 - 1))

        if l == l1 and j == j1 and n == n1:
            x += 1

        row = (index - index_first) * n_quad + n
        col = (index1 - index_first) * n_quad + n1
        M[row, col] += x

    def _estimate_kernels_and_norms(self, i, index_first, index_last, res,
                                    n_quad, method):
        # We rearrange the solution vector and compute the norms
        # Here we get phi^ij_l and the corresponding norms
        for index in range(index_first, index_last + 1):
            y = res[(index - index_first) * n_quad:(index - index_first + 1) *
                    n_quad][:, 0]

            self._phi_ijl.append((self._quad_x, y))

            if method in {'gauss', 'gauss-'}:
                self._norm_ijl.append(np.sum(y * self._quad_w))

            elif method in {'log', 'lin'}:
                # interpolation (the one we used in the scheme) norm
                self._norm_ijl.append(
                    np.sum((y[:-1] + y[1:]) / 2. * self._quad_w[:-1]))

        # Now we compute phi^ij and the corresponding norms
        self.kernels.append([])
        for j in range(0, self.n_nodes):
            index = self._ijl2index[i][j][0]
            self.kernels[i].append(
                np.array(self._phi_ijl[index]) *
                self._mark_probabilities[j][0])
            self.kernels_norms[i, j] = self._norm_ijl[index] * \
                                       self._mark_probabilities[j][0]
            index += 1

            for l in range(1, len(self.marked_components[j])):
                self.kernels[i][j] += self._phi_ijl[index] * \
                                      self._mark_probabilities[j][l]
                self.kernels_norms[i, j] += self._norm_ijl[index] * \
                                            self._mark_probabilities[j][l]
                index += 1

    def _estimate_baseline(self):
        M = np.eye(self.n_nodes) - self.kernels_norms
        self.baseline = np.dot(M, self.mean_intensity)

    def _estimate_mark_functions(self):
        self.mark_functions = []
        for i in range(0, self.n_nodes):
            self.mark_functions.append([])
            for j in range(0, self.n_nodes):
                if len(self.marked_components[j]) == 1:
                    self.mark_functions[i].append((np.array([1]),
                                                   np.array([1])))
                    continue
                y = np.zeros(0)
                x = np.zeros(0)
                n = 100
                for l in range(0, len(self.marked_components[j])):
                    index = self._ijl2index[i][j][l]
                    y = np.append(
                        y,
                        np.zeros(n) +
                        self._norm_ijl[index] / self.kernels_norms[i, j])
                    xmin = self.marked_components[j][l][0]
                    xmax = self.marked_components[j][l][1]
                    if l == 0:
                        xmin = self._mark_min[j]
                    if l == len(self.marked_components[j]) - 1:
                        xmax = self._mark_max[j]
                    x = np.append(
                        x,
                        np.arange(n) * (xmax - xmin) / (n - 1) + xmin)
                self.mark_functions[i].append((x, y))

    def get_kernel_supports(self):
        """Computes kernel support. This makes our learner compliant with
        `tick.plot.plot_hawkes_kernels` API

        Returns
        -------
        output : `np.ndarray`, shape=(n_nodes, n_nodes)
            2d array in which each entry i, j corresponds to the support of
            kernel i, j
        """
        supports = np.empty((self.n_nodes, self.n_nodes))
        for i, j in itertools.product(range(self.n_nodes), repeat=2):
            supports[i, j] = np.max(self.kernels[0][0][0])
        return supports

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
        t_values = self.kernels[i][j][0]
        y_values = self.kernels[i][j][1]

        if self.quad_method == 'log':
            with warnings.catch_warnings(record=True):
                log_t_values = np.log10(t_values)
                log_y_values = np.log10(y_values)
            log_abscissa_array = np.log10(abscissa_array)
            min_value = np.nanmin(log_abscissa_array)
            log_interpolation = np.interp(log_abscissa_array, log_t_values,
                                          log_y_values, left=min_value,
                                          right=min_value)
            kernel_values = np.power(10.0, log_interpolation)
        else:
            kernel_values = np.interp(abscissa_array, t_values, y_values,
                                      left=0, right=0)

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
        # we need to convert it to a numpy array
        return np.array(self.kernels_norms)

    def _has_been_computed_once(self):
        return self.mark_functions is not None
