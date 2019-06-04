# License: BSD 3 clause

from operator import itemgetter
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from tick.base.simulation import Simu
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
from tick.preprocessing import LongitudinalFeaturesLagger
from itertools import permutations
from copy import deepcopy
from scipy.stats import beta, norm


class SimuSCCS(Simu):
    """Simulation of a Self Control Case Series (SCCS) model. This simulator can
    produce exposure (features), outcomes (labels) and censoring data.
    The features matrices  are a `n_cases` list of numpy arrays (dense case) or
    csr_matrices (sparse case) of shape `(n_intervals, n_features)` containing
    exposures to each feature.
    Exposure can take two forms:
    - short repeated exposures (`single_exposure`): in that case, each column of the
    numpy arrays or csr matrices can contain multiple ones, each one representing an
    exposure for a particular time bucket.
    - infinite unique exposures (`multiple_exposure`): in that case, each column of the
    numpy arrays or csr matrices can only contain a single one, corresponding to the
    starting date of the exposure.

    Parameters
    ----------
    n_cases : `int`
        Number of cases to generate. A case is a sample who experience at
        least one adverse event.

    n_intervals : `int`
        Number of time intervals used to generate features and outcomes.

    n_features : `int`
        Number of features to simulate for each case.

    n_lags : `numpy.ndarray`, shape=(n_features,), dtype="uint64"
       Number of lags per feature. The model will regress labels on the
       last observed values of the features over their corresponding
       `n_lags` time intervals. `n_lags` values must be between 0 and
       `n_intervals` - 1.

    exposure_type : {'single_exposure', 'multiple_exposure'}, default='single_exposure'
       Either 'single_exposure' for infinite unique exposures or 'multiple_exposure' for
       short repeated exposures.

    distribution : {'multinomial', 'poisson'}, default='multinomial'
       Distribution used to generate the outcomes. In the 'multinomial'
       case, the Poisson process used to generate the events is conditioned
       by total the number event per sample, which is set to be equal to
       one. In that case, the simulation matches exactly the SCCS model
       hypotheses. In the 'poisson' case, the outcomes are generated from a
       Poisson process, which can result in more than one outcome tick per
       sample. In this case, the first event is kept, and the other are
       discarded.

    sparse : `boolean`, default=True
        Generate sparse or dense features.

    censoring_prob : `float`, default=0.
       Probability that a sample is censored. Should be in [0, 1]. If 0, no
       censoring is applied. When > 0, SimuSCCS simulates a censoring vector.
       In that case, the features and outcomes are simulated, then right-censored
       according to the simulated censoring dates.

    censoring_scale : `float`, default=None
       The number of censored time intervals are drawn from a Poisson
       distribution with intensity equal to `censoring_scale`. The higher,
       the more intervals will be censored. If None, no censoring is
       applied.

    coeffs : `list` containing `numpy.ndarray`, default=None
       Can be used to provide your own set of coefficients. Element `i` of
       the list should be a 1-d `numpy.ndarray` of shape (n_lags + 1), where
       `n_lags[i]` is the number of lags associated to feature `i`.
       If set to None, the simulator will generate coefficients randomly.

    hawkes_exp_kernels : `SimuHawkesExpKernels`, default=None
        Features are simulated with exponential kernel Hawkes processes.
        This parameter can be used to specify your own kernels (see
        `SimuHawkesExpKernels` documentation). If None, random kernels
        are generated. The same kernels are used to generate features for
        the whole generated population.

    n_correlations : `int`, default=0
        If `hawkes_exp_kernels` is None, random kernels are generated. This
        parameter controls the number of non-null non-diagonal kernels.

    batch_size : `int`, default=None
       When generating outcomes with Poisson distribution, the simulator will
       discard samples to which no event has occurred. In this case, the
       simulator generate successive batches of samples, until it reaches
       a total of n_samples. This parameter can be used to set the batch size.

    seed : `int`, default=None
        The seed of the random number generator

    verbose : `bool`, default=True
        If True, print things

    Examples
    --------
    >>> import numpy as np
    >>> from tick.survival import SimuSCCS
    >>> n_lags = np.repeat(2, 2).astype('uint64')
    >>> sim = SimuSCCS(n_cases=5, n_intervals=3, n_features=2, n_lags=n_lags,
    ... seed=42, sparse=False, exposure_type="multiple_exposures",
    ... verbose=False)
    >>> features, labels, outcomes, censoring, _coeffs = sim.simulate()
    >>> print(features)
    [array([[0., 0.],
           [1., 0.],
           [1., 1.]]), array([[1., 0.],
           [1., 0.],
           [1., 1.]]), array([[1., 1.],
           [1., 1.],
           [1., 1.]]), array([[0., 0.],
           [1., 1.],
           [1., 0.]]), array([[1., 0.],
           [0., 0.],
           [0., 0.]])]
    >>> print(censoring)
    [3 3 3 3 3]
    >>> print(_coeffs)
    [array([ 0.54738557, -0.15109073,  0.71345739]), array([ 1.67633284, -0.25656871, -0.25655065])]
    """

    _const_attr = [
        # user defined parameters
        '_exposure_type',
        '_outcome_distribution',
        '_censoring_prob',
        '_censoring_scale',  # redundant with prob ?
        '_batch_size',
        '_distribution',
        '_n_lags',
        # user defined or computed attributes
        '_hawkes_exp_kernel',
        '_coeffs',
        '_time_drift',
        '_features_offset'
    ]

    _attrinfos = {key: {'writable': False} for key in _const_attr}
    _attrinfos['hawkes_obj'] = {'writable': True}

    def __init__(
            self,
            n_cases,
            n_intervals,
            n_features,
            n_lags,
            time_drift=None,
            exposure_type="single_exposure",
            distribution="multinomial",
            sparse=True,
            censoring_prob=0,
            censoring_scale=None,
            coeffs=None,
            hawkes_exp_kernels=None,
            n_correlations=0,
            batch_size=None,
            seed=None,
            verbose=True,
    ):
        super(SimuSCCS, self).__init__(seed, verbose)
        self.n_cases = n_cases
        self.n_intervals = n_intervals
        self.n_features = n_features
        self._features_offset = None
        self._n_lags = None
        self.n_lags = n_lags
        self.sparse = sparse

        self.hawkes_obj = None

        # attributes with restricted value range
        self._exposure_type = None
        self.exposure_type = exposure_type

        self._distribution = None
        self.distribution = distribution

        self._censoring_prob = 0
        self.censoring_prob = censoring_prob

        self._censoring_scale = None
        self.censoring_scale = censoring_scale if censoring_scale \
            else n_intervals / 4

        self._coeffs = None
        self.coeffs = coeffs

        self._batch_size = None
        self.batch_size = batch_size

        # TODO later: add properties for these parameters
        self.n_correlations = n_correlations
        self.hawkes_exp_kernels = hawkes_exp_kernels
        self.time_drift = time_drift  # function(t), used only for the paper, allow to add a baseline
        # TODO: make a property from this baseline

    def simulate(self):
        """ Launch simulation of the data.

        Returns
        -------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : `list` of `numpy.ndarray`,
            list of length n_cases, each element of the list of
            shape=(n_intervals,)
            The labels vector

        censoring : `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.

        _coeffs : `numpy.ndarray`, shape=(n_features * (n_lags + 1),)
            The coefficients used to simulate the data.
        """
        return Simu.simulate(self)

    def _simulate(self):
        """ Loop to generate batches of samples until n_cases is reached.
        """
        n_lagged_features = int(self.n_lags.sum() + self.n_features)
        n_cases = self.n_cases
        if self._coeffs is None:
            self._set('_coeffs', np.random.normal(1e-3, 1.1,
                                                  n_lagged_features))

        features = []
        censored_features = []
        outcomes = []
        censoring = np.zeros((n_cases,), dtype="uint64")
        cases_count = 0
        while cases_count < n_cases:
            _features, _censored_features, _outcomes, _censoring, _n_samples = \
                self._simulate_batch()

            n_new_cases = _n_samples
            c = cases_count
            cases_count += n_new_cases
            n = n_cases - c if cases_count >= n_cases else n_new_cases

            features.extend(_features[0:n])
            censored_features.extend(_censored_features[0:n])
            outcomes.extend(_outcomes[0:n])
            censoring[c:c + n] = _censoring[0:n]

        return features, censored_features, outcomes, censoring, self.coeffs

    def _simulate_batch(self):
        """Simulate a batch of samples, each of which have ticked at least once.
        """
        _features, _n_samples = self.simulate_features(self.batch_size)
        _censored_features = deepcopy(_features)
        _outcomes = self.simulate_outcomes(_features)
        _censoring = np.full((_n_samples,), self.n_intervals, dtype="uint64")
        if self.censoring_prob:
            censored_idx = np.random.binomial(1, self.censoring_prob,
                                              size=_n_samples).astype("bool")
            _censoring[censored_idx] -= np.random.poisson(
                lam=self.censoring_scale,
                size=(censored_idx.sum(),)).astype("uint64")
            _censored_features = self._censor_array_list(
                _censored_features, _censoring)
            _outcomes = self._censor_array_list(_outcomes, _censoring)

            _features, _censored_features, _outcomes, censoring, _ = \
                self._filter_non_positive_samples(_features, _censored_features,
                                                  _outcomes, _censoring)

        return _features, _censored_features, _outcomes, _censoring, _n_samples

    def simulate_features(self, n_samples):
        """Simulates features, either `single_exposure` or
        `multiple_exposures` exposures.
        """
        if self.exposure_type == "single_exposure":
            features, n_samples = self._sim_single_exposures()
        elif self.exposure_type == "multiple_exposures":
            sim = self._sim_multiple_exposures_exposures
            features = [sim() for _ in range(n_samples)]
        return features, n_samples

    # We just keep it for the tests now
    # TODO later: need to be improved with Hawkes
    def _sim_multiple_exposures_exposures(self):
        features = np.zeros((self.n_intervals, self.n_features))
        while features.sum() == 0:
            # Make sure we do not generate empty feature matrix
            features = np.random.randint(
                2,
                size=(self.n_intervals, self.n_features),
            ).astype("float64")
        if self.sparse:
            features = csr_matrix(features, dtype="float64")
        return features

    def _sim_single_exposures(self):
        if not self.sparse:
            raise ValueError(
                "'single_exposure' exposures can only be simulated"
                " as sparse feature matrices")

        if self.hawkes_exp_kernels is None:
            np.random.seed(self.seed)
            decays = .002 * np.ones((self.n_features, self.n_features))
            baseline = 4 * np.random.random(self.n_features) / self.n_intervals
            mult = np.random.random(self.n_features)
            adjacency = mult * np.eye(self.n_features)

            if self.n_correlations:
                comb = list(permutations(range(self.n_features), 2))
                if len(comb) > 1:
                    idx = itemgetter(*np.random.choice(
                        range(len(comb)), size=self.n_correlations,
                        replace=False))
                    comb = idx(comb)

                for i, j in comb:
                    adjacency[i, j] = np.random.random(1)

            self._set(
                'hawkes_exp_kernels',
                SimuHawkesExpKernels(adjacency=adjacency, decays=decays,
                                     baseline=baseline, verbose=False,
                                     seed=self.seed))

        self.hawkes_exp_kernels.adjust_spectral_radius(
            .1)  # TODO later: allow to change this parameter
        hawkes = SimuHawkesMulti(self.hawkes_exp_kernels,
                                 n_simulations=self.n_cases)

        run_time = self.n_intervals
        hawkes.end_time = [1 * run_time for _ in range(self.n_cases)]
        dt = 1
        self.hawkes_exp_kernels.track_intensity(dt)
        hawkes.simulate()

        self.hawkes_obj = hawkes
        features = [[
            np.min(np.floor(f)) if len(f) > 0 else -1 for f in patient_events
        ] for patient_events in hawkes.timestamps]

        features = [
            self.to_coo(feat, (run_time, self.n_features)) for feat in features
        ]

        # Make sure patients have at least one exposure?
        exposures_filter = itemgetter(
            *[i for i, f in enumerate(features) if f.sum() > 0])
        features = exposures_filter(features)
        n_samples = len(features)

        return features, n_samples

    def simulate_outcomes(self, features):
        features, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags). \
            fit_transform(features)

        if self.distribution == "poisson":
            # TODO later: add self.max_n_events to allow for multiple outcomes
            # In this case, the multinomial simulator should use this arg too
            outcomes = self._simulate_poisson_outcomes(features, self._coeffs)
        else:
            outcomes = self._simulate_multinomial_outcomes(
                features, self._coeffs)
        return outcomes

    def _simulate_multinomial_outcomes(self, features, coeffs):
        baseline = np.zeros(self.n_intervals)
        if self.time_drift is not None:
            baseline = self.time_drift(np.arange(self.n_intervals))
        dot_products = [baseline + feat.dot(coeffs) for feat in features]

        def sim(dot_prod):
            dot_prod -= dot_prod.max()
            probabilities = np.exp(dot_prod) / np.sum(np.exp(dot_prod))
            outcomes = np.random.multinomial(1, probabilities)
            return outcomes.astype("int32")

        return [sim(dot_product) for dot_product in dot_products]

    def _simulate_poisson_outcomes(self, features, coeffs,
                                   first_tick_only=True):
        baseline = np.zeros(self.n_intervals)
        if self.time_drift is not None:
            baseline = self.time_drift(np.arange(self.n_intervals))
        dot_products = [baseline + feat.dot(coeffs) for feat in features]

        def sim(dot_prod):
            dot_prod -= dot_prod.max()
            intensities = np.exp(dot_prod)
            ticks = np.random.poisson(lam=intensities)
            if first_tick_only:
                first_tick_idx = np.argmax(ticks > 0)
                y = np.zeros_like(intensities)
                if ticks.sum() > 0:
                    y[first_tick_idx] = 1
            else:
                y = ticks
            return y.astype("int32")

        return [sim(dot_product) for dot_product in dot_products]

    @staticmethod
    def _censor_array_list(array_list, censoring):
        """Apply censoring to a list of array-like objects. Works for 1-D or 2-D
        arrays, as long as the first axis represents the time.

        Parameters
        ----------
        array_list : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features) or shape=(n_intervals,)
            The list of features matrices.

        censoring : `numpy.ndarray`, shape=(n_cases, 1), dtype="uint64"
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.

        Returns
        -------
        output : `[numpy.ndarrays]`  or `[csr_matrices]`, shape=(n_intervals, n_features)
            The list of censored features matrices.
        """

        def censor(array, censoring_idx):
            if sps.issparse(array):
                array = array.tolil()
                array[int(censoring_idx):] = 0
                array = array.tocsr()
            else:
                array[int(censoring_idx):] = 0
            return array

        return [censor(l, censoring[i]) for i, l in enumerate(array_list)]

    @staticmethod
    def _filter_non_positive_samples(features, features_censored, labels,
                                     censoring):
        """Filter out samples which don't tick in the observation window.

        Parameters
        ----------
        features : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : list of numpy.ndarray of length n_cases,
            shape=(n_intervals,)
            The list of labels matrices.
        """
        nnz = [np.nonzero(arr)[0] for arr in labels]
        positive_sample_idx = [i for i, arr in enumerate(nnz) if len(arr) > 0]
        if len(positive_sample_idx) == 0:
            raise ValueError("There should be at least one positive sample per\
             batch. Try to increase batch_size.")
        pos_samples_filter = itemgetter(*positive_sample_idx)
        return list(pos_samples_filter(features)),\
            list(pos_samples_filter(features_censored)),\
            list(pos_samples_filter(labels)),\
            censoring[positive_sample_idx],\
            np.array(positive_sample_idx, dtype="uint64")

    @staticmethod
    def to_coo(feat, shape):
        feat = np.array(feat)
        cols = np.where(feat >= 0)[0]
        rows = np.array(feat[feat >= 0])
        if len(cols) == 0:
            cols = np.random.randint(0, shape[1], 1)
            rows = np.random.randint(0, shape[0], 1)
        data = np.ones_like(cols)
        return csr_matrix((data, (rows, cols)), shape=shape, dtype="float64")

    @property
    def exposure_type(self):
        return self._exposure_type

    @exposure_type.setter
    def exposure_type(self, value):
        if value not in ["single_exposure", "multiple_exposures"]:
            raise ValueError("exposure_type can be only 'single_exposure' or "
                             "'multiple_exposures'.")
        self._set("_exposure_type", value)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if value not in ["multinomial", "poisson"]:
            raise ValueError("distribution can be only 'multinomial' or "
                             "'poisson'.")
        self._set("_distribution", value)

    @property
    def censoring_prob(self):
        return self._censoring_prob

    @censoring_prob.setter
    def censoring_prob(self, value):
        if value < 0 or value > 1:
            raise ValueError("censoring_prob value should be in [0, 1].")
        self._set("_censoring_prob", value)

    @property
    def censoring_scale(self):
        return self._censoring_scale

    @censoring_scale.setter
    def censoring_scale(self, value):
        if value < 0:
            raise ValueError("censoring_scale should be greater than 0.")
        self._set("_censoring_scale", value)

    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, value):
        offsets = [0]
        for l in value:
            if l < 0:
                raise ValueError('n_lags elements should be greater than or '
                                 'equal to 0.')
            offsets.append(offsets[-1] + l + 1)
        self._set('_n_lags', value)
        self._set('_features_offset', offsets)

    @property
    def coeffs(self):
        value = list()
        for i, l in enumerate(self.n_lags):
            start = int(self._features_offset[i])
            end = int(start + l + 1)
            value.append(self._coeffs[start:end])
        return value

    @coeffs.setter
    def coeffs(self, value):
        if value is not None:
            for i, c in enumerate(value):
                if c.shape[0] != int(self.n_lags[i] + 1):
                    raise ValueError("Coeffs %i th element should be of shape\
                     (n_lags[%i] + 1),)" % (i, self.n_lags[i]))
            value = np.hstack(value)
        self._set("_coeffs", value)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None and self.distribution == "multinomial":
            self._set("_batch_size", self.n_cases)
        elif value is None:
            self._set("_batch_size", int(min(2000, self.n_cases)))
        else:
            self._set("_batch_size", int(value))
        self._set("_batch_size", max(100, self.batch_size))


class CustomEffects:
    def __init__(self, n_intervals):
        """Class provinding flexible relative incidence curves to be used as
        coefficients in the `SimuSCCS` class.

        Parameters
        ----------
        n_intervals : `int`
            Number of time intervals used to generate features and outcomes.
        """
        self.n_intervals = n_intervals
        self._curves_type_dict = {
            1: (5, 1),
            2: (2, 2),
            3: (.5, .5),
            4: (2, 5),
            5: (1, 3)
        }

    def constant_effect(self, amplitude, cut=0):
        """Returns coefficients corresponding to a constant relative incidence
        of value equal to `amplitude`. If `cut` is greater than 0, the relative
        incidence will be null on [`cut`, `n_intervals`]
        """
        risk_curve = np.ones(self.n_intervals) * amplitude
        if cut > 0:
            risk_curve[cut:] = 1
        return risk_curve

    def bell_shaped_effect(self, amplitude, width, lag=0, cut=0):
        """Returns coefficients corresponding to a bell shaped relative
        incidence of max value equal to `amplitude`. If `cut` is greater than 0,
        the relative incidence will be null on [`cut`, `n_intervals`]. The
        effect starts at `lag` interval, and lasts `width` intervals.
        """
        self._check_params(lag, width, amplitude, cut)
        if width % 2 == 0:
            width += 1
        effect = norm(0, width / 5).pdf(np.arange(width) - int(width / 2))
        return self._create_risk_curve(effect, amplitude, cut, width, lag)

    def increasing_effect(self, amplitude, lag=0, cut=0, curvature_type=1):
        """Returns coefficients corresponding to an increasing relative
        incidence of max value equal to `amplitude`. If `cut` is greater than 0,
        the relative incidence will be null on [`cut`, `n_intervals`]. The
        effect starts at `lag` interval, and lasts `width` intervals.
        The parameter `curvature_type` controls the shape of the relative
        incidence curve, it can take values in {1, 2, 3, 4, 5}.
        """
        width = self.n_intervals
        self._check_params(lag, width, amplitude, cut)
        if curvature_type not in np.arange(5) + 1:
            raise ValueError('curvature type should be in {1, 2, 3, 4, 5}')
        a, b = self._curves_type_dict[curvature_type]
        effect = beta(a, b).cdf(np.arange(width) / width)
        return self._create_risk_curve(effect, amplitude, cut, width, lag)

    def _check_params(self, lag, width, amplitude, cut):
        if cut is not None and cut >= width:
            raise ValueError('cut should be < width')
        if lag > self.n_intervals:
            raise ValueError('n_intervals should be > lag')
        if amplitude <= 0:
            raise ValueError('amplitude should be > 0')

    def _create_risk_curve(self, effect, amplitude, cut, width, lag):
        if cut:
            effect = effect[:int(width - cut)]
        end_effect = int(lag + width - cut)
        if end_effect > self.n_intervals:
            end_effect = self.n_intervals
        effect = effect[:end_effect - lag]

        M = effect.max()
        m = effect.min()
        effect = (effect - m) / (M - m)
        effect *= (amplitude - 1)
        risk_curve = np.ones(self.n_intervals)
        risk_curve[lag:end_effect] += effect
        return risk_curve

    @staticmethod
    def negative_effect(positive_effect):
        return np.exp(-np.log(positive_effect))
