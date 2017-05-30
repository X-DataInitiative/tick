from operator import itemgetter
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from .base.simu import Simu
from tick.preprocessing.longitudinal_features_lagger\
    import LongitudinalFeaturesLagger
from warnings import warn


class SimuSCCS(Simu):
    """Simulation of a Self Control Case Series (SCCS) model. This simulator can
    produce exposure (features), outcomes (labels) and censoring data. 
     
    The features matrices  are a `n_sample` list of numpy arrays (dense case) or 
    csr_matrices (sparse case) of shape `(n_intervals, n_features)` containing
    exposures to each feature.
    Exposure can take two forms:
    - short repeated exposures: in that case, each column of the numpy arrays 
    or csr matrices can contain multiple ones, each one representing an exposure
    for a particular time bucket.
    - infinite unique exposures: in that case, each column of the numpy arrays
    or csr matrices can only contain a single one, corresponding to the starting
    date of the exposure.

    Parameters
    ----------
    n_samples : `int`,
        Number of samples to generate.
    
    n_intervals : `int`
        Number of time intervals used to generate features and outcomes.
         
    n_lags : `int`
        Number of lags to be used when simulating the outcomes. The output
        features are returned without lagging to allow the testing of lag
        misspecification in the model.

    sparse : `boolean`, default=True
        Generate sparse or dense features.
     
    exposure_type : {'infinite', 'short'}, default='infinite'
        Either 'infinite' for infinite unique exposures or 'short' for short
        repeated exposures.
        
    distribution : {'multinomial', 'poisson'}, default='multinomial'
        Dristribution used to generate the outcomes. In the 'multinomial' case,
        the Poisson process used to generate the events is conditionned by total 
        the number event per sample, which is set to be equal to one. In that
        case, the simulation matches exactly the SCCS model hypotheses. In the
        'poisson' case, the outcomes are generated from a Poisson process, which
        can result in more than one outcome tick per sample. See 
        `first_tick_only` option to filter out additional events.
        
    first_tick_only : `Boolean`, default=True
        Keep only the first event and drop the consecutive ones when simulating
        with Poisson distribution. 
        
    censoring : `Boolean`, default=True
        Simulate a censoring vector. In that case, the features and outcomes are
        simulated, then right-censored according to the simulated censoring
        dates.
        
    censoring_prob : `float`, default=.7
        Probability that a sample is censored. Should be in [0, 1].
        
    censoring_intensity : `float`, default=.9
        The number of censored time intervals are drawn from a Poisson 
        distribution with intensity equal to `censoring_intensity`. The higher,
        the more intervals will be censored. Should be greater than 0.
        
    coeffs : `numpy.ndarray` of shape (n_features * (n_lags + 1)), default=None
        Can be used to provide your own set of coefficients. If set to None, the
        simulator will generate coefficients randomly.
        
    seed : `int`
        The seed of the random number generator
        
    verbose : `bool`
        If True, print things
        
    batch_size : `int`, default=None
        When generating outcomes with Poisson distribution, the simulator will
        discard samples to which no event has occured. In this case, the
        simulator generate successive batches of samples, until it reaches 
        a total of n_samples. This parameter can be used to set the batch size.
        
    Examples
    --------
    >>> import numpy as np
    >>> from tick.simulation import SimuSCCS
    >>> sim = SimuSCCS(n_samples=2, n_intervals=3, n_features=2, n_lags=2,
    ... seed=42, sparse=False, exposure_type="short")
    >>> features, labels, censoring, coeffs = sim.simulate()
    --------------------------------------
    Launching simulation using SimuSCCS...
    Done simulating using SimuSCCS in 5.54e-03 seconds.
    >>> print(features)
    [array([[ 0.,  0.],
            [ 1.,  0.],
            [ 1.,  1.]]),
     array([[ 1.,  1.],
            [ 1.,  1.],
            [ 1.,  1.]])
    ]
    >>> print(labels)
    [array([0, 0, 1], dtype=uint64), array([0, 0, 1], dtype=uint64)]
    >>> print(censoring)
    [3 3]
    >>> print(coeffs)
    [ 0.54738557 -0.15109073  0.71345739  1.67633284 -0.25656871 -0.25655065]
     """

    _attrinfos = {
        "_exposure_type": {
            "writable": False
        },
        "_distribution": {
            "writable": False
        },
        "_censoring_prob": {
            "writable": False
        },
        "_censoring_intensity": {
            "writable": False
        },
        "_coeffs": {
            "writable": False
        },
        "_batch_size": {
            "writable": False
        }
    }

    def __init__(self, n_samples, n_intervals, n_features, n_lags, coeffs=None,
                 sparse=True, exposure_type="infinite",
                 distribution="multinomial", first_tick_only=True,
                 censoring=True, censoring_prob=.7, censoring_intensity=.9,
                 seed=None, verbose=True, batch_size=None):
        super(SimuSCCS, self).__init__(seed, verbose)
        self.n_samples = n_samples
        self.n_intervals = n_intervals
        self.n_features = n_features
        self.n_lags = n_lags
        self.sparse = sparse
        self.first_tick_only = first_tick_only
        self.censoring = censoring
        self.exposure_type = exposure_type
        self.distribution = distribution
        self.censoring_prob = censoring_prob
        self.censoring_intensity = censoring_intensity
        self.coeffs = coeffs
        self.batch_size = batch_size

    def simulate(self):
        """
        Launch simulation of the data.

        Returns
        -------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_samples, each element of the list of 
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : `list` of `numpy.ndarray`,
            list of length n_samples, each element of the list of 
            shape=(n_intervals,)
            The labels vector
            
        censoring : `numpy.ndarray`, shape=(n_samples,), dtype="uint64"
            The censoring data. This array should contain integers in 
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then 
            the observation of sample i is stopped at interval c, that is, the 
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.
        
        coeffs : `numpy.ndarray`, shape=(n_features * (n_lags + 1),)
            The coefficients used to simulate the data.
        """
        return Simu.simulate(self)

    def _simulate(self):
        """ Loop to generate batches of samples until n_samples is reached.
        """
        n_lagged_features = (self.n_lags + 1) * self.n_features
        n_samples = self.n_samples
        if self.coeffs is None:
            self.coeffs = np.random.normal(1e-3, 1.1, n_lagged_features)

        features = []
        outcomes = []
        out_censoring = np.zeros((n_samples,), dtype="uint64")
        sample_count = 0
        while sample_count < n_samples:
            X_temp, y_temp, _censoring, _ = self._simulate_batch()
            n_new_samples = len(y_temp)
            expected_count = sample_count + n_new_samples
            if expected_count >= n_samples:
                n = n_new_samples - (expected_count - n_samples)
            else:
                n = n_new_samples

            features.extend(X_temp[0:n])
            outcomes.extend(y_temp[0:n])
            out_censoring[sample_count:sample_count+n] = _censoring[0:n]
            sample_count += n_new_samples

        return features, outcomes, out_censoring, self.coeffs

    def _simulate_batch(self):
        """Simulate a batch of samples, each of which have ticked at least once.
        """
        _censoring = np.full((self.batch_size,), self.n_intervals,
                             dtype="uint64")
        # No censoring right now
        X_temp = self._simulate_sccs_features(self.batch_size)
        y_temp = self._simulate_outcomes(X_temp, _censoring)

        if self.censoring:
            censored_idx = np.random.binomial(1, self.censoring_prob,
                                              size=self.batch_size
                                              ).astype("bool")
            _censoring[censored_idx] -= np.random.poisson(
                lam=self.censoring_intensity, size=(censored_idx.sum(),)
                ).astype("uint64")
            X_temp = self._censor_array_list(X_temp, _censoring)
            y_temp = self._censor_array_list(y_temp, _censoring)

        return self._filter_non_positive_samples(X_temp, y_temp,
                                                 _censoring)

    def _simulate_sccs_features(self, n_samples):
        """Simulates features, either `infinite` or `short` exposures."""
        if self.exposure_type == "infinite":
            sim = self._sim_infinite_exposures
        elif self.exposure_type == "short":
            sim = self._sim_short_exposures

        return [sim() for _ in range(n_samples)]

    def _sim_short_exposures(self):
        features = np.random.randint(2,
                                     size=(self.n_intervals, self.n_features),
                                     ).astype("float64")
        if self.sparse:
            features = csr_matrix(features, dtype="float64")
        return features

    def _sim_infinite_exposures(self):
        if not self.sparse:
            raise ValueError("'infinite' exposures can only be simulated as \
            sparse feature matrices")
        # Select features for which there is exposure
        if self.n_features > 1:
            n_exposures = np.random.randint(1, self.n_features, 1)
            cols = np.random.choice(np.arange(self.n_features, dtype="int64"),
                                    n_exposures, replace=False)
        else:
            n_exposures = 1
            cols = np.array([0])
        # choose exposure start
        rows = np.random.randint(self.n_intervals, size=n_exposures)
        # build sparse matrix
        data = np.ones_like(cols, dtype="float64")
        exposures = csr_matrix((data, (rows, cols)),
                               shape=(self.n_intervals, self.n_features),
                               dtype="float64")
        return exposures

    def _simulate_outcomes(self, features, censoring):
        features = LongitudinalFeaturesLagger(n_lags=self.n_lags).\
            fit_transform(features, censoring)

        if self.distribution == "poisson":
            y = self._simulate_outcome_from_poisson(features, self.coeffs,
                                                    self.first_tick_only)
        else:
            y = self._simulate_outcome_from_multi(features, self.coeffs,)
        return y

    @staticmethod
    def _simulate_outcome_from_multi(features, coeffs):
        dot_products = [f.dot(coeffs) for f in features]

        def sim(dot_prod):
            dot_prod -= dot_prod.max()
            probabilities = np.exp(dot_prod) / \
                            np.sum(np.exp(dot_prod))
            y = np.random.multinomial(1, probabilities)
            return y.astype("int32")

        return [sim(dot_product) for dot_product in dot_products]

    @staticmethod
    def _simulate_outcome_from_poisson(features, coeffs, first_tick_only=True):
        dot_products = [feat.dot(coeffs) for feat in features]

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
            list of length n_samples, each element of the list of 
            shape=(n_intervals, n_features) or shape=(n_intervals,)
            The list of features matrices.
            
        censoring : `numpy.ndarray`, shape=(n_samples, 1), dtype="uint64"
            The censoring data. This array should contain integers in 
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then 
            the observation of sample i is stopped at interval c, that is, the 
            row c - 1 of the correponding matrix. The last n_intervals - c rows
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
    def _filter_non_positive_samples(features, labels, censoring):
        """Filter out samples which don't tick in the observation window.
        
        Parameters
        ----------
        features : list of numpy.ndarray or list of scipy.sparse.csr_matrix,
            list of length n_samples, each element of the list of 
            shape=(n_intervals, n_features)
            The list of features matrices.
            
        labels : list of numpy.ndarray of length n_samples,
            shape=(n_intervals,)
            The list of labels matrices.
        """
        nnz = [np.nonzero(arr)[0] for arr in labels]
        positive_sample_idx = [i for i, arr in enumerate(nnz) if
                               len(arr) > 0]
        if len(positive_sample_idx) == 0:
            raise ValueError("There should be at least one positive sample per\
             batch. Try to increase batch_size.")
        pos_samples_filter = itemgetter(*positive_sample_idx)
        return list(pos_samples_filter(features)),\
            list(pos_samples_filter(labels)),\
            censoring[positive_sample_idx],\
            np.array(positive_sample_idx, dtype="uint64")

    @property
    def exposure_type(self):
        return self._exposure_type

    @exposure_type.setter
    def exposure_type(self, value):
        if value not in ["infinite", "short"]:
            raise ValueError("exposure_type can be only 'infinite' or 'short'.")
        self._set("_exposure_type", value)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if value not in ["multinomial", "poisson"]:
            raise ValueError("distribution can be only 'multinomial' or\
             'poisson'.")
        self._set("_distribution", value)

    @property
    def censoring_prob(self):
        return self._censoring_prob

    @censoring_prob.setter
    def censoring_prob(self, value):
        if value < 0 or value > 1:
            raise ValueError("value should be in [0, 1].")
        self._set("_censoring_prob", value)

    @property
    def censoring_intensity(self):
        return self._censoring_intensity

    @censoring_intensity.setter
    def censoring_intensity(self, value):
        if value < 0:
            raise ValueError("censoring_intensity should be greater than 0.")
        self._set("_censoring_intensity", value)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is not None and \
                        value.shape != (self.n_features * (self.n_lags + 1),):
            raise ValueError("Coeffs should be of shape\
             (n_features * (n_lags + 1),)")
        self._set("_coeffs", value)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value is None and self.distribution == "multinomial":
            self._set("_batch_size", self.n_samples)
        elif value is None:
            self._set("_batch_size", int(min(2000, self.n_samples)))
        else:
            self._set("_batch_size", int(value))
        self._set("_batch_size", max(100, self.batch_size))
