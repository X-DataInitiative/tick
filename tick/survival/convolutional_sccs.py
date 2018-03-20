import numpy as np
from abc import ABC
from operator import itemgetter
from collections import namedtuple
from sklearn.model_selection import StratifiedKFold
from tick.base import Base
from tick.prox import ProxZero, ProxMulti, ProxTV, ProxEquality, \
    ProxGroupL1
from tick.solver import SVRG
from tick.survival import SimuSCCS, ModelSCCS
from tick.preprocessing import LongitudinalSamplesFilter, \
    LongitudinalFeaturesLagger
from scipy.stats import uniform
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tick.preprocessing.utils import check_longitudinal_features_consistency, \
    check_censoring_consistency

# Case classes
Bootstrap_CI = namedtuple('Bootstrap_CI', ['refit_coeffs', 'median',
                                           'lower_bound', 'upper_bound',
                                           'confidence'])

Strengths = namedtuple('Strengths', ['strength_tv', 'strength_group_l1'])


# TODO later: exploit new options of SVRG (parallel fit, variance_reduction...)
class ConvSCCS(ABC, Base):
    _const_attr = [
        # constructed attributes
        '_preprocessor_obj',
        '_model_obj',
        '_solver_obj',
        # user defined parameters
        '_n_lags',
        '_penalized_features',
        '_strength_tv',
        '_strength_group_l1',
        '_random_state',
        # computed attributes
        'n_cases',
        'n_intervals',
        'n_features',
        'n_coeffs',
        'coeffs',
        '_features_offset',
        '_fitted',
        '_step_size',
        # refit coeffs, median, and CI data
        'bootstrap_coeffs',
    ]

    _attrinfos = {key: {'writable': False} for key in _const_attr}

    def __init__(self, n_lags: np.array,
                 penalized_features: np.array,
                 strength_tv=None, strength_group_l1=None,
                 step: float = None, tol: float = 1e-5, max_iter: int = 100,
                 verbose: bool = False, print_every: int = 10,
                 record_every: int = 10, random_state: int = None):
        """ConvSCCS model. This class allows to estimate a lagged effect for
        each feature. TV and Group L1 penalties might be used to penalize the
        coefficient groups modelling the lagged effects.

        Parameters
        ----------
        n_lags : `numpy.ndarray`, shape=(n_features,), dtype="uint64"
            Number of lags per feature. The model will regress labels on the
            last observed values of the features over their corresponding
            `n_lags` time intervals. `n_lags` values must be between 0 and
            `n_intervals` - 1.

        penalized_features : `numpy.ndarray`, shape=(n_features,), dtype="bool"
            Booleans indicating whether the features should be penalised or
            not.

        strength_tv : `float`, default=None
            Strength of the TV penalization. This value should be `None` or
            greater than 0.

        strength_group_l1 : `float`, default=None
            Strength of the group Lasso penalization. This value should be
            `None` or greater than 0.

        step : `float`
            Step-size parameter, the most important parameter of the solver.
            Whenever possible, this can be automatically tuned as
            ``step = 1 / model.get_lip_max()``. Otherwise, use a
            try-an-improve approach.

        tol : `float`, default=1e-5
            The tolerance of the solver (iterations stop when the stopping
            criterion is below it).

        max_iter : `int`, default=100
            Maximum number of iterations of the solver, namely maximum
            number of epochs.

        verbose : `bool`, default=False
            If `True`, solver verboses history, otherwise nothing is
            displayed.

        random_state : `int`, default=None
            If not None, the seed of the random sampling.


        Attributes
        ----------
        n_cases : `int` (read-only)
            Number of samples

        n_intervals : `int` (read-only)
            Number of time intervals

        n_features : `int` (read-only)
            Number of features

        n_coeffs : `int` (read-only)
            Total number of coefficients of the model

        coeffs : `numpy.ndarray`, shape=(n_coeffs,), dtype="float64" (read-only)
            Coefficients of the model.

        bootstrap_coeffs : `Bootstrap_CI` (read-only)
            Bootstrap coefficients and confidence intervals of the model.
        """
        Base.__init__(self)
        # Init objects to be computed later
        self.n_cases = None
        self.n_intervals = None
        self.n_features = None
        self.n_coeffs = None
        self.coeffs = None
        self.bootstrap_coeffs = Bootstrap_CI(list(), list(), list(), list(),
                                             None)
        self._fitted = None
        self._step_size = None

        # Init user defined parameters
        self._features_offset = None
        self._n_lags = None
        self.n_lags = n_lags
        self._penalized_features = None
        self.penalized_features = penalized_features
        self._strength_tv = None
        self._strength_group_l1 = None
        self.strength_tv = strength_tv
        self.strength_group_l1 = strength_group_l1

        self._step_size = step
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose,
        self.print_every = print_every
        self.record_every = record_every
        random_state = int(np.random.randint(0, 1000, 1)[0]
                           if random_state is None else random_state)
        self._random_state = None
        self.random_state = random_state

        # Construct objects
        self._preprocessor_obj = self._construct_preprocessor_obj()
        self._model_obj = None
        self._solver_obj = self._construct_solver_obj(step, max_iter, tol,
                                                      print_every, record_every,
                                                      verbose, random_state)

    # Interface
    def fit(self, features: list, labels: list,
            censoring: np.array, bootstrap: bool = False,
            bootstrap_rep: int = 200,
            bootstrap_confidence: float = .95):
        """Fit the model according to the given training data.

        Parameters
        ----------
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

        bootstrap : `bool`, default=False
            Activate parametric bootstrap confidence intervals computation.

        bootstrap_rep : `int`, default=200
            Number of parametric bootstrap iterations

        bootstrap_confidence : `float`, default=.95
            Confidence level of the bootstrapped confidence intervals

        Returns
        -------
        output : `LearnerSCCS`
            The current instance with given data
        """
        p_features, p_labels, p_censoring = self._prefit(features, labels,
                                                         censoring)
        self._fit(project=False)
        self._postfit(p_features, p_labels, p_censoring, False,
                      bootstrap, bootstrap_rep, bootstrap_confidence)

        return self.coeffs, self.bootstrap_coeffs

    def score(self, features=None, labels=None, censoring=None):
        """Returns the negative log-likelihood of the model, using the current
        fitted coefficients on the passed data.
        If no data is passed, the negative log-likelihood is computed using the
        data used for training.

        Parameters
        ----------
        features : `None` or `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`,
            list of length n_cases, each element of the list of
            shape=(n_intervals, n_features)
            The list of features matrices.

        labels : `None` or `list` of `numpy.ndarray`,
            list of length n_cases, each element of the list of
            shape=(n_intervals,)
            The labels vector

        censoring : `None` or `numpy.ndarray`, shape=(n_cases,), dtype="uint64"
            The censoring data. This array should contain integers in
            [1, n_intervals]. If the value i is equal to n_intervals, then there
            is no censoring for sample i. If censoring = c < n_intervals, then
            the observation of sample i is stopped at interval c, that is, the
            row c - 1 of the corresponding matrix. The last n_intervals - c rows
            are then set to 0.

        Returns
        -------
        output : `float`
            The value of the negative log-likelihood
        """

        return self._score(features, labels, censoring, preprocess=True)

    def fit_kfold_cv(self, features, labels, censoring,
                     strength_tv_range: tuple=(),
                     strength_group_l1_range: tuple=(), logspace=True,
                     n_cv_iter: int= 30, n_folds: int = 3, shuffle: bool = True,
                     bootstrap: bool = False, bootstrap_rep: int = 100,
                     bootstrap_confidence: float = .95):
        """Perform a cross validation to find optimal hyperparameters given
        training data. Cross validation using stratified K-folds and random
        search, parameters being sampled uniformly in the range (logspace or
        linspace) specified by the user.

        Parameters
        ----------
        features : `list` of `numpy.ndarray` or `list` of `scipy.sparse.csr_matrix`
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

        strength_tv_range : `tuple`, shape=(2,), dtype="int"
            Range in which sampling TV penalization strength during the
            random search. `logspace=True`, range values are understood as
            powers of ten, i.e `(-5, -1)` will result in samples being in
            `10**(-5), 10**(-1)`.

        strength_group_l1_range : `tuple`, shape=(2,), dtype="int"
            Range in which sampling group L1 penalization strength during the
            random search. `logspace=True`, range values are understood as
            powers of ten, i.e `(-5, -1)` will result in samples being in
            `10**(-5), 10**(-1)`.

        logspace : `bool`
            If `True`, hyperparameters are samples in logspace.

        n_cv_iter : `int`
            Number of hyperparameters samples to draw when performing
            random search.

        n_folds : `int`
            Number of folds used to perform the cross validation.

        shuffle : `bool`
            If `True`, the data is shuffled before performing train / validation
            / test splits.

        bootstrap : `bool`, default=False
            Activate parametric bootstrap confidence intervals computation.

        bootstrap_rep : `int`, default=200
            Number of parametric bootstrap iterations

        bootstrap_confidence : `float`, default=.95
            Confidence level of the bootstrapped confidence intervals

        Returns
        -------
        output : `LearnerSCCS`
            The current instance with given data
        """
        # setup the model and preprocess the data
        p_features, p_labels, p_censoring = self._prefit(features, labels,
                                                         censoring)
        # split the data with stratified KFold
        kf = StratifiedKFold(n_folds, shuffle, self.random_state)
        labels_interval = np.nonzero(p_labels)[1]

        # Training loop
        model_global_parameters = {
            "n_intervals": self.n_intervals,
            "n_lags": self.n_lags,
            "n_features": self.n_features,
        }
        cv_tracker = CrossValidationTracker(model_global_parameters)
        generators = self._construct_generator_obj(strength_tv_range,
                                                   strength_group_l1_range,
                                                   logspace)
        # TODO later: parallelize CV
        i = 0
        while i < n_cv_iter:
            self._set('coeffs', np.zeros(self.n_coeffs))
            self._strengths = [g.rvs(1)[0] for g in generators]

            train_scores = []
            test_scores = []
            for train_index, test_index in kf.split(p_features, labels_interval):
                train = itemgetter(*train_index.tolist())
                test = itemgetter(*test_index.tolist())
                X_train, X_test = list(train(p_features)), list(test(p_features))
                y_train, y_test = list(train(p_labels)), list(test(p_labels))
                censoring_train, censoring_test = p_censoring[train_index], \
                    p_censoring[test_index]

                self._model_obj.fit(X_train, y_train, censoring_train)
                self._fit(project=False)

                train_scores.append(self._score())
                test_scores.append(self._score(X_test, y_test, censoring_test))

            cv_tracker.log_cv_iteration({'strength': self._strengths},
                                        np.array(train_scores),
                                        np.array(test_scores))
            i += 1

        best_parameters = cv_tracker.find_best_params()
        best_strength = best_parameters["strength"]

        # refit best model on all the data
        self._set('coeffs', np.zeros(self.n_coeffs))
        self._strengths = best_strength

        self._model_obj.fit(p_features, p_labels, p_censoring)
        coeffs, bootstrap_ci = self._postfit(p_features, p_labels, p_censoring,
                                             True, bootstrap, bootstrap_rep,
                                             bootstrap_confidence)

        cv_tracker.log_best_model(self._strengths, self.coeffs.tolist(),
                                  self.score(),
                                  self.bootstrap_coeffs)

        return coeffs, cv_tracker

    # Internals
    def _prefit(self, features, labels, censoring):
        n_intervals, n_features = features[0].shape
        n_cases = len(features)
        censoring = check_censoring_consistency(censoring, n_cases)
        features = check_longitudinal_features_consistency(features,
                                                           (n_intervals,
                                                            n_features),
                                                           "float64")
        labels = check_longitudinal_features_consistency(labels,
                                                         (n_intervals,),
                                                         "int32")
        self._set('n_features', n_features)
        self._set('n_intervals', n_intervals)

        features, labels, censoring = self._preprocess_data(features,
                                                            labels,
                                                            censoring)
        n_coeffs = int(np.sum(self._n_lags) + self.n_features)
        self._set('n_coeffs', n_coeffs)
        self._set('coeffs', np.zeros(n_coeffs))
        self._set('n_cases', len(features))

        # Step computation
        self._set("_model_obj", self._construct_model_obj())
        self._model_obj.fit(features, labels, censoring)
        if self.step is None:
            self.step = 1 / self._model_obj.get_lip_max()

        return features, labels, censoring

    def _fit(self, project):
        prox_obj = self._construct_prox_obj(self.coeffs, project)
        solver_obj = self._solver_obj
        model_obj = self._model_obj

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = self.coeffs

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start, step=self.step)

        self._set("coeffs", coeffs)
        self._set("_fitted", True)

        return coeffs

    def _postfit(self, p_features, p_labels, p_censoring,
                 refit, bootstrap, bootstrap_rep, bootstrap_confidence):
        # WARNING: _refit uses already preprocessed p_features, p_labels
        # and p_censoring
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        if refit:
            # refit coeffs on all the data (after Cross Validation for example)
            self._model_obj.fit(p_features, p_labels, p_censoring)
            coeffs = self._fit(project=False)
            self._set('coeffs', coeffs)

        if bootstrap:
            self._model_obj.fit(p_features, p_labels, p_censoring)
            refit_coeffs = self._fit(project=True)
            bootstrap_ci = self._bootstrap(p_features, p_labels, p_censoring,
                                           refit_coeffs, bootstrap_rep,
                                           bootstrap_confidence)
            self._set('bootstrap_coeffs', bootstrap_ci._asdict())

        return self.coeffs, self.bootstrap_coeffs

    # Utilities #
    def _preprocess_data(self, features, labels, censoring):
        for preprocessor in self._preprocessor_obj:
            features, labels, censoring = preprocessor.fit_transform(
                features,
                labels,
                censoring)
        return features, labels, censoring

    def _bootstrap(self, p_features, p_labels, p_censoring, coeffs,
                   rep, confidence):
        # WARNING: _bootstrap inputs are already preprocessed p_features,
        # p_labels and p_censoring
        if confidence <= 0 or confidence >= 1:
            raise ValueError("`bootstrap_confidence` should be in (0, 1)")
        confidence = 1 - confidence
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        bootstrap_coeffs = []
        sim = SimuSCCS(self.n_cases, self.n_intervals, self.n_features,
                       self.n_lags, coeffs=coeffs)
        # TODO later: parallelize bootstrap (everything should be pickable...)
        for k in range(rep):
            y = sim._simulate_multinomial_outcomes(p_features, coeffs)
            self._model_obj.fit(p_features, y, p_censoring)
            bootstrap_coeffs.append(self._fit(True))

        bootstrap_coeffs = np.exp(np.array(bootstrap_coeffs))
        bootstrap_coeffs.sort(axis=0)
        lower_bound = np.log(bootstrap_coeffs[
                                 int(np.floor(rep * confidence / 2))])
        upper_bound = np.log(bootstrap_coeffs[
                                 int(np.floor(rep * (1 - confidence / 2)))])
        median_coeffs = np.log(bootstrap_coeffs[
                                   int(np.floor(rep * .5))])
        return Bootstrap_CI(coeffs, median_coeffs, lower_bound,
                            upper_bound, confidence)

    def _score(self, features=None, labels=None, censoring=None,
               preprocess=False):
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        all_none = all(e is None for e in [features, labels, censoring])
        if all_none:
            loss = self._model_obj.loss(self.coeffs)
        else:
            if features is None:
                raise ValueError('Passed ``features`` is None')
            elif labels is None:
                raise ValueError('Passed ``labels`` is None')
            elif censoring is None:
                raise ValueError('Passed ``censoring`` is None')
            else:
                # Avoid calling preprocessing during CV
                if preprocess:
                    features, labels, censoring = self._preprocess_data(
                        features,
                        labels,
                        censoring)
                model = self._construct_model_obj().fit(features, labels,
                                                        censoring)
                loss = model.loss(self.coeffs)
        return loss

    # Factories #
    def _construct_preprocessor_obj(self):
        # TODO later: fix parallel preprocessing
        preprocessors = list()
        preprocessors.append(LongitudinalSamplesFilter(n_jobs=1))
        if len(self.n_lags) > 0:
            preprocessors.append(LongitudinalFeaturesLagger(self.n_lags,
                                                            n_jobs=1))
        return preprocessors

    def _construct_model_obj(self):
        return ModelSCCS(self.n_intervals, self.n_lags)

    def _construct_prox_obj(self, coeffs=None, project=False):
        n_penalized_features = len(self.penalized_features) \
            if self.penalized_features is not None else 0

        if project:
            # project future coeffs on the support of given coeffs
            if all(self.n_lags) == 0:
                proxs = [ProxZero()]
            elif coeffs is not None and any(self.n_lags) > 0:
                prox_ranges = self._detect_support(coeffs)
                proxs = [ProxEquality(0, range=r) for r in prox_ranges]
            else:
                raise ValueError("Coeffs are None. " +
                                 "Equality penalty cannot infer the "
                                 "coefficients support.")
        elif n_penalized_features > 0 and self._strength_tv is not None or \
                self._strength_group_l1 is not None:
            # TV and GroupLasso penalties
            blocks_start = np.zeros(n_penalized_features)
            blocks_end = np.zeros(n_penalized_features)
            proxs = []

            for i in self.penalized_features:
                start = int(self._features_offset[i])
                blocks_start[i] = start
                end = int(blocks_start[i] + self._n_lags[i] + 1)
                blocks_end[i] = end
                if self._strength_tv is not None:
                    proxs.append(ProxTV(self._strength_tv, range=(start, end)))

            if self._strength_group_l1 is not None:
                blocks_size = blocks_end - blocks_start
                proxs.append(ProxGroupL1(self._strength_group_l1,
                                         blocks_start.tolist(),
                                         blocks_size.tolist()))
        else:
            # Default prox: does nothing
            proxs = [ProxZero()]

        prox_obj = ProxMulti(tuple(proxs))

        return prox_obj

    def _detect_support(self, coeffs):
        """Return the ranges over which consecutive coefficients are equal in
        case at least two coefficients are equal.

        This method is used to compute the ranges for ProxEquality,
        to enforce a support corresponding to the support of `coeffs`.

         example:
         coeffs = np.array([ 1.  2.  2.  1.  1.])
         self._detect_support(coeffs)
         >>> [(1, 3), (3, 5)]
         """
        kernel = np.array([1, -1])
        groups = []
        for i in self.penalized_features:
            idx = int(self._features_offset[i])
            n_lags = int(self._n_lags[i])
            if n_lags > 0:
                acc = 1
                for change in np.convolve(coeffs[idx:idx+n_lags+1], kernel,
                                          'valid'):
                    if change:
                        if acc > 1:
                            groups.append((idx, idx + acc))
                        idx += acc
                        acc = 1
                    else:
                        acc += 1
                # Last coeff always count as a change
                if acc > 1:
                    groups.append((idx, idx + acc))
        return groups

    @staticmethod
    def _construct_solver_obj(step, max_iter, tol, print_every,
                              record_every, verbose, seed):
        # seed cannot be None in SVRG
        solver_obj = SVRG(step=step, max_iter=max_iter, tol=tol,
                          print_every=print_every,
                          record_every=record_every,
                          verbose=verbose, seed=seed)

        return solver_obj

    def _construct_generator_obj(self, strength_tv_range,
                                 strength_group_l1_range,
                                 logspace=True):
        generators = []
        if len(strength_tv_range) == 2:
            if logspace:
                generators.append(Log10UniformGenerator(*strength_tv_range))
            else:
                generators.append(uniform(strength_tv_range))
        else:
            generators.append(null_generator)

        if len(strength_group_l1_range) == 2:
            if logspace:
                generators.append(
                    Log10UniformGenerator(*strength_group_l1_range))
            else:
                generators.append(uniform(strength_group_l1_range))
        else:
            generators.append(null_generator)

        return generators

    # Properties #
    @property
    def step(self):
        return self._step_size

    @step.setter
    def step(self, value):
        self._set('_step_size', value)
        self._solver_obj.step = value

    @property
    def _strengths(self):
        return Strengths(self._strength_tv,
                         self._strength_group_l1)

    @_strengths.setter
    def _strengths(self, value):
        if len(value) == 2:
            self.strength_tv = value[0]
            self.strength_group_l1 = value[1]
        else:
            raise ValueError('strength should be a tuple of length 2.')

    @property
    def strength_tv(self):
        return self._strength_tv

    @strength_tv.setter
    def strength_tv(self, value):
        if value is None or isinstance(value, float) and value > 0:
            self._set('_strength_tv', value)
        else:
            raise ValueError(
                'strength_tv should be a float greater than zero.')

    @property
    def strength_group_l1(self):
        return self._strength_tv

    @strength_group_l1.setter
    def strength_group_l1(self, value):
        if value is None or isinstance(value, float) and value > 0:
            self._set('_strength_group_l1', value)
        else:
            raise ValueError('strength_group_l1 should be a float greater '
                             'than zero.')

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        np.random.seed(value)
        self._set('_random_state', value)

    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, value):
        offsets = [0]
        for l in value:
            offsets.append(offsets[-1] + l + 1)
        self._set('_n_lags', value)
        self._set('_features_offset', offsets)
        self._construct_preprocessor_obj()

    @property
    def penalized_features(self):
        return self._penalized_features

    @penalized_features.setter
    def penalized_features(self, value):
        self._set('_penalized_features', value)
        self._construct_preprocessor_obj()


# TODO later: put the code below somewhere else?
class CrossValidationTracker:
    def __init__(self, global_params: dict):
        self.global_params = global_params
        self.cv_params = list()
        self.cv_train_scores = list()
        self.cv_mean_train_scores = list()
        self.cv_sd_train_scores = list()
        self.cv_test_scores = list()
        self.cv_mean_test_scores = list()
        self.cv_sd_test_scores = list()
        self.best_model = {
            'strength': list(),
            'coeffs': list(),
            'bootstrap_ci': {}
        }

    def log_cv_iteration(self, cv_params, cv_train_score, cv_test_score):
        self.cv_params.append(cv_params)
        self.cv_train_scores.append(list(cv_train_score))
        self.cv_mean_train_scores.append(cv_train_score.mean())
        self.cv_sd_train_scores.append(cv_train_score.std())
        self.cv_test_scores.append(list(cv_test_score))
        self.cv_mean_test_scores.append(cv_test_score.mean())
        self.cv_sd_test_scores.append(cv_test_score.std())

    def find_best_params(self):
        # Find best parameters
        best_idx = int(np.argmin(self.cv_mean_test_scores))
        return self.cv_params[best_idx]

    def log_best_model(self, strength, coeffs, score, bootstrap_ci_dict):
        self.best_model = {
            'strength': strength,
            'coeffs': list(coeffs),
            'score': score,
            'bootstrap_ci': bootstrap_ci_dict
        }

    def todict(self):
        return {'global_model_parameters': list(self.global_params),
                'cv_params': list(self.cv_params),
                'cv_train_scores': list(self.cv_train_scores),
                'cv_mean_train_scores': list(self.cv_mean_train_scores),
                'cv_sd_train_scores': list(self.cv_sd_train_scores),
                'cv_test_scores': list(self.cv_test_scores),
                'cv_mean_test_scores': list(self.cv_mean_test_scores),
                'cv_sd_test_scores': list(self.cv_sd_test_scores),
                'best_model': self.best_model
                }

    def plot_cv_report(self, elevation=25, azimuth=35):
        group_l1_strength = [p['strength'].strength_group_l1 for p in
                             self.cv_params]
        tv_strength = [p['strength'].strength_tv for p in
                       self.cv_params]
        if not any(group_l1_strength) is None and not any(tv_strength) is None:
            return self.plot_learning_curves_contour(elevation, azimuth)
        elif not any(group_l1_strength) is None:
            return self.plot_learning_curves('Group L1')
        elif not any(tv_strength) is None:
            return self.plot_learning_curves('TV')
        else:
            raise ValueError("Logged Group L1 and TV strengths are None.")

    def plot_learning_curves(self, hyperparameter):
        #TODO: test this method
        if hyperparameter == "TV":
            strength = [p['strength'].strength_tv for p in
                        self.cv_params]
        elif hyperparameter == "Group L1":
            strength = [p['strength'].strength_group_l1 for p in
                        self.cv_params]
        else:
            raise ValueError("hyperparameter value should be either `TV` or"
                             " `Group L1`")
        x = np.log10(strength)
        order = np.argsort(x)
        m = np.array(self.cv_mean_train_scores)[order]
        sd = np.array(self.cv_sd_train_scores)[order]
        fig, ax = plt.figure()

        p1 = ax.plot(x[order], m)
        p2 = ax.fill_between(x[order], m - sd, m + sd, alpha=.3)
        min_point_train = np.min(m - sd)
        m = np.array(self.cv_mean_test_scores)[order]
        sd = np.array(self.cv_sd_test_scores)[order]
        p3 = ax.plot(x[order], m)
        p4 = ax.fill_between(x[order], m - sd, m + sd, alpha=.3)
        min_point_test = np.min(m - sd)
        min_point = min(min_point_train, min_point_test)
        p5 = plt.scatter(np.log10(strength), min_point*np.ones_like(strength))

        ax.legend([(p1[0], p2), (p3[0], p4), p5],
                  ['train score', 'test score', 'tested hyperparameters'],
                  loc='lower right')
        ax.set_title('Learning curves')
        ax.set_xlabel('Strength %s (log scale)' % hyperparameter)
        ax.set_ylabel('Loss')
        return fig, ax

    def plot_learning_curves_contour(self, elevation=25, azimuth=35):
        """‘elev’ stores the elevation angle in the z plane.
        ‘azim’ stores the azimuth angle in the x,y plane."""
        sc_train = self.cv_mean_train_scores
        sc_test = self.cv_mean_test_scores
        group_l1_strength = [p['strength'].strength_group_l1 for p in
                             self.cv_params]
        tv_strength = [p['strength'].strength_tv for p in
                       self.cv_params]
        X_tile = np.log10(tv_strength)
        Y_tile = np.log10(group_l1_strength)

        fig, axarr = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                                  sharex=True)

        ax = axarr[-1]
        ax.scatter(np.log10(tv_strength), np.log10(group_l1_strength))
        ax.set_title("Random search tested hyperparameters")
        ax.set_xlabel('Strength TV')
        ax.set_ylabel('Strength Group L1')

        names = ['train', 'test']
        cmaps = [cm.Blues, cm.Greens]

        for i, cv in enumerate([sc_train, sc_test]):
            Z = np.array(cv)
            ax = axarr[i]
            cax = ax.tricontourf(X_tile, Y_tile, Z, 50, cmap=cmaps[i])
            ax.set_title(r'Loss (%s)' % names[i])
            ax.set_xlabel("TV strength (log)")
            idx = np.where(Z == Z.min())
            x, y = (X_tile[idx][0], Y_tile[idx][0])
            ax.scatter(x, y, color="red", marker="x")
            ax.text(x, y, r'%.2f' % Z.min(), color="red", fontsize=12)

        axarr[0].set_ylabel("Group L1 strength (log)")
        plt.tight_layout()

        fig2 = plt.figure(figsize=(8, 6.5))
        ax = Axes3D(fig2)
        colors = ['blue', 'green']
        names = ['train', 'test']
        proxies = []
        proxy_names = []
        for i, cv in enumerate([sc_train, sc_test]):
            Z = np.array(cv)
            ax.plot_trisurf(X_tile, Y_tile, Z, alpha=0.3, color=colors[i])
            proxies.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=.3))
            proxy_names.append("%s score" % names[i])

        Z = np.array(sc_test)
        x, y = np.log10(
            np.array(
                self.find_best_params()['strength']).astype(
                np.float))
        idx = np.where(X_tile == x)  # should be equal to np.where(Y_tile == y)
        z = Z[idx]
        p1 = ax.scatter(x, y, z, c='red')
        proxies.append(p1)
        proxy_names.append('CV best score')

        x, y = np.log10(
            np.array(
                self.find_best_params()['strength']).astype(
                np.float))
        idx = np.where(X_tile == x)  # should be equal to np.where(Y_tile == y)
        z = Z[idx]
        p2 = ax.scatter(x, y, z, c='magenta')
        proxies.append(p2)
        proxy_names.append('CV best score')

        ax.set_xlabel("TV strength (log)")
        ax.set_ylabel("Group L1 strength (log)")
        ax.set_title("Learning surfaces")
        ax.set_zlabel("loss")
        ax.view_init(elevation, azimuth)
        plt.legend(proxies, proxy_names, loc='best')
        return fig, axarr, fig2, ax


# Generators for random search
# generator which generates nothing
DumbGenerator = namedtuple('DumbGenerator', ['rvs'])

null_generator = DumbGenerator(rvs=lambda x: [None])


class Log10UniformGenerator:
    """Generate uniformly distributed points in the log10 space."""
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.gen = uniform(0, 1)

    def rvs(self, n):
        return 10 ** (
            self.min_val + (self.max_val - self.min_val) * self.gen.rvs(n))
