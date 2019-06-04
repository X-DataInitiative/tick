import inspect
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
    check_censoring_consistency, safe_array

# Case classes
Confidence_intervals = namedtuple(
    'Confidence_intervals',
    ['refit_coeffs', 'lower_bound', 'upper_bound', 'confidence_level'])


# TODO later: exploit new options of SVRG (parallel fit, variance_reduction...)
# TODO later: add SAGA solver
class ConvSCCS(ABC, Base):
    """ConvSCCS learner, estimates lagged features effect using TV and Group L1
    penalties. These penalties constrain the coefficient groups modelling the
    lagged effects to ensure their regularity and sparsity.

    Parameters
    ----------
    n_lags : `numpy.ndarray`, shape=(n_features,), dtype="uint64"
        Number of lags per feature. The model will regress labels on the
        last observed values of the features over their corresponding
        `n_lags` time intervals. `n_lags` values must be between 0 and
        `n_intervals` - 1.

    penalized_features : `numpy.ndarray`, shape=(n_features,), dtype="bool", default=None
        Booleans indicating whether the features should be penalised or
        not. If set to None, pernalize all features.

    C_tv : `float`, default=None
        Level of TV penalization TV penalization. This value should be
        `None` or greater than 0.

    C_group_l1 : `float`, default=None
        Level of group Lasso penalization. This value should be `None` or
        greater than 0.

    step : `float`, default=None
        Step-size parameter, the most important parameter of the solver.
        If set to None, it will be automatically tuned as
        ``step = 1 / model.get_lip_max()``.

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it).

    max_iter : `int`, default=100
        Maximum number of iterations of the solver, namely maximum
        number of epochs.

    verbose : `bool`, default=False
        If `True`, solver verboses history, otherwise nothing is
        displayed.

    print_every : `int`, default=1
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True.

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``.

    random_state : `int`, default=None
        If not None, the seed of the random sampling.


    Attributes
    ----------
    n_cases : `int` (read-only)
        Number of samples with at least one outcome.

    n_intervals : `int` (read-only)
        Number of time intervals.

    n_features : `int` (read-only)
        Number of features.

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model.

    coeffs : `list` (read-only)
        List containing 1-dimensional `np.ndarray` (`dtype=float`)
        containing the coefficients of the model. Each numpy array contains
        the `(n_lags + 1)` coefficients associated with a feature. Each
        coefficient of such arrays can be interpreted as the log relative
        intensity associated with this feature, `k` periods after exposure
        start, where `k` is the index of the coefficient in the array.

    intensities : `list` (read-only)
        List containing 1-dimensional `np.ndarray` (`dtype=float`)
        containing the intensities estimated by the model.
        Each numpy array contains the relative intensities of a feature.
        Element of these arrays can be interpreted as the relative
        intensity associated with a feature, `k` periods after exposure
        start, where `k` is the index of the coefficient in the array.

    confidence_intervals : `Confidence_intervals` (read-only)
        Coefficients refitted on the model and associated confidence
        intervals computed using parametric bootstrap. Refitted coefficients
        are projected on the support of the coefficients estimated by the
        penalised model.
        Refitted coefficients and their confidence
        intervals follow the same structure as `coeffs`.

    References
    ----------
    Morel, M., Bacry, E., Gaïffas, S., Guilloux, A., & Leroy, F.
    (Submitted, 2018, January). ConvSCCS: convolutional self-controlled case
    series model for lagged adverse event detection
    """
    _const_attr = [
        # constructed attributes
        '_preprocessor_obj',
        '_model_obj',
        '_solver_obj',
        # user defined parameters
        '_n_lags',
        '_penalized_features',
        '_C_tv',
        '_C_group_l1',
        '_random_state',
        # computed attributes
        'n_cases',
        'n_intervals',
        'n_features',
        'n_coeffs',
        '_coeffs',
        '_features_offset',
        '_fitted',
        '_step_size',
        # refit _coeffs, median, and CI data
        'confidence_intervals',
        '_solver_info'
    ]

    _attrinfos = {key: {'writable': False} for key in _const_attr}

    def __init__(self, n_lags: np.array, penalized_features: np.array = None,
                 C_tv=None, C_group_l1=None, step: float = None,
                 tol: float = 1e-5, max_iter: int = 100, verbose: bool = False,
                 print_every: int = 10, record_every: int = 10,
                 random_state: int = None):
        Base.__init__(self)
        # Init objects to be computed later
        self.n_cases = None
        self.n_intervals = None
        self.n_features = None
        self.n_coeffs = None
        self._coeffs = None
        self.confidence_intervals = Confidence_intervals(
            list(), list(), list(), None)
        self._fitted = None
        self._step_size = None

        # Init user defined parameters
        self._features_offset = None
        self._n_lags = None
        self.n_lags = n_lags
        self._penalized_features = None
        self.penalized_features = penalized_features
        self._C_tv = None
        self._C_group_l1 = None
        self.C_tv = C_tv
        self.C_group_l1 = C_group_l1

        self._step_size = step
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose,
        self.print_every = print_every
        self.record_every = record_every
        random_state = int(
            np.random.randint(0, 1000, 1)[0]
            if random_state is None else random_state)
        self._random_state = None
        self.random_state = random_state

        # Construct objects
        self._preprocessor_obj = self._construct_preprocessor_obj()
        self._model_obj = None
        self._solver_info = (
          step, max_iter, tol, print_every, record_every, verbose, self.random_state)
        self._solver_obj = self._construct_solver_obj(*self._solver_info)

    # Interface
    def fit(self, features: list, labels: list, censoring: np.array,
            confidence_intervals: bool = False, n_samples_bootstrap: int = 200,
            confidence_level: float = .95):
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

        confidence_intervals : `bool`, default=False
            Activate parametric bootstrap confidence intervals computation.

        n_samples_bootstrap : `int`, default=200
            Number of parametric bootstrap iterations

        confidence_level : `float`, default=.95
            Confidence level of the bootstrapped confidence intervals

        Returns
        -------
        output : `LearnerSCCS`
            The current instance with given data
        """
        p_features, p_labels, p_censoring = self._prefit(
            features, labels, censoring)
        self._fit(project=False)
        self._postfit(p_features, p_labels, p_censoring, False,
                      confidence_intervals, n_samples_bootstrap,
                      confidence_level)

        return self.coeffs, self.confidence_intervals

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

    def fit_kfold_cv(self, features, labels, censoring, C_tv_range: tuple = (),
                     C_group_l1_range: tuple = (), logscale=True,
                     n_cv_iter: int = 30, n_folds: int = 3,
                     shuffle: bool = True, confidence_intervals: bool = False,
                     n_samples_bootstrap: int = 100,
                     confidence_level: float = .95):
        """Perform a cross validation to find optimal hyperparameters given
        training data. Cross validation using stratified K-folds and random
        search, parameters being sampled uniformly in the range (logscale or
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

        C_tv_range : `tuple`, shape=(2,), dtype="int"
            Range in which sampling TV penalization level during the
            random search. `logscale=True`, range values are understood as
            powers of ten, i.e `(0, 5)` will result in samples being in
            `10**(0), 10**(5)`.

        C_group_l1_range : `tuple`, shape=(2,), dtype="int"
            Range in which sampling group L1 penalization level during the
            random search. `logscale=True`, range values are understood as
            powers of ten, i.e `(0, 5)` will result in samples being in
            `10**(0), 10**(5)`.

        logscale : `bool`
            If `True`, hyperparameters are sampled on logscale.

        n_cv_iter : `int`
            Number of hyperparameters samples to draw when performing
            random search.

        n_folds : `int`
            Number of folds used to perform the cross validation.

        shuffle : `bool`
            If `True`, the data is shuffled before performing train / validation
            / test splits.

        confidence_intervals : `bool`, default=False
            Activate parametric bootstrap confidence intervals computation.

        n_samples_bootstrap : `int`, default=200
            Number of parametric bootstrap iterations

        confidence_level : `float`, default=.95
            Confidence level of the bootstrapped confidence intervals

        Returns
        -------
        output : `LearnerSCCS`
            The current instance with given data
        """
        # setup the model and preprocess the data
        p_features, p_labels, p_censoring = self._prefit(
            features, labels, censoring)
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
        C_tv_generator, C_group_l1_generator = self._construct_generator_obj(
            C_tv_range, C_group_l1_range, logscale)
        # TODO later: parallelize CV
        i = 0
        while i < n_cv_iter:
            self._set('_coeffs', np.zeros(self.n_coeffs))
            self.C_tv = C_tv_generator.rvs(1)[0]
            self.C_group_l1 = C_group_l1_generator.rvs(1)[0]

            train_scores = []
            test_scores = []
            for train_index, test_index in kf.split(p_features,
                                                    labels_interval):
                train = itemgetter(*train_index.tolist())
                test = itemgetter(*test_index.tolist())
                X_train, X_test = list(train(p_features)), list(
                    test(p_features))
                y_train, y_test = list(train(p_labels)), list(test(p_labels))
                censoring_train, censoring_test = p_censoring[train_index], \
                    p_censoring[test_index]

                self._model_obj.fit(X_train, y_train, censoring_train)
                self._fit(project=False)

                train_scores.append(self._score())
                test_scores.append(self._score(X_test, y_test, censoring_test))

            cv_tracker.log_cv_iteration(self.C_tv, self.C_group_l1,
                                        np.array(train_scores),
                                        np.array(test_scores))
            i += 1

        # refit best model on all the data
        best_parameters = cv_tracker.find_best_params()
        self.C_tv = best_parameters["C_tv"]
        self.C_group_l1 = best_parameters["C_group_l1"]
        self._set('_coeffs', np.zeros(self.n_coeffs))

        self._model_obj.fit(p_features, p_labels, p_censoring)
        coeffs, bootstrap_ci = self._postfit(
            p_features, p_labels, p_censoring, True, confidence_intervals,
            n_samples_bootstrap, confidence_level)

        cv_tracker.log_best_model(self.C_tv, self.C_group_l1,
                                  self._coeffs.tolist(), self.score(),
                                  self.confidence_intervals)

        return self.coeffs, cv_tracker

    def plot_intensities(self, figsize=(10, 6), sharex=False, sharey=False):
        """Plot intensities estimated by the penalized model. The intensities
        subfigures are plotted on two columns.

        Parameters
        ----------
        figsize : `tuple`, default=(10, 6)
        Size of the figure

        sharex : `bool`, default=False
        Constrain the x axes to have the same range.

        sharey : `bool`, default=False
        Constrain the y axes to have the same range.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        Figure to be plotted

        axarr : `numpy.ndarray`, `dtype=object`
        `matplotlib.axes._subplots.AxesSubplot` objects associated to each
        intensity subplot.
        """
        n_rows = int(np.ceil(self.n_features / 2))
        remove_last_plot = self.n_features % 2 != 0

        fig, axarr = plt.subplots(n_rows, 2, sharex=sharex, sharey=sharey,
                                  figsize=figsize)
        for i, c in enumerate(self.coeffs):
            self._plot_intensity(axarr[i // 2][i % 2], c, None, None)
        plt.suptitle('Estimated (penalized) relative risks')
        axarr[0][1].legend(loc='upper right')
        [ax[0].set_ylabel('Relative incidence') for ax in axarr]
        [ax.set_xlabel('Time after exposure start') for ax in axarr[-1]]
        if remove_last_plot:
            fig.delaxes(axarr[-1][-1])
        return fig, axarr

    def plot_confidence_intervals(self, figsize=(10, 6), sharex=False,
                                  sharey=False):
        """Plot intensities estimated by the penalized model. The intensities
        subfigures are plotted on two columns.

        Parameters
        ----------
        figsize : `tuple`, default=(10, 6)
        Size of the figure

        sharex : `bool`, default=False
        Constrain the x axes to have the same range.

        sharey : `bool`, default=False
        Constrain the y axes to have the same range.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        Figure to be plotted

        axarr : `numpy.ndarray`, `dtype=object`
        `matplotlib.axes._subplots.AxesSubplot` objects associated to each
        intensity subplot
        """
        n_rows = int(np.ceil(self.n_features / 2))
        remove_last_plot = (self.n_features % 2 != 0)

        fig, axarr = plt.subplots(n_rows, 2, sharex=sharex, sharey=sharey,
                                  figsize=figsize)
        ci = self.confidence_intervals
        coeffs = ci['refit_coeffs']
        lb = ci['lower_bound']
        ub = ci['upper_bound']
        for i, c in enumerate(coeffs):
            self._plot_intensity(axarr[i // 2][i % 2], c, lb[i], ub[i])
        plt.suptitle('Estimated relative risks with 95% confidence bands')
        axarr[0][1].legend(loc='best')
        [ax[0].set_ylabel('Relative incidence') for ax in axarr]
        [ax.set_xlabel('Time after exposure start') for ax in axarr[-1]]
        if remove_last_plot:
            fig.delaxes(axarr[-1][-1])
        return fig, axarr

    @staticmethod
    def _plot_intensity(ax, coeffs, upper_bound, lower_bound):
        n_coeffs = len(coeffs)
        if n_coeffs > 1:
            x = np.arange(n_coeffs)
            ax.step(x, np.exp(coeffs), label="Estimated RI")
            if upper_bound is not None and lower_bound is not None:
                ax.fill_between(x, np.exp(lower_bound), np.exp(upper_bound),
                                alpha=.5, color='orange', step='pre',
                                label="95% boostrap CI")
        elif n_coeffs == 1:
            if upper_bound is not None and lower_bound is not None:
                ax.errorbar(0, coeffs, yerr=(np.exp(lower_bound),
                                             np.exp(upper_bound)), fmt='o',
                            ecolor='orange')
            else:
                ax.scatter([0], np.exp(coeffs), label="Estimated RI")
        return ax

    # Internals
    def _prefit(self, features, labels, censoring):
        n_intervals, n_features = features[0].shape
        if any(self.n_lags > n_intervals - 1):
            raise ValueError('`n_lags` should be < `n_intervals` - 1, where '
                             '`n_intervals` is the number of rows of the '
                             'feature matrices.')
        n_cases = len(features)
        censoring = check_censoring_consistency(censoring, n_cases)
        features = check_longitudinal_features_consistency(
            features, (n_intervals, n_features), "float64")
        labels = check_longitudinal_features_consistency(
            labels, (n_intervals,), "int32")
        self._set('n_features', n_features)
        self._set('n_intervals', n_intervals)

        features, labels, censoring = self._preprocess_data(
            features, labels, censoring)
        n_coeffs = int(np.sum(self._n_lags) + self.n_features)
        self._set('n_coeffs', n_coeffs)
        self._set('_coeffs', np.zeros(n_coeffs))
        self._set('n_cases', len(features))

        # Step computation
        self._set("_model_obj", self._construct_model_obj())
        self._model_obj.fit(features, labels, censoring)
        if self.step is None:
            self.step = 1 / self._model_obj.get_lip_max()

        return features, labels, censoring

    def _fit(self, project):
        prox_obj = self._construct_prox_obj(self._coeffs, project)
        solver_obj = self._solver_obj
        model_obj = self._model_obj

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = self._coeffs

        # Launch the solver
        _coeffs = solver_obj.solve(coeffs_start, step=self.step)

        self._set("_coeffs", _coeffs)
        self._set("_fitted", True)

        return _coeffs

    def _postfit(self, p_features, p_labels, p_censoring, refit, bootstrap,
                 n_samples_bootstrap, confidence_level):
        # WARNING: _refit uses already preprocessed p_features, p_labels
        # and p_censoring
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        if refit:
            # refit _coeffs on all the data (after Cross Validation for example)
            self._model_obj.fit(p_features, p_labels, p_censoring)
            coeffs = self._fit(project=False)
            self._set('_coeffs', coeffs)

        if bootstrap:
            self._model_obj.fit(p_features, p_labels, p_censoring)
            _refit_coeffs = self._fit(project=True)
            confidence_intervals = self._bootstrap(
                p_features, p_labels, p_censoring, _refit_coeffs,
                n_samples_bootstrap, confidence_level)
            self._set('confidence_intervals', confidence_intervals._asdict())

        return self.coeffs, self.confidence_intervals

    # Utilities #
    def _preprocess_data(self, features, labels, censoring):
        for preprocessor in self._preprocessor_obj:
            features, labels, censoring = preprocessor.fit_transform(
                features, labels, censoring)
        return features, labels, censoring

    def _bootstrap(self, p_features, p_labels, p_censoring, coeffs, rep,
                   confidence):
        # WARNING: _bootstrap inputs are already preprocessed p_features,
        # p_labels and p_censoring
        # Coeffs here are assumed to be an array (same object than self._coeffs)
        if confidence <= 0 or confidence >= 1:
            raise ValueError("`confidence_level` should be in (0, 1)")
        confidence = 1 - confidence
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        bootstrap_coeffs = []
        sim = SimuSCCS(self.n_cases, self.n_intervals, self.n_features,
                       self.n_lags, coeffs=self._format_coeffs(coeffs))
        # TODO later: parallelize bootstrap (everything should be pickable...)
        for k in range(rep):
            y = sim._simulate_multinomial_outcomes(p_features, coeffs)
            self._model_obj.fit(p_features, y, p_censoring)
            bootstrap_coeffs.append(self._fit(True))

        bootstrap_coeffs = np.exp(np.array(bootstrap_coeffs))
        bootstrap_coeffs.sort(axis=0)
        lower_bound = np.log(bootstrap_coeffs[int(
            np.floor(rep * confidence / 2))])
        upper_bound = np.log(bootstrap_coeffs[int(
            np.floor(rep * (1 - confidence / 2)))])
        return Confidence_intervals(
            self._format_coeffs(coeffs), self._format_coeffs(lower_bound),
            self._format_coeffs(upper_bound), confidence)

    def _score(self, features=None, labels=None, censoring=None,
               preprocess=False):
        if not self._fitted:
            raise RuntimeError('You must fit the model first')

        all_none = all(e is None for e in [features, labels, censoring])
        if all_none:
            loss = self._model_obj.loss(self._coeffs)
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
                        features, labels, censoring)
                model = self._construct_model_obj().fit(
                    features, labels, censoring)
                loss = model.loss(self._coeffs)
        return loss

    def _format_coeffs(self, coeffs):
        value = list()
        for i, l in enumerate(self.n_lags):
            start = int(self._features_offset[i])
            end = int(start + l + 1)
            value.append(coeffs[start:end])
        return value

    # Factories #
    def _construct_preprocessor_obj(self):
        # TODO later: fix parallel preprocessing
        preprocessors = list()
        preprocessors.append(LongitudinalSamplesFilter(n_jobs=1))
        if len(self.n_lags) > 0:
            preprocessors.append(
                LongitudinalFeaturesLagger(self.n_lags, n_jobs=1))
        return preprocessors

    def _construct_model_obj(self):
        return ModelSCCS(self.n_intervals, self.n_lags)

    def _construct_prox_obj(self, coeffs=None, project=False):
        n_penalized_features = len(self.penalized_features) \
            if self.penalized_features is not None else 0

        if project:
            # project future _coeffs on the support of given _coeffs
            if all(self.n_lags == 0):
                proxs = [ProxZero()]
            elif coeffs is not None:
                prox_ranges = self._detect_support(coeffs)
                proxs = [ProxEquality(0, range=r) for r in prox_ranges]
            else:
                raise ValueError("Coeffs are None. " +
                                 "Equality penalty cannot infer the "
                                 "coefficients support.")
        elif n_penalized_features > 0 and self._C_tv is not None or \
                self._C_group_l1 is not None:
            # TV and GroupLasso penalties
            blocks_start = np.zeros(n_penalized_features)
            blocks_end = np.zeros(n_penalized_features)
            proxs = []

            for i in self.penalized_features:
                start = int(self._features_offset[i])
                blocks_start[i] = start
                end = int(blocks_start[i] + self._n_lags[i] + 1)
                blocks_end[i] = end
                if self._C_tv is not None:
                    proxs.append(ProxTV(1 / self._C_tv, range=(start, end)))

            if self._C_group_l1 is not None:
                blocks_size = blocks_end - blocks_start
                proxs.append(
                    ProxGroupL1(1 / self._C_group_l1, blocks_start.tolist(),
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
        to enforce a support corresponding to the support of `_coeffs`.

         example:
         _coeffs = np.array([ 1.  2.  2.  1.  1.])
         self._detect_support(_coeffs)
         >>> [(1, 3), (3, 5)]
         """
        kernel = np.array([1, -1])
        groups = []
        for i in self.penalized_features:
            idx = int(self._features_offset[i])
            n_lags = int(self._n_lags[i])
            if n_lags > 0:
                acc = 1
                for change in np.convolve(coeffs[idx:idx + n_lags + 1], kernel,
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
    def _construct_solver_obj(step, max_iter, tol, print_every, record_every,
                              verbose, seed):
        # TODO: we might want to use SAGA also later... (might be faster here)
        # seed cannot be None in SVRG
        solver_obj = SVRG(step=step, max_iter=max_iter, tol=tol,
                          print_every=print_every, record_every=record_every,
                          verbose=verbose, seed=seed)

        return solver_obj


    @staticmethod
    def _construct_solver_obj_with_class(
            step, max_iter, tol, print_every, record_every, verbose, seed, clazz=SVRG):
        """All creatioon of solver by class type, removes values from constructor parameter
           list that do not exist on the class construct to be called
         """
        # inspect must be first assign
        _, _, _, kvs = inspect.getargvalues(inspect.currentframe())
        constructor_map = kvs.copy()
        args = inspect.getfullargspec(clazz.__init__)[0]
        for k, v in kvs.items():
            if k not in args:
                del constructor_map[k]
        return SVRG(**constructor_map)


    def _construct_generator_obj(self, C_tv_range, C_group_l1_range,
                                 logspace=True):
        generators = []
        if len(C_tv_range) == 2:
            if logspace:
                generators.append(Log10UniformGenerator(*C_tv_range))
            else:
                generators.append(uniform(C_tv_range))
        else:
            generators.append(null_generator)

        if len(C_group_l1_range) == 2:
            if logspace:
                generators.append(Log10UniformGenerator(*C_group_l1_range))
            else:
                generators.append(uniform(C_group_l1_range))
        else:
            generators.append(null_generator)

        return generators

    # Properties #
    @property
    def step(self):
        return self._step_size

    @step.setter
    def step(self, value):
        if value > 0:
            self._set('_step_size', value)
            self._solver_obj.step = value
        else:
            raise ValueError("step should be greater than 0.")

    @property
    def C_tv(self):
        return self._C_tv

    @C_tv.setter
    def C_tv(self, value):
        if value is None or isinstance(value, float) and value > 0:
            self._set('_C_tv', value)
        else:
            raise ValueError('C_tv should be a float greater than zero.')

    @property
    def C_group_l1(self):
        return self._C_group_l1

    @C_group_l1.setter
    def C_group_l1(self, value):
        if value is None or isinstance(value, float) and value > 0:
            self._set('_C_group_l1', value)
        else:
            raise ValueError('C_group_l1 should be a float greater '
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
            if l < 0:
                raise ValueError('n_lags elements should be greater than or '
                                 'equal to 0.')
            offsets.append(offsets[-1] + l + 1)
        value = safe_array(value, dtype=np.uint64)
        self._set('_n_lags', value)
        self._set('_features_offset', offsets)
        self._construct_preprocessor_obj()

    @property
    def penalized_features(self):
        return self._penalized_features

    @penalized_features.setter
    def penalized_features(self, value):
        if value is None:
            value = [True] * len(self.n_lags)
        self._set('_penalized_features', value)
        self._construct_preprocessor_obj()

    @property
    def coeffs(self):
        return self._format_coeffs(self._coeffs)

    @coeffs.setter
    def coeffs(self, value):
        raise ValueError('coeffs cannot be set')

    @property
    def intensities(self):
        return [np.exp(c) for c in self.coeffs]

    @intensities.setter
    def intensities(self, value):
        raise ValueError('intensities cannot be set')


# TODO later: put the code below somewhere else?
class CrossValidationTracker:
    def __init__(self, model_params: dict):
        self.model_params = model_params
        self.kfold_train_scores = list()
        self.kfold_mean_train_scores = list()
        self.kfold_sd_train_scores = list()
        self.kfold_test_scores = list()
        self.kfold_mean_test_scores = list()
        self.kfold_sd_test_scores = list()
        # TODO later: make this class usable for any parameters
        # self.parameter_names = parameter_names
        # self.best_model = {
        #     '_coeffs': list(),
        #     'confidence_interval': {},
        #     **{name: list() for name in parameter_names}
        # }
        # for field in parameter_names:
        #     setattr(field + '_history', list())
        self.C_tv_history = list()
        self.C_group_l1_history = list()

    def log_cv_iteration(self, C_tv, C_group_l1, kfold_train_scores,
                         kfold_test_scores):
        self.kfold_train_scores.append(list(kfold_train_scores))
        self.kfold_mean_train_scores.append(kfold_train_scores.mean())
        self.kfold_sd_train_scores.append(kfold_train_scores.std())
        self.kfold_test_scores.append(list(kfold_test_scores))
        self.kfold_mean_test_scores.append(kfold_test_scores.mean())
        self.kfold_sd_test_scores.append(kfold_test_scores.std())
        self.C_tv_history.append(C_tv)
        self.C_group_l1_history.append(C_group_l1)

    def find_best_params(self):
        # Find best parameters
        best_idx = int(np.argmin(self.kfold_mean_test_scores))
        best_C_tv = self.C_tv_history[best_idx]
        best_C_group_l1 = self.C_group_l1_history[best_idx]
        return {'C_tv': best_C_tv, 'C_group_l1': best_C_group_l1}

    def log_best_model(self, C_tv, C_group_l1, coeffs, score,
                       confidence_interval_dict):
        self.best_model = {
            'C_tv': C_tv,
            'C_group_l1': C_group_l1,
            '_coeffs': list(coeffs),
            'score': score,
            'confidence_intervals': confidence_interval_dict
        }

    def todict(self):
        return {
            'global_model_parameters': list(self.model_params),
            'C_tv_history': list(self.C_tv_history),
            'C_group_l1_history': list(self.C_group_l1_history),
            'kfold_train_scores': list(self.kfold_train_scores),
            'kfold_mean_train_scores': list(self.kfold_mean_train_scores),
            'kfold_sd_train_scores': list(self.kfold_sd_train_scores),
            'kfold_test_scores': list(self.kfold_test_scores),
            'kfold_mean_test_scores': list(self.kfold_mean_test_scores),
            'kfold_sd_test_scores': list(self.kfold_sd_test_scores),
            'best_model': self.best_model
        }

    def plot_cv_report(self, elevation=25, azimuth=35):
        if len(self.C_group_l1_history) > 0 and len(self.C_tv_history) > 0:
            return self.plot_learning_curves_contour(elevation, azimuth)
        elif len(self.C_group_l1_history) == 0:
            return self.plot_learning_curves('Group L1')
        elif len(self.C_tv_history) == 0:
            return self.plot_learning_curves('TV')
        else:
            raise ValueError("Logged Group L1 and TV penalisation levels "
                             "history are empty.")

    def plot_learning_curves(self, hyperparameter):
        if hyperparameter == "TV":
            C = self.C_tv_history
        elif hyperparameter == "Group L1":
            C = self.C_group_l1_history
        else:
            raise ValueError("hyperparameter value should be either `TV` or"
                             " `Group L1`")
        x = np.log10(C)
        order = np.argsort(x)
        m = np.array(self.kfold_mean_train_scores)[order]
        sd = np.array(self.kfold_sd_train_scores)[order]
        fig = plt.figure()
        ax = plt.gca()
        p1 = ax.plot(x[order], m)
        p2 = ax.fill_between(x[order], m - sd, m + sd, alpha=.3)
        min_point_train = np.min(m - sd)
        m = np.array(self.kfold_mean_test_scores)[order]
        sd = np.array(self.kfold_sd_test_scores)[order]
        p3 = ax.plot(x[order], m)
        p4 = ax.fill_between(x[order], m - sd, m + sd, alpha=.3)
        min_point_test = np.min(m - sd)
        min_point = min(min_point_train, min_point_test)
        p5 = plt.scatter(np.log10(C), min_point * np.ones_like(C))

        ax.legend([(p1[0], p2), (p3[0], p4), p5],
                  ['train score', 'test score', 'tested hyperparameters'],
                  loc='lower right')
        ax.set_title('Learning curves')
        ax.set_xlabel('C %s (log scale)' % hyperparameter)
        ax.set_ylabel('Loss')
        return fig, ax

    def plot_learning_curves_contour(self, elevation=25, azimuth=35):
        """‘elev’ stores the elevation angle in the z plane.
        ‘azim’ stores the azimuth angle in the x,y plane."""
        sc_train = self.kfold_mean_train_scores
        sc_test = self.kfold_mean_test_scores
        X_tile = np.log10(self.C_tv_history)
        Y_tile = np.log10(self.C_group_l1_history)

        fig, axarr = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                                  sharex=True)

        ax = axarr[-1]
        ax.scatter(X_tile, Y_tile)
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
            ax.set_xlabel("TV level (log)")
            idx = np.where(Z == Z.min())
            x, y = (X_tile[idx][0], Y_tile[idx][0])
            ax.scatter(x, y, color="red", marker="x")
            ax.text(x, y, r'%.2f' % Z.min(), color="red", fontsize=12)

        axarr[0].set_ylabel("Group L1 level (log)")
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
        best_params = self.find_best_params()
        x = np.log10(best_params['C_tv'])
        y = np.log10(best_params['C_group_l1'])
        idx = np.where(X_tile == x)  # should be equal to np.where(Y_tile == y)
        z = Z[idx]
        p1 = ax.scatter(x, y, z, c='red')
        proxies.append(p1)
        proxy_names.append('CV best score')

        ax.set_xlabel("TV level (log)")
        ax.set_ylabel("Group L1 level (log)")
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
