from tick.inference.base import LearnerOptim
from tick.optim.prox import ProxTV, ProxMulti, ProxZero, ProxEquality, ProxL1
from tick.optim.solver import BFGS, SVRG
from tick.optim.model import ModelSCCS
from tick.preprocessing import LongitudinalFeaturesProduct,\
    LongitudinalFeaturesLagger
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from operator import itemgetter
from scipy.misc import comb
from itertools import chain


class LearnerSCCS(LearnerOptim):
    _attrinfos = {
        "_preprocessor_obj": {
            "writable": False
        },
        "coeffs": {
            "writable": False
        },
        "n_intervals": {
            "writable": False
        },
        "n_lags": {
            "writable": False
        },
        "feature_products": {
            "writable": False
        },
        "n_features": {
            "writable": False
        },
    }  # TODO: add refit_coeffs and refit_CI

    _solvers = {
        'svrg': SVRG,
        'bfgs': BFGS,
    }

    _penalties = {
        'none': ProxZero,
        'TV': ProxTV,
        'Equality': ProxEquality,
        'L1-TV': [ProxL1, ProxTV]
    }

    def __init__(self, n_intervals: int, n_lags: int, n_features: int,
                 strength: float, penalty='TV', solver="svrg",
                 feature_products=False, feature_type="infinite",
                 step=None, tol=1e-5, max_iter=100, verbose=True,
                 warm_start=False, print_every=10, record_every=10,
                 random_state=None):
        self.n_intervals = int(n_intervals)
        self.n_lags = int(n_lags)
        if feature_type in ["infinite", "short"]:  # TODO: property
            self.feature_type = feature_type
        else:
            raise ValueError("``feature_type`` should be either ``infinite`` or\
             ``short``.")
        self.feature_products = feature_products
        self.n_features = int(comb(n_features, 2) + n_features
                              if feature_products
                              else n_features)

        LearnerOptim.__init__(self, penalty="none", C=None, solver=solver,
                              step=step, tol=tol, max_iter=max_iter,
                              verbose=verbose, warm_start=warm_start,
                              print_every=print_every,
                              record_every=record_every, sdca_ridge_strength=0,
                              elastic_net_ratio=0,
                              random_state=random_state
                              )

        allowed_penalties = list(self._penalties.keys())
        allowed_penalties.sort()
        if penalty not in allowed_penalties:
            raise ValueError("``penalty`` must be one of %s, got %s" %
                             (', '.join(allowed_penalties), penalty))
        self._set('penalty', penalty)
        self.strength = strength  # TODO property and second parameter for L1 pen ?
        self._preprocessor_obj = self._construct_preprocessor_obj()

    def fit(self, features: np.ndarray, labels: np.array,
            censoring: np.array): # TODO doc quotes ``
        """Fit the model according to the given training data.

        Parameters
        ----------
        features : List[{2d array, csr matrix of shape (n_intervals, n_features)}]
        The features matrix

        labels : List[{1d array, csr matrix of shape (n_intervals, 1)]
            The labels vector

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `LearnerCoxReg`
            The current instance with given data
        """
        pp = self._preprocessor_obj
        model_obj = self._model_obj

        features, labels, censoring, _ = pp.transform(features, labels,
                                                      censoring)

        self._compute_step(features, labels, censoring)

        n_grouped_cols = self.n_lags + 1
        groups = [(int(n_grouped_cols * i), int(n_grouped_cols * (i + 1)))
                  for i in range(self.n_features)]
        prox_obj = self._construct_prox_multi_obj(self.penalty, groups)
        self._set("_prox_obj", prox_obj)

        coeffs = self._fit(prox_obj)
        self._set("coeffs", coeffs)
        self._set("_fitted", True)

        return self

    def score(self, features=None, labels=None, censoring=None):
        # TODO TEST
        """Returns the negative log-likelihood of the model, using the current
        fitted coefficients on the passed data.
        If no data is passed, the negative log-likelihood is computed using the
        data used for training.

        Parameters
        ----------
        features : `None` or `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        labels : `None` or `numpy.array`, shape = (n_samples,)
            Observed labels

        censoring : `None` or `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `float`
            The value of the negative log-likelihood
        """
        if self._fitted:
            all_none = all(e is None for e in [features, labels, censoring])
            if all_none:
                return self._model_obj.loss(self.coeffs)
            else:
                if features is None:
                    raise ValueError('Passed ``features`` is None')
                elif labels is None:
                    raise ValueError('Passed ``labels`` is None')
                elif censoring is None:
                    raise ValueError('Passed ``censoring`` is None')
                else:
                    model = self._construct_model_obj().fit(features, labels,
                                                            censoring)
                    return model.loss(self.coeffs)
        else:
            raise RuntimeError('You must fit the model first')

    def fit_KFold_CV(self, features, labels, censoring, strength_list,
                     n_splits=3):
        # TODO: Write DOC
        # TODO: add option for stratified KFold, and refits to compute bootstrap
        # preprocess all the data
        n_features = self.n_features
        pp = self._preprocessor_obj
        features, labels, censoring, _ = pp.transform(features, labels,
                                                      censoring)
        # Compute lip const on all the data
        # TODO: is it a good to do so ?
        self._compute_step(features, labels, censoring)

        # split the data
        kf = KFold(n_splits=n_splits) # TODO: stratified KFold here
        scores = []

        # Construct prox here
        self._set("_prox_obj",
                  self._construct_prox_multi_obj("TV", groups))

        # Training loop
        for strength in strength_list:
            # self.strength = strength
            # create prox instance
            self._set("strength", strength) # TODO: use a setter here to update prox_obj

            kfold_scores_train = []
            kfold_scores_test = []
            for train_index, test_index in kf.split(features):
                train = itemgetter(*train_index.tolist())
                test = itemgetter(*test_index.tolist())
                X_train, X_test = list(train(features)), list(test(features))
                y_train, y_test = list(train(labels)), list(test(labels))
                censoring_train, censoring_test = censoring[train_index], \
                                                  censoring[test_index]
                self.model_obj.fit(X_train, y_train, censoring_train)
                coeffs = self._fit(self.prox_obj)

                kfold_scores_train.append(self.score())
                kfold_scores_test.append(self.score(X_test, y_test,
                                                    censoring_test))

            scores.append({
                "n_intervals": self.n_intervals,
                "n_lags": self.n_lags,
                "n_features": self.n_features,
                "feature_products": self.feature_products,
                "feature_type": self.feature_type,
                "strength": self.strength,
                "train": {
                    "mean": np.mean(kfold_scores_train),
                    "var": np.var(kfold_scores_train),
                    "sd": np.sqrt(np.var(kfold_scores_train)),
                    "kfold_scores": kfold_scores_train
                },
                "test": {
                    "mean": np.mean(kfold_scores_test),
                    "var": np.var(kfold_scores_test),
                    "sd": np.sqrt(np.var(kfold_scores_test)),
                    "kfold_scores": kfold_scores_test
                }
            })
        # Find best parameters and refit on full data
        best_idx = np.argmin([s["test"]["mean"] for s in scores])
        best_parameters = scores[best_idx]
        best_strength = best_parameters["strength"]

        # refit best model on all the data
        self._set("strength",
                  best_strength)  # TODO: code a setter to update the prox_obj too

        self.model_obj.fit(X_train, y_train, censoring_train)
        coeffs = self._fit(self.prox_obj)
        self._set("coeffs", coeffs)
        best_model = {
            "n_intervals": self.n_intervals,
            "n_lags": self.n_lags,
            "n_features": self.n_features,
            "feature_products": self.feature_products,
            "feature_type": self.feature_type,
            "strength": self.strength,
            "coeffs": coeffs.tolist()
        }
        return coeffs, scores, best_model

    def refit(self, coeffs):
        # TODO TEST
        # if not self._fitted:
        #     raise RuntimeError('You must fit the model first')
        groups = self._detect_change_points(coeffs)
        prox_obj = self._construct_prox_multi_obj('Equality', groups)
        coeffs = self._fit(prox_obj)
        return coeffs

    def bootstrap_CI(self, coeffs, rep, confidence):
        coeffs = []
        for k in range(rep):
            X, y, c, _ = SimuSCCS(coeffs)
            coeffs.append(self.refit(coeffs, X, y, c))  # TODO update refit
        coeffs = np.array(coeffs)
        coeffs.sort(axis=1)  # TODO check axis
        lower_bound = coeffs[np.ceil(rep * confidence / 2)]
        upper_bound = coeffs[np.ceil(rep * (1 - confidence / 2))]
        return lower_bound, upper_bound

    def _compute_step(self, features, labels, censoring):
        self.model_obj.fit(features, labels, censoring)
        if self.step is None:  # TODO property + flag for step (if none, compute it)
            step = 1 / self.model_obj.get_lip_max()
        self.step = step
        return step

    def _fit(self, prox_obj):
        solver_obj = self._solver_obj
        model_obj = self._model_obj

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = None
        if self.warm_start and self.coeffs is not None:
            coeffs = self.coeffs
            # ensure starting point has the right format
            if coeffs.shape == (model_obj.n_coeffs,):
                coeffs_start = coeffs

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start, step=self.step)

        return coeffs

    def _detect_change_points(self, coeffs):
        # TODO TEST
        coeffs = coeffs.reshape((self.n_features, self.n_lags + 1))
        kernel = np.array([1, -1])
        groups = []
        for l in range(self.n_features):
            idx = 0
            acc = 1
            for change in np.convolve(coeffs[l, :], kernel, 'valid') != 0:
                if change:
                    groups.append((idx, idx + acc))
                    idx += acc
                    acc = 1
                else:
                    acc += 1
            groups.append((idx, 4))
        return groups

    def _construct_preprocessor_obj(self):
        # TODO: add a filter for useless cases here ?
        lagger = LongitudinalFeaturesLagger(self.n_lags)
        features_product = LongitudinalFeaturesProduct(self.exposure_type)
        return lagger, features_product

    def _construct_model_obj(self):
        return ModelSCCS(self.n_intervals, self.n_lags)

    def _construct_prox_multi_obj(self, penalty, groups):
        if penalty == "none":
            proxs = [ProxZero()]
        elif penalty == "TV":
            proxs = (ProxTV(self.strength, range=grp) for grp in groups)
        elif penalty == "L1_TV":
            # This is a flatmap
            proxs = chain.from_iterable(
                self._prox_L1_TV(self.strength, range=grp) for grp in groups)
        elif penalty == "Equality":
            proxs = (ProxEquality(0, range=grp) for grp in groups)

        prox_obj = ProxMulti(tuple(proxs))

        return prox_obj

    @staticmethod
    def _prox_L1_TV(strength_L1, strength_TV, range):
        if range[1] < (range[0] + 1):
            raise ValueError("range[1] should be > range[0]")  # TODO: useless check ?

        proxL1 = ProxL1(strength_L1, range=(range[0], range[0]+1))
        proxTV = ProxTV(strength_TV, range=range)

        return proxL1, proxTV
