from tick.inference.base import LearnerOptim
from tick.optim.prox import ProxTV, ProxMulti, ProxZero
from tick.optim.solver import BFGS, SVRG
from tick.optim.model import ModelSCCS
from tick.preprocessing import LongitudinalFeaturesProduct,\
    LongitudinalFeaturesLagger
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter
from scipy.misc import comb



class LearnerLMTV(LearnerOptim):
    _attrinfos = {
        "_preprocessor_obj": {
            "writable": False
        },
        "coeffs": {
            "writable": False
        }
    }

    _solvers = {
        'svrg': SVRG,
        'bfgs': BFGS,
    }

    _penalties = {
        'none': ProxZero,
        'multi-tv': ProxZero,
    }

    def __init__(self, n_intervals: int, n_lags: int, n_features: int,
                 strength: float, solver="svrg",
                 step=None, tol=1e-5, max_iter=100,
                 verbose=True, warm_start=False, print_every=10,
                 record_every=10, random_state=None, feature_products=False,
                 feature_type="infinite"):
        if feature_type in ["infinite", "short"]:
            self.feature_type = feature_type
        else:
            raise ValueError("``feature_type`` should be either ``infinite`` or\
             ``short``.")
        self.n_intervals = int(
            n_intervals)  # TODO: set this to writable = false
        self.n_lags = int(n_lags)  # TODO: set this to writable = false
        self.feature_products = feature_products  # TODO: set this to writable = false
        self.n_features = int(comb(n_features, 2) + n_features
                              if feature_products
                              else n_features)  # TODO: set this to writable = false
        LearnerOptim.__init__(self, penalty="none", C=None, solver=solver,
                              step=step, tol=tol, max_iter=max_iter,
                              verbose=verbose, warm_start=warm_start,
                              print_every=print_every,
                              record_every=record_every, sdca_ridge_strength=0,
                              elastic_net_ratio=0,
                              random_state=random_state
                              )
        self.penalty = "multi-tv"
        self.strength = strength
        self._preprocessor_obj = self._construct_preprocessor_obj()

    def _construct_model_obj(self):
        return ModelSCCS(self.n_intervals, self.n_lags)

    def _construct_prox_obj(self, penalty, elastic_net_ratio, extra_prox_kwarg):
        # Parameters of the penalty

        if penalty == "none":
            prox_obj = ProxZero()
        elif penalty != "multi-tv":
            raise ValueError("``penalty`` must be ``multi-tv`` or ``none``")
        else:
            n_grouped_cols = self.n_lags + 1
            proxs = [ProxTV(self.strength, range=(int(n_grouped_cols * i),
                                                  int(n_grouped_cols * (i + 1)))
                            ) for i in range(self.n_features)]
            prox_obj = ProxMulti(proxs)

        return prox_obj

    def _construct_preprocessor_obj(self):
        lagger = LongitudinalFeaturesLagger(self.n_lags)
        features_product = LongitudinalFeaturesProduct(self.exposure_type)
        return lagger, features_product

    def fit(self, features: np.ndarray, labels: np.array,
            censoring: np.array):
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
        # The fit from Model calls the _set_data below

        solver_obj = self._solver_obj
        model_obj = self._model_obj
        n_features = self.n_features
        self._set("_prox_obj",
                  self._construct_prox_obj("multi-tv", None, None))
        prox_obj = self._prox_obj
        pp = self._preprocessor_obj

        features, labels, censoring, _ = pp.transform(features, labels,
                                                      censoring)

        # Pass the data to the model
        model_obj.fit(features, labels, censoring)

        if self.step is None:
            self.step = 1 / model_obj.get_lip_max()

        # TODO: update prox here with the dim

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

        # Get the learned coefficients
        self._set("coeffs", coeffs)
        self._set("_fitted", True)
        return self

    def score(self, features=None, labels=None, censoring=None):
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
        # preprocess all the data
        n_features = self.n_features
        pp = self._preprocessor_obj
        features, labels, censoring, _ = pp.transform(features, labels,
                                                      censoring)

        # get model instance
        solver_obj = self._solver_obj
        model_obj = self._model_obj

        # Get lip constant here
        if self.step is None:
            model_obj.fit(features, labels, censoring)
            self.step = 1 / model_obj.get_lip_max()

        # split the data
        kf = KFold(n_splits=n_splits)
        scores = []

        # Training loop
        for strength in strength_list:
            # self.strength = strength
            # create prox instance
            self._set("strength", strength)
            self._set("_prox_obj",
                      self._construct_prox_obj("multi-tv", None, None))
            prox_obj = self._prox_obj

            kfold_scores_train = []
            kfold_scores_test = []
            for train_index, test_index in kf.split(features):
                train = itemgetter(*train_index.tolist())
                test = itemgetter(*test_index.tolist())
                X_train, X_test = list(train(features)), list(test(features))
                y_train, y_test = list(train(labels)), list(test(labels))
                censoring_train, censoring_test = censoring[train_index], \
                                                  censoring[test_index]
                model_obj.fit(X_train, y_train, censoring_train)
                solver_obj.set_model(model_obj)
                solver_obj.set_prox(prox_obj)
                coeffs = solver_obj.solve(step=self.step)

                kfold_scores_train.append(model_obj.loss(coeffs))
                model_obj.fit(X_test, y_test, censoring_test)
                kfold_scores_test.append(model_obj.loss(coeffs))

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
        self._set("_prox_obj",
                  self._construct_prox_obj("multi-tv", None, None))
        prox_obj = self._prox_obj
        model_obj.fit(features, labels, censoring)
        solver_obj.set_model(model_obj)
        solver_obj.set_prox(prox_obj)
        coeffs = solver_obj.solve(step=self.step)
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
