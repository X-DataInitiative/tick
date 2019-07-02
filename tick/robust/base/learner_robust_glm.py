# License: BSD 3 clause

from warnings import warn
import numpy as np
from tick.base import Base
from tick.base.learner import LearnerGLM
from tick.prox import ProxZero, ProxL1, ProxL2Sq, ProxElasticNet, \
    ProxSlope, ProxMulti

class LearnerRobustGLM(LearnerGLM):
    """Learner for a Robust Generalized Linear Model (GML).
    This is a GLM with sample intercepts (one for each sample).
    Slope penalization is used on the individual intercepts.
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant.

    Parameters
    ----------
    C_sample_intercepts : `float`
        Level of penalization of the ProxSlope penalization used for detection
        of outliers. Ideally, this should be equal to n_samples / noise_level,
        where noise_level is an estimated noise level

    C : `float`, default=1e3
        Level of penalization of the model weights

    fdr : `float`, default=0.05
        Target false discovery rate for the detection of outliers, namely for
        the detection of non-zero entries in ``sample_intercepts``

    penalty : {'none', 'l1', 'l2', 'elasticnet', 'slope'}, default='l2'
        The penalization to use. Default 'l2', namely ridge penalization.

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model, namely a global intercept

    refit : 'bool', default=False
        Not implemented yet

    solver : 'gd', 'agd'
        The name of the solver to use. For now, only gradient descent and
        accelerated gradient descent are available

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning.

    tol : `float`, default=1e-7
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it).

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``
        
    Other Parameters
    ----------------
    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty only.

    slope_fdr : `float`, default=0.05
        Target false discovery rate for the detection of detection of non-zero
        entries in the model weights.
        Used in the 'slope' penalty only.

    Attributes
    ----------
    weights : `numpy.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    sample_intercepts : `numpy.array`, shape=(n_samples,)
        Sample intercepts. This should be a sparse vector, since a non-zero
        entry means that the sample is an outlier.

    intercept : `float` or `None`
        The intercept, if ``fit_intercept=True``, otherwise `None`

    coeffs : `numpy.array`, shape=(n_features + n_samples + 1,)
        The full array of coefficients of the model. Namely, this is simply
        the concatenation of ``weights``, ``sample_intercepts``
        and ``intercept``
    """
    _attrinfos = {
        "_fit_intercept": {
            "writable": False
        },
        "sample_intercepts": {
            "writable": False
        },
        "weights": {
            "writable": False
        },
        "intercept": {
            "writable": False
        },
        "_prox_intercepts_obj": {
            "writable": False
        },
        "coeffs": {
            "writable": False
        }
    }

    _solvers = {
        'gd': 'GD',
        'agd': 'AGD',
    }

    _solvers_with_linesearch = ['gd', 'agd']
    _solvers_with_step = ['gd', 'agd']

    _penalties = {
        'none': ProxZero,
        'l1': ProxL1,
        'l2': ProxL2Sq,
        'elasticnet': ProxElasticNet,
        'slope': ProxSlope
    }

    def __init__(self, C_sample_intercepts, C=1e3, fdr=0.05, penalty='l2',
                 fit_intercept=True, refit=False, solver='agd',
                 warm_start=False, step=None, tol=1e-5, max_iter=100,
                 verbose=True, print_every=10, record_every=10,
                 elastic_net_ratio=0.95, slope_fdr=0.05):

        extra_model_kwargs = {'fit_intercept': fit_intercept}

        LearnerGLM.__init__(
            self, fit_intercept=fit_intercept, penalty=penalty, C=C,
            solver=solver, step=step, tol=tol, max_iter=max_iter,
            verbose=verbose, warm_start=warm_start, print_every=print_every,
            record_every=record_every, elastic_net_ratio=elastic_net_ratio,
            random_state=None, blocks_start=None, blocks_length=None)

        self._prox_intercepts_obj = ProxSlope(1e-2)
        self.C_sample_intercepts = C_sample_intercepts
        self.fdr = fdr
        self.refit = refit

        if 'slope_fdr' in self._actual_kwargs or \
                        penalty == 'slope':
            self.slope_fdr = slope_fdr

        self.sample_intercepts = None
        self.coeffs = None

    def fit(self, X: object, y: np.array):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`,, shape=(n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : `np.array`, shape=(n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : LearnerGLM
            The fitted instance of the model
        """
        solver_obj = self._solver_obj
        model_obj = self._model_obj
        prox_obj = self._prox_obj
        prox_intercepts_obj = self._prox_intercepts_obj
        fit_intercept = self.fit_intercept

        X = LearnerGLM._safe_array(X)
        y = LearnerGLM._safe_array(y)

        # Pass the data to the model
        model_obj.fit(X, y)

        n_samples = model_obj.n_samples
        n_features = model_obj.n_features

        if self.step is None:
            self.step = 1. / model_obj.get_lip_best()

        # Range of the sample intercepts prox is always the same
        if fit_intercept:
            prox_intercepts_obj.range = (n_features + 1,
                                         n_features + n_samples + 1)
        else:
            prox_intercepts_obj.range = (n_features, n_features + n_samples)
        prox_obj.range = (0, n_features)

        if self.penalty == 'none':
            # No penalization is used on the model weights, so the prox applied
            # overall is only ProxSlope on the sample intercepts
            solver_prox = prox_intercepts_obj
        else:
            solver_prox = ProxMulti([prox_obj, prox_intercepts_obj])

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(solver_prox)
        # Make sure that there is no linesearch
        solver_obj.linesearch = False

        coeffs_start = None
        if self.warm_start and self.coeffs is not None:
            if self.coeffs.shape == (model_obj.n_coeffs,):
                coeffs_start = self.coeffs
            else:
                raise ValueError('Cannot warm start, coeffs don\'t have the '
                                 'right shape')

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start)

        self._set("coeffs", coeffs)
        self._set("weights", coeffs[:n_features])
        if fit_intercept:
            self._set("intercept", coeffs[n_features])
            self._set("sample_intercepts",
                      coeffs[(n_features + 1):(n_features + n_samples + 1)])
        else:
            self._set("intercept", None)
            self._set("sample_intercepts",
                      coeffs[n_features:(n_features + n_samples)])
        self._set("_fitted", True)
        return self

    def predict(self, X):
        """Not available. This model is helpful to estimate and detect
        outliers. It cannot, for now, predict the label based on non-observed
        features.
        """
        raise NotImplementedError("Not available for now.")

    def score(self, X):
        """Not available. This model is helpful to estimate and detect
        outliers. Score computation makes no sense in this setting.
        """
        raise NotImplementedError("Not available for now.")

    def get_params(self):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : `dict`
            Parameter names mapped to their values.
        """
        dd = {
            'fit_intercept': self.fit_intercept,
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'step': self.step,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'print_every': self.print_every,
            'record_every': self.record_every,
            'sdca_ridge_strength': self.sdca_ridge_strength,
            'elastic_net_ratio': self.elastic_net_ratio,
            'random_state': self.random_state,
            'blocks_start': self.blocks_start,
            'blocks_length': self.blocks_length,
        }
        return dd

    def set_params(self, **kwargs):
        """
        Set the parameters for this learner.

        Parameters
        ----------
        **kwargs :
            Named arguments to update in the learner

        Returns
        -------
        output : `LearnerRobustGLM`
            self with updated parameters
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        return self

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("intercept", None)
        dd.pop("weights", None)
        return dd

    @property
    def C_sample_intercepts(self):
        return 1. / self._prox_intercepts_obj.strength

    @C_sample_intercepts.setter
    def C_sample_intercepts(self, val):
        if val is None:
            raise ValueError("``C_sample_intercepts`` cannot be `None`")
        elif val == 0.:
            raise ValueError("``C_sample_intercepts`` cannot be 0.")
        elif val <= 0:
            raise ValueError(
                "``C_sample_intercepts`` must be positive, got %s" % str(val))
        elif np.isinf(val):
            raise ValueError(
                "``C_sample_intercepts`` must be a finite number, got %s" %
                str(val))
        else:
            strength = 1. / val
            self._prox_intercepts_obj.strength = strength

    @property
    def fdr(self):
        return self._prox_intercepts_obj.fdr

    @fdr.setter
    def fdr(self, val):
        if val is None:
            raise ValueError("``fdr`` cannot be `None`")
        elif np.isinf(val):
            raise ValueError(
                "``fdr`` must be a finite number, got %s" % str(val))
        elif val <= 0 or val >= 1:
            raise ValueError("``fdr`` must be in (0, 1), got %s" % str(val))
        else:
            self._prox_intercepts_obj.fdr = val

    @property
    def slope_fdr(self):
        if self.penalty == 'slope':
            return self._prox_obj.fdr
        else:
            return None

    @slope_fdr.setter
    def slope_fdr(self, val):
        if self.penalty == 'slope':
            if val is None:
                raise ValueError("``slope_fdr`` cannot be `None`")
            elif np.isinf(val):
                raise ValueError(
                    "``slope_fdr`` must be a finite number, got %s" % str(val))
            elif val <= 0 or val >= 1:
                raise ValueError(
                    "``slope_fdr`` must be in (0, 1), got %s" % str(val))
            else:
                self._prox_obj.fdr = val
        else:
            warn('Penalty "%s" has no ``slope_fdr`` attribute' % self.penalty,
                 RuntimeWarning)

    @property
    def refit(self):
        return False

    @refit.setter
    def refit(self, val):
        if val:
            raise NotImplementedError('``refit`` can only be set to `False` '
                                      'for now')
