# License: BSD 3 clause

from warnings import warn

import numpy as np
from tick.base import Base

from tick.base_model import ModelLipschitz
from .learner_optim import LearnerOptim


class LearnerGLM(LearnerOptim):
    """Learner for a Generalized Linear Model (GML).
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : 'none', 'l1', 'l2', 'elasticnet', 'tv', 'binarsity', default='l2'
        The penalization to use. Default 'l2', namely ridge penalization.

    solver : 'gd', 'agd', 'bfgs', 'svrg', 'sdca'
        The name of the solver to use

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used in 'gd', 'agd', 'sgd'
        and 'svrg' solvers

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

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
    sdca_ridge_strength : `float`, default=1e-3
        It controls the strength of the additional ridge penalization. Used in
        'sdca' solver

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : int seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers
        
    blocks_start : `numpy.array`, shape=(n_features,), default=None
        The indices of the first column of each binarized feature blocks. It
        corresponds to the ``feature_indices`` property of the
        ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty
        
    blocks_length : `numpy.array`, shape=(n_features,), default=None
        The length of each binarized feature blocks. It corresponds to the
        ``n_values`` property of the ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    Attributes
    ----------
    weights : np.array, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    intercept : `float` or None
        The intercept, if ``fit_intercept=True``, otherwise `None`
    """

    _attrinfos = {
        "_fit_intercept": {
            "writable": False
        },
        "weights": {
            "writable": False
        },
        "intercept": {
            "writable": False
        },
    }

    def __init__(self, fit_intercept=True, penalty='l2', C=1e3, solver="svrg",
                 step=None, tol=1e-5, max_iter=100, verbose=True,
                 warm_start=False, print_every=10, record_every=10,
                 sdca_ridge_strength=1e-3, elastic_net_ratio=0.95,
                 random_state=None, blocks_start=None, blocks_length=None):

        extra_model_kwargs = {'fit_intercept': fit_intercept}

        LearnerOptim.__init__(
            self, penalty=penalty, C=C, solver=solver, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, warm_start=warm_start,
            print_every=print_every, record_every=record_every,
            sdca_ridge_strength=sdca_ridge_strength,
            elastic_net_ratio=elastic_net_ratio, random_state=random_state,
            extra_model_kwargs=extra_model_kwargs, blocks_start=blocks_start,
            blocks_length=blocks_length)

        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept = None

    @property
    def fit_intercept(self):
        return self._model_obj.fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, val: bool):
        self._model_obj.fit_intercept = val

    def fit(self, X: object, y: np.array):
        """
        Fit the model according to the given training data.

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
        fit_intercept = self.fit_intercept

        # Pass the data to the model
        model_obj.fit(X, y)

        if self.step is None and self.solver in self._solvers_with_step:
            if self.solver in self._solvers_with_linesearch:
                self._solver_obj.linesearch = True
            elif self.solver == 'svrg':
                if isinstance(self._model_obj, ModelLipschitz):
                    self.step = 1. / self._model_obj.get_lip_max()
                else:
                    warn('SVRG step needs to be tuned manually',
                         RuntimeWarning)
                    self.step = 1.
            elif self.solver == 'sgd':
                warn('SGD step needs to be tuned manually', RuntimeWarning)
                self.step = 1.

        # Determine the range of the prox
        # User cannot specify a custom range if he is using learners
        if fit_intercept:
            # Don't penalize the intercept (intercept is the last coeff)
            prox_obj.range = (0, model_obj.n_coeffs - 1)
        else:
            prox_obj.range = (0, model_obj.n_coeffs)

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = None
        if self.warm_start and self.weights is not None:
            if self.fit_intercept and self.intercept is not None:
                coeffs = np.hstack((self.weights, self.intercept))
            else:
                coeffs = self.weights
            # ensure starting point has the right format
            if coeffs is not None and coeffs.shape == (model_obj.n_coeffs,):
                coeffs_start = coeffs
            else:
                raise ValueError('Cannot warm start, coeffs don\'t have the '
                                 'right shape')

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start)

        # Get the learned coefficients
        if fit_intercept:
            self._set("weights", coeffs[:-1])
            self._set("intercept", coeffs[-1])
        else:
            self._set("weights", coeffs)
            self._set("intercept", None)
        self._set("_fitted", True)
        return self

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
        output : `LearnerGLM`
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
