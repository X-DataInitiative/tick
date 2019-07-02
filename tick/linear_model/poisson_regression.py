# License: BSD 3 clause

import numpy as np

from tick.base import actual_kwargs
from tick.base.learner import LearnerGLM
from .model_poisreg import ModelPoisReg

class PoissonRegression(LearnerGLM):
    """Poisson regression learner, with exponential link function.
    It supports several solvers and several penalizations.
    Note that for this model, there is no way to tune
    automatically the `step` of the solver. Thus, the default for `step`
    might work, or not, so that several values should be tried out.

    Parameters
    ----------
    step : `float`, default=1e-3
        Step-size to be used for the solver. For Poisson regression there is no
        way to tune it automatically. The default tuning might work, or not...

    C : `float`, default=1e3
        Level of penalization

    penalty : {'l1', 'l2', 'elasticnet', 'tv'}, default='l2'
        The penalization to use. Default is ridge penalization

    solver : {'gd', 'agd', 'bfgs', 'svrg', 'sgd'}, default='svrg'
        The name of the solver to use

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    tol : `float`, default=1e-6
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=False
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=1
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
        Used in 'elasticnet' penalty

    random_state : `int` seed, RandomState instance, or None (default)
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
    weights : `numpy.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    intercept : `float` or None
        The intercept, if ``fit_intercept=True``, otherwise `None`
    """

    _solvers = {
        'gd': 'GD',
        'agd': 'AGD',
        'sgd': 'SGD',
        'svrg': 'SVRG',
        'bfgs': 'BFGS',
    }

    _attrinfos = {"_actual_kwargs": {"writable": False}}

    @actual_kwargs
    def __init__(self, step=1e-3, fit_intercept=True, penalty='l2', C=1e3,
                 tol=1e-5, max_iter=100, solver='svrg', verbose=False,
                 warm_start=False, print_every=10, record_every=1,
                 elastic_net_ratio=0.95, random_state=None, blocks_start=None,
                 blocks_length=None):

        self._actual_kwargs = PoissonRegression.__init__.actual_kwargs

        # If 'bfgs' is specified then there is no step. Solver can't be changed
        # after creation of the objected, so hard-replacing step is okay.
        if solver == 'bfgs':
            step = None

        LearnerGLM.__init__(
            self, step=step, fit_intercept=fit_intercept, penalty=penalty, C=C,
            solver=solver, tol=tol, max_iter=max_iter, verbose=verbose,
            warm_start=warm_start, print_every=print_every,
            record_every=record_every, elastic_net_ratio=elastic_net_ratio,
            random_state=random_state, blocks_start=blocks_start,
            blocks_length=blocks_length)

    def _construct_model_obj(self, fit_intercept=True):
        return ModelPoisReg(fit_intercept=fit_intercept, link='exponential')

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
        self : `PoissonRegression`
            The fitted instance of the model
        """
        return LearnerGLM.fit(self, X, y)

    def loglik(self, X, y):
        """Compute the minus log-likelihood of the model, using the given
        features matrix and labels, with the intercept and model weights
        currently fitted in the object.
        Minus log-likelihood is computed, so that smaller is better.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`,, shape=(n_samples, n_features)
            Features matrix

        y : `np.array`, shape=(n_samples,)
            Labels vector relative to X

        Returns
        -------
        output : `float`
            Value of the minus log-likelihood
        """
        if not self._fitted:
            raise ValueError("You must call ``fit`` before")
        else:
            fit_intercept = self.fit_intercept
            model = self._construct_model_obj(fit_intercept)
            coeffs = self.weights
            if fit_intercept:
                coeffs = np.append(coeffs, self.intercept)
            return model.fit(X, y).loss(coeffs)

    def decision_function(self, X):
        """Decision function for given samples. This is simply given by
        the predicted means of each sample.
        Predicted mean in this model for a features vector x is simply given by
        exp(x.dot(weights) + intercept)

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Value of the decision function of each sample points
        """
        if not self._fitted:
            raise ValueError("You must call ``fit`` before")
        else:
            X = self._safe_array(X, dtype=X.dtype)
            z = X.dot(self.weights)
            if self.intercept:
                z += self.intercept
            return np.exp(z)

    def predict(self, X):
        """Predict label for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Returns predicted labels
        """
        return np.rint(self.decision_function(X))
