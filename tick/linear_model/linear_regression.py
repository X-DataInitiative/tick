# License: BSD 3 clause

from tick.base import actual_kwargs
from tick.base.learner import LearnerGLM
from .model_linreg import ModelLinReg

class LinearRegression(LearnerGLM):
    """
    Linear regression learner, with many choices of penalization and
    solvers.

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : {'none', 'l1', 'l2', 'elasticnet', 'tv'}, default='l2'
        The penalization to use. Default is ridge penalization

    solver : {'gd', 'agd', 'svrg'}, default='svrg'
        The name of the solver to use

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used in fista, sgd and svrg
        solvers

    tol : `float`, default=1e-5
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

    random_state : `int` seed, `RandomState` instance, or `None` (default)
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
    weights : `np.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    intercept : `float` or `None`
        The intercept, if ``fit_intercept=True``, otherwise `None`
    """

    _solvers = {'gd': 'GD', 'agd': 'AGD', 'svrg': 'SVRG'}

    _attrinfos = {"_actual_kwargs": {"writable": False}}

    @actual_kwargs
    def __init__(self, fit_intercept=True, penalty='l2', C=1e3, solver='svrg',
                 step=None, tol=1e-5, max_iter=100, verbose=False,
                 warm_start=False, print_every=10, record_every=1,
                 elastic_net_ratio=0.95, random_state=None, blocks_start=None,
                 blocks_length=None):

        self._actual_kwargs = LinearRegression.__init__.actual_kwargs
        LearnerGLM.__init__(
            self, fit_intercept=fit_intercept, penalty=penalty, C=C,
            solver=solver, step=step, tol=tol, max_iter=max_iter,
            verbose=verbose, warm_start=warm_start, print_every=print_every,
            record_every=record_every, elastic_net_ratio=elastic_net_ratio,
            random_state=random_state, blocks_start=blocks_start,
            blocks_length=blocks_length)

    def _construct_model_obj(self, fit_intercept=True):
        return ModelLinReg(fit_intercept)

    def predict(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix to predict for.

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Returns predicted values.
        """
        if not self._fitted:
            raise ValueError("You must call ``fit`` before")
        else:
            X = self._safe_array(X, dtype=X.dtype)
            z = X.dot(self.weights)
            if self.intercept:
                z += self.intercept
            return z

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the fitted linear
        regression model, computed on the given features matrix and labels.

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix.

        y : `np.ndarray`, shape = (n_samples,)
            Labels vector.

        Returns
        -------
        score : `float`
            R^2 of self.predict(X) against y
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
