# License: BSD 3 clause

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from tick.base import Base
from tick.prox import ProxZero, ProxL1, ProxL2Sq, ProxElasticNet, \
    ProxTV, ProxBinarsity
from tick.preprocessing.utils import safe_array

class LearnerOptim(ABC, Base):
    """Learner for all models that are inferred with a `tick.solver`
    and a `tick.prox`
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : 'none', 'l1', 'l2', 'elasticnet', 'tv', 'binarsity', default='l2'
        The penalization to use. Default 'l2', namely is ridge penalization.

    solver : 'gd', 'agd', 'bfgs', 'svrg', 'sdca'
        The name of the solver to use

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
    """


    _attrinfos = {
        "solver": {
            "writable": False
        },
        "_solver_obj": {
            "writable": False
        },
        "penalty": {
            "writable": False
        },
        "_prox_obj": {
            "writable": False
        },
        "_model_obj": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        },
        "_C": {
            "writable": False
        },
        "random_state": {
            "writable": False
        },
        "_warm_start": {
            "writable": False
        },
        "_actual_kwargs": {
            "writable": False
        },
    }

    _solvers = {
        'gd': 'GD',
        'agd': 'AGD',
        'sgd': 'SGD',
        'svrg': 'SVRG',
        'bfgs': 'BFGS',
        'sdca': 'SDCA'
    }
    _solvers_with_linesearch = ['gd', 'agd']
    _solvers_with_step = ['gd', 'agd', 'svrg', 'sgd']
    _solvers_stochastic = ['sgd', 'svrg', 'sdca']
    _penalties = {
        'none': ProxZero,
        'l1': ProxL1,
        'l2': ProxL2Sq,
        'elasticnet': ProxElasticNet,
        'tv': ProxTV,
        'binarsity': ProxBinarsity
    }

    def __init__(self, penalty='l2', C=1e3, solver="svrg", step=None, tol=1e-5,
                 max_iter=100, verbose=True, warm_start=False, print_every=10,
                 record_every=10, sdca_ridge_strength=1e-3,
                 elastic_net_ratio=0.95, random_state=None, blocks_start=None,
                 blocks_length=None, extra_model_kwargs=None,
                 extra_prox_kwarg=None):

        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}

        # Construct the model
        if extra_model_kwargs is None:
            extra_model_kwargs = {}
        self._model_obj = self._construct_model_obj(**extra_model_kwargs)

        # Construct the solver. The solver is created at creation of the
        # learner, and cannot be instantiated again (using another solver type)
        # afterwards.
        self.solver = solver
        self._set_random_state(random_state)
        self._solver_obj = self._construct_solver_obj(
            solver, step, max_iter, tol, print_every, record_every, verbose,
            sdca_ridge_strength)

        # Construct the prox. The prox is created at creation of the
        # learner, and cannot be instantiated again (using another prox type)
        # afterwards.
        self.penalty = penalty
        if extra_prox_kwarg is None:
            extra_prox_kwarg = {}
        self._prox_obj = self._construct_prox_obj(penalty, elastic_net_ratio,
                                                  blocks_start, blocks_length,
                                                  extra_prox_kwarg)

        # Set C after creating prox to set prox strength
        if 'C' in self._actual_kwargs or penalty != 'none':
            # Print self.C = C
            self.C = C

        self.record_every = record_every
        self.step = step
        self._fitted = False
        self.warm_start = warm_start

        if 'sdca_ridge_strength' in self._actual_kwargs or solver == 'sdca':
            self.sdca_ridge_strength = sdca_ridge_strength

        if 'elastic_net_ratio' in self._actual_kwargs or \
                        penalty == 'elasticnet':
            self.elastic_net_ratio = elastic_net_ratio

        if 'blocks_start' in self._actual_kwargs or penalty == 'binarsity':
            self.blocks_start = blocks_start

        if 'blocks_length' in self._actual_kwargs or penalty == 'binarsity':
            self.blocks_length = blocks_length

    @abstractmethod
    def _construct_model_obj(self, **kwargs):
        pass

    def _construct_solver_obj(self, solver, step, max_iter, tol, print_every,
                              record_every, verbose, sdca_ridge_strength):
        # Parameters of the solver
        from tick.solver import AGD, GD, BFGS, SGD, SVRG, SDCA
        solvers = {
            'AGD': AGD,
            'BFGS': BFGS,
            'GD': GD,
            'SGD': SGD,
            'SVRG': SVRG,
            'SDCA': SDCA
        }
        solver_args = []
        solver_kwargs = {
            'max_iter': max_iter,
            'tol': tol,
            'print_every': print_every,
            'record_every': record_every,
            'verbose': verbose
        }

        allowed_solvers = list(self._solvers.keys())
        allowed_solvers.sort()
        if solver not in self._solvers:
            raise ValueError("``solver`` must be one of %s, got %s" %
                             (', '.join(allowed_solvers), solver))
        else:
            if solver in self._solvers_with_step:
                solver_kwargs['step'] = step
            if solver in self._solvers_stochastic:
                solver_kwargs['seed'] = self._seed
            if solver == 'sdca':
                solver_args += [sdca_ridge_strength]

            solver_obj = solvers[self._solvers[solver]](*solver_args, **solver_kwargs)

        return solver_obj

    def _construct_prox_obj(self, penalty, elastic_net_ratio, blocks_start,
                            blocks_length, extra_prox_kwarg):
        # Parameters of the penalty
        penalty_args = []

        allowed_penalties = list(self._penalties.keys())
        allowed_penalties.sort()
        if penalty not in allowed_penalties:
            raise ValueError("``penalty`` must be one of %s, got %s" %
                             (', '.join(allowed_penalties), penalty))

        else:
            if penalty != 'none':
                # strength will be set by setting C afterwards
                penalty_args += [0]
            if penalty == 'elasticnet':
                penalty_args += [elastic_net_ratio]
            if penalty == 'binarsity':
                if blocks_start is None:
                    raise ValueError(
                        "Penalty '%s' requires ``blocks_start``, got %s" %
                        (penalty, str(blocks_start)))
                elif blocks_length is None:
                    raise ValueError(
                        "Penalty '%s' requires ``blocks_length``, got %s" %
                        (penalty, str(blocks_length)))
                else:
                    penalty_args += [blocks_start, blocks_length]

            prox_obj = self._penalties[penalty](*penalty_args,
                                                **extra_prox_kwarg)

        return prox_obj

    @property
    def warm_start(self):
        return self._warm_start

    @warm_start.setter
    def warm_start(self, val):
        if val is True and self.solver == 'sdca':
            raise ValueError('SDCA cannot be warm started')
        self._warm_start = val

    @property
    def max_iter(self):
        return self._solver_obj.max_iter

    @max_iter.setter
    def max_iter(self, val):
        self._solver_obj.max_iter = val

    @property
    def verbose(self):
        return self._solver_obj.verbose

    @verbose.setter
    def verbose(self, val):
        self._solver_obj.verbose = val

    @property
    def tol(self):
        return self._solver_obj.tol

    @tol.setter
    def tol(self, val):
        self._solver_obj.tol = val

    @property
    def step(self):
        if self.solver in self._solvers_with_step:
            return self._solver_obj.step
        else:
            return None

    @step.setter
    def step(self, val):
        if self.solver in self._solvers_with_step:
            self._solver_obj.step = val
        elif val is not None:
            warn('Solver "%s" has no settable step' % self.solver,
                 RuntimeWarning)

    def _set_random_state(self, val):
        if self.solver in self._solvers_stochastic:
            if val is not None and val < 0:
                raise ValueError(
                    'random_state must be positive, got %s' % str(val))
            self.random_state = val
        else:
            if val is not None:
                warn('Solver "%s" has no settable random_state' % self.solver,
                     RuntimeWarning)
            self.random_state = None

    @property
    def _seed(self):
        if self.solver in self._solvers_stochastic:
            if self.random_state is None:
                return -1
            else:
                return self.random_state
        else:
            warn('Solver "%s" has no _seed' % self.solver, RuntimeWarning)

    @property
    def print_every(self):
        return self._solver_obj.print_every

    @print_every.setter
    def print_every(self, val):
        self._solver_obj.print_every = val

    @property
    def record_every(self):
        return self._solver_obj.record_every

    @record_every.setter
    def record_every(self, val):
        self._solver_obj.record_every = val

    @property
    def C(self):
        if self.penalty == 'none':
            return 0
        elif np.isinf(self._prox_obj.strength):
            return 0
        elif self._prox_obj.strength == 0:
            return None
        else:
            return 1. / self._prox_obj.strength

    @C.setter
    def C(self, val):
        if val is None:
            strength = 0.
        elif val <= 0:
            raise ValueError("``C`` must be positive, got %s" % str(val))
        else:
            strength = 1. / val

        if self.penalty != 'none':
            self._prox_obj.strength = strength
        else:
            if val is not None:
                warn('You cannot set C for penalty "%s"' % self.penalty,
                     RuntimeWarning)

    @property
    def elastic_net_ratio(self):
        if self.penalty == 'elasticnet':
            return self._prox_obj.ratio
        else:
            return None

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, val):
        if self.penalty == 'elasticnet':
            self._prox_obj.ratio = val
        else:
            warn(
                'Penalty "%s" has no elastic_net_ratio attribute' %
                self.penalty, RuntimeWarning)

    @property
    def blocks_start(self):
        if self.penalty == 'binarsity':
            return self._prox_obj.blocks_start
        else:
            return None

    @blocks_start.setter
    def blocks_start(self, val):
        if self.penalty == 'binarsity':
            if type(val) is list:
                val = np.array(val, dtype=np.uint64)
            if val.dtype is not np.uint64:
                val = val.astype(np.uint64)
            self._prox_obj.blocks_start = val
        else:
            warn('Penalty "%s" has no blocks_start attribute' % self.penalty,
                 RuntimeWarning)

    @property
    def blocks_length(self):
        if self.penalty == 'binarsity':
            return self._prox_obj.blocks_length
        else:
            return None

    @blocks_length.setter
    def blocks_length(self, val):
        if self.penalty == 'binarsity':
            if type(val) is list:
                val = np.array(val, dtype=np.uint64)
            if val.dtype is not np.uint64:
                val = val.astype(np.uint64)
            self._prox_obj.blocks_length = val
        else:
            warn('Penalty "%s" has no blocks_length attribute' % self.penalty,
                 RuntimeWarning)

    @property
    def sdca_ridge_strength(self):
        if self.solver == 'sdca':
            return self._solver_obj._solver.get_l_l2sq()
        else:
            return None

    @sdca_ridge_strength.setter
    def sdca_ridge_strength(self, val):
        if self.solver == 'sdca':
            self._solver_obj.l_l2sq = val
        else:
            warn(
                'Solver "%s" has no sdca_ridge_strength attribute' %
                self.solver, RuntimeWarning)

    @staticmethod
    def _safe_array(X, dtype="float64"):
        return safe_array(X, dtype)
