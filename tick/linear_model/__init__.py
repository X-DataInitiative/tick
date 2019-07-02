# License: BSD 3 clause

import tick.base
import tick.base_model.build.base_model

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .poisson_regression import PoissonRegression

from .model_linreg import ModelLinReg
from .model_logreg import ModelLogReg
from .model_hinge import ModelHinge
from .model_smoothed_hinge import ModelSmoothedHinge
from .model_quadratic_hinge import ModelQuadraticHinge
from .model_poisreg import ModelPoisReg

from .simu_linreg import SimuLinReg
from .simu_logreg import SimuLogReg
from .simu_poisreg import SimuPoisReg

__all__ = [
    'LinearRegression', 'LogisticRegression', 'LogisticRegression',
    'ModelLinReg', 'ModelLogReg', 'ModelPoisReg', 'ModelHinge',
    'ModelSmoothedHinge', 'ModelQuadraticHinge', 'SimuLinReg', 'SimuLogReg',
    'SimuPoisReg'
]
