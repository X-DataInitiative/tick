# License: BSD 3 clause

import tick.base

from .logreg import ModelLogReg
from .linreg import ModelLinReg
from .hinge import ModelHinge
from .smoothed_hinge import ModelSmoothedHinge
from .quadratic_hinge import ModelQuadraticHinge
from .poisreg import ModelPoisReg

__all__ = ["ModelLinReg",
           "ModelLogReg",
           "ModelPoisReg",
           'ModelHinge',
           'ModelSmoothedHinge',
           'ModelQuadraticHinge',
           ]
