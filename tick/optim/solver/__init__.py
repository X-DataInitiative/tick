import tick.base

from .gd import GD
from .agd import AGD
from .bfgs import BFGS
from .scpg import SCPG
from .sgd import SGD
from .svrg import SVRG
from .sdca import SDCA
from .gfb import GFB
from .adagrad import AdaGrad

__all__ = ["GD", "AGD", "BFGS", "SCPG", "SGD", "SVRG", "SDCA", "GFB",
           "AdaGrad"]
