# License: BSD 3 clause

from abc import abstractmethod
from . import Model
import numpy as np

__author__ = 'Stephane Gaiffas'


class ModelSelfConcordant(Model):
    """An abstract base class for a model that implements the
    self-concordant constant

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    def __init__(self, dtype=np.float64):
        Model.__init__(self, dtype=dtype)

    @property
    def _sc_constant(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using "
                             "``sc_constant``")
        return self._get_sc_constant()

    @abstractmethod
    def _get_sc_constant(self) -> float:
        pass
