# License: BSD 3 clause

import numpy as np

from .base import ModelHawkes, ModelFirstOrder, ModelSelfConcordant, \
    LOSS_AND_GRAD
from .build.model import ModelHawkesFixedExpKernLogLikList as \
    _ModelHawkesFixedExpKernLogLik


from tickmodel.build.model import ModelCustomBasic as _ModelCustomBasic

class ModelCustomBasic(ModelFirstOrder):

    _attrinfos = {
        "decay": {
            "cpp_setter": "set_decay"
        },
        "_end_times": {},
        "data": {}
    }

    def __init__(self, decay: float, MaxN_of_f : int, n_threads: int = 1):
        ModelFirstOrder.__init__(self)
        self.decay = decay
        self._model = _ModelCustomBasic(decay, MaxN_of_f, n_threads)
        print(self._model)


    def fit(self, events, global_n, end_times=None):

        self._end_times = end_times
        return ModelFirstOrder.fit(self, events, global_n)

    def _set_data(self, events, global_n):
        #self._set("data", events, global_n)

        end_times = self._end_times
        if end_times is None:
            end_times = max(map(max, events))

        self._model.set_data(events, global_n, end_times)


    def _loss(self, coeffs):
        return self._model.loss(coeffs)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._model.grad(coeffs, out)
        return out

    def _get_n_coeffs(self):
        return self._model.get_n_coeffs()

    @property
    def decays(self):
        return self.decay

    @property
    def _epoch_size(self):
        # This gives the typical size of an epoch when using a
        # stochastic optimization algorithm
        return self.n_jumps

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_jumps

    @property
    def n_jumps(self):
        return self._model.get_n_total_jumps()
