# License: BSD 3 clause

from abc import ABC, abstractmethod
from tick.base import Base
from multiprocessing import cpu_count


class LongitudinalPreprocessor(ABC, Base):
    """An abstract base class for a longitudinal data preprocessors

    Parameters
    ----------
    n_jobs : `int`, default=-1
        Number of tasks to run in parallel. If set to -1, the number of tasks is
        set to the number of cores.
    """

    _attrinfos = {'n_jobs': {'writable': True}}

    def __init__(self, n_jobs=1):
        Base.__init__(self)
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    @abstractmethod
    def fit(self, features, labels, censoring) -> None:
        pass

    @abstractmethod
    def transform(self, features, labels, censoring) -> tuple:
        pass

    def fit_transform(self, features, labels=None, censoring=None) -> tuple:
        self.fit(features, labels, censoring)
        return self.transform(features, labels, censoring)
