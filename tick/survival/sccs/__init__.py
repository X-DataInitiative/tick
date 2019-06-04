# License: BSD 3 clause

import tick.base

from .batch_convolutional_sccs import BatchConvSCCS
from .stream_convolutional_sccs import StreamConvSCCS

__all__ = [
    "BatchConvSCCS", "StreamConvSCCS"
]
