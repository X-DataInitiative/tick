# -*- coding: utf-8 -*-
"""tick module
"""
# License: BSD 3 clause

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tick")
except PackageNotFoundError:
    __version__ = "unknown"

import tick.base
