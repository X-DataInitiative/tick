#!/usr/bin/env python

# Convenience script to extract indices type while compiling

try:
    import numpy as np
    from scipy.sparse import sputils

    sparsearray_type = sputils.get_index_dtype()

    if sparsearray_type == np.int64:
        print("-DTICK_SPARSE_INDICES_INT64")
    else:
    	print("-DTICK_SPARSE_INDICES_INT32")
except ImportError as e:
    if is_building_tick and numpy_available:
        print(e)
        warnings.warn("scipy is not installed, unable to determine "
                      "sparse array integer type (assuming 32 bits)\n")