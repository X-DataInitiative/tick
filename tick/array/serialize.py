# License: BSD 3 clause

import os

import numpy as np
import scipy

from tick.array.build.array import (
    tick_float_array_to_file,
    tick_float_array2d_to_file,
    tick_float_sparse2d_to_file,
    tick_double_array_to_file,
    tick_double_array2d_to_file,
    tick_double_sparse2d_to_file,
    tick_float_array_from_file,
    tick_float_array2d_from_file,
    tick_float_sparse2d_from_file,
    tick_double_array_from_file,
    tick_double_array2d_from_file,
    tick_double_sparse2d_from_file,
    tick_float_colmaj_sparse2d_to_file,
    tick_float_colmaj_sparse2d_from_file,
    tick_double_colmaj_sparse2d_to_file,
    tick_double_colmaj_sparse2d_from_file,
    tick_double_colmaj_array2d_to_file,
    tick_double_colmaj_array2d_from_file,
    tick_float_colmaj_array2d_to_file,
    tick_float_colmaj_array2d_from_file,
)


def serialize_array(array, filepath):
    """Save an array on disk on a format that tick C++ modules can read

    This method is intended to be used by developpers only, mostly for
    benchmarking in C++ on real datasets imported from Python

    Parameters
    ----------
    array : `np.ndarray` or `scipy.sparse.csr_matrix`
        1d or 2d array

    filepath : `str`
        Path where the array will be stored

    Returns
    -------
    path : `str`
        Global path of the serialized array
    """
    if array.dtype not in [np.float32, np.float64]:
        raise ValueError('Only float32/64 arrays can be serrialized')

    if array.dtype == "float32":
        if isinstance(array, np.ndarray):
            if len(array.shape) == 1:
                serializer = tick_float_array_to_file
            elif len(array.shape) == 2:
                if array.flags['F_CONTIGUOUS']:
                    serializer = tick_float_colmaj_array2d_to_file
                else:
                    serializer = tick_float_array2d_to_file
            else:
                raise ValueError('Only 1d and 2d arrays can be serrialized')
        else:
            if len(array.shape) == 2:
                if isinstance(array, scipy.sparse.csc.csc_matrix):
                    serializer = tick_float_colmaj_sparse2d_to_file
                else:
                    serializer = tick_float_sparse2d_to_file
            else:
                raise ValueError('Only 2d sparse arrays can be serrialized')
    elif array.dtype == "float64" or array.dtype == "double":
        if isinstance(array, np.ndarray):
            if len(array.shape) == 1:
                serializer = tick_double_array_to_file
            elif len(array.shape) == 2:
                if array.flags['F_CONTIGUOUS']:
                    serializer = tick_double_colmaj_array2d_to_file
                else:
                    serializer = tick_double_array2d_to_file
            else:
                raise ValueError('Only 1d and 2d arrays can be serrialized')
        else:
            if len(array.shape) == 2:
                if isinstance(array, scipy.sparse.csc.csc_matrix):
                    serializer = tick_double_colmaj_sparse2d_to_file
                else:
                    serializer = tick_double_sparse2d_to_file
            else:
                raise ValueError('Only 2d sparse arrays can be serrialized')
    else:
        raise ValueError('Unhandled serrialization type')

    serializer(filepath, array)
    return os.path.abspath(filepath)


def load_array(filepath, array_type='dense', array_dim=1, dtype="float64",
               major="row"):
    """Loaf an array from disk from a format that tick C++ modules can read

    This method is intended to be used by developpers only, mostly for
    benchmarking in C++ on real datasets imported from Python

    Parameters
    ----------
    filepath : `str`
        Path where the array was stored

    array_type : {'dense', 'sparse'}, default='dense'
        Expected type of the array

    array_dim : `int`
        Expected dimension of the array

    dtype : {'float64', 'float32'}
        Number type of the array

    major : {'row', 'col'}
        Used to associate correct templated C++ class

    Returns
    -------
    array : `np.ndarray` or `scipy.sparse.csr_matrix`
        1d or 2d array
    """
    abspath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError('File {} does not exists'.format(abspath))

    if dtype == "float32":
        if array_type == 'dense':
            if array_dim == 1:
                reader = tick_float_array_from_file
            elif array_dim == 2:
                if major == "col":
                    reader = tick_float_colmaj_array2d_from_file
                else:
                    reader = tick_float_array2d_from_file
            else:
                raise ValueError('Only 1d and 2d arrays can be loaded')
        elif array_type == 'sparse':
            if array_dim == 2:
                if major == "col":
                    reader = tick_float_colmaj_sparse2d_from_file
                else:
                    reader = tick_float_sparse2d_from_file
            else:
                raise ValueError('Only 2d sparse arrays can be loaded')
        else:
            raise ValueError('Cannot load this class of array')
    elif dtype == "float64" or dtype == "double":
        if array_type == 'dense':
            if array_dim == 1:
                reader = tick_double_array_from_file
            elif array_dim == 2:
                if major == "col":
                    reader = tick_double_colmaj_array2d_from_file
                else:
                    reader = tick_double_array2d_from_file
            else:
                raise ValueError('Only 1d and 2d arrays can be loaded')
        elif array_type == 'sparse':
            if array_dim == 2:
                if major == "col":
                    reader = tick_double_colmaj_sparse2d_from_file
                else:
                    reader = tick_double_sparse2d_from_file
            else:
                raise ValueError('Only 2d sparse arrays can be loaded')
        else:
            raise ValueError('Cannot load this class of array')
    else:
        raise ValueError('Unhandled serrialization type')

    return reader(filepath)
