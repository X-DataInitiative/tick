import numpy as np

from tick.base import Base


def extract_dtype(dtype_or_object_with_dtype):
    import six
    if (isinstance(dtype_or_object_with_dtype, six.string_types)
            or isinstance(dtype_or_object_with_dtype, np.dtype)):
        return np.dtype(dtype_or_object_with_dtype)
    elif hasattr(dtype_or_object_with_dtype, 'dtype'):
        return np.dtype(dtype_or_object_with_dtype.dtype)
    else:
        raise ValueError("unsupported type used for prox creation, "
                         "expects dtype or class with dtype , type: {}".format(
                             dtype_or_object_with_dtype.__class__.__name__))


def get_typed_class(clazz, dtype_or_object_with_dtype, dtype_map):
    clazz.dtype = extract_dtype(dtype_or_object_with_dtype)
    if np.dtype(clazz.dtype) not in dtype_map:
        raise ValueError("dtype does not exist in type map for {}".format(
            clazz.__class__.__name__))
    return dtype_map[np.dtype(clazz.dtype)]


def copy_with(clazz, ignore_fields: list = None):
    """Copies clazz, temporarily sets values to None to avoid copying.
       not thread safe
    """
    from copy import deepcopy

    if not isinstance(clazz, Base):
        raise ValueError("Only objects inheriting from Base class should be"
                         "copied with copy_with.")

    fields = {}
    for field in ignore_fields:
        if hasattr(clazz, field) and getattr(clazz, field) is not None:
            fields[field] = getattr(clazz, field)
            clazz._set(field, None)

    new_clazz = deepcopy(clazz)

    for field in fields:
        clazz._set(field, fields[field])

    return new_clazz
