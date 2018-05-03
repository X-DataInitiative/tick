


import numpy as np

def extract_dtype(dtype_or_object_with_dtype):
    import six
    should_update_prox = False
    local_dtype = None
    if (isinstance(dtype_or_object_with_dtype, six.string_types)
            or isinstance(dtype_or_object_with_dtype, np.dtype)):
        local_dtype = np.dtype(dtype_or_object_with_dtype)
    elif hasattr(dtype_or_object_with_dtype, 'dtype'):
        local_dtype = np.dtype(dtype_or_object_with_dtype.dtype)
    else:
        raise ValueError(("""
         unsupported type used for prox creation,
         expects dtype or class with dtype , type:
         """ + clazz.__class__.__name__).strip())
    return local_dtype

def get_typed_class(clazz, dtype_or_object_with_dtype, dtype_map):

    clazz.dtype = extract_dtype(dtype_or_object_with_dtype)
    if np.dtype(clazz.dtype) not in dtype_map:
        raise ValueError("""dtype does not exist in
          type map for """ + clazz.__class__.__name__.strip())
    return dtype_map[np.dtype(clazz.dtype)]

def copy_with(clazz, ignore_fields : dict = None):
    """Copies clazz, temporarily sets values to None to avoid copying
       not thread safe
    """
    from copy import deepcopy

    fields = {}
    for field in ignore_fields:
        if hasattr(clazz, field) and getattr(clazz, field) is not None:
            fields[field] = getattr(clazz, field)
            clazz._set(field, None)

    new_clazz = deepcopy(clazz)

    for field in fields:
        clazz._set(field, fields[field])

    return new_clazz
