# License: BSD 3 clause

from inspect import signature


def actual_kwargs(function):
    """
    Decorator that provides the wrapped function with an attribute
    'actual_kwargs'
    containing just those keyword arguments actually passed in to the function.

    References
    ----------
    http://stackoverflow.com/questions/1408818/getting-the-the-keyword
    -arguments-actually-passed-to-a-python-method

    Notes
    -----
    We override the signature of the decorated function to ensure it will be
    displayed correctly in sphinx
    """

    original_signature = signature(function)

    def inner(*args, **kwargs):
        inner.actual_kwargs = kwargs
        return function(*args, **kwargs)

    inner.__signature__ = original_signature

    return inner
