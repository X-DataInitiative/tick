# License: BSD 3 clause


def kaplan_meier(timestamps, event_observed):
    """Computes the Kaplan-Meier survival function estimation
    given by:

    .. math::
        S{(t)} = \\prod_{i:t_{(i)}<t}^n \\left(1 - \\frac{d_i}{n_i} \\right)

    where

      - :math:`d_i` are the number of deaths at :math:`t_{i}`
      - :math:`n_i` are the number of patients alive just before :math:`t_{i}`

    Parameters
    ----------
    timestamps : `numpy.array`
        Timestamps for each observation

    event_observed : `numpy.array`
        Bool array denoting if the death event was observed or not

    Returns
    -------
    output : `numpy.array`
        The computed Kaplan-Meier survival function estimation
    """
    import numpy as np

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)

    if isinstance(event_observed, list):
        event_observed = np.array(event_observed)

    timestamps_observed = timestamps[event_observed == 1]
    unique_timestamps_observed = np.concatenate((np.zeros(1), np.unique(timestamps_observed)))

    return np.cumprod(
        np.fromiter((1.0 - np.sum(t == timestamps_observed) / np.sum(t <= timestamps)
                     for t in unique_timestamps_observed), dtype='float',
                    count=unique_timestamps_observed.size))


def nelson_aalen(timestamps, event_observed):
    """Computes the Nelson-Aalen cumulative hazard rate estimation
    given by:

    .. math::
        \\Lambda{(t_{i})} = \\sum_{j=1}^i \\frac{d_j}{n_j}

    where

      - :math:`d_j` are the number of deaths at :math:`t_{j}`
      - :math:`n_j` are the number of patients alive just before :math:`t_{j}`

    Parameters
    ----------
    timestamps : `numpy.array`
        Timestamps for each observation

    event_observed : `numpy.array`
        Bool array denoting if the death event was observed or not

    Returns
    -------
    output : `numpy.array`
        The computed Nelson-Aalen cumulative hazard rate
    """
    import numpy as np

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)

    if isinstance(event_observed, list):
        event_observed = np.array(event_observed)

    timestamps_observed = timestamps[event_observed == 1]
    unique_timestamps_observed = np.concatenate((np.zeros(1), np.unique(timestamps_observed)))

    return np.cumsum(
        np.fromiter((np.sum(t == timestamps_observed) / np.sum(t <= timestamps)
                     for t in unique_timestamps_observed), dtype='float',
                    count=unique_timestamps_observed.size))
