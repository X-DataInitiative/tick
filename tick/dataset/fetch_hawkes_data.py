# License: BSD 3 clause

from tick.dataset import fetch_tick_dataset


def fetch_hawkes_bund_data():
    """Load Hawkes formatted bund data from
    https://github.com/X-DataInitiative/tick-datasets/tree/master/hawkes/bund

    This data is meant to be fitted with Hawkes processes. It contains for each
    day 4 time series representing:

        1. Mid-price movement up
        2. Mid-price movement down
        3. Buyer initiated trades that do not move the mid-price
        4. Seller initiated trades that do not move the mid-price

    Returns
    -------
    output : `list` of `list` of `np.ndarray`, dim=(20, 4, _)
        List of 20 days of 4 timestamps data.
    """
    dataset = 'hawkes/bund/bund.npz'
    return [list(timestamps) for _, timestamps in fetch_tick_dataset(dataset)]
