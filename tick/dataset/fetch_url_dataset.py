# License: BSD 3 clause
import os
import tarfile

import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file

from tick.dataset.download_helper import download_dataset, get_data_home

dataset_path = 'url/url_svmlight.tar.gz'
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'
_N_FEATURES = 3231961


def load_url_dataset_day(cache_path, days):
    """Loads url dataset from a tar file

    Parameters
    ----------
    cache_path : `str`
        Path to the tar file

    days : `list` or `range`
        Days to be loaded

    Returns
    -------
    X : `np.ndarray`
        A sparse matrix containing the features

    y : `np.ndarray`
        An array containing the labels
    """
    tar_file = tarfile.open(cache_path, "r:gz")

    X, y = None, None

    for day in days:
        data_filename = 'url_svmlight/Day{}.svm'.format(day)
        with tar_file.extractfile(data_filename) as data_file:
            X_day, y_day = load_svmlight_file(data_file,
                                              n_features=_N_FEATURES)

        if X is None:
            X, y = X_day, y_day
        else:
            X = scipy.sparse.vstack((X, X_day))
            y = np.hstack((y, y_day))

    return X, y


def download_url_dataset(data_home=None, verbose=False):
    """Downloads URL dataset and stores it locally

    Parameters
    ----------
    data_home : `str`, optional, default=None
        Specify a download and cache folder for the datasets. If None
        and not configured with TICK_DATASETS environement variable
        all tick datasets are stored in '~/tick_datasets' subfolders.

    verbose : `bool`, default=True
        If True, download progress bar will be printed

    Returns
    -------
    cache_path : `str`
        File path of the downloaded data
    """
    return download_dataset(dataset_url, dataset_path, data_home=data_home,
                            verbose=verbose)


def fetch_url_dataset(n_days=120, data_home=None, verbose=True):
    """Loads URL dataset

    Uses cache if this dataset has already been downloaded.

    Parameters
    ----------
    data_home : `str`, optional, default=None
        Specify a download and cache folder for the datasets. If None
        and not configured with TICK_DATASETS environement variable
        all tick datasets are stored in '~/tick_datasets' subfolders.

    verbose : `bool`, default=True
        If True, download progress bar will be printed

    Returns
    -------
    X : `np.ndarray`
        A sparse matrix containing the features

    y : `np.ndarray`
        An array containing the labels
    """
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)

    dataset = None
    if os.path.exists(cache_path):
        try:
            dataset = load_url_dataset_day(cache_path, range(n_days))
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if dataset is None:
        download_url_dataset(data_home=data_home, verbose=verbose)
        dataset = load_url_dataset_day(cache_path, range(n_days))

    return dataset
