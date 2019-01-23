# License: BSD 3 clause
import os
import tarfile

import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file

from tick.dataset.download_helper import download_dataset, get_data_home

dataset_path = 'url/url_svmlight.tar.gz'
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'


def load_url_dataset_day(cache_path, days):
    tar_file = tarfile.open(cache_path, "r:gz")

    n_features = 3231961

    X, y = None, None

    for day in days:
        data_filename = 'url_svmlight/Day{}.svm'.format(day)
        with tar_file.extractfile(data_filename) as data_file:
            X_day, y_day = load_svmlight_file(data_file, n_features=n_features)

        if X is None:
            X, y = X_day, y_day
        else:
            X = scipy.sparse.vstack((X, X_day))
            y = np.hstack((y, y_day))

    return X, y


def fetch_url_dataset(n_days=120, data_home=None, verbose=True):
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)

    dataset = None
    print(cache_path)
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


def download_url_dataset(data_home=None, verbose=False):
    download_dataset(dataset_url, dataset_path, data_home=data_home,
                     verbose=verbose)