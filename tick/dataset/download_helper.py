# License: BSD 3 clause

"""Helper to download and cache datasets from the tick_datasets github
repository

Inspired from
https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/datasets/twenty_newsgroups.py
"""
import logging
from urllib.request import urlopen

import os

import shutil

import numpy as np
from sklearn.datasets import load_svmlight_file
import math

logger = logging.getLogger(__name__)

BASE_URL = ("https://raw.githubusercontent.com/X-DataInitiative/tick-datasets"
            "/master/%s")


def download_tick_dataset(dataset_path, data_home=None, verbose=True):
    """Downloads dataset from tick_datasets github repository and store it
    locally

    Parameters
    ----------
    dataset_path : `str`
        Dataset path on tick_datasets github repository. For example
        "binary/adult/adult.trn.bz2" for adult train dataset

    data_home : `str`, optional, default=None
        Specify a download and cache folder for the datasets. If None,
        all tick datasets are stored in '~/tick_datasets' subfolders.

    verbose : `bool`, default=True
        If True, download progress bar will be printed

    Returns
    -------
    cache_path : `str`
        File path of the downloaded data
    """
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)
    cache_dir = os.path.dirname(cache_path)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_url = BASE_URL % dataset_path
    if verbose:
        logger.warning("Downloading dataset from %s", file_url)
    opener = urlopen(file_url)
    chunk_size = 4096
    with open(cache_path, 'wb') as f:
        n_chunks = 0
        file_size = opener.length
        while True:
            data = opener.read(chunk_size)
            if data:
                percent = chunk_size * n_chunks / file_size
                if verbose:
                    progress_bar(percent, length=file_size)
                f.write(data)
                n_chunks += 1
            else:
                if verbose:
                    progress_bar(1, length=file_size)
                break

    return cache_path


def fetch_tick_dataset(dataset_path, data_home=None, verbose=True):
    """Fetch dataset from tick_datasets github repository.
     
    Uses cache if this dataset has already been downloaded.

    Parameters
    ----------
    dataset_path : `str`
        Dataset path on tick_datasets github repository. For example
        "binary/adult/adult.trn.bz2" for adult train dataset

    data_home : `str`, optional, default=None
        Specify a download and cache folder for the datasets. If None,
        all tick datasets are stored in '~/tick_datasets' subfolders.

    verbose : `bool`, default=True
        If True, download progress bar will be printed

    Returns
    -------
    output : `np.ndarray` or `dict` or `tuple`
        Dataset. Its format will depend on queried dataset.
    """
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)

    dataset = None
    if os.path.exists(cache_path):
        try:
            dataset = load_dataset(dataset_path, data_home=data_home)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if dataset is None:
        download_tick_dataset(dataset_path, data_home=data_home,
                              verbose=verbose)
        dataset = load_dataset(dataset_path, data_home=data_home)

    return dataset


def load_dataset(dataset_path, data_home=None):
    """Load dataset from given path

    Parameters
    ----------
    dataset_path : `str`
        Dataset relative path

    data_home : `str`, optional, default=None
        Specify a download and cache folder for the datasets. If None,
        all tick datasets are stored in '~/tick_datasets' subfolders.

    Returns
    -------
    output : `np.ndarray` or `dict` or `tuple`
        Dataset. Its format will depend on queried dataset.
    """
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)

    if cache_path.endswith(".npz"):
        dataset = np.load(cache_path)
        # If we have only one numpy array we return it directly otherwise
        # we return the row dictionary
        if len(dataset.keys()) == 1:
            key_0 = dataset.keys()[0]
            dataset = dataset[key_0]
        else:
            dataset = dataset.items()
    else:
        dataset = load_svmlight_file(cache_path)

    return dataset


def get_data_home(data_home=None):
    """Return the path of the tick data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.
    By default the data dir is set to a folder named 'tick_datasets'
    in the user home folder.
    Alternatively, it can be set by the 'TICK_DATASETS' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = os.environ.get('TICK_DATASETS',
                                   os.path.join('~', 'tick_datasets'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_dataset(dataset_path, data_home=None):
    """Clear dataset from cache folder
    """
    data_home = get_data_home(data_home)
    cache_path = os.path.join(data_home, dataset_path)

    if os.path.exists(cache_path):
        os.remove(cache_path)


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def convert_size(size_bytes):
    """Convert raw bytes into human readable size

    References
    ----------
    http://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    """
    if size_bytes == 0:
        return '0B'
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '%s %s' % (s, size_name[i])


def progress_bar(progress, width=40, length=None):
    """Print progress bar to sys.stdout

    Parameters
    ----------
    progress : `float`
        Reached progress between 0 (just started) and 1 (finished)

    width : `int`
        Total width of the progress bar

    length : `int`
        Size in bytes of the downloaded file
    """
    if length:
        size = "(%s)" % convert_size(length)
    else:
        size = ''

    n_bars = int(progress * width)
    bar = "[%s%s]" % ("=" * n_bars, " " * (width - n_bars))
    print("\r%s %s %2d%%" % (size, bar, progress * 100), flush=True, end="")
    if progress == 1:
        print()
