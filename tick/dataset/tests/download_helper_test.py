# License: BSD 3 clause

import os
import unittest
import socket

from tick.dataset.download_helper import fetch_tick_dataset, clear_dataset, \
    get_data_home


def is_connected():
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        with socket.create_connection(("www.google.com", 80)):
            return True
    except OSError:
        pass
    return False


class Test(unittest.TestCase):
    def test_fetch_tick_dataset(self):
        """...Test dataset is correctly downloaded or fetched from cache
        """

        if is_connected():

            dataset_path = "binary/adult/adult.tst.bz2"

            # start with no cache
            clear_dataset(dataset_path)

            # download dataset from github repo
            features, labels = fetch_tick_dataset(dataset_path, verbose=False)
            self.assertEqual(features.shape, (16281, 123))
            self.assertEqual(labels.shape, (16281,))

            cache_path = os.path.join(get_data_home(), dataset_path)
            file_modification_time_1 = os.path.getmtime(cache_path)

            # fetch dataset from cache
            features, labels = fetch_tick_dataset(dataset_path)
            self.assertEqual(features.shape, (16281, 123))
            self.assertEqual(labels.shape, (16281,))

            # ensure the same file has been used
            file_modification_time_2 = os.path.getmtime(cache_path)
            self.assertEqual(file_modification_time_1,
                             file_modification_time_2)


if __name__ == "__main__":
    unittest.main()
