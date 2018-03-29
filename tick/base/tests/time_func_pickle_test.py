# License: BSD 3 clause

import unittest
import numpy as np
import pickle

from tick.base import TimeFunction


class Test(unittest.TestCase):
    def test_pickle(self):
        """...Test TimeFunction's pickling ability
        """
        T = np.array([0.0, 1.0, 2.0])
        Y = np.array([1.0, 0.0, -1.0])

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterLinear, dt=0.2)

        recon = pickle.loads(pickle.dumps(tf))

        self.assertEqual(tf.value(1), recon.value(1))
        self.assertEqual(tf.value(2), recon.value(2))
        self.assertEqual(tf.value(1.5), recon.value(1.5))
        self.assertEqual(tf.value(0.75), recon.value(0.75))


if __name__ == "__main__":
    unittest.main()
