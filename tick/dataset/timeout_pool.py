# License: BSD 3 clause
import multiprocessing
import time
from functools import partial
from multiprocessing import util
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import Pool

import numpy as np

# util.get_logger().setLevel(util.DEBUG)
from scipy.sparse import csr_matrix


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print_args = [arg for arg in args
                      if not isinstance(arg, np.ndarray)
                      and not isinstance(arg, csr_matrix)]
        print("Aborting due to timeout {} for {}"
              .format(timeout, print_args))
        p.terminate()
        # raise


def apply_async_with_timeout(pool, func, args, timeout=None):
    async_results = []
    for arg in args:
        abortable_func = partial(abortable_worker, func, timeout=timeout)
        async_results += [pool.apply_async(abortable_func, arg)]
    pool.close()

    results = [result.get() for result in async_results]
    return results


if __name__ == '__main__':
    def heavy_worker(work_time):
        time.sleep(work_time)
        return work_time


    pool = Pool(3)
    args = [(1,), (4,), (1.5,)]
    print(apply_async_with_timeout(pool, heavy_worker, args, timeout=2))
