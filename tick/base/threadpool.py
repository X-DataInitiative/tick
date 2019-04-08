# License: BSD 3 clause

import threading
import queue


class ThreadPool(object):
    def __init__(self, with_lock=False, max_threads=8):
        object.__init__(self)
        self._Qin = queue.Queue()
        self._max_threads = max_threads
        self._n_works = 0
        self._Qerr = queue.Queue()
        if (with_lock): self.lock = threading.Lock()
        else: self.lock = None

    def add_work(self, callable, *args, **kwargs):
        self._n_works += 1
        self._Qin.put(
            dict(callable=callable, args=args, kwargs=kwargs, me=self))

    def _worker(self, n):
        while True:
            try:
                work = self._Qin.get(False)
                work['callable'](*work['args'], **work['kwargs'])
                self._Qin.task_done()
            except queue.Empty:
                break
        #            except:
        #        print "merde"
        #self._Qerr.put("Error");


#break

    def start(self):
        for n in range(min(self._n_works, self._max_threads)):
            t = threading.Thread(target=self._worker, args=(n,))
            t.start()

        self._Qin.join()
