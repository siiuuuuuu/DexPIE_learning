import multiprocessing as mp
import time

class LIFOQueue:
    """Inter-process shared LIFO circular buffer queue, O(1)."""
    def __init__(self, maxsize: int = 3):
        self.maxsize = maxsize
        self._mgr   = mp.Manager()
        self._buf   = self._mgr.list([None] * maxsize)   # Circular buffer.
        self._head  = self._mgr.Value('i', -1)           # Stack top points to newest data; -1 means empty.
        self._count = self._mgr.Value('i', 0)
        self._cond  = self._mgr.Condition()              # Use Condition instead of a raw lock.

    # ---------- Public API ----------
    def put(self, item):
        with self._cond:# Thread-safe access; only one process can operate on the queue.
            self._head.value = (self._head.value + 1) % self.maxsize# Wrap to 0 and overwrite oldest data when maxsize is exceeded.
            self._buf[self._head.value] = item
            if self._count.value < self.maxsize:
                self._count.value += 1
            self._cond.notify()          # Wake one waiting get().

    def get(self):
        with self._cond:
            while self._count.value == 0:# Block while the queue is empty.
                self._cond.wait()        # Atomically release the lock and block safely.
            item = self._buf[self._head.value]# Read the stack top after waking and reacquiring the lock.
            self._head.value = (self._head.value - 1) % self.maxsize
            self._count.value -= 1
            return item

    def get_nowait(self):
        with self._cond:
            if self._count.value == 0:
                raise EOFError('queue empty')
            return self.get()

    def qsize(self):
        with self._cond:
            return self._count.value

    def empty(self):
        return self.qsize() == 0

    def full(self):
        return self.qsize() == self.maxsize

    def close(self):
        self._mgr.shutdown()   # Release Manager subprocess.


if __name__ == '__main__':
    q = LIFOQueue(3)
    for v in [1,2,3,4,5]:
        q.put(v)
    print([q.get() for _ in range(3)])   # -> [4, 3, 2], true LIFO.
    q.close()
