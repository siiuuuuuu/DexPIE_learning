import multiprocessing as mp
import time

class LIFOQueue:
    """进程间共享的 LIFO 循环缓冲队列，O(1)"""
    def __init__(self, maxsize: int = 3):
        self.maxsize = maxsize
        self._mgr   = mp.Manager()
        self._buf   = self._mgr.list([None] * maxsize)   # 循环缓冲
        self._head  = self._mgr.Value('i', -1)           # 指向栈顶为最新数据，-1 表示空 
        self._count = self._mgr.Value('i', 0)
        self._cond  = self._mgr.Condition()              # 用 Condition 代替裸锁

    # ---------- 公共 API ----------
    def put(self, item):
        with self._cond:#确保线程安全，只有一个进程能操作队列
            self._head.value = (self._head.value + 1) % self.maxsize#当超过maxsize时自动回到0覆盖最老的数据
            self._buf[self._head.value] = item
            if self._count.value < self.maxsize:
                self._count.value += 1
            self._cond.notify()          # 唤醒一个等待的 get()，即其进入wait状态

    def get(self):
        with self._cond:
            while self._count.value == 0:#队列为空时阻塞等待
                self._cond.wait()        # 原子地释放锁并安全阻塞
            item = self._buf[self._head.value]#获取栈顶元素（唤醒后获得锁继续执行）
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
        self._mgr.shutdown()   # 释放 Manager 子进程


if __name__ == '__main__':
    q = LIFOQueue(3)
    for v in [1,2,3,4,5]:
        q.put(v)
    print([q.get() for _ in range(3)])   # -> [4, 3, 2]  真正的 LIFO
    q.close()