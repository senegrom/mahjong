"""Stopped learning must not leave blocked producers retaining training tensors."""
import threading
import time
import unittest

from neural.prefetch import Prefetcher


class PrefetchTests(unittest.TestCase):
    def test_order_and_repeated_exhaustion(self):
        for values in ([], range(10)):
            with Prefetcher(values, lambda x: x * 2, depth=1) as stream:
                self.assertEqual(list(stream), [x * 2 for x in values])
                self.assertEqual(list(stream), [])
            self.assertFalse(stream.thread.is_alive())
            self.assertTrue(stream.queue.empty())

    def test_full_queue_is_cancelled_and_drained(self):
        prepared = threading.Event()
        def prepare(x):
            if x == 1: prepared.set()
            return bytearray(1024)
        stream = Prefetcher(range(100), prepare, depth=1)
        self.assertTrue(prepared.wait(1))  # producer has more data than queue capacity
        start = time.monotonic()
        stream.close(); stream.close()
        self.assertLess(time.monotonic() - start, 1)
        self.assertFalse(stream.thread.is_alive())
        self.assertTrue(stream.queue.empty())

    def test_consumer_failure_joins_worker(self):
        with self.assertRaisesRegex(RuntimeError, 'consumer'):
            with Prefetcher(range(100), lambda x: x, depth=1) as stream:
                for _ in stream:
                    raise RuntimeError('consumer')
        self.assertFalse(stream.thread.is_alive())
        self.assertTrue(stream.queue.empty())

    def test_producer_failure_preserves_order_and_joins_worker(self):
        def prepare(x):
            if x == 2: raise ValueError('producer')
            return x
        with Prefetcher(range(100), prepare, depth=1) as stream:
            self.assertEqual(next(stream), 0); self.assertEqual(next(stream), 1)
            with self.assertRaisesRegex(ValueError, 'producer'): next(stream)
            with self.assertRaises(StopIteration): next(stream)
        self.assertFalse(stream.thread.is_alive())

    def test_bad_depth_rejected_before_worker_creation(self):
        for depth in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): Prefetcher([], lambda x: x, depth)
