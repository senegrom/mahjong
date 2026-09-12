"""Bounded minibatch preparation with explicit ownership of its worker.

Use as a context manager (or call close in finally) when a consumer can stop
before exhaustion. Closing waits for the current prepare call, cancels further
work, drains queued tensors, and joins the owned thread. It cannot interrupt
an arbitrary blocking prepare function; training uses finite tensor gathers.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

Item = TypeVar('Item')
Ready = TypeVar('Ready')


class Prefetcher(Iterator[Ready]):
    def __init__(self, items: Iterable[Item], prepare: Callable[[Item], Ready], depth: int = 3) -> None:
        if type(depth) is not int or depth <= 0:
            raise ValueError('prefetch depth must be a positive integer')
        self.items = iter(items)
        self.prepare = prepare
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._closed = False
        self.thread = threading.Thread(target=self._run, daemon=True, name='mahjong-prefetch')
        self.thread.start()

    def _put(self, kind: str, payload=None) -> bool:
        while not self._stop.is_set():
            try:
                self.queue.put((kind, payload), timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = next(self.items)
                except StopIteration:
                    break
                if not self._put('ready', self.prepare(item)):
                    return
        except BaseException as error:  # handed to the consuming thread
            self._put('error', error)
            return
        self._put('end')

    def close(self) -> None:
        """Idempotent; release queued minibatches even after a consumer exception."""
        if self._closed:
            return
        self._stop.set()
        self.thread.join()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self._closed = True
        self.items = iter(())
        self.prepare = None

    def __enter__(self) -> Prefetcher:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __iter__(self) -> Prefetcher:
        return self

    def __next__(self) -> Ready:
        if self._closed:
            raise StopIteration
        try:
            kind, payload = self.queue.get()
        except BaseException:
            self.close()
            raise
        if kind == 'ready':
            return payload
        self.close()
        if kind == 'error':
            raise payload
        raise StopIteration
