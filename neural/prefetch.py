"""Preparing the next minibatch while the card works on this one.

A training step gathers a few thousand sparse rows on the host, makes them
dense on the card and runs the network. Done in turn, the card idles
through the gather and the host through the step. This runs the gather a
few steps ahead in a thread of its own; NumPy and PyTorch both let go of
the interpreter lock for the work that matters, so the two overlap.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

Item = TypeVar("Item")
Ready = TypeVar("Ready")


class Prefetcher(Iterator[Ready]):
    """Yields `prepare(item)` for each item, in order, prepared up to
    `depth` items ahead in a background thread. An exception in
    `prepare` is raised from the loop that consumes."""

    _END = object()

    def __init__(self, items: Iterable[Item], prepare: Callable[[Item], Ready], depth: int = 3) -> None:
        self.items = iter(items)
        self.prepare = prepare
        self.queue: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            for item in self.items:
                self.queue.put(("ready", self.prepare(item)))
        except BaseException as error:  # noqa: BLE001 - handed to the consumer
            self.queue.put(("error", error))
        self.queue.put(("end", self._END))

    def __iter__(self) -> Prefetcher:
        return self

    def __next__(self) -> Ready:
        kind, payload = self.queue.get()
        if kind == "ready":
            return payload
        if kind == "error":
            raise payload
        raise StopIteration
