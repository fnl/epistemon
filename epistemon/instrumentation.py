"""Performance instrumentation for measuring indexing operations."""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = defaultdict(list)

    def record(self, operation: str, duration: float) -> None:
        self.timings[operation].append(duration)

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for operation, durations in self.timings.items():
            if durations:
                result[operation] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                }
        return result

    def log_summary(self) -> None:
        summary = self.summary()
        if not summary:
            return

        logger.info("Performance Summary:")
        for operation, stats in summary.items():
            logger.info(
                "  %s: count=%d, total=%.4fs, avg=%.4fs, min=%.4fs, max=%.4fs",
                operation,
                stats["count"],
                stats["total"],
                stats["avg"],
                stats["min"],
                stats["max"],
            )


_global_metrics: PerformanceMetrics | None = None


def enable_instrumentation() -> None:
    global _global_metrics
    _global_metrics = PerformanceMetrics()


def disable_instrumentation() -> None:
    global _global_metrics
    _global_metrics = None


def get_metrics() -> PerformanceMetrics | None:
    return _global_metrics


@contextmanager
def measure(operation: str) -> Generator[None, None, None]:
    if _global_metrics is None:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _global_metrics.record(operation, duration)
