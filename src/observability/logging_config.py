"""
Observability — structured logging configuration.
Sets up a rich, production-style logger used across the whole platform.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from src.config.settings import OBS_CONFIG


# Logger factory

def get_logger(name: str) -> logging.Logger:
    """Return a named logger wired to the platform handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, OBS_CONFIG.log_level, logging.INFO))
    logger.propagate = False
    return logger


# Metrics collector (in-memory, lightweight)

class MetricsCollector:
    """Lightweight in-process metrics store.

    Tracks per-session and global counters for agent calls, tool calls,
    errors, and latency.  For a production deployment replace with
    Prometheus / OpenTelemetry exporters.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            "total_requests": 0,
            "total_errors": 0,
            "total_tool_calls": 0,
            "agent_calls": {},
            "tool_calls": {},
            "latencies_ms": [],
        }
        self._logger = get_logger(__name__)

    # counters

    def record_request(self) -> None:
        self._data["total_requests"] += 1

    def record_error(self, context: str = "") -> None:
        self._data["total_errors"] += 1
        self._logger.error("Error recorded | context=%s", context)

    def record_tool_call(self, tool_name: str) -> None:
        self._data["total_tool_calls"] += 1
        self._data["tool_calls"][tool_name] = (
            self._data["tool_calls"].get(tool_name, 0) + 1
        )

    def record_agent_call(self, agent_name: str) -> None:
        self._data["agent_calls"][agent_name] = (
            self._data["agent_calls"].get(agent_name, 0) + 1
        )

    def record_latency(self, latency_ms: float) -> None:
        self._data["latencies_ms"].append(latency_ms)
        # keep only last 500 samples
        if len(self._data["latencies_ms"]) > 500:
            self._data["latencies_ms"] = self._data["latencies_ms"][-500:]

    # read

    def get_summary(self) -> dict[str, Any]:
        latencies = self._data["latencies_ms"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "total_requests": self._data["total_requests"],
            "total_errors": self._data["total_errors"],
            "total_tool_calls": self._data["total_tool_calls"],
            "agent_calls": dict(self._data["agent_calls"]),
            "tool_calls": dict(self._data["tool_calls"]),
            "avg_latency_ms": round(avg_latency, 2),
        }


# Tracing helpers

@contextmanager
def trace_node(name: str, metrics: MetricsCollector | None = None) -> Generator[None, None, None]:
    """Context manager that logs node entry/exit and records latency."""
    logger = get_logger("tracer")
    logger.info("▶ Node [%s] started", name)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        logger.exception("✗ Node [%s] failed: %s", name, exc)
        if metrics:
            metrics.record_error(context=name)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("■ Node [%s] finished in %.1f ms", name, elapsed_ms)
        if metrics:
            metrics.record_latency(elapsed_ms)


def traced_agent(agent_name: str, metrics: MetricsCollector | None = None) -> Callable:
    """Decorator that wraps an agent method with tracing and metrics."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(f"agent.{agent_name}")
            logger.info("Agent [%s] invoked", agent_name)
            if metrics:
                metrics.record_agent_call(agent_name)
            with trace_node(agent_name, metrics):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# Singleton

METRICS = MetricsCollector()
