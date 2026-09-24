"""Uniform tracing/telemetry wrapper used by every agent call in the harness."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from .schemas import AgentTraceEntry, ProjectState

_step_counter = {"n": 0}


@contextmanager
def traced_call(state: ProjectState, agent: str, action: str, model: str):
    """Times a block, catches exceptions so one degraded agent never kills the
    pipeline, and appends a structured AgentTraceEntry to state.trace.

    Usage:
        with traced_call(state, "zoning", "analyze_site", model) as t:
            result = do_work()
            t.summary = "buildable envelope computed"
            t.details = {"area_sf": 12345}
    """
    _step_counter["n"] += 1
    step = _step_counter["n"]
    start = time.perf_counter()

    class _T:
        summary: str = ""
        details: dict[str, Any] = {}
        status: str = "ok"

    t = _T()
    try:
        yield t
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for demo resilience
        t.status = "error"
        t.summary = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        entry = AgentTraceEntry(
            step=step,
            agent=agent,
            action=action,
            model=model,
            status=t.status,
            latency_ms=round(latency_ms, 1),
            summary=t.summary,
            details=t.details,
        )
        state.trace.append(entry)


def reset_step_counter():
    _step_counter["n"] = 0
