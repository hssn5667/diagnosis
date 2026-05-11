"""
shared/sse_utils.py
-------------------
Shared SSE (Server-Sent Events) utilities used by every agent microservice
and the orchestrator.

Design rules:
  - Every SSE event is a JSON object with at least {"type": str}
  - Agents emit:  status → (processing steps) → result | error
  - Orchestrator proxies per-agent streams and emits a final "complete" event
  - All helpers are sync-friendly generators OR async generators
  - Never swallow exceptions — always emit an error event before re-raising

Event schema (all fields optional except "type"):
  {"type": "status",   "node": str, "message": str, "step": int, "total": int}
  {"type": "progress", "node": str, "message": str, "pct": float}   # 0-100
  {"type": "result",   "node": str, "data": dict,  "summary": str}
  {"type": "error",    "node": str, "message": str, "fatal": bool}
  {"type": "complete", "node": str, "data": dict,  "elapsed_ms": int}
  {"type": "token",    "node": str, "token": str}                   # LLM tokens
  {"type": "final",    "data": dict}                                 # orchestrator only
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Iterator


# ── Low-level formatting ───────────────────────────────────────────────────────

def sse_line(data: dict) -> str:
    """Format a dict as a single SSE data line (ends with double newline)."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def sse_done() -> str:
    """The canonical end-of-stream sentinel."""
    return "data: [DONE]\n\n"


# ── Event constructors ─────────────────────────────────────────────────────────

def evt_status(node: str, message: str, step: int = 0, total: int = 0) -> str:
    return sse_line({"type": "status", "node": node,
                     "message": message, "step": step, "total": total})


def evt_progress(node: str, message: str, pct: float) -> str:
    return sse_line({"type": "progress", "node": node,
                     "message": message, "pct": round(pct, 1)})


def evt_result(node: str, data: dict, summary: str = "") -> str:
    return sse_line({"type": "result", "node": node,
                     "data": data, "summary": summary})


def evt_error(node: str, message: str, fatal: bool = False) -> str:
    return sse_line({"type": "error", "node": node,
                     "message": message, "fatal": fatal})


def evt_complete(node: str, data: dict, elapsed_ms: int = 0) -> str:
    return sse_line({"type": "complete", "node": node,
                     "data": data, "elapsed_ms": elapsed_ms})


def evt_token(node: str, token: str) -> str:
    return sse_line({"type": "token", "node": node, "token": token})


def evt_final(data: dict) -> str:
    return sse_line({"type": "final", "data": data})


# ── FastAPI StreamingResponse headers ─────────────────────────────────────────

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection":    "keep-alive",
    "X-Accel-Buffering": "no",   # disables nginx buffering
}


# ── Async generator: proxy a remote agent's SSE stream ────────────────────────

async def proxy_agent_stream(
    client,          # httpx.AsyncClient
    url: str,
    payload: dict,
    node_name: str,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """
    Open an SSE stream to a downstream agent, yield each line as-is to the
    client, and capture the final "result" / "complete" event data.

    Yields:
        raw SSE strings (already formatted with "data: ...\n\n")

    Returns (via StopIteration value):
        The data dict from the last "result" or "complete" event, or None.

    Usage in an async for loop:
        result = None
        async for chunk in proxy_agent_stream(client, url, payload, "diagnosis"):
            yield chunk   # pass through to client
            # result is populated internally
    """
    try:
        async with client.stream(
            "POST", url,
            json=payload,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                yield evt_error(node_name,
                                f"Agent returned HTTP {response.status_code}: "
                                f"{body[:200].decode('utf-8', errors='replace')}",
                                fatal=False)
                return

            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                payload_str = raw_line[6:]   # strip "data: "
                if payload_str == "[DONE]":
                    break
                # Pass the line through verbatim — client sees all agent events
                yield f"data: {payload_str}\n\n"

    except Exception as exc:
        yield evt_error(node_name, f"Stream connection failed: {exc}", fatal=False)


async def collect_agent_stream(
    client,
    url: str,
    payload: dict,
    node_name: str,
    timeout: float = 60.0,
) -> tuple[list[str], dict | None]:
    """
    Like proxy_agent_stream but COLLECTS events instead of yielding them.
    Returns (all_sse_lines, last_result_data).
    Use this when you need the full agent result before continuing the graph.
    """
    lines: list[str] = []
    last_result: dict | None = None

    try:
        async with client.stream(
            "POST", url,
            json=payload,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                err = evt_error(node_name,
                                f"HTTP {response.status_code}: "
                                f"{body[:200].decode('utf-8', errors='replace')}",
                                fatal=False)
                lines.append(err)
                return lines, None

            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                payload_str = raw_line[6:]
                if payload_str == "[DONE]":
                    break
                sse = f"data: {payload_str}\n\n"
                lines.append(sse)

                # Track the latest result/complete payload
                try:
                    evt = json.loads(payload_str)
                    if evt.get("type") in ("result", "complete") and "data" in evt:
                        last_result = evt["data"]
                except (json.JSONDecodeError, KeyError):
                    pass

    except Exception as exc:
        err = evt_error(node_name, f"Stream failed: {exc}", fatal=False)
        lines.append(err)

    return lines, last_result


# ── Timing helper ──────────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer."""
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    def elapsed_s(self) -> float:
        return round(time.perf_counter() - self._start, 2)