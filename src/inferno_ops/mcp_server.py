"""InfernoOps MCP server: exposes the four rack telemetry tools over MCP.

Thin wrapper only — no business logic here. Each tool delegates straight to
the pure functions in tools.py, the same functions the Claude agent (Day 6)
drives directly. The server keeps its own live simulator/buffer (it has no
Streamlit session to borrow one from) so it works standalone against an
MCP client or inspector.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.logging_config import configure_logging
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import TelemetrySnapshot
from inferno_ops.tools import ADJUST_FLOW_RATE_SCHEMA, DETECT_THROTTLE_EVENT_SCHEMA
from inferno_ops.tools import GENERATE_RCA_SCHEMA, READ_RACK_METRICS_SCHEMA
from inferno_ops.tools import adjust_flow_rate as _adjust_flow_rate
from inferno_ops.tools import detect_throttle_event as _detect_throttle_event
from inferno_ops.tools import generate_rca as _generate_rca
from inferno_ops.tools import read_rack_metrics as _read_rack_metrics

logger = logging.getLogger(__name__)

_config: InfernoConfig = load_config()
_sim = RackSimulator(config=_config, seed=_config.sim_seed)
_buffer: list[TelemetrySnapshot] = []
_lock = threading.Lock()

_WARM_START_TICKS = 5


def _trim_buffer() -> None:
    """Cap ``_buffer`` at the configured rolling-history length (per GPU)."""
    maxlen = _config.dashboard_buffer_maxlen * _config.gpu_count
    if len(_buffer) > maxlen:
        del _buffer[: len(_buffer) - maxlen]


def _tick() -> None:
    """Advance the simulator one step and append to the shared buffer."""
    with _lock:
        _buffer.extend(_sim.step())
        _trim_buffer()


for _ in range(_WARM_START_TICKS):
    _tick()


def _tick_loop() -> None:
    """Background loop: keep the buffer live for as long as the server runs."""
    while True:
        time.sleep(_config.refresh_interval_s)
        _tick()


mcp = FastMCP("inferno-ops")


@mcp.tool(name="read_rack_metrics", description=READ_RACK_METRICS_SCHEMA["description"])
def read_rack_metrics() -> list[dict[str, Any]]:
    """Return the most recent telemetry snapshot for every GPU in the rack."""
    with _lock:
        return _read_rack_metrics(list(_buffer))


@mcp.tool(name="detect_throttle_event", description=DETECT_THROTTLE_EVENT_SCHEMA["description"])
def detect_throttle_event() -> list[dict[str, Any]]:
    """Flag any GPU whose clock dropped beyond the configured threshold recently."""
    with _lock:
        return _detect_throttle_event(list(_buffer), _config)


@mcp.tool(name="generate_rca", description=GENERATE_RCA_SCHEMA["description"])
def generate_rca(gpu_id: int) -> dict[str, Any]:
    """Generate a structured root-cause record for one GPU.

    Args:
        gpu_id: Index of the GPU to analyze.
    """
    with _lock:
        return _generate_rca(list(_buffer), gpu_id, _config)


@mcp.tool(name="adjust_flow_rate", description=ADJUST_FLOW_RATE_SCHEMA["description"])
def adjust_flow_rate(gpu_id: int, delta_lpm: float) -> dict[str, Any]:
    """Bump one GPU's coolant flow rate by a given amount (liters per minute).

    Args:
        gpu_id: Index of the GPU to adjust.
        delta_lpm: Change in flow rate, liters per minute. May be negative.
    """
    with _lock:
        return _adjust_flow_rate(_sim, gpu_id, delta_lpm)


def main() -> None:
    """Start the background tick loop and run the MCP server (stdio transport)."""
    configure_logging()
    threading.Thread(target=_tick_loop, daemon=True).start()
    logger.info("InfernoOps MCP server starting (stdio transport)")
    mcp.run()


if __name__ == "__main__":
    main()
