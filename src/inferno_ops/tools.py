"""Pure telemetry-analysis tools exposed to the Claude agent via tool use.

Each function operates only on a passed-in buffer of ``TelemetrySnapshot``
objects: deterministic, no Streamlit, no network. The ``*_SCHEMA`` dicts
describe the Anthropic tool-use ``input_schema`` for what an LLM caller may
pass — the buffer itself is injected by the calling code, never by the model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.telemetry import TelemetrySnapshot


def _group_by_gpu(
    buffer: Sequence[TelemetrySnapshot],
) -> dict[int, list[TelemetrySnapshot]]:
    """Group a flat telemetry buffer into per-GPU lists sorted by tick."""
    grouped: dict[int, list[TelemetrySnapshot]] = defaultdict(list)
    for snap in buffer:
        grouped[snap.gpu_id].append(snap)
    for snaps in grouped.values():
        snaps.sort(key=lambda s: s.tick)
    return grouped


def _snapshot_to_dict(snap: TelemetrySnapshot) -> dict[str, Any]:
    """Convert a ``TelemetrySnapshot`` to a JSON-serializable dict."""
    return {
        "gpu_id": snap.gpu_id,
        "tick": snap.tick,
        "temp_c": snap.temp_c,
        "clock_mhz": snap.clock_mhz,
        "power_w": snap.power_w,
        "flow_lpm": snap.flow_lpm,
        "throttled": snap.throttled,
    }


READ_RACK_METRICS_SCHEMA: dict[str, Any] = {
    "name": "read_rack_metrics",
    "description": (
        "Return the latest telemetry snapshot for every GPU in the rack "
        "(temperature, clock speed, power draw, coolant flow rate)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def read_rack_metrics(buffer: Sequence[TelemetrySnapshot]) -> list[dict[str, Any]]:
    """Return the most recent snapshot for each GPU in the buffer.

    Args:
        buffer: Flat telemetry history, e.g. accumulated ``RackSimulator.step()``
            output across many ticks.

    Returns:
        One dict per GPU (sorted by ``gpu_id``), each the latest-tick reading.
    """
    grouped = _group_by_gpu(buffer)
    return [_snapshot_to_dict(snaps[-1]) for _, snaps in sorted(grouped.items())]


DETECT_THROTTLE_EVENT_SCHEMA: dict[str, Any] = {
    "name": "detect_throttle_event",
    "description": (
        "Flag any GPU whose clock speed dropped beyond the configured "
        "throttle threshold within a recent time window."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def detect_throttle_event(
    buffer: Sequence[TelemetrySnapshot],
    config: InfernoConfig | None = None,
) -> list[dict[str, Any]]:
    """Flag GPUs whose clock dropped beyond the configured threshold recently.

    For each GPU, looks at the last ``config.throttle_detect_window_ticks``
    entries (or fewer, if the buffer is shorter), computes the drop from the
    peak clock in that window to the latest clock, and flags the GPU when
    that drop fraction is >= ``config.throttle_clock_drop_pct`` (inclusive).

    Args:
        buffer: Flat telemetry history to analyze.
        config: Runtime configuration; defaults to ``load_config()``.

    Returns:
        One dict per flagged GPU: gpu_id, tick, drop_pct, peak_clock_mhz,
        latest_clock_mhz. Sorted by gpu_id. Empty if nothing is flagged.
    """
    cfg = config or load_config()
    grouped = _group_by_gpu(buffer)
    flagged: list[dict[str, Any]] = []

    for gpu_id, snaps in sorted(grouped.items()):
        window = snaps[-cfg.throttle_detect_window_ticks :]
        if not window:
            continue
        peak_clock = max(s.clock_mhz for s in window)
        latest = window[-1]
        if peak_clock <= 0:
            continue
        drop_pct = (peak_clock - latest.clock_mhz) / peak_clock
        if drop_pct >= cfg.throttle_clock_drop_pct:
            flagged.append(
                {
                    "gpu_id": gpu_id,
                    "tick": latest.tick,
                    "drop_pct": round(drop_pct, 4),
                    "peak_clock_mhz": peak_clock,
                    "latest_clock_mhz": latest.clock_mhz,
                }
            )

    return flagged


TOOLS: list[dict[str, Any]] = [READ_RACK_METRICS_SCHEMA, DETECT_THROTTLE_EVENT_SCHEMA]
