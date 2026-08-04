"""Pure telemetry-analysis tools exposed to the Claude agent via tool use.

Each function operates only on a passed-in buffer of ``TelemetrySnapshot``
objects: deterministic, no Streamlit, no network. The ``*_SCHEMA`` dicts
describe the Anthropic tool-use ``input_schema`` for what an LLM caller may
pass — the buffer itself is injected by the calling code, never by the model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import EventRecord, TelemetrySnapshot


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


ADJUST_FLOW_RATE_SCHEMA: dict[str, Any] = {
    "name": "adjust_flow_rate",
    "description": (
        "Simulate bumping the coolant pump for one GPU by a given amount "
        "(liters per minute). Persists into subsequent readings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "gpu_id": {"type": "integer", "description": "Index of the GPU to adjust."},
            "delta_lpm": {
                "type": "number",
                "description": "Change in flow rate, liters per minute. May be negative.",
            },
        },
        "required": ["gpu_id", "delta_lpm"],
    },
}


def adjust_flow_rate(sim: RackSimulator, gpu_id: int, delta_lpm: float) -> dict[str, Any]:
    """Bump a GPU's coolant flow rate on a live simulator and record the action.

    Args:
        sim: The running simulator instance to mutate.
        gpu_id: Index of the GPU to adjust.
        delta_lpm: Change in flow rate, liters per minute (may be negative).

    Returns:
        An ``EventRecord`` (as a dict) describing the action taken.
    """
    snap = sim.adjust_flow(gpu_id, delta_lpm)
    record = EventRecord(
        event_type="flow_adjustment",
        gpu_id=gpu_id,
        tick=snap.tick,
        data={
            "delta_lpm": delta_lpm,
            "new_flow_lpm": snap.flow_lpm,
        },
    )
    return asdict(record)


GENERATE_RCA_SCHEMA: dict[str, Any] = {
    "name": "generate_rca",
    "description": (
        "Generate a structured root-cause record for one GPU: temperature "
        "delta, flow rate, power draw, suspected cause, and a recommended "
        "action, based on a recent window of telemetry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "gpu_id": {"type": "integer", "description": "Index of the GPU to analyze."},
        },
        "required": ["gpu_id"],
    },
}


def generate_rca(
    buffer: Sequence[TelemetrySnapshot],
    gpu_id: int,
    config: InfernoConfig | None = None,
) -> dict[str, Any]:
    """Generate a structured root-cause record for one GPU.

    Looks at the GPU's most recent ``config.throttle_detect_window_ticks``
    snapshots and derives a deterministic suspected cause and recommended
    action from temperature trend, flow rate, and power draw.

    Args:
        buffer: Flat telemetry history to analyze.
        gpu_id: Index of the GPU to analyze.
        config: Runtime configuration; defaults to ``load_config()``.

    Returns:
        An ``EventRecord`` (as a dict) whose ``data`` field contains
        temp_delta_c, flow_lpm, power_w, suspected_cause, recommended_action.

    Raises:
        ValueError: If the buffer has no entries for ``gpu_id``.
    """
    cfg = config or load_config()
    grouped = _group_by_gpu(buffer)
    snaps = grouped.get(gpu_id, [])
    if not snaps:
        raise ValueError(f"No telemetry for gpu_id={gpu_id}")

    window = snaps[-cfg.throttle_detect_window_ticks :]
    latest = window[-1]
    temp_delta_c = round(latest.temp_c - window[0].temp_c, 2)
    rack_flows = [s[-1].flow_lpm for s in grouped.values() if s]
    median_flow = sorted(rack_flows)[len(rack_flows) // 2]

    if temp_delta_c > 0 and latest.flow_lpm < median_flow:
        flow_deficit = round(median_flow - latest.flow_lpm, 2)
        suspected_cause = (
            f"insufficient coolant flow: {latest.flow_lpm} L/min vs rack "
            f"median {median_flow} L/min while temp rose {temp_delta_c}C "
            f"over the last {len(window)} ticks"
        )
        recommended_action = (
            f"increase coolant flow rate by {flow_deficit} L/min "
            f"(from {latest.flow_lpm} to at least {median_flow} L/min)"
        )
    elif temp_delta_c > 0 and latest.power_w > window[0].power_w:
        power_delta_w = round(latest.power_w - window[0].power_w, 2)
        flow_bump = round(temp_delta_c * cfg.rca_power_case_flow_bump_per_c, 2)
        suspected_cause = (
            f"sustained high power draw: {window[0].power_w}W to "
            f"{latest.power_w}W (+{power_delta_w}W) over the last "
            f"{len(window)} ticks, temp rose {temp_delta_c}C"
        )
        recommended_action = (
            f"verify workload/power delivery (power draw +{power_delta_w}W); "
            f"as an interim mitigation, increase coolant flow by "
            f"{flow_bump} L/min to offset the {temp_delta_c}C rise"
        )
    else:
        suspected_cause = (
            f"no significant anomaly detected: temp change {temp_delta_c}C, "
            f"flow {latest.flow_lpm} L/min (rack median {median_flow} L/min), "
            f"power {latest.power_w}W"
        )
        recommended_action = (
            f"monitor only — temp change {temp_delta_c}C and flow "
            f"{latest.flow_lpm} L/min are within normal range, no "
            f"corrective action required"
        )

    record = EventRecord(
        event_type="root_cause_analysis",
        gpu_id=gpu_id,
        tick=latest.tick,
        data={
            "temp_delta_c": temp_delta_c,
            "flow_lpm": latest.flow_lpm,
            "power_w": latest.power_w,
            "suspected_cause": suspected_cause,
            "recommended_action": recommended_action,
        },
    )
    return asdict(record)


TOOLS: list[dict[str, Any]] = [
    READ_RACK_METRICS_SCHEMA,
    DETECT_THROTTLE_EVENT_SCHEMA,
    ADJUST_FLOW_RATE_SCHEMA,
    GENERATE_RCA_SCHEMA,
]
