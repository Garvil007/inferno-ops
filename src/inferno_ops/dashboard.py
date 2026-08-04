"""Pure dashboard-shaping helpers: no Streamlit, no session state.

Kept separate from app.py so temperature-status/color logic and chart-data
shaping stay testable without a Streamlit script run context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from inferno_ops.config import InfernoConfig
from inferno_ops.telemetry import TelemetrySnapshot

_STATUS_EMOJI = {
    "normal": "\U0001f7e2",
    "warning": "\U0001f7e1",
    "throttle": "\U0001f7e0",
    "critical": "\U0001f534",
}

_RACK_HEALTH_EMOJI = {
    "healthy": "\U0001f7e2",
    "warning": "\U0001f7e1",
    "critical": "\U0001f534",
}

_TEMP_STATUS_TO_RACK_HEALTH = {
    "normal": "healthy",
    "warning": "warning",
    "throttle": "warning",
    "critical": "critical",
}

_RACK_HEALTH_SEVERITY = {"healthy": 0, "warning": 1, "critical": 2}


def temp_status(temp_c: float, config: InfernoConfig) -> str:
    """Classify a temperature reading against configured thresholds.

    Args:
        temp_c: GPU core temperature in Celsius.
        config: Runtime configuration holding the threshold values.

    Returns:
        One of "normal", "warning", "throttle", "critical" (ascending
        severity), based on ``config.temp_warning_c`` / ``temp_throttle_c``
        / ``temp_critical_c``.
    """
    if temp_c >= config.temp_critical_c:
        return "critical"
    if temp_c >= config.temp_throttle_c:
        return "throttle"
    if temp_c >= config.temp_warning_c:
        return "warning"
    return "normal"


def status_emoji(status: str) -> str:
    """Map a temperature status to a colored-square emoji indicator.

    Args:
        status: A status returned by ``temp_status``.

    Returns:
        A colored-square emoji standing in for that severity level.
    """
    return _STATUS_EMOJI[status]


def latest_snapshot_per_gpu(
    buffer: Sequence[TelemetrySnapshot],
) -> dict[int, TelemetrySnapshot]:
    """Reduce a flat telemetry buffer to each GPU's most recent snapshot.

    Args:
        buffer: Flat, tick-ordered telemetry history spanning many GPUs.

    Returns:
        Mapping of gpu_id to its latest snapshot in the buffer.
    """
    latest: dict[int, TelemetrySnapshot] = {}
    for snap in buffer:
        current = latest.get(snap.gpu_id)
        if current is None or snap.tick > current.tick:
            latest[snap.gpu_id] = snap
    return latest


def rack_health_status(latest: dict[int, TelemetrySnapshot], config: InfernoConfig) -> str:
    """Derive one overall rack-health tier from every GPU's temperature status.

    Args:
        latest: Mapping of gpu_id to its latest snapshot, as returned by
            ``latest_snapshot_per_gpu``.
        config: Runtime configuration holding the temperature thresholds.

    Returns:
        The worst-case of "healthy", "warning", "critical" across all GPUs
        (temp_status "throttle" collapses into "warning"). "healthy" if
        ``latest`` is empty (no readings yet).
    """
    worst = "healthy"
    for snap in latest.values():
        status = _TEMP_STATUS_TO_RACK_HEALTH[temp_status(snap.temp_c, config)]
        if _RACK_HEALTH_SEVERITY[status] > _RACK_HEALTH_SEVERITY[worst]:
            worst = status
    return worst


def rack_health_emoji(status: str) -> str:
    """Map a rack-health status to a colored-square emoji indicator.

    Args:
        status: A status returned by ``rack_health_status``.

    Returns:
        A colored-square emoji standing in for that severity level.
    """
    return _RACK_HEALTH_EMOJI[status]


def throttled_gpu_ids(latest: dict[int, TelemetrySnapshot]) -> list[int]:
    """List every GPU currently throttled, per the simulator's own flag.

    Args:
        latest: Mapping of gpu_id to its latest snapshot, as returned by
            ``latest_snapshot_per_gpu``.

    Returns:
        Sorted gpu_ids where ``snap.throttled`` is True. Empty if none.
    """
    return sorted(gpu_id for gpu_id, snap in latest.items() if snap.throttled)


def compute_pue(latest: dict[int, TelemetrySnapshot], config: InfernoConfig) -> float | None:
    """Estimate current Power Usage Effectiveness from the latest telemetry.

    Models total facility power as IT power plus a coolant-pump power draw
    proportional to total flow rate plus a fixed facility overhead:
    ``(it_power + pump_power + overhead) / it_power``.

    Args:
        latest: Mapping of gpu_id to its latest snapshot, as returned by
            ``latest_snapshot_per_gpu``.
        config: Runtime configuration holding the PUE model coefficients.

    Returns:
        PUE rounded to ``config.pue_round_digits``, or ``None`` if there is
        no telemetry yet or IT power draw is zero (PUE undefined).
    """
    if not latest:
        return None
    it_power_w = sum(snap.power_w for snap in latest.values())
    if it_power_w <= 0:
        return None
    total_flow_lpm = sum(snap.flow_lpm for snap in latest.values())
    pump_power_w = total_flow_lpm * config.pue_pump_power_per_lpm_w
    facility_power_w = it_power_w + pump_power_w + config.pue_fixed_overhead_w
    return round(facility_power_w / it_power_w, config.pue_round_digits)


def temp_series_by_gpu(buffer: Sequence[TelemetrySnapshot]) -> dict[str, list[float]]:
    """Shape a flat telemetry buffer into per-GPU temperature series for charting.

    Args:
        buffer: Flat, tick-ordered telemetry history spanning many GPUs.

    Returns:
        Mapping of ``"gpu_{id}"`` label to that GPU's ``temp_c`` readings in
        tick order, suitable for ``st.line_chart``.
    """
    grouped: dict[int, list[TelemetrySnapshot]] = defaultdict(list)
    for snap in buffer:
        grouped[snap.gpu_id].append(snap)
    return {
        f"gpu_{gpu_id}": [s.temp_c for s in sorted(snaps, key=lambda s: s.tick)]
        for gpu_id, snaps in sorted(grouped.items())
    }
