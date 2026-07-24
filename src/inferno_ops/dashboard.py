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
