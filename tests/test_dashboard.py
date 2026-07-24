"""Tests for pure dashboard-shaping helpers (no Streamlit involved)."""

from __future__ import annotations

from inferno_ops.config import InfernoConfig
from inferno_ops.dashboard import (
    latest_snapshot_per_gpu,
    status_emoji,
    temp_series_by_gpu,
    temp_status,
)
from inferno_ops.telemetry import TelemetrySnapshot


def _make_config(**overrides: object) -> InfernoConfig:
    """Build a test config with sane defaults, overridable per test."""
    base = {
        "anthropic_api_key": None,
        "temp_warning_c": 75.0,
        "temp_throttle_c": 85.0,
        "temp_critical_c": 95.0,
    }
    base.update(overrides)
    return InfernoConfig(**base)


def _snap(gpu_id: int, tick: int, temp_c: float) -> TelemetrySnapshot:
    """Build a synthetic snapshot with only the fields tests care about set explicitly."""
    return TelemetrySnapshot(
        gpu_id=gpu_id,
        tick=tick,
        temp_c=temp_c,
        clock_mhz=1800.0,
        power_w=250.0,
        flow_lpm=14.0,
        throttled=False,
    )


def test_temp_status_returns_normal_below_warning_threshold() -> None:
    """A cool reading below the warning threshold is classified normal."""
    config = _make_config()
    assert temp_status(60.0, config) == "normal"


def test_temp_status_returns_warning_at_warning_threshold() -> None:
    """A reading at or above temp_warning_c but below temp_throttle_c is warning."""
    config = _make_config()
    assert temp_status(75.0, config) == "warning"


def test_temp_status_returns_throttle_at_throttle_threshold() -> None:
    """A reading at or above temp_throttle_c but below temp_critical_c is throttle."""
    config = _make_config()
    assert temp_status(85.0, config) == "throttle"


def test_temp_status_returns_critical_at_critical_threshold() -> None:
    """A reading at or above temp_critical_c is critical."""
    config = _make_config()
    assert temp_status(95.0, config) == "critical"


def test_status_emoji_returns_distinct_emoji_per_status() -> None:
    """Every status maps to a distinct emoji indicator."""
    statuses = ["normal", "warning", "throttle", "critical"]
    emojis = {status_emoji(status) for status in statuses}
    assert len(emojis) == len(statuses)


def test_latest_snapshot_per_gpu_picks_highest_tick_per_gpu() -> None:
    """Given multiple ticks per GPU, only the most recent snapshot is kept."""
    buffer = [
        _snap(gpu_id=0, tick=1, temp_c=50.0),
        _snap(gpu_id=0, tick=2, temp_c=55.0),
        _snap(gpu_id=1, tick=1, temp_c=60.0),
    ]

    latest = latest_snapshot_per_gpu(buffer)

    assert latest[0].tick == 2
    assert latest[0].temp_c == 55.0
    assert latest[1].tick == 1


def test_latest_snapshot_per_gpu_empty_buffer_returns_empty_dict() -> None:
    """An empty buffer (dashboard hasn't ticked yet) yields no snapshots."""
    assert latest_snapshot_per_gpu([]) == {}


def test_temp_series_by_gpu_orders_readings_by_tick() -> None:
    """Each GPU's temperature series is ordered by tick, not insertion order."""
    buffer = [
        _snap(gpu_id=0, tick=2, temp_c=55.0),
        _snap(gpu_id=0, tick=1, temp_c=50.0),
    ]

    series = temp_series_by_gpu(buffer)

    assert series["gpu_0"] == [50.0, 55.0]


def test_temp_series_by_gpu_keys_are_sorted_by_gpu_id() -> None:
    """Multiple GPUs produce one series each, keyed and ordered by gpu_id."""
    buffer = [
        _snap(gpu_id=1, tick=1, temp_c=60.0),
        _snap(gpu_id=0, tick=1, temp_c=50.0),
    ]

    series = temp_series_by_gpu(buffer)

    assert list(series.keys()) == ["gpu_0", "gpu_1"]
