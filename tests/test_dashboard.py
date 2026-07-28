"""Tests for pure dashboard-shaping helpers (no Streamlit involved)."""

from __future__ import annotations

from inferno_ops.config import InfernoConfig
from inferno_ops.dashboard import (
    compute_pue,
    latest_snapshot_per_gpu,
    rack_health_emoji,
    rack_health_status,
    status_emoji,
    temp_series_by_gpu,
    temp_status,
    throttled_gpu_ids,
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


def _snap(
    gpu_id: int,
    tick: int,
    temp_c: float,
    power_w: float = 250.0,
    flow_lpm: float = 14.0,
    throttled: bool = False,
) -> TelemetrySnapshot:
    """Build a synthetic snapshot with only the fields tests care about set explicitly."""
    return TelemetrySnapshot(
        gpu_id=gpu_id,
        tick=tick,
        temp_c=temp_c,
        clock_mhz=1800.0,
        power_w=power_w,
        flow_lpm=flow_lpm,
        throttled=throttled,
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


def test_rack_health_status_all_normal_is_healthy() -> None:
    """A rack with every GPU below warning is healthy."""
    config = _make_config()
    latest = {0: _snap(0, 1, 50.0), 1: _snap(1, 1, 55.0)}

    assert rack_health_status(latest, config) == "healthy"


def test_rack_health_status_one_warning_gpu_makes_rack_warning() -> None:
    """One GPU at warning level is enough to mark the whole rack warning."""
    config = _make_config()
    latest = {0: _snap(0, 1, 50.0), 1: _snap(1, 1, 78.0)}

    assert rack_health_status(latest, config) == "warning"


def test_rack_health_status_throttle_tier_collapses_to_warning() -> None:
    """A GPU at throttle-tier temp (below critical) still reads as rack warning."""
    config = _make_config()
    latest = {0: _snap(0, 1, 87.0)}

    assert rack_health_status(latest, config) == "warning"


def test_rack_health_status_one_critical_gpu_makes_rack_critical() -> None:
    """A single critical GPU dominates even if every other GPU is healthy."""
    config = _make_config()
    latest = {0: _snap(0, 1, 50.0), 1: _snap(1, 1, 96.0)}

    assert rack_health_status(latest, config) == "critical"


def test_rack_health_status_empty_latest_is_healthy() -> None:
    """No readings yet (first tick) defaults to healthy, not an error."""
    config = _make_config()

    assert rack_health_status({}, config) == "healthy"


def test_rack_health_emoji_returns_distinct_emoji_per_status() -> None:
    """Every rack-health status maps to a distinct emoji indicator."""
    statuses = ["healthy", "warning", "critical"]
    emojis = {rack_health_emoji(status) for status in statuses}
    assert len(emojis) == len(statuses)


def test_throttled_gpu_ids_returns_only_throttled_gpus_sorted() -> None:
    """Only GPUs with throttled=True are reported, sorted by gpu_id."""
    latest = {
        2: _snap(2, 1, 90.0, throttled=True),
        0: _snap(0, 1, 50.0, throttled=False),
        1: _snap(1, 1, 88.0, throttled=True),
    }

    assert throttled_gpu_ids(latest) == [1, 2]


def test_throttled_gpu_ids_empty_when_none_throttled() -> None:
    """A fully healthy rack reports no throttled GPUs."""
    latest = {0: _snap(0, 1, 50.0, throttled=False)}

    assert throttled_gpu_ids(latest) == []


def test_compute_pue_matches_expected_ratio_for_known_inputs() -> None:
    """PUE = (IT power + pump power + overhead) / IT power, for known values."""
    config = _make_config(
        pue_pump_power_per_lpm_w=10.0,
        pue_fixed_overhead_w=100.0,
        pue_round_digits=3,
    )
    latest = {
        0: _snap(0, 1, 50.0, power_w=200.0, flow_lpm=10.0),
        1: _snap(1, 1, 50.0, power_w=200.0, flow_lpm=10.0),
    }
    # it_power=400, pump_power=20*10=200, overhead=100 -> (400+200+100)/400 = 1.75
    assert compute_pue(latest, config) == 1.75


def test_compute_pue_rounds_to_configured_digits() -> None:
    """The PUE result respects config.pue_round_digits."""
    config = _make_config(
        pue_pump_power_per_lpm_w=1.0,
        pue_fixed_overhead_w=1.0,
        pue_round_digits=2,
    )
    latest = {0: _snap(0, 1, 50.0, power_w=300.0, flow_lpm=7.0)}

    result = compute_pue(latest, config)

    assert result == round(result, 2)


def test_compute_pue_empty_latest_returns_none() -> None:
    """No telemetry yet (first tick) means PUE is undefined, not zero."""
    config = _make_config()

    assert compute_pue({}, config) is None


def test_compute_pue_zero_it_power_returns_none() -> None:
    """Zero IT power draw would divide by zero; PUE is undefined instead."""
    config = _make_config()
    latest = {0: _snap(0, 1, 50.0, power_w=0.0)}

    assert compute_pue(latest, config) is None
