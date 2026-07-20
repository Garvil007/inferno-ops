"""Tests for pure telemetry-analysis tools."""

from __future__ import annotations

from inferno_ops.config import InfernoConfig
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import TelemetrySnapshot
from inferno_ops.tools import detect_throttle_event, read_rack_metrics


def _make_config(**overrides: object) -> InfernoConfig:
    """Build a test config with sane defaults, overridable per test."""
    base = {
        "anthropic_api_key": None,
        "gpu_count": 4,
        "sim_seed": 7,
        "temp_throttle_c": 85.0,
        "throttle_clock_drop_pct": 0.25,
        "throttle_detect_window_ticks": 5,
    }
    base.update(overrides)
    return InfernoConfig(**base)


def _snap(gpu_id: int, tick: int, clock_mhz: float, temp_c: float = 50.0) -> TelemetrySnapshot:
    """Build a synthetic snapshot with only the fields tests care about set explicitly."""
    return TelemetrySnapshot(
        gpu_id=gpu_id,
        tick=tick,
        temp_c=temp_c,
        clock_mhz=clock_mhz,
        power_w=250.0,
        flow_lpm=14.0,
        throttled=False,
    )


def test_read_rack_metrics_returns_latest_snapshot_per_gpu() -> None:
    """Only the highest-tick reading survives for each GPU."""
    buffer = [
        _snap(0, 1, 1800.0),
        _snap(0, 2, 1795.0),
        _snap(1, 1, 1810.0),
        _snap(1, 2, 1805.0),
        _snap(1, 3, 1800.0),
    ]

    result = read_rack_metrics(buffer)

    assert [r["gpu_id"] for r in result] == [0, 1]
    assert result[0]["tick"] == 2
    assert result[0]["clock_mhz"] == 1795.0
    assert result[1]["tick"] == 3
    assert result[1]["clock_mhz"] == 1800.0


def test_detect_throttle_event_clear_case_flags_gpu() -> None:
    """A clock drop well past the threshold is flagged."""
    config = _make_config()
    buffer = [_snap(0, t, 1800.0) for t in range(1, 4)] + [_snap(0, 4, 1200.0)]

    result = detect_throttle_event(buffer, config)

    assert len(result) == 1
    assert result[0]["gpu_id"] == 0
    assert result[0]["drop_pct"] > config.throttle_clock_drop_pct


def test_detect_throttle_event_clean_case_flags_nothing() -> None:
    """Stable clocks within a window never get flagged."""
    config = _make_config()
    buffer = [
        _snap(gpu_id, t, 1800.0 + (t % 2))
        for gpu_id in range(config.gpu_count)
        for t in range(1, 6)
    ]

    result = detect_throttle_event(buffer, config)

    assert result == []


def test_detect_throttle_event_borderline_at_threshold_is_flagged() -> None:
    """A drop exactly equal to the configured threshold is flagged (inclusive)."""
    config = _make_config(throttle_clock_drop_pct=0.25)
    peak = 1600.0
    latest = peak * (1 - 0.25)  # exactly 25% drop
    buffer = [_snap(0, 1, peak), _snap(0, 2, latest)]

    result = detect_throttle_event(buffer, config)

    assert len(result) == 1
    assert result[0]["gpu_id"] == 0
    assert result[0]["drop_pct"] == 0.25


def test_detect_throttle_event_just_below_threshold_not_flagged() -> None:
    """A drop just under the threshold is not flagged."""
    config = _make_config(throttle_clock_drop_pct=0.25)
    peak = 1600.0
    latest = peak * (1 - 0.24)
    buffer = [_snap(0, 1, peak), _snap(0, 2, latest)]

    result = detect_throttle_event(buffer, config)

    assert result == []


def test_tools_work_against_real_simulator_output() -> None:
    """Both tools operate correctly on a buffer produced by RackSimulator.

    The simulator freezes a GPU's clock once throttled, so a throttle event
    is only visible in a *recent* window right around the transition tick —
    check as it happens rather than on one final full-history snapshot.
    """
    config = _make_config(gpu_count=4, temp_throttle_c=60.0, sim_seed=5)
    sim = RackSimulator(config=config, seed=5)

    buffer: list[TelemetrySnapshot] = []
    previously_throttled: set[int] = set()
    detected_at_transition = False

    for _ in range(200):
        buffer.extend(sim.step())

        metrics = read_rack_metrics(buffer)
        assert len(metrics) == config.gpu_count
        assert all("clock_mhz" in m for m in metrics)

        flagged_ids = {f["gpu_id"] for f in detect_throttle_event(buffer, config)}
        newly_throttled = {
            s.gpu_id for s in buffer[-config.gpu_count :] if s.throttled
        } - previously_throttled
        if newly_throttled and newly_throttled <= flagged_ids:
            detected_at_transition = True
        previously_throttled |= {s.gpu_id for s in buffer[-config.gpu_count :] if s.throttled}

    assert detected_at_transition
