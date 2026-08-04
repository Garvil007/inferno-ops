"""Tests for pure telemetry-analysis tools."""

from __future__ import annotations

import pytest

from inferno_ops.config import InfernoConfig
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import TelemetrySnapshot
from inferno_ops.tools import (
    adjust_flow_rate,
    detect_throttle_event,
    generate_rca,
    read_rack_metrics,
)


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


def _snap(
    gpu_id: int,
    tick: int,
    clock_mhz: float,
    temp_c: float = 50.0,
    power_w: float = 250.0,
    flow_lpm: float = 14.0,
) -> TelemetrySnapshot:
    """Build a synthetic snapshot with only the fields tests care about set explicitly."""
    return TelemetrySnapshot(
        gpu_id=gpu_id,
        tick=tick,
        temp_c=temp_c,
        clock_mhz=clock_mhz,
        power_w=power_w,
        flow_lpm=flow_lpm,
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


def test_generate_rca_returns_expected_shape_and_fields() -> None:
    """RCA record is a structured EventRecord dict with the required data fields."""
    config = _make_config()
    buffer = [
        _snap(0, 1, 1800.0, temp_c=50.0),
        _snap(0, 2, 1780.0, temp_c=55.0),
        _snap(0, 3, 1760.0, temp_c=60.0),
    ]

    record = generate_rca(buffer, gpu_id=0, config=config)

    assert record["event_type"] == "root_cause_analysis"
    assert record["gpu_id"] == 0
    assert record["tick"] == 3
    data = record["data"]
    assert set(data.keys()) == {
        "temp_delta_c",
        "flow_lpm",
        "power_w",
        "suspected_cause",
        "recommended_action",
    }
    assert isinstance(data["temp_delta_c"], float)
    assert isinstance(data["flow_lpm"], float)
    assert isinstance(data["power_w"], float)
    assert isinstance(data["suspected_cause"], str) and data["suspected_cause"]
    assert isinstance(data["recommended_action"], str) and data["recommended_action"]
    assert data["temp_delta_c"] == 10.0


def test_generate_rca_raises_for_unknown_gpu() -> None:
    """Analyzing a GPU absent from the buffer is a clear error, not silent output."""
    config = _make_config()
    buffer = [_snap(0, 1, 1800.0)]

    with pytest.raises(ValueError):
        generate_rca(buffer, gpu_id=99, config=config)


def _flow_scenario_data() -> dict[str, object]:
    """Build the insufficient-flow RCA scenario and return its data dict."""
    config = _make_config()
    buffer = [
        _snap(0, t, 1800.0, temp_c=50.0 + t, flow_lpm=10.0) for t in range(1, 6)
    ] + [
        _snap(1, 1, 1800.0, flow_lpm=14.0),
        _snap(2, 1, 1800.0, flow_lpm=14.0),
    ]
    return generate_rca(buffer, gpu_id=0, config=config)["data"]


def _power_scenario_data() -> dict[str, object]:
    """Build the high-power-draw RCA scenario and return its data dict."""
    config = _make_config()
    buffer = [
        _snap(0, t, 1800.0, temp_c=50.0 + t, power_w=250.0 + 10.0 * t, flow_lpm=16.0)
        for t in range(1, 6)
    ] + [
        _snap(1, 1, 1800.0, flow_lpm=14.0),
        _snap(2, 1, 1800.0, flow_lpm=14.0),
    ]
    return generate_rca(buffer, gpu_id=0, config=config)["data"]


def _healthy_scenario_data() -> dict[str, object]:
    """Build the no-anomaly RCA scenario and return its data dict."""
    config = _make_config()
    buffer = [_snap(0, t, 1800.0, temp_c=50.0, flow_lpm=14.0) for t in range(1, 6)]
    return generate_rca(buffer, gpu_id=0, config=config)["data"]


def test_generate_rca_flow_scenario_cites_real_numbers_and_concrete_bump() -> None:
    """Insufficient-flow scenario: report cites the exact flow deficit and target."""
    data = _flow_scenario_data()

    assert "10.0" in data["suspected_cause"]
    assert "14.0" in data["suspected_cause"]
    assert "4.0" in data["suspected_cause"]  # temp_delta_c: 55.0 - 51.0
    assert "4.0" in data["recommended_action"]  # flow deficit: 14.0 - 10.0
    assert "10.0" in data["recommended_action"]
    assert "14.0" in data["recommended_action"]


def test_generate_rca_power_scenario_cites_real_numbers_and_flow_bump() -> None:
    """High-power scenario: report cites the power delta and a numeric interim flow bump."""
    data = _power_scenario_data()

    assert "260.0" in data["suspected_cause"]  # window[0] power: 250 + 10*1
    assert "300.0" in data["suspected_cause"]  # latest power: 250 + 10*5
    assert "40.0" in data["suspected_cause"]  # power delta
    assert "+40.0" in data["recommended_action"]
    assert "L/min" in data["recommended_action"]


def test_generate_rca_healthy_scenario_cites_real_numbers_no_action() -> None:
    """No-anomaly scenario: report still cites real numbers, but recommends no action."""
    data = _healthy_scenario_data()

    assert "0.0" in data["suspected_cause"]  # temp_delta_c
    assert "14.0" in data["suspected_cause"]
    assert "no corrective action required" in data["recommended_action"]


def test_generate_rca_three_scenarios_produce_mutually_distinct_reports() -> None:
    """The flow, power, and healthy scenarios never collide on generic template text."""
    flow_data = _flow_scenario_data()
    power_data = _power_scenario_data()
    healthy_data = _healthy_scenario_data()

    causes = {flow_data["suspected_cause"], power_data["suspected_cause"], healthy_data["suspected_cause"]}
    actions = {
        flow_data["recommended_action"],
        power_data["recommended_action"],
        healthy_data["recommended_action"],
    }
    assert len(causes) == 3
    assert len(actions) == 3


def test_adjust_flow_rate_changes_later_readings_for_targeted_gpu() -> None:
    """Bumping flow on one GPU measurably raises its subsequent flow readings."""
    config = _make_config(gpu_count=4, sim_seed=11)
    sim = RackSimulator(config=config, seed=11)

    sim.step()  # let it settle a bit
    baseline = {s.gpu_id: s.flow_lpm for s in sim.step()}

    record = adjust_flow_rate(sim, gpu_id=1, delta_lpm=5.0)

    assert record["event_type"] == "flow_adjustment"
    assert record["gpu_id"] == 1
    assert record["data"]["delta_lpm"] == 5.0
    assert record["data"]["new_flow_lpm"] > baseline[1]

    later = {s.gpu_id: s.flow_lpm for s in sim.step()}

    assert later[1] > baseline[1] + 2.0
    for gpu_id in (0, 2, 3):
        assert abs(later[gpu_id] - baseline[gpu_id]) < 2.0
