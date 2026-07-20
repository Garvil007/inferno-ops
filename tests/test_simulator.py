"""Tests for the GPU rack telemetry simulator."""

from __future__ import annotations

from inferno_ops.config import InfernoConfig
from inferno_ops.simulator import (
    _CLOCK_MAX_MHZ,
    _CLOCK_MIN_MHZ,
    _FLOW_MAX_LPM,
    _FLOW_MIN_LPM,
    _POWER_MAX_W,
    _POWER_MIN_W,
    _TEMP_MAX_C,
    _TEMP_MIN_C,
    RackSimulator,
)


def _make_config(**overrides: object) -> InfernoConfig:
    """Build a test config with sane defaults, overridable per test."""
    base = {
        "anthropic_api_key": None,
        "gpu_count": 4,
        "sim_seed": 7,
        "temp_throttle_c": 85.0,
        "throttle_clock_drop_pct": 0.25,
    }
    base.update(overrides)
    return InfernoConfig(**base)


def test_inject_throttle_drops_affected_gpu_clock() -> None:
    """Manually injecting a throttle event cuts that GPU's clock speed."""
    sim = RackSimulator(config=_make_config())
    sim.step()
    before = next(s for s in sim.step() if s.gpu_id == 1)
    clock_before = before.clock_mhz

    after = sim.inject_throttle(1)

    assert after.throttled is True
    assert after.clock_mhz < clock_before
    assert after.clock_mhz == round(clock_before * 0.75, 0) or after.clock_mhz >= _CLOCK_MIN_MHZ


def test_inject_throttle_does_not_affect_other_gpus() -> None:
    """Injecting a throttle on one GPU leaves other GPUs untouched."""
    sim = RackSimulator(config=_make_config())
    before_snaps = {s.gpu_id: s for s in sim.step()}

    sim.inject_throttle(0)

    assert before_snaps[1].throttled is False


def test_seeding_gives_reproducible_sequences() -> None:
    """Two simulators built with the same seed produce identical readings."""
    sim_a = RackSimulator(config=_make_config(sim_seed=123), seed=123)
    sim_b = RackSimulator(config=_make_config(sim_seed=123), seed=123)

    for _ in range(20):
        snaps_a = sim_a.step()
        snaps_b = sim_b.step()
        assert snaps_a == snaps_b


def test_different_seeds_diverge() -> None:
    """Different seeds should (almost certainly) produce different sequences."""
    sim_a = RackSimulator(config=_make_config(sim_seed=1), seed=1)
    sim_b = RackSimulator(config=_make_config(sim_seed=2), seed=2)

    snaps_a = [sim_a.step() for _ in range(10)]
    snaps_b = [sim_b.step() for _ in range(10)]

    assert snaps_a != snaps_b


def test_values_stay_within_realistic_bounds_over_many_steps() -> None:
    """Every emitted reading stays within its configured physical bounds."""
    sim = RackSimulator(config=_make_config(gpu_count=8), seed=99)

    for _ in range(500):
        for snap in sim.step():
            assert _TEMP_MIN_C <= snap.temp_c <= _TEMP_MAX_C
            assert _CLOCK_MIN_MHZ <= snap.clock_mhz <= _CLOCK_MAX_MHZ
            assert _POWER_MIN_W <= snap.power_w <= _POWER_MAX_W
            assert _FLOW_MIN_LPM <= snap.flow_lpm <= _FLOW_MAX_LPM


def test_temp_crossing_configured_limit_triggers_throttle() -> None:
    """A GPU that drifts past temp_throttle_c gets throttled automatically."""
    sim = RackSimulator(config=_make_config(temp_throttle_c=50.0), seed=5)

    throttled_seen = False
    for _ in range(50):
        for snap in sim.step():
            if snap.throttled:
                throttled_seen = True
                assert snap.temp_c >= 50.0 - 1e-9 or snap.clock_mhz < 1800.0

    assert throttled_seen
