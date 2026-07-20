"""Rack-of-GPUs telemetry simulator.

Pure simulation: no Claude, no Streamlit. Produces a bounded, correlated
random walk per GPU (temp -> power -> clock) plus a throttle model that
fires when temperature crosses the configured limit.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.telemetry import TelemetrySnapshot

logger = logging.getLogger(__name__)

_TEMP_MIN_C = 35.0
_TEMP_MAX_C = 100.0
_TEMP_STEP_C = 1.5

_CLOCK_BASE_MHZ = 1800.0
_CLOCK_MIN_MHZ = 300.0
_CLOCK_MAX_MHZ = 2100.0
_CLOCK_STEP_MHZ = 20.0

_POWER_BASE_W = 250.0
_POWER_MIN_W = 80.0
_POWER_MAX_W = 400.0

_FLOW_BASE_LPM = 14.0
_FLOW_MIN_LPM = 8.0
_FLOW_MAX_LPM = 20.0
_FLOW_STEP_LPM = 0.4


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


@dataclass
class _GpuState:
    """Mutable per-GPU simulation state (internal, not exposed to callers)."""

    gpu_id: int
    temp_c: float
    clock_mhz: float
    power_w: float
    flow_lpm: float
    throttled: bool = False


class RackSimulator:
    """Simulates a rack of GPUs with bounded, correlated telemetry drift."""

    def __init__(self, config: InfernoConfig | None = None, seed: int | None = None) -> None:
        """Build a rack simulator.

        Args:
            config: Runtime configuration; defaults to ``load_config()``.
            seed: RNG seed. Falls back to ``config.sim_seed`` when omitted.
        """
        self.config = config or load_config()
        self._rng = random.Random(seed if seed is not None else self.config.sim_seed)
        self._tick = 0
        self._gpus = [
            _GpuState(
                gpu_id=i,
                temp_c=self._rng.uniform(45.0, 55.0),
                clock_mhz=_CLOCK_BASE_MHZ,
                power_w=_POWER_BASE_W,
                flow_lpm=_FLOW_BASE_LPM,
            )
            for i in range(self.config.gpu_count)
        ]

    def step(self) -> list[TelemetrySnapshot]:
        """Advance the simulation by one tick and return a snapshot per GPU."""
        self._tick += 1
        snapshots = []
        for gpu in self._gpus:
            self._drift(gpu)
            self._apply_throttle_check(gpu)
            snapshots.append(self._to_snapshot(gpu))
        return snapshots

    def inject_throttle(self, gpu_id: int) -> TelemetrySnapshot:
        """Manually force a throttle event on one GPU, bypassing the temp check.

        Args:
            gpu_id: Index of the GPU to throttle.

        Returns:
            The updated snapshot for that GPU.

        Raises:
            IndexError: If ``gpu_id`` is out of range.
        """
        gpu = self._gpus[gpu_id]
        self._throttle(gpu)
        return self._to_snapshot(gpu)

    def _drift(self, gpu: _GpuState) -> None:
        """Apply one tick of bounded, correlated random walk to a GPU's state."""
        temp_delta = self._rng.uniform(-_TEMP_STEP_C, _TEMP_STEP_C)
        gpu.temp_c = _clamp(gpu.temp_c + temp_delta, _TEMP_MIN_C, _TEMP_MAX_C)

        temp_pressure = (gpu.temp_c - 50.0) / 50.0
        target_power = _POWER_BASE_W + temp_pressure * 100.0
        gpu.power_w = _clamp(
            gpu.power_w + (target_power - gpu.power_w) * 0.3 + self._rng.uniform(-5.0, 5.0),
            _POWER_MIN_W,
            _POWER_MAX_W,
        )

        if not gpu.throttled:
            power_pressure = (gpu.power_w - _POWER_BASE_W) / _POWER_BASE_W
            target_clock = _CLOCK_BASE_MHZ - power_pressure * 300.0
            clock_delta = _clamp(
                target_clock - gpu.clock_mhz, -_CLOCK_STEP_MHZ, _CLOCK_STEP_MHZ
            )
            gpu.clock_mhz = _clamp(gpu.clock_mhz + clock_delta, _CLOCK_MIN_MHZ, _CLOCK_MAX_MHZ)

        flow_delta = self._rng.uniform(-_FLOW_STEP_LPM, _FLOW_STEP_LPM)
        gpu.flow_lpm = _clamp(gpu.flow_lpm + flow_delta, _FLOW_MIN_LPM, _FLOW_MAX_LPM)

    def _apply_throttle_check(self, gpu: _GpuState) -> None:
        """Throttle the GPU if temperature crossed the configured limit."""
        if gpu.temp_c >= self.config.temp_throttle_c and not gpu.throttled:
            self._throttle(gpu)
        elif gpu.temp_c < self.config.temp_throttle_c and gpu.throttled:
            gpu.throttled = False

    def _throttle(self, gpu: _GpuState) -> None:
        """Cut a GPU's clock speed by the configured percentage and mark it throttled."""
        gpu.throttled = True
        gpu.clock_mhz = _clamp(
            gpu.clock_mhz * (1.0 - self.config.throttle_clock_drop_pct),
            _CLOCK_MIN_MHZ,
            _CLOCK_MAX_MHZ,
        )
        logger.info("GPU %d throttled at %.1fC, clock now %.0fMHz", gpu.gpu_id, gpu.temp_c, gpu.clock_mhz)

    def _to_snapshot(self, gpu: _GpuState) -> TelemetrySnapshot:
        """Convert internal mutable state to an immutable public snapshot."""
        return TelemetrySnapshot(
            gpu_id=gpu.gpu_id,
            tick=self._tick,
            temp_c=round(gpu.temp_c, 1),
            clock_mhz=round(gpu.clock_mhz, 0),
            power_w=round(gpu.power_w, 1),
            flow_lpm=round(gpu.flow_lpm, 2),
            throttled=gpu.throttled,
        )


def _main() -> None:
    """Run the simulator standalone and print evolving readings."""
    from inferno_ops.logging_config import configure_logging

    configure_logging()
    sim = RackSimulator()
    for _ in range(10):
        for snap in sim.step():
            print(
                f"tick={snap.tick} gpu={snap.gpu_id} temp={snap.temp_c}C "
                f"clock={snap.clock_mhz}MHz power={snap.power_w}W "
                f"flow={snap.flow_lpm}L/min throttled={snap.throttled}"
            )
        time.sleep(0.2)


if __name__ == "__main__":
    _main()
