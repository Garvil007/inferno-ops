"""Shared telemetry data types used across simulator, dashboard, and agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetrySnapshot:
    """One GPU's readings at a single simulation tick.

    Attributes:
        gpu_id: Index of the GPU within the rack.
        tick: Monotonic step counter since simulation start.
        temp_c: Core temperature in Celsius.
        clock_mhz: Core clock speed in MHz.
        power_w: Power draw in Watts.
        flow_lpm: Coolant flow rate in liters per minute.
        throttled: Whether this GPU is throttled on this tick.
    """

    gpu_id: int
    tick: int
    temp_c: float
    clock_mhz: float
    power_w: float
    flow_lpm: float
    throttled: bool
