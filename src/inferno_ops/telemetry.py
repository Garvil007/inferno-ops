"""Shared telemetry data types used across simulator, dashboard, and agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class EventRecord:
    """A structured, agent-triggered or agent-generated rack event.

    Shared shape for both action records (e.g. a manual flow adjustment) and
    analysis records (e.g. a root-cause finding) so tool output stays uniform.

    Attributes:
        event_type: Short identifier for the kind of event, e.g.
            "flow_adjustment" or "root_cause_analysis".
        gpu_id: GPU the event pertains to.
        tick: Simulation tick the event was recorded or generated at.
        data: Event-type-specific structured payload (JSON-serializable).
    """

    event_type: str
    gpu_id: int
    tick: int
    data: dict[str, Any]
