"""Single source of truth for InfernoOps configuration.

All thresholds, limits, and tunables live here. No magic numbers elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class InfernoConfig:
    """Runtime configuration for InfernoOps.

    Attributes:
        anthropic_api_key: API key for the Anthropic SDK, read from env.
        model_name: Claude model identifier used by the agent.
        refresh_interval_s: Seconds between dashboard telemetry refreshes.
        temp_warning_c: GPU temperature (Celsius) that triggers a warning.
        temp_throttle_c: GPU temperature (Celsius) at which thermal throttling begins.
        temp_critical_c: GPU temperature (Celsius) considered critical.
        throttle_clock_drop_pct: Fraction (0-1) to cut clock speed by when throttled.
        throttle_detect_window_ticks: Recent-tick window size for throttle detection.
        sim_seed: Seed for the telemetry simulator's RNG, for reproducible demos.
        gpu_count: Number of simulated GPUs.
        agent_max_tokens: Max output tokens for a single agent completion call.
        agent_max_tool_iterations: Max tool-use round-trips per agent cycle,
            to bound the loop if the model keeps requesting tools.
        dashboard_buffer_maxlen: Max telemetry snapshots kept per GPU in the
            dashboard's rolling history buffer.
        dashboard_decision_log_maxlen: Max agent decision entries kept in the
            dashboard's scrolling event log.
        pue_pump_power_per_lpm_w: Modeled cooling-pump power draw per liter/min
            of total coolant flow, used to estimate PUE from telemetry.
        pue_fixed_overhead_w: Modeled fixed facility overhead power (lighting,
            networking, etc.), used to estimate PUE from telemetry.
        pue_round_digits: Decimal places to round the computed PUE readout to.
        dashboard_chat_history_maxlen: Max chat messages (user + assistant
            combined) kept in the dashboard's chat history.
        rca_power_case_flow_bump_per_c: Suggested coolant-flow bump (L/min per
            degree C of temp rise) generate_rca recommends as an interim
            mitigation when the suspected cause is high power draw rather
            than insufficient flow.
    """

    anthropic_api_key: str | None
    model_name: str = "claude-sonnet-4-6"
    refresh_interval_s: float = 2.0
    temp_warning_c: float = 75.0
    temp_throttle_c: float = 85.0
    temp_critical_c: float = 95.0
    throttle_clock_drop_pct: float = 0.25
    throttle_detect_window_ticks: int = 5
    sim_seed: int = 42
    gpu_count: int = 8
    agent_max_tokens: int = 2048
    agent_max_tool_iterations: int = 6
    dashboard_buffer_maxlen: int = 200
    dashboard_decision_log_maxlen: int = 50
    pue_pump_power_per_lpm_w: float = 15.0
    pue_fixed_overhead_w: float = 500.0
    pue_round_digits: int = 3
    dashboard_chat_history_maxlen: int = 40
    rca_power_case_flow_bump_per_c: float = 0.5


def load_config() -> InfernoConfig:
    """Build an ``InfernoConfig`` from environment variables and defaults.

    Returns:
        A populated, immutable ``InfernoConfig`` instance.
    """
    return InfernoConfig(anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"))
