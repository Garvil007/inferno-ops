"""InfernoOps agent: wires the Claude client to the rack telemetry tools.

Each cycle the agent should: read metrics, detect throttling, and if found,
run root-cause analysis and recommend a concrete cooling action. No
Streamlit, no simulator internals beyond the shared telemetry buffer.
"""

from __future__ import annotations

import json
import logging

import anthropic
from anthropic.types import Message

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.telemetry import TelemetrySnapshot
from inferno_ops.tools import TOOLS, read_rack_metrics

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the InfernoOps agent, monitoring a rack of GPUs in an \
immersion-cooled AI data center. Each cycle you receive the latest \
telemetry snapshot for every GPU (temperature, clock speed, power draw, \
coolant flow rate).

Your job each cycle, in order:
1. Call read_rack_metrics to see the latest reading for every GPU.
2. Call detect_throttle_event to check whether any GPU's clock has \
dropped beyond the configured threshold recently.
3. If a throttle event is found: call generate_rca for the affected \
GPU(s) to get a structured root-cause record, then call adjust_flow_rate \
with a concrete flow-rate bump for that GPU to address the likely cause.
4. If nothing is throttled, say so plainly and stop — do not take \
cooling action on a healthy rack.

Always explain your reasoning in plain language and cite the real \
numbers from the tool results (exact temperatures, clock speeds, flow \
rates) — never a vague description like "high temperature". Do not \
guess at a cause without checking the RCA record first.
"""


def build_client(config: InfernoConfig) -> anthropic.Anthropic:
    """Build an Anthropic client from configured credentials.

    Args:
        config: Runtime configuration holding the API key.

    Returns:
        A configured ``anthropic.Anthropic`` client.

    Raises:
        RuntimeError: If no API key is available.
    """
    if not config.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
    return anthropic.Anthropic(api_key=config.anthropic_api_key)


def call_agent(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    buffer: list[TelemetrySnapshot],
) -> Message:
    """Send one metrics snapshot to Claude and let it request rack tools.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration (model name, max tokens).
        buffer: Flat telemetry history to summarize as the latest snapshot.

    Returns:
        The raw API response. Check ``.stop_reason`` for ``"tool_use"``
        and iterate ``.content`` for ``tool_use`` blocks to see which
        tools (if any) Claude requested.
    """
    snapshot = read_rack_metrics(buffer)
    user_message = (
        "Latest rack telemetry snapshot (one reading per GPU):\n"
        f"{json.dumps(snapshot, indent=2)}"
    )

    response = client.messages.create(
        model=config.model_name,
        max_tokens=config.agent_max_tokens,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )
    logger.info("agent call stop_reason=%s", response.stop_reason)
    return response


def _main() -> None:
    """Run the rack simulator briefly, then send one snapshot to Claude."""
    from inferno_ops.logging_config import configure_logging
    from inferno_ops.simulator import RackSimulator

    configure_logging()
    config = load_config()
    client = build_client(config)

    sim = RackSimulator(config=config, seed=config.sim_seed)
    buffer: list[TelemetrySnapshot] = []
    for _ in range(20):
        buffer.extend(sim.step())

    response = call_agent(client, config, buffer)

    for block in response.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "tool_use":
            print(f"tool_use: {block.name} input={block.input}")

    if response.stop_reason != "tool_use":
        logger.warning("expected stop_reason='tool_use', got %r", response.stop_reason)


if __name__ == "__main__":
    _main()
