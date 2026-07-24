"""InfernoOps agent: wires the Claude client to the rack telemetry tools.

Each cycle the agent should: read metrics, detect throttling, and if found,
run root-cause analysis and recommend a concrete cooling action. No
Streamlit, no simulator internals beyond the shared telemetry buffer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import anthropic
from anthropic.types import Message, MessageParam

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import TelemetrySnapshot
from inferno_ops.tools import (
    TOOLS,
    adjust_flow_rate,
    detect_throttle_event,
    generate_rca,
    read_rack_metrics,
)

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

_ToolFn = Callable[[dict[str, Any]], Any]


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


def _build_tool_dispatch(
    buffer: list[TelemetrySnapshot],
    sim: RackSimulator,
    config: InfernoConfig,
) -> dict[str, _ToolFn]:
    """Bind the four rack tools to a specific buffer/simulator/config.

    Args:
        buffer: Flat telemetry history the read/analysis tools operate on.
        sim: Live simulator instance the action tool mutates.
        config: Runtime configuration passed to the analysis tools.

    Returns:
        Mapping of tool name to a callable taking the model-supplied
        ``tool_use`` input dict and returning that tool's JSON-serializable
        result.
    """
    return {
        "read_rack_metrics": lambda _input: read_rack_metrics(buffer),
        "detect_throttle_event": lambda _input: detect_throttle_event(buffer, config),
        "adjust_flow_rate": lambda tool_input: adjust_flow_rate(
            sim, tool_input["gpu_id"], tool_input["delta_lpm"]
        ),
        "generate_rca": lambda tool_input: generate_rca(
            buffer, tool_input["gpu_id"], config
        ),
    }


def _execute_tool_call(block: Any, dispatch: dict[str, _ToolFn]) -> dict[str, Any]:
    """Run one requested tool call and package the result as a tool_result block.

    Args:
        block: A ``tool_use`` content block from the model's response.
        dispatch: Tool-name-to-callable mapping from ``_build_tool_dispatch``.

    Returns:
        An Anthropic ``tool_result`` content block dict. On tool failure,
        ``is_error`` is set so the model can see and react to the failure
        instead of the process crashing.
    """
    try:
        fn = dispatch[block.name]
        result = fn(block.input)
        print(f"  tool_use: {block.name} input={block.input}")
        print(f"  tool_result: {json.dumps(result)}")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result),
        }
    except Exception as exc:  # noqa: BLE001 - must surface any tool failure to the model
        logger.exception("tool call %s failed", block.name)
        print(f"  tool_error: {block.name} -> {exc}")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Tool {block.name} failed: {exc}",
            "is_error": True,
        }


def run_agent_loop(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    sim: RackSimulator,
    buffer: list[TelemetrySnapshot],
) -> Message | None:
    """Run one full agent cycle: send metrics, execute requested tools, repeat.

    Loops on the Anthropic Messages API until the model stops requesting
    tools (``stop_reason != "tool_use"``) or ``config.agent_max_tool_iterations``
    round-trips are exhausted, whichever comes first. Each round-trip may
    contain multiple ``tool_use`` blocks; all their results are returned to
    the model in a single follow-up message, as the API requires.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration (model name, max tokens, iteration cap).
        sim: Live simulator instance, mutated by action tools like
            ``adjust_flow_rate``.
        buffer: Flat telemetry history to summarize as the latest snapshot.

    Returns:
        The final ``Message`` once the model stops requesting tools, or
        ``None`` if the API call failed or the iteration cap was hit before
        a final answer was produced.
    """
    dispatch = _build_tool_dispatch(buffer, sim, config)
    snapshot = read_rack_metrics(buffer)
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                "Latest rack telemetry snapshot (one reading per GPU):\n"
                f"{json.dumps(snapshot, indent=2)}"
            ),
        }
    ]

    for iteration in range(1, config.agent_max_tool_iterations + 1):
        try:
            response = client.messages.create(
                model=config.model_name,
                max_tokens=config.agent_max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error("agent API call failed on iteration %d: %s", iteration, exc)
            return None

        logger.info(
            "agent call iteration=%d stop_reason=%s", iteration, response.stop_reason
        )
        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text)

        if response.stop_reason != "tool_use":
            return response

        messages.append({"role": "assistant", "content": response.content})
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = [_execute_tool_call(b, dispatch) for b in tool_use_blocks]
        messages.append({"role": "user", "content": tool_results})

    logger.warning(
        "hit agent_max_tool_iterations=%d without a final answer",
        config.agent_max_tool_iterations,
    )
    return None


def _run_scenario(label: str, config: InfernoConfig, client: anthropic.Anthropic) -> None:
    """Run one demo scenario end to end and print its trace.

    Args:
        label: Human-readable scenario name for the printed header.
        config: Scenario-specific configuration (e.g. a low throttle
            threshold to force a throttle event).
        client: Configured Anthropic client shared across scenarios.
    """
    print(f"\n=== {label} ===")
    sim = RackSimulator(config=config, seed=config.sim_seed)
    buffer: list[TelemetrySnapshot] = []
    for _ in range(20):
        buffer.extend(sim.step())

    response = run_agent_loop(client, config, sim, buffer)
    if response is None:
        print("agent cycle did not complete (see logs)")


def _main() -> None:
    """Run the agent once against a throttle scenario and a healthy scenario."""
    from dataclasses import replace

    from inferno_ops.logging_config import configure_logging

    configure_logging()
    base_config = load_config()
    client = build_client(base_config)

    throttle_config = replace(base_config, temp_throttle_c=60.0, sim_seed=5, gpu_count=4)
    healthy_config = replace(base_config, temp_throttle_c=95.0, sim_seed=1, gpu_count=4)

    _run_scenario("Throttle scenario", throttle_config, client)
    _run_scenario("Healthy scenario", healthy_config, client)


if __name__ == "__main__":
    _main()
