"""InfernoOps agent: wires the Claude client to the rack telemetry tools.

Each cycle the agent should: read metrics, detect throttling, and if found,
run root-cause analysis and recommend a concrete cooling action. No
Streamlit, no simulator internals beyond the shared telemetry buffer.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import Message, MessageParam

from inferno_ops.config import InfernoConfig, load_config
from inferno_ops.dashboard import latest_snapshot_per_gpu
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

# Shared grounding rule: the one place this requirement is stated. Every
# prompt below interpolates this instead of restating its own version, so
# the "cite real numbers, no filler" bar can't drift between monitoring and
# chat behavior.
GROUNDING_RULE = (
    "Always cite the exact numbers from tool results (temperatures, clock "
    "speeds, flow rates, power draw) — never a vague description like "
    '"high temperature" or a generic recommendation like "increase '
    'cooling". When you call generate_rca, relay its suspected_cause and '
    "recommended_action verbatim (they already cite specific figures and a "
    "concrete corrective action) rather than paraphrasing them into "
    "something vaguer. Never guess at a cause or a number you have not "
    "seen in a tool result."
)

SYSTEM_PROMPT = f"""\
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

{GROUNDING_RULE}
"""

CHAT_SYSTEM_PROMPT = f"""\
You are the InfernoOps agent, answering an operator's questions about a \
rack of GPUs in an immersion-cooled AI data center. You have four tools: \
read_rack_metrics, detect_throttle_event, generate_rca, and \
adjust_flow_rate. They read from and act on the same live telemetry and \
simulator the monitoring dashboard uses — there is no separate data \
source.

For every question, call whichever tools you need to ground your answer \
before answering. If a question implies a corrective action (e.g. "fix \
GPU 3"), you may call adjust_flow_rate, but say plainly what you did and \
why. Keep answers concise and conversational — a few sentences, not a \
report.

{GROUNDING_RULE}
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
        "adjust_flow_rate": lambda tool_input: _guarded_adjust_flow_rate(
            buffer, sim, tool_input["gpu_id"], tool_input["delta_lpm"]
        ),
        "generate_rca": lambda tool_input: generate_rca(buffer, tool_input["gpu_id"], config),
    }


def _guarded_adjust_flow_rate(
    buffer: list[TelemetrySnapshot],
    sim: RackSimulator,
    gpu_id: int,
    delta_lpm: float,
) -> dict[str, Any]:
    """Apply a flow adjustment only if the target GPU is currently throttled.

    Hardening for the live-demo steady state: the model is instructed not to
    act on a healthy rack, but this makes that a code-enforced guarantee
    instead of a prompt-only hope, so a model that misbehaves can't spam
    real flow adjustments on a GPU that doesn't need them. Falls open (allows
    the adjustment) if the GPU has no telemetry yet, since there is nothing
    to judge against.

    Args:
        buffer: Flat telemetry history to check the GPU's current state in.
        sim: Live simulator instance to mutate if the adjustment proceeds.
        gpu_id: Index of the GPU to adjust.
        delta_lpm: Change in flow rate, liters per minute.

    Returns:
        The normal ``adjust_flow_rate`` result if applied, or a
        ``{"skipped": True, ...}`` dict explaining why it was withheld.
    """
    latest = latest_snapshot_per_gpu(buffer)
    snap = latest.get(gpu_id)
    if snap is not None and not snap.throttled:
        logger.info("skipped adjust_flow_rate for GPU %d: not currently throttled", gpu_id)
        return {
            "skipped": True,
            "gpu_id": gpu_id,
            "reason": (
                f"GPU {gpu_id} is not currently throttled (latest reading: "
                f"{snap.temp_c}C, {snap.clock_mhz}MHz) — no action taken to "
                "avoid unnecessary changes on a healthy rack."
            ),
        }
    return adjust_flow_rate(sim, gpu_id, delta_lpm)


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


def _run_tool_loop(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    sim: RackSimulator,
    buffer: list[TelemetrySnapshot],
    system: str,
    messages: list[MessageParam],
    trace: list[dict[str, Any]] | None = None,
) -> Message | None:
    """Drive the Anthropic tool-use round-trip loop to completion.

    Loops on the Anthropic Messages API until the model stops requesting
    tools (``stop_reason != "tool_use"``) or ``config.agent_max_tool_iterations``
    round-trips are exhausted, whichever comes first. Each round-trip may
    contain multiple ``tool_use`` blocks; all their results are returned to
    the model in a single follow-up message, as the API requires. Shared core
    for both the monitoring cycle (``run_agent_loop``) and chat Q&A
    (``answer_chat_question``) — both drive the same four tools against the
    same live buffer/simulator, only ``system`` and the seed ``messages``
    differ.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration (model name, max tokens, iteration cap).
        sim: Live simulator instance, mutated by action tools like
            ``adjust_flow_rate``.
        buffer: Flat telemetry history the tools operate on.
        system: System prompt for this loop's purpose.
        messages: Seed message list (mutated in place as the loop proceeds).
        trace: Optional list to append ``{"name", "input", "result"}`` records
            to for every tool call executed this cycle, in order. Left
            untouched (``None``) by default so existing callers are unaffected.

    Returns:
        The final ``Message`` once the model stops requesting tools, or
        ``None`` if the API call failed or the iteration cap was hit before
        a final answer was produced.
    """
    dispatch = _build_tool_dispatch(buffer, sim, config)

    for iteration in range(1, config.agent_max_tool_iterations + 1):
        try:
            response = client.messages.create(
                model=config.model_name,
                max_tokens=config.agent_max_tokens,
                system=system,
                # TOOLS is a plain JSON-schema dict list — matches the API's
                # actual wire shape, just not the SDK's precise ToolParam union.
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error("agent API call failed on iteration %d: %s", iteration, exc)
            return None

        logger.info("agent call iteration=%d stop_reason=%s", iteration, response.stop_reason)
        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text)

        if response.stop_reason != "tool_use":
            return response

        messages.append({"role": "assistant", "content": response.content})
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = [_execute_tool_call(b, dispatch) for b in tool_use_blocks]
        # tool_results is a list of plain tool_result dicts — matches the
        # API's actual wire shape, just not the SDK's precise content union.
        messages.append({"role": "user", "content": tool_results})  # type: ignore[typeddict-item]

        if trace is not None:
            for block, result in zip(tool_use_blocks, tool_results, strict=True):
                trace.append(
                    {
                        "name": block.name,
                        "input": block.input,
                        "result": result["content"],
                        "is_error": result.get("is_error", False),
                    }
                )

    logger.warning(
        "hit agent_max_tool_iterations=%d without a final answer",
        config.agent_max_tool_iterations,
    )
    return None


def run_agent_loop(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    sim: RackSimulator,
    buffer: list[TelemetrySnapshot],
    trace: list[dict[str, Any]] | None = None,
) -> Message | None:
    """Run one full monitoring cycle: send metrics, execute requested tools, repeat.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration (model name, max tokens, iteration cap).
        sim: Live simulator instance, mutated by action tools like
            ``adjust_flow_rate``.
        buffer: Flat telemetry history to summarize as the latest snapshot.
        trace: Optional list to append tool-call records to. See
            ``_run_tool_loop``.

    Returns:
        The final ``Message`` once the model stops requesting tools, or
        ``None`` if the API call failed or the iteration cap was hit before
        a final answer was produced.
    """
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
    return _run_tool_loop(client, config, sim, buffer, SYSTEM_PROMPT, messages, trace)


@dataclass(frozen=True)
class AgentDecision:
    """One dashboard-loggable summary of a single agent cycle.

    Attributes:
        timestamp: Wall-clock time the cycle completed, ``HH:MM:SS``.
        detected: What the agent found (e.g. a throttled GPU with its
            reading), or a healthy/failure statement.
        action: The concrete action taken (e.g. a flow-rate adjustment), or
            "No action taken".
        explanation: The agent's own narrated reasoning, taken verbatim from
            its final text response.
    """

    timestamp: str
    detected: str
    action: str
    explanation: str


def _detected_summary(trace: list[dict[str, Any]]) -> str:
    """Build the "what was detected" line from a tool-call trace.

    Args:
        trace: Tool-call records collected by ``run_agent_loop``.

    Returns:
        A human-readable summary drawn from the ``detect_throttle_event`` and
        ``generate_rca`` calls in ``trace``, or "No throttle detected" if
        neither ran or neither found anything.
    """
    for record in trace:
        if record["name"] == "generate_rca" and not record["is_error"]:
            rca = json.loads(record["result"])
            return f"GPU {record['input'].get('gpu_id')} root cause: {rca.get('data', rca)}"
    for record in trace:
        if record["name"] == "detect_throttle_event" and not record["is_error"]:
            result = json.loads(record["result"])
            if result:
                return f"Throttle detected: {result}"
    return "No throttle detected"


def _action_summary(trace: list[dict[str, Any]]) -> str:
    """Build the "what action was taken" line from a tool-call trace.

    Args:
        trace: Tool-call records collected by ``run_agent_loop``.

    Returns:
        A human-readable summary of the ``adjust_flow_rate`` call in
        ``trace``, or "No action taken" if none ran.
    """
    for record in trace:
        if record["name"] == "adjust_flow_rate" and not record["is_error"]:
            result = json.loads(record["result"])
            if result.get("skipped"):
                continue  # guardrail withheld the action; not a real change
            gpu_id = record["input"].get("gpu_id")
            delta = record["input"].get("delta_lpm")
            return f"Adjusted GPU {gpu_id} coolant flow by {delta} L/min"
    return "No action taken"


def _explanation_summary(response: Message | None) -> str:
    """Extract the agent's final narrated explanation from its response.

    Args:
        response: The final ``Message`` from ``run_agent_loop``, or ``None``
            if the cycle failed.

    Returns:
        The concatenated text of the response, or a failure statement.
    """
    if response is None:
        return "Agent cycle did not complete (API error or iteration cap; see logs)."
    text_blocks = [b.text for b in response.content if b.type == "text" and b.text]
    return " ".join(text_blocks) if text_blocks else "(agent gave no narrated explanation)"


def summarize_decision(response: Message | None, trace: list[dict[str, Any]]) -> AgentDecision:
    """Reduce one agent cycle's response and tool trace to a loggable decision.

    Args:
        response: The final ``Message`` from ``run_agent_loop``, or ``None``
            if the cycle failed.
        trace: Tool-call records collected during that same cycle.

    Returns:
        A timestamped ``AgentDecision`` summarizing what was detected, what
        action was taken, and why.
    """
    return AgentDecision(
        timestamp=time.strftime("%H:%M:%S"),
        detected=_detected_summary(trace),
        action=_action_summary(trace),
        explanation=_explanation_summary(response),
    )


def run_agent_cycle(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    sim: RackSimulator,
    buffer: list[TelemetrySnapshot],
) -> AgentDecision:
    """Run one agent cycle and reduce it to a single dashboard-loggable decision.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration.
        sim: Live simulator instance, mutated by action tools.
        buffer: Flat telemetry history for this cycle.

    Returns:
        A timestamped ``AgentDecision`` for this cycle, safe to append
        directly to a scrolling dashboard event log.
    """
    trace: list[dict[str, Any]] = []
    response = run_agent_loop(client, config, sim, buffer, trace=trace)
    return summarize_decision(response, trace)


def answer_chat_question(
    client: anthropic.Anthropic,
    config: InfernoConfig,
    sim: RackSimulator,
    buffer: list[TelemetrySnapshot],
    question: str,
    prior_messages: list[MessageParam] | None = None,
) -> str:
    """Answer an operator's question, grounded in the same tools and buffer.

    Drives the same ``_run_tool_loop`` core as ``run_agent_loop`` — the same
    four tools against the same live ``buffer``/``sim`` — so chat answers
    can never diverge from what the monitoring cycle sees.

    Args:
        client: Configured Anthropic client.
        config: Runtime configuration.
        sim: Live simulator instance, mutated by action tools if the model
            chooses to call one (e.g. in response to "fix GPU 3").
        buffer: Flat telemetry history the tools operate on.
        question: The operator's new question.
        prior_messages: Earlier turns in this chat, already in Anthropic
            ``{"role", "content"}`` shape, oldest first. ``None``/empty for
            the first question in a session.

    Returns:
        The agent's final narrated answer, or a graceful failure statement
        if the cycle didn't complete (API error or iteration cap) — never
        raises.
    """
    messages: list[MessageParam] = list(prior_messages or [])
    messages.append({"role": "user", "content": question})
    response = _run_tool_loop(client, config, sim, buffer, CHAT_SYSTEM_PROMPT, messages)
    return _explanation_summary(response)


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
