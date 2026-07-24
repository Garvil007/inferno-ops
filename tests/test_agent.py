"""Tests for the agentic tool-use loop in agent.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from inferno_ops.agent import run_agent_loop
from inferno_ops.config import InfernoConfig
from inferno_ops.simulator import RackSimulator
from inferno_ops.telemetry import TelemetrySnapshot


def _make_config(**overrides: object) -> InfernoConfig:
    """Build a test config with sane defaults, overridable per test."""
    base = {
        "anthropic_api_key": "test-key",
        "gpu_count": 4,
        "sim_seed": 7,
        "temp_throttle_c": 85.0,
        "throttle_clock_drop_pct": 0.25,
        "throttle_detect_window_ticks": 5,
        "agent_max_tool_iterations": 6,
    }
    base.update(overrides)
    return InfernoConfig(**base)


def _text_block(text: str) -> SimpleNamespace:
    """Build a fake Anthropic ``text`` content block."""
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_input: dict[str, Any], block_id: str = "tool_1") -> SimpleNamespace:
    """Build a fake Anthropic ``tool_use`` content block."""
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _message(content: list[SimpleNamespace], stop_reason: str) -> SimpleNamespace:
    """Build a fake Anthropic ``Message`` response."""
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class _FakeMessages:
    """Stand-in for ``client.messages`` that replays canned responses in order."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = iter(responses)

    def create(self, **_kwargs: object) -> SimpleNamespace:
        return next(self._responses)


class _FakeClient:
    """Stand-in for ``anthropic.Anthropic`` exposing only what the loop uses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = _FakeMessages(responses)


def _seeded_buffer(config: InfernoConfig) -> tuple[RackSimulator, list[TelemetrySnapshot]]:
    """Build a simulator + buffer pair for tests that call real tool functions."""
    sim = RackSimulator(config=config, seed=config.sim_seed)
    buffer: list[TelemetrySnapshot] = []
    for _ in range(5):
        buffer.extend(sim.step())
    return sim, buffer


def test_loop_executes_multiple_sequential_tool_calls_then_returns_final_text() -> None:
    """Two tool_use round-trips followed by a final text response terminate the loop."""
    config = _make_config()
    sim, buffer = _seeded_buffer(config)

    responses = [
        _message([_tool_use_block("read_rack_metrics", {}, "id1")], "tool_use"),
        _message([_tool_use_block("detect_throttle_event", {}, "id2")], "tool_use"),
        _message([_text_block("Rack looks fine, no action needed.")], "end_turn"),
    ]
    client = _FakeClient(responses)

    result = run_agent_loop(client, config, sim, buffer)

    assert result is not None
    assert result.stop_reason == "end_turn"
    assert result.content[0].text == "Rack looks fine, no action needed."


def test_loop_terminates_immediately_on_healthy_first_response() -> None:
    """A single non-tool_use response ends the loop in one iteration."""
    config = _make_config()
    sim, buffer = _seeded_buffer(config)

    responses = [_message([_text_block("Healthy rack, no action needed.")], "end_turn")]
    client = _FakeClient(responses)

    result = run_agent_loop(client, config, sim, buffer)

    assert result is not None
    assert result.stop_reason == "end_turn"


def test_loop_stops_and_warns_when_iteration_cap_is_hit() -> None:
    """The loop returns None if the cap is hit before a final answer."""
    config = _make_config(agent_max_tool_iterations=2)
    sim, buffer = _seeded_buffer(config)

    responses = [
        _message([_tool_use_block("read_rack_metrics", {}, "id1")], "tool_use"),
        _message([_tool_use_block("read_rack_metrics", {}, "id2")], "tool_use"),
    ]
    client = _FakeClient(responses)

    result = run_agent_loop(client, config, sim, buffer)

    assert result is None


def test_tool_execution_error_is_reported_as_tool_result_not_a_crash() -> None:
    """A tool that raises (e.g. unknown gpu_id) yields an is_error tool_result, loop continues."""
    config = _make_config()
    sim, buffer = _seeded_buffer(config)

    responses = [
        _message(
            [_tool_use_block("generate_rca", {"gpu_id": 999}, "id1")], "tool_use"
        ),
        _message([_text_block("Could not analyze that GPU.")], "end_turn"),
    ]
    client = _FakeClient(responses)

    result = run_agent_loop(client, config, sim, buffer)

    assert result is not None
    assert result.stop_reason == "end_turn"


def test_multiple_tool_use_blocks_in_one_response_are_all_executed() -> None:
    """A single response requesting two tools at once gets both results returned together."""
    config = _make_config()
    sim, buffer = _seeded_buffer(config)

    responses = [
        _message(
            [
                _tool_use_block("read_rack_metrics", {}, "id1"),
                _tool_use_block("detect_throttle_event", {}, "id2"),
            ],
            "tool_use",
        ),
        _message([_text_block("Done.")], "end_turn"),
    ]
    client = _FakeClient(responses)

    result = run_agent_loop(client, config, sim, buffer)

    assert result is not None
    assert result.stop_reason == "end_turn"
