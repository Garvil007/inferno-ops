"""Tests for the MCP tool server: schema correctness and in-process tool calls.

Uses FastMCP's list_tools()/call_tool() directly (no subprocess/stdio) —
the same in-process surface an MCP inspector or client drives over the wire.
"""

from __future__ import annotations

import asyncio

from inferno_ops.mcp_server import mcp


def _list_tools():
    return asyncio.run(mcp.list_tools())


def _call_tool(name: str, arguments: dict) -> tuple:
    return asyncio.run(mcp.call_tool(name, arguments))


def test_list_tools_exposes_exactly_the_four_rack_tools() -> None:
    """The server advertises read_rack_metrics, detect_throttle_event, generate_rca, and
    adjust_flow_rate."""
    tools = _list_tools()
    names = {t.name for t in tools}

    assert names == {
        "read_rack_metrics",
        "detect_throttle_event",
        "generate_rca",
        "adjust_flow_rate",
    }


def test_list_tools_have_descriptions_and_input_schemas() -> None:
    """Every tool has a non-empty description and a JSON Schema input shape."""
    tools = _list_tools()

    for tool in tools:
        assert tool.description
        assert tool.inputSchema["type"] == "object"


def test_generate_rca_and_adjust_flow_rate_require_gpu_id() -> None:
    """The parameterized tools declare gpu_id (and delta_lpm) as required inputs."""
    tools = {t.name: t for t in _list_tools()}

    assert tools["generate_rca"].inputSchema["required"] == ["gpu_id"]
    assert set(tools["adjust_flow_rate"].inputSchema["required"]) == {"gpu_id", "delta_lpm"}


def test_call_read_rack_metrics_returns_one_entry_per_gpu() -> None:
    """Calling read_rack_metrics over MCP returns the same shape as the direct function."""
    from inferno_ops.mcp_server import _config

    _content, structured = _call_tool("read_rack_metrics", {})

    readings = structured["result"]
    assert len(readings) == _config.gpu_count
    assert all("temp_c" in r and "clock_mhz" in r for r in readings)


def test_call_detect_throttle_event_returns_a_list() -> None:
    """detect_throttle_event returns a (possibly empty) list, never an error."""
    _content, structured = _call_tool("detect_throttle_event", {})

    assert isinstance(structured["result"], list)


def test_call_generate_rca_returns_structured_root_cause_record() -> None:
    """generate_rca for a valid gpu_id returns the same EventRecord shape tools.py produces."""
    _content, structured = _call_tool("generate_rca", {"gpu_id": 0})

    assert structured["event_type"] == "root_cause_analysis"
    assert structured["gpu_id"] == 0
    assert "suspected_cause" in structured["data"]


def test_call_adjust_flow_rate_mutates_the_same_simulator_read_tool_sees() -> None:
    """A flow adjustment via MCP persists on the live simulator — same instance, not duplicated.

    adjust_flow_rate mutates the simulator's internal state; that only shows
    up in read_rack_metrics once the next tick appends a fresh snapshot
    (buffer entries are historical, not retroactively edited) — so this
    ticks once more afterward, exactly as a real client would see it evolve.
    """
    from inferno_ops.mcp_server import _tick

    _content, adjust_result = _call_tool("adjust_flow_rate", {"gpu_id": 0, "delta_lpm": 5.0})
    assert adjust_result["event_type"] == "flow_adjustment"
    assert adjust_result["data"]["delta_lpm"] == 5.0

    _tick()
    _content, read_result = _call_tool("read_rack_metrics", {})
    gpu_0 = next(r for r in read_result["result"] if r["gpu_id"] == 0)
    assert gpu_0["flow_lpm"] >= adjust_result["data"]["new_flow_lpm"] - 1.0
