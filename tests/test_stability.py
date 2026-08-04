"""Long-run stability stress test: scripted substitute for a live 20+ minute demo run.

Replays a large multiple of every dashboard.py-shaped rolling collection's
configured cap (buffer, decision log, chat history) through the same
session-state shapes app.py uses, with a fake Anthropic client (no API
cost/latency), and asserts nothing grows unbounded.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from inferno_ops.agent import run_agent_cycle
from inferno_ops.config import InfernoConfig
from inferno_ops.simulator import RackSimulator

_HEALTHY_RESPONSE = SimpleNamespace(
    content=[SimpleNamespace(type="text", text="Rack looks fine, no action needed.")],
    stop_reason="end_turn",
)


class _AlwaysHealthyMessages:
    """Fake ``client.messages`` that always answers with the same healthy text turn.

    Unlike the canned-response fakes in test_agent.py (finite, scenario-scripted),
    this one never runs out — needed to drive hundreds of consecutive cycles.
    """

    def create(self, **_kwargs: object) -> SimpleNamespace:
        return _HEALTHY_RESPONSE


class _AlwaysHealthyClient:
    """Fake ``anthropic.Anthropic`` wired to ``_AlwaysHealthyMessages``."""

    def __init__(self) -> None:
        self.messages = _AlwaysHealthyMessages()


def _make_config(**overrides: object) -> InfernoConfig:
    """Build a small-but-representative config so caps are exercised many times over."""
    base = {
        "anthropic_api_key": "test-key",
        "gpu_count": 4,
        "sim_seed": 3,
        "dashboard_buffer_maxlen": 50,
        "dashboard_decision_log_maxlen": 20,
        "dashboard_chat_history_maxlen": 10,
    }
    base.update(overrides)
    return InfernoConfig(**base)


def test_long_run_keeps_buffer_decision_log_and_chat_history_bounded() -> None:
    """Simulates hundreds of refresh ticks; every capped collection stays at its cap.

    Scripted substitute for babysitting a literal 20+ minute browser session:
    replays far more ticks than any cap needs to be exceeded several times
    over, using the exact same deque(maxlen=...) shapes app.py's
    st.session_state uses.
    """
    config = _make_config()
    sim = RackSimulator(config=config, seed=config.sim_seed)
    client = _AlwaysHealthyClient()

    buffer: deque = deque(maxlen=config.dashboard_buffer_maxlen * config.gpu_count)
    decision_log: deque = deque(maxlen=config.dashboard_decision_log_maxlen)
    chat_history: deque = deque(maxlen=config.dashboard_chat_history_maxlen)

    ticks = (config.dashboard_buffer_maxlen + config.dashboard_decision_log_maxlen) * 5

    for i in range(ticks):
        buffer.extend(sim.step())
        assert len(buffer) <= config.dashboard_buffer_maxlen * config.gpu_count

        decision = run_agent_cycle(client, config, sim, list(buffer))
        decision_log.append(decision)
        assert len(decision_log) <= config.dashboard_decision_log_maxlen

        if i % 3 == 0:
            chat_history.append({"role": "user", "content": f"question {i}"})
            chat_history.append({"role": "assistant", "content": "answer"})
            assert len(chat_history) <= config.dashboard_chat_history_maxlen

    assert len(buffer) == config.dashboard_buffer_maxlen * config.gpu_count
    assert len(decision_log) == config.dashboard_decision_log_maxlen
    assert len(chat_history) == config.dashboard_chat_history_maxlen
    assert all(decision.action == "No action taken" for decision in decision_log)
