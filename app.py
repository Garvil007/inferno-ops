"""InfernoOps Streamlit dashboard entrypoint.

Orchestration only: no business logic here. Day 10 scope — a chat panel that
answers operator questions using the same tools/buffer as the live-refresh
fragment, alongside the Day 7-9 rolling telemetry buffer, per-GPU metrics,
chart, rack health/PUE readout, throttle alert, and agent decision log.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import streamlit as st

from inferno_ops.agent import AgentDecision, answer_chat_question, build_client, run_agent_cycle
from inferno_ops.config import load_config
from inferno_ops.dashboard import (
    compute_pue,
    latest_snapshot_per_gpu,
    rack_health_emoji,
    rack_health_status,
    status_emoji,
    temp_series_by_gpu,
    temp_status,
    throttled_gpu_ids,
)
from inferno_ops.logging_config import configure_logging
from inferno_ops.simulator import RackSimulator

configure_logging()
logger = logging.getLogger(__name__)

config = load_config()

if "sim" not in st.session_state:
    st.session_state.sim = RackSimulator(config=config, seed=config.sim_seed)
    st.session_state.buffer = deque(maxlen=config.dashboard_buffer_maxlen * config.gpu_count)
    st.session_state.decision_log = deque(maxlen=config.dashboard_decision_log_maxlen)
    st.session_state.chat_history = deque(maxlen=config.dashboard_chat_history_maxlen)
    try:
        st.session_state.client = build_client(config)
    except RuntimeError as exc:
        logger.warning("agent client unavailable: %s", exc)
        st.session_state.client = None
    logger.info("dashboard state initialized: gpu_count=%d", config.gpu_count)

st.set_page_config(page_title="InfernoOps", page_icon="\U0001f525", layout="wide")
st.title("InfernoOps")
st.caption(f"model: {config.model_name} | refresh: {config.refresh_interval_s}s")


@st.fragment(run_every=config.refresh_interval_s)
def render_dashboard() -> None:
    """Advance the simulator one tick, invoke the agent, and render the live view.

    Isolated as a fragment so only this block reruns on each tick. The
    simulator, buffer, client, and decision log all live in
    ``st.session_state``, initialized once above, so history survives every
    rerun instead of resetting.
    """
    sim: RackSimulator = st.session_state.sim
    buffer: deque = st.session_state.buffer
    client = st.session_state.client
    decision_log: deque[AgentDecision] = st.session_state.decision_log

    buffer.extend(sim.step())

    latest = latest_snapshot_per_gpu(buffer)

    throttled = throttled_gpu_ids(latest)
    if throttled:
        st.error(f"\U0001f6a8 THROTTLE ALERT — GPU(s) {throttled} currently throttled")

    health = rack_health_status(latest, config)
    pue = compute_pue(latest, config)
    status_col, pue_col = st.columns(2)
    with status_col:
        st.metric(label="Rack health", value=f"{rack_health_emoji(health)} {health.title()}")
    with pue_col:
        st.metric(
            label="PUE",
            value=f"{pue:.{config.pue_round_digits}f}" if pue is not None else "N/A",
        )

    columns = st.columns(len(latest))
    for col, gpu_id in zip(columns, sorted(latest)):
        snap = latest[gpu_id]
        status = temp_status(snap.temp_c, config)
        with col:
            st.metric(
                label=f"{status_emoji(status)} GPU {gpu_id}",
                value=f"{snap.temp_c}°C",
                delta=f"{snap.clock_mhz:.0f} MHz | {snap.flow_lpm:.1f} L/min",
                delta_color="off",
            )

    st.line_chart(temp_series_by_gpu(buffer))

    if client is not None:
        decision = run_agent_cycle(client, config, sim, list(buffer))
    else:
        decision = AgentDecision(
            timestamp=time.strftime("%H:%M:%S"),
            detected="N/A",
            action="N/A",
            explanation="Agent unavailable: ANTHROPIC_API_KEY missing from environment.",
        )
    decision_log.append(decision)

    st.subheader("Agent decision log")
    with st.container(height=300):
        for entry in reversed(decision_log):
            st.markdown(
                f"**{entry.timestamp}** — detected: {entry.detected} | "
                f"action: {entry.action}\n\n_{entry.explanation}_"
            )
            st.divider()


render_dashboard()

st.subheader("Ask the agent")
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input('Ask about the rack (e.g. "why did GPU 3 throttle?")')
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.client is not None:
        prior_messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in list(st.session_state.chat_history)[:-1]
        ]
        answer = answer_chat_question(
            st.session_state.client,
            config,
            st.session_state.sim,
            list(st.session_state.buffer),
            question,
            prior_messages,
        )
    else:
        answer = "Agent unavailable: ANTHROPIC_API_KEY missing from environment."

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
