"""InfernoOps Streamlit dashboard entrypoint.

Orchestration only: no business logic here. Day 7 scope — wire the
simulator into a session-state-backed rolling telemetry buffer, show
per-GPU temperature metrics with color logic, and chart temperature over
time. No agent/Claude wiring yet.
"""

from __future__ import annotations

import logging
from collections import deque

import streamlit as st

from inferno_ops.config import load_config
from inferno_ops.dashboard import (
    latest_snapshot_per_gpu,
    status_emoji,
    temp_series_by_gpu,
    temp_status,
)
from inferno_ops.logging_config import configure_logging
from inferno_ops.simulator import RackSimulator

configure_logging()
logger = logging.getLogger(__name__)

config = load_config()

if "sim" not in st.session_state:
    st.session_state.sim = RackSimulator(config=config, seed=config.sim_seed)
    st.session_state.buffer = deque(maxlen=config.dashboard_buffer_maxlen * config.gpu_count)
    logger.info("dashboard state initialized: gpu_count=%d", config.gpu_count)

st.set_page_config(page_title="InfernoOps", page_icon="\U0001f525", layout="wide")
st.title("InfernoOps")
st.caption(f"model: {config.model_name} | refresh: {config.refresh_interval_s}s")


@st.fragment(run_every=config.refresh_interval_s)
def render_dashboard() -> None:
    """Advance the simulator one tick and render the live rack view.

    Isolated as a fragment so only this block reruns on each tick. The
    simulator and buffer live in ``st.session_state``, initialized once
    above, so history survives every rerun instead of resetting.
    """
    sim: RackSimulator = st.session_state.sim
    buffer: deque = st.session_state.buffer

    buffer.extend(sim.step())

    latest = latest_snapshot_per_gpu(buffer)
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


render_dashboard()
