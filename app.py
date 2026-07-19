"""InfernoOps Streamlit dashboard entrypoint.

Orchestration only: no business logic here. Day 1 scope — verify the
project skeleton, config loading, logging, and a live-updating fragment.
"""

from __future__ import annotations

import logging
import random
import time

import streamlit as st

from inferno_ops.config import load_config
from inferno_ops.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

config = load_config()

if "api_key_logged" not in st.session_state:
    logger.info(
        "Anthropic API key %s",
        "loaded from environment" if config.anthropic_api_key else "MISSING from environment",
    )
    st.session_state["api_key_logged"] = True

st.set_page_config(page_title="InfernoOps", page_icon="🔥", layout="centered")
st.title("InfernoOps")
st.caption(f"model: {config.model_name} | refresh: {config.refresh_interval_s}s")


@st.fragment(run_every=config.refresh_interval_s)
def render_live_metric() -> None:
    """Render a single metric that updates on its own timer.

    Isolated as a fragment so only this block reruns on each tick —
    the rest of the page never freezes or flickers.
    """
    temp_c = round(random.uniform(60.0, 90.0), 1)
    st.metric(label="GPU 0 Temp (°C)", value=temp_c)
    st.caption(f"last tick: {time.strftime('%H:%M:%S')}")


render_live_metric()
