"""
Streamlit UI — sidebar configuration panel.

Renders model selector, API key inputs, use-case selector, and
(for AI News) the timeframe picker + fetch button.
Returns a dict of user-selected controls consumed by main.py.
"""
from __future__ import annotations

import os

import streamlit as st

from src.config.settings import APP_CONFIG
from src.observability.logging_config import get_logger

logger = get_logger(__name__)


def load_streamlit_ui() -> dict:
    """Render sidebar controls and return the user's selections.

    Returns
    -------
    dict
        Keys: selected_llm, selected_groq_model, model_api_key,
              selected_usecase, Tavily_api_key (when needed),
              timeframe (AI News only).
    """
    st.set_page_config(
        page_title=f"🤖 {APP_CONFIG.page_title}",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.header(f"🤖 {APP_CONFIG.page_title}")

    controls: dict = {}

    with st.sidebar:
        st.title("⚙️ Configuration")

        # LLM selection
        controls["selected_llm"] = st.selectbox(
            "🧠 LLM Provider", APP_CONFIG.llm_options
        )

        if controls["selected_llm"] == "Groq":
            controls["selected_groq_model"] = st.selectbox(
                "📦 Model", APP_CONFIG.groq_model_options
            )
            controls["model_api_key"] = st.text_input(
                "🔑 Groq API Key",
                type="password",
                help="Get your key at https://console.groq.com/keys",
            )
            os.environ["GROQ_API_KEY"] = controls["model_api_key"]
            if not controls["model_api_key"]:
                st.warning("⚠️ Enter your Groq API key to continue.")

        # Use-case selection
        controls["selected_usecase"] = st.selectbox(
            "🎯 Use Case", APP_CONFIG.usecase_options
        )
        usecase = controls["selected_usecase"]

        # Tavily key (for web-search and news use-cases)
        if usecase in ("Chatbot with Web", "AI News", "Multi-Agent Research"):
            controls["Tavily_api_key"] = st.text_input(
                "🔑 Tavily API Key",
                type="password",
                help="Get your key at https://app.tavily.com/home",
            )
            os.environ["TAVILY_API_KEY"] = controls["Tavily_api_key"]
            if not controls.get("Tavily_api_key"):
                st.warning("⚠️ Enter your Tavily API key for web search features.")

        # AI News timeframe
        if usecase == "AI News":
            st.subheader("📰 News Options")
            controls["timeframe"] = st.selectbox(
                "📅 Timeframe", ["Daily", "Weekly", "Monthly"], index=0
            )
            if st.button("🔍 Fetch Latest AI News", use_container_width=True):
                st.session_state["IsFetchButtonClicked"] = True
                st.session_state["timeframe"] = controls["timeframe"]

        # Observability panel
        with st.expander("📊 Session Metrics", expanded=False):
            from src.observability.logging_config import METRICS
            summary = METRICS.get_summary()
            st.metric("Total Requests", summary["total_requests"])
            st.metric("Total Tool Calls", summary["total_tool_calls"])
            st.metric("Errors", summary["total_errors"])
            st.metric("Avg Latency (ms)", summary["avg_latency_ms"])
            if summary["agent_calls"]:
                st.write("**Agent calls:**", summary["agent_calls"])

    return controls