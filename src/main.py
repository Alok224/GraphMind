"""
Main entrypoint — AgenticAI Platform.

Orchestrates:
1. UI loading (sidebar controls)
2. Memory initialisation per session
3. LLM initialisation (cached)
4. Graph compilation (cached)
5. Response display (streaming where supported)
"""
from __future__ import annotations

import streamlit as st

from src.graph.graph_builder import get_graph
from src.llm.provider import get_llm_from_user_input
from src.memory.memory_manager import MemoryManager
from src.observability.logging_config import get_logger
from src.ui.display_result import DisplayResultStreamlit
from src.ui.loadui import load_streamlit_ui

logger = get_logger(__name__)


def _get_memory_manager() -> MemoryManager:
    """Return the singleton MemoryManager stored in Streamlit session state."""
    if "memory_manager" not in st.session_state:
        st.session_state["memory_manager"] = MemoryManager()
    return st.session_state["memory_manager"]


def _get_session_id() -> str:
    """Return a stable session ID for this browser tab."""
    if "session_id" not in st.session_state:
        import uuid
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def load_langgraph_agenticai_application() -> None:
    """Top-level function called by app.py; runs the full Streamlit loop."""

    # Render sidebar, collect user controls
    user_input = load_streamlit_ui()
    if not user_input:
        st.error("Failed to load UI configuration.")
        return

    # Determine user message
    if st.session_state.get("IsFetchButtonClicked"):
        user_message: str = st.session_state.get("timeframe", "Daily")
        st.session_state["IsFetchButtonClicked"] = False   # reset after consuming
    else:
        user_message = st.chat_input("💬 Enter your message…") or ""

    if not user_message:
        # Show existing chat history while waiting for input
        _render_chat_history()
        return

    # Validate API key
    if not user_input.get("model_api_key"):
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        return

    # Initialise LLM (cached)
    try:
        llm = get_llm_from_user_input(user_input)
    except ValueError as exc:
        st.error(str(exc))
        return

    # Select use-case & get compiled graph (cached)
    usecase: str = user_input.get("selected_usecase", "Basic Chatbot")
    try:
        graph = get_graph(usecase=usecase, llm=llm)
    except Exception as exc:
        st.error(f"Graph compilation failed: {exc}")
        logger.exception("Graph compilation error for usecase=%r", usecase)
        return

    # Resolve session memory
    memory_manager = _get_memory_manager()
    session_id = _get_session_id()
    session_memory = memory_manager.get_or_create(session_id)

    # Render previous messages (chat history)
    _render_chat_history(session_memory)

    # Run graph and display result
    try:
        DisplayResultStreamlit(
            usecase=usecase,
            graph=graph,
            user_message=user_message,
            session_memory=session_memory,
        ).display_result_on_ui()
    except Exception as exc:
        st.error(f"Error during execution: {exc}")
        logger.exception("Execution error | usecase=%r message=%r", usecase, user_message)


def _render_chat_history(session_memory=None) -> None:
    """Replay all stored messages into the chat UI (history scrollback)."""
    if session_memory is None:
        return
    from langchain_core.messages import AIMessage, HumanMessage
    for msg in session_memory.get_messages():
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)