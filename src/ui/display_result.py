"""
Display Result — renders agent responses in the Streamlit chat UI.

Streaming
---------
For Basic Chatbot and Chatbot with Web, responses are streamed token-by-token
using graph.stream() so the user sees output appearing live.

For AI News, a spinner is shown while the pipeline runs (the news fetch
is inherently batch), and the final markdown file is rendered.

For Multi-Agent Research, streaming is used with agent transition indicators
so the user can watch the supervisor route and specialist agents respond.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph.state import AgentState
from src.memory.memory_manager import SessionMemory
from src.observability.logging_config import METRICS, get_logger

logger = get_logger(__name__)


class DisplayResultStreamlit:
    """Renders graph outputs into the Streamlit chat interface.

    Parameters
    ----------
    usecase : str
        Active use-case name.
    graph : Any
        Compiled LangGraph graph.
    user_message : str
        The raw user query or timeframe string.
    session_memory : SessionMemory
        The active session memory object; used to persist turns.
    """

    def __init__(
        self,
        usecase: str,
        graph: Any,
        user_message: str,
        session_memory: SessionMemory,
    ) -> None:
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.memory = session_memory

    # Public entry point

    def display_result_on_ui(self) -> None:
        METRICS.record_request()
        if self.usecase == "Basic Chatbot":
            self._render_basic_chatbot()
        elif self.usecase == "Chatbot with Web":
            self._render_tool_chatbot()
        elif self.usecase == "AI News":
            self._render_ai_news()
        elif self.usecase == "Multi-Agent Research":
            self._render_multi_agent()
        else:
            st.error(f"Unknown use-case: {self.usecase!r}")

    # Render helpers

    def _render_basic_chatbot(self) -> None:
        """Streaming basic chatbot with persistent memory."""
        with st.chat_message("user"):
            st.write(self.user_message)

        # Build graph input that includes conversation history
        initial_state = self.memory.build_graph_state(self.user_message)

        full_response = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            # Stream node events
            for event in self.graph.stream(initial_state, stream_mode="values"):
                messages = event.get("messages", [])
                if messages:
                    last = messages[-1]
                    if isinstance(last, AIMessage) and last.content:
                        full_response = last.content
                        placeholder.markdown(full_response)

            placeholder.markdown(full_response)

        if full_response:
            self.memory.add_user_message(self.user_message)
            self.memory.add_ai_message(full_response)
            logger.info("BasicChatbot | memory depth=%d", len(self.memory))

    def _render_tool_chatbot(self) -> None:
        """Tool-augmented chatbot with tool-call visibility and streaming."""
        with st.chat_message("user"):
            st.write(self.user_message)

        initial_state = self.memory.build_graph_state(self.user_message)

        full_response = ""
        tool_calls_shown: set[str] = set()

        with st.spinner("🔍 Searching and reasoning…"):
            result: AgentState = self.graph.invoke(initial_state)

        for msg in result.get("messages", []):
            if isinstance(msg, HumanMessage):
                pass  # already shown above
            elif isinstance(msg, ToolMessage):
                tool_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else id(msg)
                if tool_id not in tool_calls_shown:
                    with st.chat_message("ai", avatar="🔧"):
                        with st.expander("🛠️ Tool Result", expanded=False):
                            st.code(str(msg.content)[:2000], language="text")
                    tool_calls_shown.add(tool_id)
            elif isinstance(msg, AIMessage) and msg.content:
                full_response = msg.content

        with st.chat_message("assistant"):
            st.markdown(full_response)

        if full_response:
            self.memory.add_user_message(self.user_message)
            self.memory.add_ai_message(full_response)

    def _render_ai_news(self) -> None:
        """News pipeline renderer — shows spinner, then renders the summary file."""
        frequency = self.user_message.lower()
        with st.spinner(f"📰 Fetching and summarising {frequency} AI news…"):
            self.graph.invoke({"messages": [HumanMessage(content=frequency)]})

        summary_path = Path(f"./AINews/{frequency}_summary.md")
        try:
            content = summary_path.read_text(encoding="utf-8")
            st.markdown(content, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"Summary file not found: {summary_path}")
        except Exception as exc:
            st.error(f"Error displaying news: {exc}")

    def _render_multi_agent(self) -> None:
        """Multi-agent research renderer with agent transition indicators."""
        with st.chat_message("user"):
            st.write(self.user_message)

        initial_state = self.memory.build_graph_state(self.user_message)

        agent_display_names = {
            "supervisor": "🎯 Supervisor",
            "chat_agent": "💬 Chat Agent",
            "tool_agent": "🛠️ Tool Agent",
            "research_agent": "🔬 Research Agent",
            "formatter_agent": "✍️ Formatter Agent",
            "tools": "⚙️ Tool Execution",
        }

        full_response = ""
        status_container = st.empty()

        for event in self.graph.stream(initial_state, stream_mode="values"):
            current = event.get("current_agent", "")
            if current:
                display = agent_display_names.get(current, current)
                status_container.info(f"Active: {display}")

            messages = event.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage) and last.content:
                    full_response = last.content

        status_container.empty()

        with st.chat_message("assistant"):
            if full_response:
                st.markdown(full_response)
            else:
                st.warning("No response generated.")

        if full_response:
            self.memory.add_user_message(self.user_message)
            self.memory.add_ai_message(full_response)