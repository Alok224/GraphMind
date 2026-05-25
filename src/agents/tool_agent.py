"""
Tool Agent — binds all registered tools to the LLM and handles execution.

The agent uses LangChain's tool-binding so the model can autonomously
decide which tool(s) to call.  The ToolNode (added to the graph separately)
handles the actual tool execution; this node just produces the tool-call
messages.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node
from src.tools.tool_registry import get_tools

logger = get_logger(__name__)

TOOL_SYSTEM_PROMPT = (
    "You are a smart AI assistant with access to powerful tools:\n"
    "- TavilySearch: current web search for news and facts\n"
    "- Wikipedia: encyclopaedic reference lookups\n"
    "- Arxiv: scientific paper search\n"
    "- Calculator: mathematical expression evaluation\n\n"
    "Choose the RIGHT tool for each task. "
    "If multiple tools are needed, call them in the most logical order. "
    "Always synthesise a clear final answer from the tool results."
)


class ToolAgent:
    """Agent that uses tool-binding to call registered tools as needed."""

    def __init__(self, llm: Any) -> None:
        self._tools = get_tools()
        self._llm = llm
        # Attempt to bind tools; if binding fails, keep the plain LLM
        try:
            self._llm_with_tools = llm.bind_tools(self._tools)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Binding tools to LLM failed: %s", exc)
            METRICS.record_error(context="tool_bind_failure")
            self._llm_with_tools = None

    def process(self, state: AgentState) -> AgentState:
        """Produce tool-call messages or a final answer."""
        with trace_node("tool_agent", METRICS):
            METRICS.record_agent_call("tool_agent")
            messages = state.get("messages", [])

            if not messages or messages[0].type != "system":
                messages = [SystemMessage(content=TOOL_SYSTEM_PROMPT)] + list(messages)

            try:
                if self._llm_with_tools is None:
                    # No tool-bound LLM available — call plain LLM
                    response = self._llm.invoke(messages)
                else:
                    response = self._llm_with_tools.invoke(messages)
            except Exception as exc:  # handle tool-call validation / runtime errors
                logger.warning("Tool-enabled LLM invoke failed: %s", exc)
                METRICS.record_error(context="tool_invoke_failure")
                # Fallback: invoke the base LLM without tool bindings
                response = self._llm.invoke(messages)

            logger.info(
                "ToolAgent | tool_calls=%s",
                [tc.get("name") for tc in getattr(response, "tool_calls", [])],
            )
            return {**state, "messages": [response], "current_agent": "tool_agent"}