"""
Research Agent — multi-source deep research via Tavily search.

Performs multiple targeted searches and synthesises a structured report.
This is distinct from the basic ToolAgent: it drives the search strategy
itself rather than letting the LLM decide which single tool to invoke.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node
from src.tools.tool_registry import get_search_only_tools

logger = get_logger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are a senior research analyst with access to real-time web search.

For the user's research request:
1. Identify 2-3 distinct sub-questions or angles.
2. Search for each and collect evidence.
3. Synthesise a well-structured markdown report with:
   - Executive summary (3-5 sentences)
   - Key findings (bullet points with sources)
   - Conclusion

Always cite your sources inline as markdown links."""


class ResearchAgent:
    """Deep research agent using tool-augmented LLM calls."""

    def __init__(self, llm: Any) -> None:
        self._tools = get_search_only_tools()
        self._llm = llm
        try:
            self._llm_with_tools = llm.bind_tools(self._tools)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Binding search tools to LLM failed: %s", exc)
            METRICS.record_error(context="research_tool_bind_failure")
            self._llm_with_tools = None

    def process(self, state: AgentState) -> AgentState:
        """Run a multi-step research loop and return a structured report."""
        with trace_node("research_agent", METRICS):
            METRICS.record_agent_call("research_agent")
            messages = state.get("messages", [])

            if not messages or messages[0].type != "system":
                messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + list(messages)

            # Single LLM call — tools_condition in the graph will trigger
            # the ToolNode if the model emits tool_calls.
            try:
                if self._llm_with_tools is None:
                    response = self._llm.invoke(messages)
                else:
                    response = self._llm_with_tools.invoke(messages)
            except Exception as exc:
                logger.warning("ResearchAgent tool-enabled invoke failed: %s", exc)
                METRICS.record_error(context="research_tool_invoke_failure")
                response = self._llm.invoke(messages)

            logger.info(
                "ResearchAgent | tool_calls=%s",
                [tc.get("name") for tc in getattr(response, "tool_calls", [])],
            )

            return {**state, "messages": [response], "current_agent": "research_agent"}