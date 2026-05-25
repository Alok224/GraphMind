"""
Chat Agent — general-purpose conversational agent.

Maintains full message history from the state so previous turns
(loaded by MemoryManager) naturally influence the response.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node

logger = get_logger(__name__)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "Remember everything discussed in the conversation so far. "
    "Give clear, concise, and accurate answers."
)


class ChatAgent:
    """Context-aware chat agent backed by the full message history."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        """Invoke the LLM with full history and return the AI reply."""
        with trace_node("chat_agent", METRICS):
            METRICS.record_agent_call("chat_agent")
            messages = state.get("messages", [])

            # Prepend system prompt if not already present
            if not messages or messages[0].type != "system":
                messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)] + list(messages)

            response: AIMessage = self.llm.invoke(messages)
            logger.info("ChatAgent | tokens_used=%s", getattr(response, "usage_metadata", "N/A"))

            return {**state, "messages": [response], "current_agent": "chat_agent"}
