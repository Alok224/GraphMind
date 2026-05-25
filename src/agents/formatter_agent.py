"""
Formatter Agent — post-processes and polishes raw agent responses.

Sits at the end of the multi-agent pipeline and ensures every response
is well-structured, consistently formatted markdown.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node

logger = get_logger(__name__)

FORMAT_SYSTEM_PROMPT = """You are a professional content editor and formatter.

Your job is to take the raw AI assistant response and return a polished, 
well-structured version. Rules:
- Use clear headings where appropriate
- Use bullet points for lists
- Bold key terms
- Keep the content accurate — do NOT add or remove factual information
- Fix any grammar or spelling issues
- Ensure the response directly addresses the user's original question
- Maintain a professional but friendly tone"""


class FormatterAgent:
    """Refines and structures the final response before it reaches the user."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        """Format the most recent AI message in the state."""
        with trace_node("formatter_agent", METRICS):
            METRICS.record_agent_call("formatter_agent")
            messages = state.get("messages", [])

            # Find the last AI message to format
            last_ai = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
                None,
            )
            if not last_ai:
                logger.info("FormatterAgent | no AI message to format; skipping")
                return {**state, "current_agent": "formatter_agent"}

            # Also include the original user question for context
            last_human = next(
                (m for m in reversed(messages) if m.type == "human"), None
            )
            user_question = last_human.content if last_human else ""

            prompt_messages = [
                SystemMessage(content=FORMAT_SYSTEM_PROMPT),
                {
                    "role": "user",
                    "content": (
                        f"Original user question: {user_question}\n\n"
                        f"Raw response to format:\n{last_ai.content}"
                    ),
                },
            ]

            formatted: AIMessage = self.llm.invoke(prompt_messages)
            logger.info("FormatterAgent | formatted response (%d chars)", len(formatted.content))

            return {
                **state,
                "messages": [formatted],
                "formatted_response": formatted.content,
                "current_agent": "formatter_agent",
            }
