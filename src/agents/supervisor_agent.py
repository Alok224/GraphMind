"""
Supervisor Agent — the entry point for the multi-agent graph.

Responsibility
--------------
Classify the incoming user query and set ``state["next_agent"]`` to
the name of the most appropriate specialist agent node.

Routing logic uses the LLM to produce a structured decision so that
the routing is semantic, not just keyword-based.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node

logger = get_logger(__name__)

ROUTING_SYSTEM_PROMPT = """You are a supervisor AI that routes user requests to the best specialist agent.

Available agents:
- chat_agent       : General conversation, Q&A, explanations, writing help
- tool_agent       : Queries requiring web search, Wikipedia lookup, arXiv papers, or calculations
- news_agent       : AI / technology news summaries (daily, weekly, monthly)
- research_agent   : Deep multi-source research tasks, comparisons, literature reviews
- formatter_agent  : Formatting or restructuring existing content (tables, markdown, reports)

Respond ONLY with a JSON object: {"agent": "<agent_name>", "reason": "<one sentence>"}
Do not add any other text.
"""


class SupervisorAgent:
    """Routes the user query to the appropriate specialist agent."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def route(self, state: AgentState) -> AgentState:
        """Classify the user's intent and set next_agent in state.

        Falls back to ``chat_agent`` if the LLM response cannot be parsed.
        """
        with trace_node("supervisor", METRICS):
            messages = state.get("messages", [])
            if not messages:
                logger.warning("Supervisor received empty message list; defaulting to chat_agent")
                return {**state, "next_agent": "chat_agent", "current_agent": "supervisor"}

            # Only look at the latest human message to keep the prompt small
            last_human = next(
                (m for m in reversed(messages) if m.type == "human"), None
            )
            user_text = last_human.content if last_human else str(messages[-1])

            try:
                response = self.llm.invoke([
                    SystemMessage(content=ROUTING_SYSTEM_PROMPT),
                    {"role": "user", "content": f"User query: {user_text}"},
                ])
                raw = response.content.strip()
                # Extract JSON even if the model wraps it in fences
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    next_agent = decision.get("agent", "chat_agent")
                    reason = decision.get("reason", "")
                else:
                    raise ValueError(f"No JSON found in: {raw!r}")

                logger.info(
                    "Supervisor routed to [%s] | reason: %s", next_agent, reason
                )
                METRICS.record_agent_call("supervisor")

            except Exception as exc:
                logger.warning(
                    "Supervisor routing failed (%s); falling back to chat_agent", exc
                )
                next_agent = "chat_agent"

            return {**state, "next_agent": next_agent, "current_agent": "supervisor"}