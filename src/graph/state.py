"""
Shared LangGraph state definitions.

AgentState is the single state TypedDict flowing through every graph.
It extends the original Typestate with fields for:
- conversational memory / history
- multi-agent routing signals
- tool execution tracking
- news pipeline data
- observability metadata
"""
from __future__ import annotations

from typing import Annotated, Any, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Unified state for all graph variants.

    Fields
    ------
    messages : Annotated[List[AnyMessage], add_messages]
        Full message history; LangGraph's ``add_messages`` reducer
        appends new messages rather than replacing the list.
    next_agent : str
        The name of the next agent node the supervisor routes to.
        Used as the conditional-edge key.
    current_agent : str
        Name of the agent currently executing (for tracing).
    tool_results : List[dict]
        Accumulated results from tool calls in this turn.
    news_data : List[dict]
        Raw news articles fetched by the news agent.
    summary : str
        Summarised news content produced by the news agent.
    frequency : str
        Timeframe string for the news pipeline (daily/weekly/monthly).
    filename : str
        Path where the news summary was persisted.
    formatted_response : str
        Final polished response from the formatter agent.
    error : Optional[str]
        Last error message, populated on handled failures.
    metadata : dict[str, Any]
        Free-form observability payload (latency, token counts, etc.).
    session_id : str
        Identifier linking this state to a SessionMemory instance.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    next_agent: str
    current_agent: str
    tool_results: List[dict]
    news_data: List[dict]
    summary: str
    frequency: str
    filename: str
    formatted_response: str
    error: Optional[str]
    metadata: dict[str, Any]
    session_id: str