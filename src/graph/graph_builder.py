"""
Graph Builder — compiles LangGraph StateGraphs for each use-case.

Use-cases
---------
Basic Chatbot          : chat_agent only (with memory)
Chatbot with Web       : tool_agent → ToolNode loop (with memory)
AI News                : news_agent three-step pipeline
Multi-Agent Research   : supervisor → specialist agents → formatter

All compiled graphs are cached (module-level dict) so rerunning
the Streamlit app within the same session doesn't recompile.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from src.agents.chat_agent import ChatAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.news_agent import NewsAgent
from src.agents.research_agent import ResearchAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.tool_agent import ToolAgent
from src.graph.state import AgentState
from src.observability.logging_config import get_logger
from src.tools.tool_registry import build_tool_node, get_tools

logger = get_logger(__name__)

# ── Compiled graph cache ──────────────────────────────────────────────────────
# Key: (usecase, model_name) so different models get their own compiled graph.
_graph_cache: dict[tuple[str, str], Any] = {}


def _model_key(llm: Any) -> str:
    """Extract a stable string key from the LLM object for caching."""
    return getattr(llm, "model_name", str(id(llm)))


# ── Individual graph factories ────────────────────────────────────────────────

def _build_basic_chatbot(llm: Any) -> Any:
    """Simple chat agent with memory (no tools)."""
    agent = ChatAgent(llm=llm)
    g = StateGraph(AgentState)
    g.add_node("chat_agent", agent.process)
    g.add_edge(START, "chat_agent")
    g.add_edge("chat_agent", END)
    return g.compile()


def _build_chatbot_with_web(llm: Any) -> Any:
    """Tool-augmented chatbot: assistant ↔ tool_node loop."""
    tools = get_tools()
    tool_node = build_tool_node(tools)
    agent = ToolAgent(llm=llm)

    g = StateGraph(AgentState)
    g.add_node("assistant", agent.process)
    g.add_node("tools", tool_node)

    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "assistant")

    return g.compile()


def _build_ai_news(llm: Any) -> Any:
    """Three-node sequential news pipeline."""
    news_agent = NewsAgent(llm=llm)

    g = StateGraph(AgentState)
    g.add_node("fetch_news", news_agent.fetch_news)
    g.add_node("summarize_news", news_agent.summarize_news)
    g.add_node("save_results", news_agent.save_result)

    g.add_edge(START, "fetch_news")
    g.add_edge("fetch_news", "summarize_news")
    g.add_edge("summarize_news", "save_results")
    g.add_edge("save_results", END)

    return g.compile()


def _build_multi_agent_research(llm: Any) -> Any:
    """
    Full multi-agent graph:

        START
          │
       supervisor  ← classifies intent and sets next_agent
          │
     ┌────┴────────────────────────────┐
     │         │           │           │
    chat      tool      research   formatter
     │         │           │
     │        tools      tools
     │         │           │
     └────────►└───────────┘
                          │
                      formatter
                          │
                         END
    """
    supervisor = SupervisorAgent(llm=llm)
    chat_agent = ChatAgent(llm=llm)
    tool_agent = ToolAgent(llm=llm)
    research_agent = ResearchAgent(llm=llm)
    formatter_agent = FormatterAgent(llm=llm)

    tools = get_tools()
    tool_node = build_tool_node(tools)

    g = StateGraph(AgentState)

    # ── nodes ────────────────────────────────────────────────────────────────
    g.add_node("supervisor", supervisor.route)
    g.add_node("chat_agent", chat_agent.process)
    g.add_node("tool_agent", tool_agent.process)
    g.add_node("research_agent", research_agent.process)
    g.add_node("formatter_agent", formatter_agent.process)
    g.add_node("tools", tool_node)

    # ── entry ─────────────────────────────────────────────────────────────────
    g.add_edge(START, "supervisor")

    # ── supervisor → specialist ───────────────────────────────────────────────
    def supervisor_router(state: AgentState) -> str:
        return state.get("next_agent", "chat_agent")

    g.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "chat_agent": "chat_agent",
            "tool_agent": "tool_agent",
            "news_agent": "chat_agent",      # news goes through chat for multi-agent flow
            "research_agent": "research_agent",
            "formatter_agent": "formatter_agent",
        },
    )

    # ── tool agents loop ──────────────────────────────────────────────────────
    g.add_conditional_edges("tool_agent", tools_condition, {"tools": "tools", END: "formatter_agent"})
    g.add_edge("tools", "tool_agent")

    g.add_conditional_edges("research_agent", tools_condition, {"tools": "tools", END: "formatter_agent"})

    # ── terminal edges ────────────────────────────────────────────────────────
    g.add_edge("chat_agent", "formatter_agent")
    g.add_edge("formatter_agent", END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_GRAPH_FACTORIES = {
    "Basic Chatbot": _build_basic_chatbot,
    "Chatbot with Web": _build_chatbot_with_web,
    "AI News": _build_ai_news,
    "Multi-Agent Research": _build_multi_agent_research,
}


def get_graph(usecase: str, llm: Any) -> Any:
    """Return a compiled LangGraph for *usecase*, using a cache.

    Parameters
    ----------
    usecase : str
        One of the keys in ``_GRAPH_FACTORIES``.
    llm : Any
        Initialised language model (ChatGroq or compatible).

    Returns
    -------
    Compiled LangGraph graph ready to ``.invoke()`` or ``.stream()``.
    """
    cache_key = (usecase, _model_key(llm))
    if cache_key not in _graph_cache:
        factory = _GRAPH_FACTORIES.get(usecase)
        if factory is None:
            raise ValueError(
                f"Unknown use-case: {usecase!r}. "
                f"Available: {list(_GRAPH_FACTORIES)}"
            )
        logger.info("Compiling graph for usecase=%r model=%s", usecase, _model_key(llm))
        _graph_cache[cache_key] = factory(llm)
    else:
        logger.debug("Graph cache hit | usecase=%r", usecase)

    return _graph_cache[cache_key]
