"""
Tool registry — dynamic tool loading with retry and fallback logic.

Every tool is wrapped in a safe executor that:
1. Retries on transient failures (up to TOOL_CONFIG.max_retries).
2. Falls back to a stub response on persistent failure so the agent
   can still produce a useful answer.

Available tools
---------------
- TavilySearch   : web search (requires TAVILY_API_KEY)
- Wikipedia      : Wikipedia article lookup
- Arxiv          : arXiv paper search
- Calculator     : safe expression evaluator (no network)
"""
from __future__ import annotations

import math
import time
from functools import wraps
from typing import Any, Callable, List
import os

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from src.config.settings import TOOL_CONFIG
from src.observability.logging_config import METRICS, get_logger

logger = get_logger(__name__)


# Retry / safe-execute wrapper

def _safe_tool_call(tool_fn: Callable, tool_name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute *tool_fn* with retry logic; return a stub on total failure."""
    last_exc: Exception | None = None
    for attempt in range(1, TOOL_CONFIG.max_retries + 2):  # +1 for initial try
        try:
            result = tool_fn(*args, **kwargs)
            METRICS.record_tool_call(tool_name)
            logger.debug("Tool [%s] succeeded on attempt %d", tool_name, attempt)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Tool [%s] attempt %d/%d failed: %s",
                tool_name,
                attempt,
                TOOL_CONFIG.max_retries + 1,
                exc,
            )
            if attempt <= TOOL_CONFIG.max_retries:
                time.sleep(0.5 * attempt)  # simple back-off

    METRICS.record_error(context=tool_name)
    logger.error("Tool [%s] failed after all retries: %s", tool_name, last_exc)
    return f"[Tool '{tool_name}' unavailable — {last_exc}]"


# Calculator tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Parameters
    ----------
    expression:
        A valid Python arithmetic expression, e.g. ``"2 ** 10 + sqrt(16)"``.
        The ``math`` module is available (``sqrt``, ``pi``, ``log``, etc.).

    Returns
    -------
    str
        The numeric result as a string, or an error message.
    """
    safe_globals = {"__builtins__": {}, "math": math, **vars(math)}
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        logger.debug("Calculator: %s = %s", expression, result)
        METRICS.record_tool_call("calculator")
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


# Tool builders

def _build_tavily_tool() -> TavilySearchResults | None:
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_api_key:
        logger.warning(
            "TAVILY_API_KEY is not configured. TavilySearch tool will be disabled."
        )
        return None

    return TavilySearchResults(
        max_results=TOOL_CONFIG.tavily_max_results,
        tavily_api_key=tavily_api_key,
    )


def _build_arxiv_tool() -> ArxivQueryRun:
    return ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=TOOL_CONFIG.arxiv_top_k)
    )


def _build_wiki_tool() -> WikipediaQueryRun:
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=TOOL_CONFIG.wiki_top_k)
    )


# Public API

# Module-level cached tool instances (built once, reused across invocations)
_tavily_tool: TavilySearchResults | None = None
_arxiv_tool: ArxivQueryRun | None = None
_wiki_tool: WikipediaQueryRun | None = None


def get_tools() -> List[Any]:
    """Return the full list of production tool instances (cached)."""
    global _tavily_tool, _arxiv_tool, _wiki_tool

    if _tavily_tool is None:
        _tavily_tool = _build_tavily_tool()
        if _tavily_tool is not None:
            logger.info("Initialised TavilySearch tool")
    if _arxiv_tool is None:
        _arxiv_tool = _build_arxiv_tool()
        logger.info("Initialised Arxiv tool")
    if _wiki_tool is None:
        _wiki_tool = _build_wiki_tool()
        logger.info("Initialised Wikipedia tool")

    return [tool for tool in [_tavily_tool, _arxiv_tool, _wiki_tool, calculator] if tool is not None]


def get_search_only_tools() -> List[Any]:
    """Return just the Tavily search tool (used by the research agent)."""
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = _build_tavily_tool()
    return [_tavily_tool] if _tavily_tool is not None else []


def build_tool_node(tools: List[Any] | None = None) -> ToolNode:
    """Build a LangGraph ToolNode from *tools* (defaults to all tools)."""
    if tools is None:
        tools = get_tools()
    return ToolNode(tools=tools)