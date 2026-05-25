"""
News Agent — fetches and summarises AI news using Tavily.

Preserves the original three-step pipeline (fetch → summarise → save)
and integrates it cleanly into the AgentState schema.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from src.graph.state import AgentState
from src.observability.logging_config import METRICS, get_logger, trace_node

logger = get_logger(__name__)

_TIME_RANGE_MAP = {"daily": "d", "weekly": "w", "monthly": "m"}
_DAYS_MAP = {"daily": 1, "weekly": 7, "monthly": 30}

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Summarize AI news articles into clean markdown. For each article:
- Date in **YYYY-MM-DD** format (IST timezone)
- 1-2 sentence concise summary
- Source URL as a hyperlink
Sort by date descending (newest first).

Format each entry as:
### [Date]
- [Summary sentence](URL)
"""),
    ("user", "Articles:\n{articles}"),
])

NEWS_DIR = Path("./AINews")


class NewsAgent:
    """Three-step news pipeline: fetch → summarise → save."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self._tavily = TavilyClient()

    # Fetch

    def fetch_news(self, state: AgentState) -> AgentState:
        """Fetch AI news articles from Tavily for the requested timeframe."""
        with trace_node("news_agent.fetch", METRICS):
            METRICS.record_agent_call("news_agent")
            messages = state.get("messages", [])
            frequency = messages[0].content.strip().lower() if messages else "daily"
            frequency = frequency if frequency in _TIME_RANGE_MAP else "daily"

            logger.info("NewsAgent | fetching %s news", frequency)
            METRICS.record_tool_call("tavily_news")

            response = self._tavily.search(
                query="Top Artificial Intelligence (AI) technology news India and globally",
                topic="news",
                time_range=_TIME_RANGE_MAP[frequency],
                include_answer="advanced",
                max_results=20,
                days=_DAYS_MAP[frequency],
            )

            news_data = response.get("results", [])
            logger.info("NewsAgent | fetched %d articles", len(news_data))

            return {
                **state,
                "news_data": news_data,
                "frequency": frequency,
                "current_agent": "news_agent",
            }

    # Summarise

    def summarize_news(self, state: AgentState) -> AgentState:
        """Summarise fetched articles with the LLM."""
        with trace_node("news_agent.summarise", METRICS):
            news_data = state.get("news_data", [])
            if not news_data:
                return {**state, "summary": "No news articles found for the selected period."}

            articles_str = "\n\n".join(
                f"Content: {item.get('content', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Date: {item.get('published_date', '')}"
                for item in news_data
            )

            response = self.llm.invoke(SUMMARIZE_PROMPT.format(articles=articles_str))
            summary: str = response.content
            logger.info("NewsAgent | summary generated (%d chars)", len(summary))

            return {**state, "summary": summary}

    # Save

    def save_result(self, state: AgentState) -> AgentState:
        """Persist the summary to an AINews markdown file."""
        with trace_node("news_agent.save", METRICS):
            frequency = state.get("frequency", "daily")
            summary = state.get("summary", "")

            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            filename = NEWS_DIR / f"{frequency}_summary.md"

            filename.write_text(
                f"# {frequency.capitalize()} AI News Summary\n\n{summary}",
                encoding="utf-8",
            )
            logger.info("NewsAgent | saved summary to %s", filename)

            return {**state, "filename": str(filename)}