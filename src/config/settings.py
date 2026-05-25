"""
Central configuration settings for the AgenticAI platform.
Loads from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Application-level configuration."""
    page_title: str = "GraphMind: LangGraph Agentic AI Platform"
    llm_options: List[str] = field(default_factory=lambda: ["Groq"])
    usecase_options: List[str] = field(default_factory=lambda: [
        "Basic Chatbot",
        "Chatbot with Web",
        "AI News",
        "Multi-Agent Research",
    ])
    groq_model_options: List[str] = field(default_factory=lambda: [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b"
    ])


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    max_history_messages: int = 20          # messages kept in short-term memory
    session_ttl_seconds: int = 3600         # 1 hour session lifetime


@dataclass
class ObservabilityConfig:
    """Observability / logging configuration."""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_langsmith: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "agentic-chatbot")


@dataclass
class ToolConfig:
    """Tool execution configuration."""
    max_retries: int = 2
    tavily_max_results: int = 5
    arxiv_top_k: int = 3
    wiki_top_k: int = 3


# Singleton instances accessed project-wide
APP_CONFIG = AppConfig()
MEMORY_CONFIG = MemoryConfig()
OBS_CONFIG = ObservabilityConfig()
TOOL_CONFIG = ToolConfig()