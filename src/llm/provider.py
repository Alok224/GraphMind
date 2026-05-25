"""
LLM provider — initialises and caches the language model.

The model instance is cached per (api_key, model_name) pair so that
switching use-cases inside a single Streamlit session does NOT re-create
the model object and burn time on network handshakes.
"""
from __future__ import annotations

import streamlit as st
from langchain_groq import ChatGroq

from src.observability.logging_config import get_logger

logger = get_logger(__name__)

# Module-level LRU cache keyed by (api_key, model_name)
_llm_cache: dict[tuple[str, str], ChatGroq] = {}


def get_llm(api_key: str, model_name: str) -> ChatGroq:
    """Return a cached ChatGroq instance for the given credentials.

    Parameters
    ----------
    api_key : str
        Groq API key.
    model_name : str
        Model identifier, e.g. ``"llama-3.3-70b-versatile"``.

    Returns
    -------
    ChatGroq
        A ready-to-invoke language model.

    Raises
    ------
    ValueError
        If the API key is empty.
    """
    if not api_key:
        raise ValueError("Groq API key must not be empty.")

    cache_key = (api_key, model_name)
    if cache_key not in _llm_cache:
        logger.info("Initialising ChatGroq | model=%s", model_name)
        _llm_cache[cache_key] = ChatGroq(model_name=model_name, api_key=api_key)
    else:
        logger.debug("Reusing cached ChatGroq | model=%s", model_name)

    return _llm_cache[cache_key]


def get_llm_from_user_input(user_controls_input: dict) -> ChatGroq:
    """Convenience wrapper that reads keys from the Streamlit sidebar dict."""
    api_key: str = user_controls_input.get("model_api_key", "")
    model_name: str = user_controls_input.get(
        "selected_groq_model", "llama-3.3-70b-versatile"
    )
    if not api_key:
        st.error("⚠️ Please enter your Groq API key.")
        st.stop()
    return get_llm(api_key=api_key, model_name=model_name)