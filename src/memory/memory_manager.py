"""
Memory system — per-session short-term conversational memory.

Architecture
------------
MemoryManager  : session registry; one SessionMemory per Streamlit session.
SessionMemory  : ordered message buffer, capped at MEMORY_CONFIG.max_history_messages.

Both classes are designed to be held in st.session_state so they survive
Streamlit reruns within the same browser tab without hitting any external
storage.  Swap the persistence layer here if you need Redis/SQLite later.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Deque, List

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

from src.config.settings import MEMORY_CONFIG
from src.observability.logging_config import get_logger

logger = get_logger(__name__)


class SessionMemory:
    """Short-term message buffer for one user session.

    Parameters
    ----------
    session_id:
        Unique identifier for this session.
    max_messages:
        Maximum number of messages to retain (older messages are evicted).
    """

    def __init__(
        self,
        session_id: str,
        max_messages: int = MEMORY_CONFIG.max_history_messages,
    ) -> None:
        self.session_id = session_id
        self.max_messages = max_messages
        self._messages: Deque[AnyMessage] = deque(maxlen=max_messages)
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()

    # write

    def add_message(self, message: AnyMessage) -> None:
        """Append a message; oldest message is auto-evicted when cap is hit."""
        self._messages.append(message)
        self.last_accessed = time.time()
        logger.debug(
            "Session %s | +%s | depth=%d",
            self.session_id,
            type(message).__name__,
            len(self._messages),
        )

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    def clear(self) -> None:
        self._messages.clear()
        logger.info("Session %s | memory cleared", self.session_id)

    # read

    def get_messages(self) -> List[AnyMessage]:
        """Return all retained messages in chronological order."""
        self.last_accessed = time.time()
        return list(self._messages)

    def get_context_window(self, last_n: int | None = None) -> List[AnyMessage]:
        """Return the most recent *last_n* messages (or all if None)."""
        msgs = list(self._messages)
        if last_n is not None:
            return msgs[-last_n:]
        return msgs

    def build_graph_state(
        self, new_user_message: str, system_prompt: str | None = None
    ) -> dict:
        """Build a LangGraph-compatible state dict including history.

        Parameters
        ----------
        new_user_message:
            The current user turn to append.
        system_prompt:
            Optional system instruction prepended before history.
        """
        history = self.get_messages()
        messages: List[AnyMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.extend(history)
        messages.append(HumanMessage(content=new_user_message))
        return {"messages": messages}

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SessionMemory(id={self.session_id!r}, "
            f"messages={len(self._messages)}/{self.max_messages})"
        )


class MemoryManager:
    """Registry of all active SessionMemory objects.

    Held as a singleton in st.session_state["memory_manager"].
    Automatically evicts sessions that have exceeded their TTL.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str | None = None) -> SessionMemory:
        """Return an existing session or create a new one.

        Parameters
        ----------
        session_id:
            Provide an existing ID to resume that session.
            Pass ``None`` (or omit) to always create a fresh session.
        """
        self._evict_stale()

        if session_id and session_id in self._sessions:
            logger.debug("MemoryManager | resumed session %s", session_id)
            return self._sessions[session_id]

        new_id = session_id or str(uuid.uuid4())
        session = SessionMemory(session_id=new_id)
        self._sessions[new_id] = session
        logger.info("MemoryManager | created session %s", new_id)
        return session

    def _evict_stale(self) -> None:
        """Remove sessions whose TTL has expired."""
        now = time.time()
        stale = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_accessed > MEMORY_CONFIG.session_ttl_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
            logger.info("MemoryManager | evicted stale session %s", sid)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)