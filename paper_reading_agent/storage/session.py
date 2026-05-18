from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..agent.core import PaperReadingAgent
from ..config import AppConfig
from ..llm.types import Message
from ..pdf_parser.models import Paper


@dataclass
class SessionState:
    paper: Paper | None = None
    report: str = ""
    conversation_history: list[Message] = field(default_factory=list)


class SessionManager:
    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.from_env()
        self._sessions: dict[str, SessionState] = {}
        self._agents: dict[str, PaperReadingAgent] = {}

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = SessionState()
        self._agents[session_id] = PaperReadingAgent(self._config)
        return session_id

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def get_agent(self, session_id: str) -> PaperReadingAgent | None:
        return self._agents.get(session_id)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        agent = self._agents.pop(session_id, None)
        if agent and agent._vector_store:
            agent._vector_store.clear()

    def get_or_create(self, session_id: str) -> tuple[SessionState, PaperReadingAgent]:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
            self._agents[session_id] = PaperReadingAgent(self._config)
        return self._sessions[session_id], self._agents[session_id]
