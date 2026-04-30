from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class PushType(str, Enum):
    PROGRESS = "progress"
    RESULT = "result"
    QUESTION = "question"


class ContextType(str, Enum):
    CLARIFICATION = "clarification"
    REFINEMENT = "refinement"


@dataclass
class SessionResponse:
    session_id: str


@dataclass
class QueryResponse:
    transcript: str
    reply_text: str
    reply_audio_base64: str
    reply_audio_format: str
    session_id: str


@dataclass
class PushRequest:
    type: PushType
    summary: str


@dataclass
class PushResponse:
    queued: bool


@dataclass
class PendingResponse:
    pending: bool
    type: Optional[PushType] = None
    reply_text: Optional[str] = None
    reply_audio_base64: Optional[str] = None
    reply_audio_format: Optional[str] = None


@dataclass
class ContextResponse:
    update: bool
    type: Optional[ContextType] = None
    content: Optional[str] = None
    context_version: Optional[int] = None