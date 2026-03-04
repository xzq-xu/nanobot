"""Type definitions for nanobot gateway bridge."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """WebSocket event types."""
    DELTA = "delta"
    TOOL = "tool"
    LIFECYCLE = "lifecycle"
    FINAL = "final"


class LifecyclePhase(str, Enum):
    """Lifecycle phases."""
    START = "start"
    END = "end"
    ERROR = "error"


@dataclass
class ChatRequest:
    """Chat request payload."""
    message: str
    session_id: str = "cli:direct"
    stream: bool = True


@dataclass
class ChatResponse:
    """Chat response payload."""
    run_id: str
    content: str = ""
    error: str | None = None


@dataclass
class WSEvent:
    """WebSocket event."""
    type: EventType
    session_key: str
    run_id: str
    data: dict[str, Any] | None = None


@dataclass
class HealthResponse:
    """Health check response."""
    status: str
    version: str
    nanobot_version: str
