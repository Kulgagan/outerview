"""Events exchanged by the workers and the Brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class Command(str, Enum):
    START = "start"
    REPEAT = "repeat"
    END = "end"
    QUIT = "quit"


@dataclass(slots=True)
class Event:
    at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class Utterance(Event):
    samples: Any = None
    sample_rate: int = 16_000


@dataclass(slots=True)
class UserTurn(Event):
    text: str = ""
    started_at: float | None = None
    ended_at: float | None = None


@dataclass(slots=True)
class CommandEvent(Event):
    command: Command = Command.START


@dataclass(slots=True)
class BargeIn(Event):
    pass


@dataclass(slots=True)
class ClockTick(Event):
    pass


@dataclass(slots=True)
class ScreenUpdate(Event):
    code: str = ""
    run_status: str = "none"
    failing_case: str = ""
    stable: bool = False


@dataclass(slots=True)
class Trigger(Event):
    kind: str = ""
    detail: str = ""

