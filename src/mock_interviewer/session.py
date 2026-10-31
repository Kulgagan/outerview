"""In-memory interview records. Nothing here persists between sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class Phase(str, Enum):
    IDLE = "idle"
    INTRO = "intro"
    CLARIFY = "clarify"
    APPROACH = "approach"
    CODING = "coding"
    TESTING = "testing"
    FOLLOWUP = "followup"
    WRAPUP = "wrapup"
    REPORT = "report"


@dataclass(slots=True)
class Problem:
    title: str = "Practice problem"
    statement: str = "Explain an approach to solve the problem on your screen."
    examples: list[str] = field(default_factory=list)
    constraints: str = ""
    language: str = "Python"
    notes: str = "Ask clarifying questions and discuss the simplest approach first."


@dataclass(slots=True)
class Turn:
    speaker: str
    text: str
    t_start: float
    t_end: float
    phase: Phase


@dataclass(slots=True)
class Snapshot:
    t: float
    code: str


@dataclass(slots=True)
class RunEvent:
    t: float
    status: str
    failing_case: str = ""


@dataclass(slots=True)
class InterviewEvent:
    t: float
    tag: str
    text: str


@dataclass(slots=True)
class InterviewSession:
    started_at: float = field(default_factory=time.monotonic)
    problem: Problem = field(default_factory=Problem)
    turns: list[Turn] = field(default_factory=list)
    code_snapshots: list[Snapshot] = field(default_factory=list)
    runs: list[RunEvent] = field(default_factory=list)
    interviewer_events: list[InterviewEvent] = field(default_factory=list)
    phase_log: list[tuple[Phase, float]] = field(default_factory=list)
    phase: Phase = Phase.IDLE

    def set_phase(self, phase: Phase, at: float | None = None) -> None:
        when = time.monotonic() if at is None else at
        if self.phase is phase:
            return
        self.phase = phase
        self.phase_log.append((phase, when))

    def elapsed(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.started_at

    def add_turn(self, speaker: str, text: str, start: float, end: float) -> None:
        self.turns.append(Turn(speaker, text, start, end, self.phase))

    def add_snapshot(self, code: str, at: float | None = None) -> None:
        if code and (not self.code_snapshots or self.code_snapshots[-1].code != code):
            self.code_snapshots.append(Snapshot(time.monotonic() if at is None else at, code))

    def add_run(self, status: str, failing_case: str = "", at: float | None = None) -> None:
        if status != "none":
            self.runs.append(RunEvent(time.monotonic() if at is None else at, status, failing_case))

