"""Small worker threads used only by the live audio runtime."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
import time

from .events import ClockTick, Command, CommandEvent, UserTurn, Utterance
from .stt import Transcriber


class STTWorker(Thread):
    def __init__(self, utterances: Queue, output: Queue, transcriber: Transcriber) -> None:
        super().__init__(name="stt", daemon=True)
        self.utterances = utterances
        self.output = output
        self.transcriber = transcriber
        self.stop_event = Event()

    def stop(self) -> None:
        self.stop_event.set()
        self.utterances.put(None)

    def run(self) -> None:  # pragma: no cover - needs model
        while not self.stop_event.is_set():
            utterance = self.utterances.get()
            if utterance is None:
                return
            if not isinstance(utterance, Utterance):
                continue
            started = time.monotonic()
            text = self.transcriber.transcribe(utterance.samples, utterance.sample_rate)
            if text:
                self.output.put(UserTurn(text=text, started_at=started, ended_at=time.monotonic()))


class Clock(Thread):
    def __init__(self, output: Queue) -> None:
        super().__init__(name="clock", daemon=True)
        self.output = output
        self.stop_event = Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:  # pragma: no cover - timing worker
        while not self.stop_event.wait(1.0):
            self.output.put(ClockTick())


class TerminalCommands(Thread):
    """Daemon input helper: Enter is the non-STT start fallback."""

    def __init__(self, output: Queue) -> None:
        super().__init__(name="terminal-commands", daemon=True)
        self.output = output

    def run(self) -> None:  # pragma: no cover - interactive terminal
        while True:
            try:
                line = input()
            except EOFError:
                return
            if not line.strip():
                self.output.put(CommandEvent(command=Command.START))

