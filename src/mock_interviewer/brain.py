"""The deterministic interview orchestrator and phase machine."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
from typing import Callable, Protocol
import re
import time

from .config import AppConfig
from .events import BargeIn, ClockTick, Command, CommandEvent, Event as RuntimeEvent, ScreenUpdate, Trigger, UserTurn
from .llm import LLMClient, LLMUnavailable, SentenceSplitter, TaggedReply, prompt_text, sanitize_spoken_text
from .scoring import Judge, Scorecard, score_session
from .session import InterviewEvent, InterviewSession, Phase, Problem
from .text import has_complexity, match_command, word_count


class Speaker(Protocol):
    def submit(self, text: str) -> None: ...

    def stop(self) -> None: ...


@dataclass(slots=True)
class BrainResult:
    quit_requested: bool = False
    report: Scorecard | None = None


class Brain:
    """Owns all state transitions and never lets an LLM make control decisions."""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        speaker: Speaker,
        *,
        problem_loader: Callable[[], Problem] | None = None,
        judge: Judge | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.llm = llm
        self.speaker = speaker
        self.problem_loader = problem_loader or (lambda: Problem())
        self.judge = judge
        self.clock = clock
        self.session = InterviewSession(started_at=clock())
        self.cancel_llm = Event()
        self._llm_busy = False
        self._last_candidate_at = self.session.started_at
        self._last_interjection_at = -float("inf")
        self._trigger_at: dict[str, float] = {}
        self._finished = False
        self._approach_explained = False
        self.result = BrainResult()

    @property
    def llm_busy(self) -> bool:
        return self._llm_busy

    @property
    def phase(self) -> Phase:
        return self.session.phase

    def process(self, event: RuntimeEvent) -> BrainResult:
        if isinstance(event, CommandEvent):
            return self._handle_command(event.command, event.at)
        if isinstance(event, BargeIn):
            self.cancel_llm.set()
            self.speaker.stop()
            return self.result
        if isinstance(event, UserTurn):
            return self._handle_user_turn(event)
        if isinstance(event, ScreenUpdate):
            return self._handle_screen(event)
        if isinstance(event, Trigger):
            self._handle_trigger(event)
            return self.result
        if isinstance(event, ClockTick):
            self._handle_tick(event.at)
        return self.result

    def _handle_command(self, command: Command, at: float) -> BrainResult:
        if command is Command.START and self.phase is Phase.IDLE:
            self._start(at)
        elif command is Command.REPEAT and self.phase is not Phase.IDLE:
            self._speak(f"The problem is: {self.session.problem.statement}", "ANSWER", at)
        elif command is Command.END and self.phase not in {Phase.IDLE, Phase.REPORT}:
            self._finish(at, "[WRAP] Let's wrap up here. I will prepare your feedback.")
        elif command is Command.QUIT and self.phase is Phase.IDLE:
            self.result.quit_requested = True
        return self.result

    def _start(self, at: float) -> None:
        self.session.started_at = at
        self._last_candidate_at = at
        self._transition(Phase.INTRO, at)
        try:
            self.session.problem = self.problem_loader()
        except Exception as exc:
            self.session.problem = Problem(notes=f"Screen intake unavailable: {exc}")
        opening = self.session.problem.statement.strip().replace("\n", " ")
        if len(opening) > 360:
            opening = opening[:357].rsplit(" ", 1)[0] + "..."
        self._speak(f"Here is the problem: {opening} Please ask any clarifying questions before proposing an approach.", "ANSWER", at)
        self._transition(Phase.CLARIFY, at)

    def _handle_user_turn(self, event: UserTurn) -> BrainResult:
        text = event.text.strip()
        if not text:
            return self.result
        command = match_command(text, idle=self.phase is Phase.IDLE)
        if command is not None:
            return self._handle_command(command, event.at)
        if self.phase is Phase.IDLE:
            return self.result
        start, end = event.started_at or event.at, event.ended_at or event.at
        self.session.add_turn("candidate", text, start, end)
        self._last_candidate_at = end
        if word_count(text) < 3:
            return self.result
        phase_before = self.phase
        self._advance_from_candidate(text, event.at)
        if phase_before is Phase.FOLLOWUP:
            self._finish(event.at, "[WRAP] Thanks. That completes the follow-up. I will prepare your feedback.")
            return self.result
        if re.search(r"\b(?:hint|stuck|help)\b", text, flags=re.I):
            self._handle_trigger(Trigger(kind="hint_requested", detail=text, at=event.at))
        elif end - start > 30:
            self._handle_trigger(Trigger(kind="long_monologue", at=event.at))
        else:
            self._reply(f"CANDIDATE: {text}", event.at)
        return self.result

    def _advance_from_candidate(self, text: str, at: float) -> None:
        lower = text.lower()
        if self.phase is Phase.CLARIFY and any(token in lower for token in ("brute force", "approach", "i would", "we could")):
            self._approach_explained = True
            self._transition(Phase.APPROACH, at)
        if self.phase is Phase.APPROACH and any(token in lower for token in ("let me code", "start coding", "implement", "write the code")):
            self._transition(Phase.CODING, at)
        if self.phase is Phase.CODING and any(token in lower for token in ("done", "ready to test", "test this")):
            self._transition(Phase.TESTING, at)
            self._prompt_missing_completion_items(at)
        if self.phase is Phase.TESTING and any(token in lower for token in ("confident", "looks correct", "passed")):
            self._transition(Phase.FOLLOWUP, at)

    def _handle_screen(self, event: ScreenUpdate) -> BrainResult:
        self.session.add_snapshot(event.code, event.at)
        self.session.add_run(event.run_status, event.failing_case, event.at)
        line_count = len([line for line in event.code.splitlines() if line.strip()])
        if self.phase in {Phase.CLARIFY, Phase.APPROACH} and line_count >= 4:
            if not self._approach_explained:
                self._handle_trigger(Trigger(kind="code_before_plan", at=event.at))
            self._transition(Phase.CODING, event.at)
        elif self.phase is Phase.CODING and event.run_status != "none":
            self._transition(Phase.TESTING, event.at)
            self._prompt_missing_completion_items(event.at)
        if event.run_status in {"wrong_answer", "runtime_error", "time_limit", "compile_error"}:
            self._handle_trigger(Trigger(kind="failing_run", detail=event.run_status, at=event.at))
        elif event.run_status == "accepted" and self.phase is Phase.TESTING:
            self._transition(Phase.FOLLOWUP, event.at)
            self._reply("TRIGGER: accepted run; ask one short follow-up.", event.at)
        return self.result

    def _handle_tick(self, at: float) -> None:
        if self.phase in {Phase.IDLE, Phase.REPORT}:
            return
        elapsed = at - self.session.started_at
        if elapsed >= self.config.session.minutes * 60:
            self._finish(at, "[WRAP] We are at time. I will prepare your feedback.")
            return
        if elapsed >= self.config.session.minutes * 60 * 0.8:
            self._handle_trigger(Trigger(kind="time_80", at=at))
        elif elapsed >= self.config.session.minutes * 60 * 0.5:
            self._handle_trigger(Trigger(kind="time_50", at=at))
        if self.phase is Phase.CLARIFY and elapsed >= 300:
            self._transition(Phase.APPROACH, at)
        silence = at - self._last_candidate_at
        if self.phase is Phase.APPROACH and silence >= 25:
            self._handle_trigger(Trigger(kind="dead_air_approach", at=at))
        elif self.phase in {Phase.CODING, Phase.TESTING} and silence >= 60:
            self._handle_trigger(Trigger(kind="dead_air_coding", at=at))

    def _handle_trigger(self, event: Trigger) -> None:
        cooldowns = {
            "dead_air_approach": 30,
            "dead_air_coding": 60,
            "code_before_plan": float("inf"),
            "failing_run": 1,
            "time_50": float("inf"),
            "time_80": float("inf"),
            "hint_requested": 0,
            "no_complexity": float("inf"),
            "done_without_testing": float("inf"),
            "long_monologue": 45,
        }
        cooldown = cooldowns.get(event.kind, 30)
        last = self._trigger_at.get(event.kind, -float("inf"))
        if event.at - last < cooldown or event.at - self._last_interjection_at < 30:
            return
        if event.kind != "hint_requested" and self._llm_busy:
            return
        self._trigger_at[event.kind] = event.at
        self._last_interjection_at = event.at
        self._reply(f"TRIGGER: {event.kind}. {event.detail}", event.at)

    def _prompt_missing_completion_items(self, at: float) -> None:
        candidate_text = " ".join(turn.text for turn in self.session.turns if turn.speaker == "candidate")
        if not has_complexity(candidate_text):
            self._handle_trigger(Trigger(kind="no_complexity", at=at))
        if not re.search(r"\b(?:test|edge case|example)\b", candidate_text, flags=re.I):
            self._handle_trigger(Trigger(kind="done_without_testing", at=at))

    def _reply(self, prompt: str, at: float) -> None:
        if self._llm_busy or self.phase in {Phase.IDLE, Phase.REPORT}:
            return
        self._llm_busy = True
        self.cancel_llm.clear()
        raw_tag = ""
        tag: str | None = None
        discard_model_content = False
        splitter = SentenceSplitter()
        spoken_parts: list[str] = []
        try:
            for token in self.llm.stream_chat(
                self._messages(prompt),
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                cancel=self.cancel_llm,
            ):
                if self.cancel_llm.is_set():
                    return
                if tag is None:
                    raw_tag += token
                    # Give a leaked reasoning block enough room to close before deciding
                    # that the model omitted its control tag.
                    if "]" not in raw_tag and len(raw_tag) < 512:
                        continue
                    parsed = TaggedReply.parse(raw_tag)
                    tag = self._authorize_tag(parsed.tag)
                    remainder = parsed.text
                    if tag == "PASS":
                        continue
                    if parsed.tag.startswith("HINT") and tag != parsed.tag:
                        discard_model_content = True
                        remainder = "What property of the input should you examine next?"
                    if remainder:
                        for sentence in splitter.feed(remainder):
                            self._submit_sentence(sentence, spoken_parts)
                    continue
                if tag != "PASS" and not discard_model_content:
                    for sentence in splitter.feed(token):
                        self._submit_sentence(sentence, spoken_parts)
            if tag is None:
                tag = "PROBE"
                for sentence in splitter.feed(raw_tag):
                    self._submit_sentence(sentence, spoken_parts)
            if tag != "PASS" and not discard_model_content:
                for sentence in splitter.finish():
                    self._submit_sentence(sentence, spoken_parts)
        except LLMUnavailable as exc:
            tag = "ACK"
            self._submit_sentence("My local model is unavailable. Please check llama-server, then try again.", spoken_parts)
            self.session.problem.notes += f"\nLLM error: {exc}"
        finally:
            self._llm_busy = False
        if tag and not self.cancel_llm.is_set():
            spoken = " ".join(spoken_parts).strip()
            self.session.interviewer_events.append(InterviewEvent(at, tag, spoken))
            if spoken:
                self.session.add_turn("interviewer", spoken, at, self.clock())

    def _authorize_tag(self, tag: str) -> str:
        if not tag.startswith("HINT"):
            return tag
        level = int(tag[-1])
        previous = max((int(event.tag[-1]) for event in self.session.interviewer_events if event.tag.startswith("HINT")), default=0)
        if level > previous + 1:
            return "PROBE"
        return tag

    def _submit_sentence(self, text: str, spoken_parts: list[str]) -> None:
        clean = sanitize_spoken_text(text)
        if clean:
            spoken_parts.append(clean)
            self.speaker.submit(clean)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        recent = self.session.turns[-10:]
        transcript = "\n".join(f"{turn.speaker.upper()}: {turn.text}" for turn in recent)
        code = self.session.code_snapshots[-1].code if self.session.code_snapshots else ""
        allowed_hint = max((int(event.tag[-1]) for event in self.session.interviewer_events if event.tag.startswith("HINT")), default=0) + 1
        system = (
            prompt_text("persona.txt")
            + f"\nYou may only give hints at level {min(allowed_hint, 3)} or below.\n"
            f"PROBLEM: {self.session.problem.statement}\nPRIVATE NOTES: {self.session.problem.notes}\n"
            f"STATE: phase={self.phase.value}; elapsed={int(self.session.elapsed())}s; hints={allowed_hint - 1}; last_run={self.session.runs[-1].status if self.session.runs else 'none'}\n"
            f"LATEST CODE:\n{code[-6000:]}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": transcript + "\n" + prompt}]

    def _speak(self, text: str, tag: str, at: float) -> None:
        clean = sanitize_spoken_text(text)
        if not clean:
            return
        self.speaker.submit(clean)
        self.session.interviewer_events.append(InterviewEvent(at, tag, clean))
        self.session.add_turn("interviewer", clean, at, self.clock())

    def _transition(self, phase: Phase, at: float) -> None:
        self.session.set_phase(phase, at)

    def _finish(self, at: float, spoken: str) -> None:
        if self._finished:
            return
        self._finished = True
        parsed = TaggedReply.parse(spoken)
        self._transition(Phase.WRAPUP, at)
        self._speak(parsed.text, parsed.tag, at)
        try:
            self.result.report = score_session(self.session, self.config.session.bar, self.judge)
        except Exception as exc:
            self.session.problem.notes += f"\nJudge unavailable; report used deterministic metrics only: {exc}"
            self.result.report = score_session(self.session, self.config.session.bar)
        self._transition(Phase.REPORT, self.clock())

    def run(self, events: Queue, shutdown: Event) -> BrainResult:
        """Consume queue events until the live runner requests shutdown."""
        while not shutdown.is_set() and not self.result.quit_requested:
            if self.result.report is not None:
                return self.result
            try:
                event = events.get(timeout=0.25)
            except Empty:
                self.process(ClockTick(at=self.clock()))
                continue
            self.process(event)
        return self.result

