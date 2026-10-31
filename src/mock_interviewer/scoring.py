"""Evidence-gated interview scorecard and deterministic metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Protocol

from .llm import LLMClient, prompt_text
from .session import InterviewSession, Phase
from .text import filler_count, has_complexity, word_count


CHECKLIST: dict[str, tuple[str, ...]] = {
    "communication": (
        "clarified_before_coding",
        "restated_problem_or_example",
        "explained_approach_before_coding",
        "narrated_while_coding",
        "discussed_tradeoffs",
        "responded_to_hints",
    ),
    "problem_solving": (
        "stated_brute_force_and_complexity",
        "reached_near_optimal_solution",
        "stated_correct_time_and_space_complexity",
        "hints_within_bar",
        "considered_second_approach",
        "handled_followup",
    ),
    "technical_competency": (
        "final_code_compiles",
        "solved_problem",
        "at_most_one_wrong_submission",
        "readable_structure",
        "idiomatic_data_structures",
        "finished_within_time",
    ),
    "testing": (
        "proposed_two_test_cases",
        "named_two_real_edge_cases",
        "walked_through_an_example",
        "ran_or_simulated_tests",
        "found_own_bug_or_no_bug_existed",
        "did_not_declare_done_without_testing",
    ),
}


@dataclass(frozen=True, slots=True)
class BarProfile:
    name: str
    max_hint_level_for_credit: int
    strong_hire_average: float


BAR_PROFILES = {
    "intern": BarProfile("intern", 1, 3.5),
    "new_grad": BarProfile("new_grad", 0, 3.5),
    "senior": BarProfile("senior", 0, 3.5),
}


@dataclass(slots=True)
class Metrics:
    time_to_first_clarification: float | None
    clarifications: int
    time_to_first_code: float | None
    talk_ratio_coding: float
    longest_silence_coding: float
    words_per_minute: float
    filler_rate: float
    hints_by_level: dict[str, int]
    first_hint_at: float | None
    complexity_stated: bool
    brute_force_before_code: bool
    run_count: int
    final_status: str
    phase_seconds: dict[str, float]


@dataclass(slots=True)
class Scorecard:
    scores: dict[str, int]
    checklist: dict[str, dict[str, bool]]
    evidence: dict[str, dict[str, str]]
    metrics: Metrics
    verdict: str


class Judge(Protocol):
    def judge(self, session: InterviewSession, metrics: Metrics) -> dict[str, Any]: ...


class LLMJudge:
    """A non-live judge that can use a larger local model if configured."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def judge(self, session: InterviewSession, metrics: Metrics) -> dict[str, Any]:
        transcript = "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns)
        final_code = session.code_snapshots[-1].code if session.code_snapshots else ""
        payload = {
            "checklist": CHECKLIST,
            "metrics": asdict(metrics),
            "problem_notes": session.problem.notes,
            "transcript": transcript,
            "final_code": final_code,
        }
        messages = [
            {"role": "system", "content": prompt_text("judge.txt")},
            {"role": "user", "content": json.dumps(payload)},
        ]
        return self.llm.json_chat(messages)


def calculate_metrics(session: InterviewSession, now: float | None = None) -> Metrics:
    end = now if now is not None else session.elapsed() + session.started_at
    candidate_turns = [turn for turn in session.turns if turn.speaker == "candidate"]
    coding_turns = [turn for turn in candidate_turns if turn.phase is Phase.CODING]
    coding_duration = _phase_duration(session, Phase.CODING, end)
    coding_talk = sum(max(0.0, turn.t_end - turn.t_start) for turn in coding_turns)
    total_words = sum(word_count(turn.text) for turn in candidate_turns)
    elapsed_minutes = max(session.elapsed(end), 1.0) / 60
    total_fillers = sum(filler_count(turn.text) for turn in candidate_turns)
    all_text = " ".join(turn.text for turn in candidate_turns)
    code_at = session.code_snapshots[0].t - session.started_at if session.code_snapshots else None
    clarification_turns = [turn for turn in candidate_turns if turn.phase is Phase.CLARIFY and "?" in turn.text]
    first_clarification = clarification_turns[0].t_start - session.started_at if clarification_turns else None
    hints = {f"HINT{level}": 0 for level in range(1, 4)}
    hint_times: list[float] = []
    for event in session.interviewer_events:
        if event.tag in hints:
            hints[event.tag] += 1
            hint_times.append(event.t)
    final_status = session.runs[-1].status if session.runs else "none"
    before_code = " ".join(turn.text for turn in candidate_turns if code_at is None or turn.t_start - session.started_at <= code_at)
    return Metrics(
        time_to_first_clarification=first_clarification,
        clarifications=len(clarification_turns),
        time_to_first_code=code_at,
        talk_ratio_coding=min(1.0, coding_talk / coding_duration) if coding_duration else 0.0,
        longest_silence_coding=_longest_silence(coding_turns, session, end),
        words_per_minute=total_words / elapsed_minutes,
        filler_rate=(total_fillers / total_words) if total_words else 0.0,
        hints_by_level=hints,
        first_hint_at=(min(hint_times) - session.started_at) if hint_times else None,
        complexity_stated=has_complexity(all_text),
        brute_force_before_code=("brute force" in before_code.lower()),
        run_count=len(session.runs),
        final_status=final_status,
        phase_seconds={phase.value: _phase_duration(session, phase, end) for phase in Phase},
    )


def score_session(session: InterviewSession, bar: str, judge: Judge | None = None) -> Scorecard:
    metrics = calculate_metrics(session)
    raw = judge.judge(session, metrics) if judge is not None else {}
    checklist, evidence = _evidence_gated_checklist(raw)
    _apply_deterministic_answers(checklist, evidence, session, metrics, bar)
    scores = {dimension: _band(sum(items.values())) for dimension, items in checklist.items()}
    _apply_hard_caps(scores, session, metrics)
    return Scorecard(scores, checklist, evidence, metrics, verdict(scores, bar))


def _evidence_gated_checklist(raw: dict[str, Any]) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, str]]]:
    checklist = {dimension: {item: False for item in items} for dimension, items in CHECKLIST.items()}
    evidence = {dimension: {item: "" for item in items} for dimension, items in CHECKLIST.items()}
    answers = raw.get("checklist", raw) if isinstance(raw, dict) else {}
    if not isinstance(answers, dict):
        return checklist, evidence
    for dimension, items in checklist.items():
        response = answers.get(dimension, {})
        if not isinstance(response, dict):
            continue
        for item in items:
            answer = response.get(item, {})
            if not isinstance(answer, dict):
                continue
            quote = str(answer.get("evidence", "")).strip()
            # A judge must cite candidate speech; unsupported generosity is ignored.
            items[item] = bool(answer.get("yes", False)) and bool(quote)
            evidence[dimension][item] = quote if items[item] else ""
    return checklist, evidence


def _apply_deterministic_answers(
    checklist: dict[str, dict[str, bool]],
    evidence: dict[str, dict[str, str]],
    session: InterviewSession,
    metrics: Metrics,
    bar: str,
) -> None:
    profile = BAR_PROFILES.get(bar, BAR_PROFILES["intern"])
    final = metrics.final_status
    tech = checklist["technical_competency"]
    has_code = bool(session.code_snapshots)
    tech["final_code_compiles"] = final not in {"compile_error", "runtime_error"} and has_code
    tech["at_most_one_wrong_submission"] = has_code and sum(run.status == "wrong_answer" for run in session.runs) <= 1
    tech["finished_within_time"] = has_code and session.phase in {Phase.WRAPUP, Phase.REPORT}
    if final == "accepted":
        tech["solved_problem"] = True
    problem = checklist["problem_solving"]
    problem["hints_within_bar"] = max((int(tag[-1]) for tag, count in metrics.hints_by_level.items() if count), default=0) <= profile.max_hint_level_for_credit
    testing = checklist["testing"]
    testing["ran_or_simulated_tests"] = metrics.run_count > 0
    if metrics.complexity_stated:
        evidence["problem_solving"].setdefault("stated_correct_time_and_space_complexity", "")


def _apply_hard_caps(scores: dict[str, int], session: InterviewSession, metrics: Metrics) -> None:
    candidate_text = " ".join(turn.text.lower() for turn in session.turns if turn.speaker == "candidate")
    mentioned_testing = any(word in candidate_text for word in ("test", "edge", "case", "example"))
    if "done" in candidate_text and not mentioned_testing:
        scores["testing"] = min(scores["testing"], 1)
    if _phase_duration(session, Phase.CODING, session.started_at + session.elapsed()) > 0 and (
        metrics.talk_ratio_coding == 0 or metrics.longest_silence_coding > 60
    ):
        scores["communication"] = min(scores["communication"], 1)


def _band(points: int) -> int:
    if points <= 1:
        return 1
    if points <= 3:
        return 2
    if points <= 5:
        return 3
    return 4


def verdict(scores: dict[str, int], bar: str = "intern") -> str:
    average = sum(scores.values()) / len(scores)
    low = min(scores.values())
    profile = BAR_PROFILES.get(bar, BAR_PROFILES["intern"])
    if average < 2.0:
        return "Strong no hire"
    if low == 1 or average < 2.5:
        return "Leaning no hire"
    if average < 3.0:
        return "Leaning hire"
    if average >= profile.strong_hire_average and low >= 3:
        return "Strong hire"
    return "Hire"


def _phase_duration(session: InterviewSession, phase: Phase, end: float) -> float:
    log = [(logged_phase, at) for logged_phase, at in session.phase_log]
    total = 0.0
    for index, (logged_phase, at) in enumerate(log):
        if logged_phase is phase:
            next_at = log[index + 1][1] if index + 1 < len(log) else end
            total += max(0.0, next_at - at)
    return total


def _longest_silence(turns, session: InterviewSession, end: float) -> float:
    coding_start = next((at for phase, at in session.phase_log if phase is Phase.CODING), None)
    if coding_start is None:
        return 0.0
    coding_end = coding_start + _phase_duration(session, Phase.CODING, end)
    cursor, longest = coding_start, 0.0
    for turn in turns:
        longest = max(longest, turn.t_start - cursor)
        cursor = max(cursor, turn.t_end)
    return max(longest, coding_end - cursor)

