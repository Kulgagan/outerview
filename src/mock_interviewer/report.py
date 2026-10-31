"""Terminal and Markdown report rendering."""

from __future__ import annotations

from pathlib import Path

from .scoring import Scorecard
from .session import InterviewSession


def feedback(scorecard: Scorecard) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    improvements: list[str] = []
    # Start with one signal from each dimension so the summary is actionable.
    for dimension, items in scorecard.checklist.items():
        for item, passed in items.items():
            if item == "hints_within_bar" and not any(scorecard.metrics.hints_by_level.values()):
                continue
            label = item.replace("_", " ")
            quote = scorecard.evidence[dimension].get(item, "")
            if passed and len(strengths) < 3 and not any(entry.startswith(label.title()) for entry in strengths):
                strengths.append(f"{label.title()}" + (f": “{quote}”" if quote else ""))
                break
            if not passed and len(improvements) < 3:
                improvements.append(f"Practice {label.replace('_', ' ')}.")
                break
    while len(strengths) < 3:
        strengths.append("Complete another timed interview to collect more evidence.")
    while len(improvements) < 3:
        improvements.append("Continue practicing with a fresh problem.")
    return strengths, improvements


def markdown_report(session: InterviewSession, scorecard: Scorecard) -> str:
    metrics = scorecard.metrics
    strengths, improvements = feedback(scorecard)
    lines = [
        "# Mock interview report",
        "",
        f"## {scorecard.verdict}",
        "",
        "| Dimension | Score |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name.replace('_', ' ').title()} | {score} / 4 |" for name, score in scorecard.scores.items())
    lines += ["", "## Timeline", ""]
    for phase, seconds in metrics.phase_seconds.items():
        if seconds:
            lines.append(f"- {phase}: {seconds:.1f}s")
    lines += ["", "## Metrics", "", f"- Runs: {metrics.run_count} ({metrics.final_status})", f"- Talk ratio while coding: {metrics.talk_ratio_coding:.0%}", f"- Longest coding silence: {metrics.longest_silence_coding:.1f}s", f"- Filler rate: {metrics.filler_rate:.1%}", f"- Hints: {metrics.hints_by_level}", "", "## Evidence", ""]
    for dimension, items in scorecard.checklist.items():
        for item, passed in items.items():
            if passed:
                quote = scorecard.evidence[dimension].get(item, "")
                lines.append(f"- {dimension}: {item.replace('_', ' ')}" + (f" — “{quote}”" if quote else ""))
    lines += ["", "## Strengths", ""] + [f"- {item}" for item in strengths]
    lines += ["", "## Improve next", ""] + [f"- {item}" for item in improvements]
    if session.code_snapshots:
        lines += ["", "## Final code", "", "```", session.code_snapshots[-1].code, "```"]
    lines += ["", "## Interviewer notes", "", session.problem.notes]
    return "\n".join(lines) + "\n"


def print_report(session: InterviewSession, scorecard: Scorecard) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover - requirements boundary
        print(markdown_report(session, scorecard))
        return
    console = Console()
    table = Table(title=scorecard.verdict)
    table.add_column("Dimension")
    table.add_column("Score", justify="right")
    for dimension, score in scorecard.scores.items():
        table.add_row(dimension.replace("_", " ").title(), f"{score} / 4")
    console.print(table)
    console.print(
        f"Runs: {scorecard.metrics.run_count} ({scorecard.metrics.final_status}) | "
        f"Coding talk ratio: {scorecard.metrics.talk_ratio_coding:.0%} | "
        f"Longest silence: {scorecard.metrics.longest_silence_coding:.1f}s"
    )
    strengths, improvements = feedback(scorecard)
    console.print("Strengths:")
    for item in strengths:
        console.print(f"- {item}")
    console.print("Improve next:")
    for item in improvements:
        console.print(f"- {item}")


def save_report(path: str | Path, session: InterviewSession, scorecard: Scorecard) -> None:
    Path(path).write_text(markdown_report(session, scorecard), encoding="utf-8")

