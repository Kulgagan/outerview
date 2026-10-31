"""Transcript normalization and deterministic command handling."""

from __future__ import annotations

import re

from .events import Command


VOCABULARY_PROMPT = (
    "hash map, two pointers, sliding window, time complexity, space complexity, "
    "Big O, O(n), O(n squared), binary search, DFS, BFS, heap, recursion"
)

_COMMANDS: list[tuple[re.Pattern[str], Command]] = [
    (re.compile(r"^\s*start\s*[.!]?$", re.I), Command.START),
    (re.compile(r"\brepeat (?:the )?(?:question|problem)\b", re.I), Command.REPEAT),
    (re.compile(r"\b(?:end (?:the )?interview|stop)\b", re.I), Command.END),
    (re.compile(r"^\s*quit\s*[.!]?$", re.I), Command.QUIT),
]


def match_command(text: str, *, idle: bool = False) -> Command | None:
    """Return an allowed explicit voice command, without involving the LLM."""
    for pattern, command in _COMMANDS:
        if pattern.search(text):
            if command is Command.QUIT and not idle:
                return None
            return command
    return None


def normalize_complexity(text: str) -> str:
    """Normalize common speech-to-text variants without rewriting normal speech."""
    normalized = text
    rules = [
        (r"\b(?:big\s*)?oh\s+of\s+n\s+squared\b", "O(n^2)"),
        (r"\b(?:big\s*)?o(?:h)?\s+of\s+n\s+squared\b", "O(n^2)"),
        (r"\b(?:big\s*)?oh\s+of\s+n\s+cubed\b", "O(n^3)"),
        (r"\b(?:big\s*)?oh\s+of\s+log\s+n\b", "O(log n)"),
        (r"\b(?:big\s*)?oh\s+of\s+n\s+log\s+n\b", "O(n log n)"),
        (r"\b(?:big\s*)?oh\s+of\s+n\b", "O(n)"),
        (r"\bconstant(?:\s+time)?\b", "O(1)"),
    ]
    for pattern, replacement in rules:
        normalized = re.sub(pattern, replacement, normalized, flags=re.I)
    return normalized


def has_complexity(text: str) -> bool:
    return bool(re.search(r"O\([^)]*\)", text))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def filler_count(text: str) -> int:
    return len(re.findall(r"\b(?:um+|uh+|like|you know)\b", text, flags=re.I))

