"""Local OpenAI-compatible LLM client, prompt loading, and stream parsing."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
from pathlib import Path
import re
from threading import Event
from typing import Any, Iterator, Protocol, Sequence

from .config import LLMConfig
from .session import Problem


VALID_TAGS = frozenset({"PASS", "ACK", "ANSWER", "PROBE", "CHALLENGE", "HINT1", "HINT2", "HINT3", "WRAP"})


class LLMClient(Protocol):
    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        cancel: Event | None = None,
    ) -> Iterator[str]: ...

    def json_chat(self, messages: Sequence[dict[str, Any]], *, cancel: Event | None = None) -> dict[str, Any]: ...


class LLMUnavailable(RuntimeError):
    pass


class OpenAICompatLLM:
    """A deliberately small client for local llama-server deployments."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        cancel: Event | None = None,
    ) -> Iterator[str]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - guarded by requirements
            raise LLMUnavailable("Install httpx to use the local LLM.") from exc
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                with client.stream("POST", self.endpoint, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if cancel is not None and cancel.is_set():
                            return
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            token = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                            continue
                        if token:
                            yield str(token)
        except Exception as exc:  # httpx errors differ across installed versions
            raise LLMUnavailable(f"Local LLM request failed: {exc}") from exc

    def json_chat(self, messages: Sequence[dict[str, Any]], *, cancel: Event | None = None) -> dict[str, Any]:
        text = "".join(self.stream_chat(messages, max_tokens=800, temperature=0.1, cancel=cancel))
        return parse_json_object(text)

    def extract_problem(self, image_bytes: bytes, mime_type: str = "image/png") -> Problem:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": prompt_text("problem_extract.txt")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the coding problem from this screen."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ],
            },
        ]
        data = self.json_chat(messages)
        return validate_problem(data)

    def observe_code(self, image_bytes: bytes, mime_type: str = "image/png") -> dict[str, str]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": prompt_text("observer.txt")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the current editor and result area."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ],
            },
        ]
        data = self.json_chat(messages)
        status = str(data.get("run_status", "none"))
        if status not in {"none", "running", "accepted", "wrong_answer", "runtime_error", "time_limit", "compile_error"}:
            raise ValueError(f"Unsupported run status: {status}")
        return {"code": str(data.get("code", "")), "run_status": status, "failing_case": str(data.get("failing_case", ""))}

    def generate_notes(self, problem: Problem) -> str:
        messages = [
            {"role": "system", "content": prompt_text("notes.txt")},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "title": problem.title,
                        "statement": problem.statement,
                        "examples": problem.examples,
                        "constraints": problem.constraints,
                    }
                ),
            },
        ]
        return sanitize_spoken_text("".join(self.stream_chat(messages, max_tokens=500, temperature=0.2)))


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM did not return a JSON object")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM JSON response must be an object")
    return data


def validate_problem(data: dict[str, Any]) -> Problem:
    title = str(data.get("title", "")).strip()
    statement = str(data.get("statement", "")).strip()
    if not title or not statement:
        raise ValueError("Problem extraction needs non-empty title and statement")
    examples = data.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("Problem examples must be a list")
    return Problem(
        title=title,
        statement=statement,
        examples=[str(example) for example in examples],
        constraints=str(data.get("constraints", "")),
        language=str(data.get("language", "Python")),
    )


def prompt_text(name: str) -> str:
    return (Path(__file__).with_name("prompts") / name).read_text(encoding="utf-8")


def sanitize_spoken_text(text: str) -> str:
    """Remove accidental reasoning markup and punctuation unsuitable for speech."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    clean = re.sub(r"</?think>", "", clean, flags=re.I)
    clean = re.sub(r"[`*_#]", "", clean)
    clean = re.sub(r"[\U0001F300-\U0001FAFF]", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


@dataclass(frozen=True, slots=True)
class TaggedReply:
    tag: str
    text: str

    @classmethod
    def parse(cls, raw: str) -> "TaggedReply":
        cleaned = sanitize_spoken_text(raw)
        match = re.match(r"^\[([A-Z0-9]+)\]\s*", cleaned)
        if not match or match.group(1) not in VALID_TAGS:
            return cls("PROBE", cleaned)
        return cls(match.group(1), cleaned[match.end() :].strip())


class SentenceSplitter:
    """Flush short completed sentences while an LLM stream is still arriving."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        result: list[str] = []
        while True:
            stripped = self.buffer.rstrip()
            boundary = re.search(r"[.!?](?:\s+|$)", stripped)
            clause = re.search(r"[,;:](?:\s+|$)", stripped)
            chosen = boundary if boundary and len(stripped[: boundary.end()].split()) >= 5 else None
            if chosen is None and clause and len(stripped[: clause.end()].split()) >= 12:
                chosen = clause
            if chosen is None:
                break
            result.append(stripped[: chosen.end()].strip())
            self.buffer = stripped[chosen.end() :]
        return result

    def finish(self) -> list[str]:
        final = sanitize_spoken_text(self.buffer)
        self.buffer = ""
        return [final] if final else []


class RuleBasedLLM:
    """Offline fallback intended for text-mode plumbing and tests, never judging."""

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        cancel: Event | None = None,
    ) -> Iterator[str]:
        text = str(messages[-1].get("content", "")).lower() if messages else ""
        if "repeat" in text:
            reply = "[ANSWER] Please restate the problem in your own words, then ask any clarification you need."
        elif "hint" in text:
            reply = "[HINT1] What information would make each new value easier to process?"
        elif "done" in text or "test" in text:
            reply = "[PROBE] How would you test that with an edge case?"
        else:
            reply = "[PROBE] Talk me through the simplest approach first."
        for word in re.findall(r"\S+\s*", reply):
            if cancel is not None and cancel.is_set():
                return
            yield word

    def json_chat(self, messages: Sequence[dict[str, Any]], *, cancel: Event | None = None) -> dict[str, Any]:
        return {}

