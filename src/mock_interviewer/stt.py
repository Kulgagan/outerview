"""faster-whisper wrapper kept isolated from the real-time audio worker."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import STTConfig
from .text import VOCABULARY_PROMPT, normalize_complexity


class Transcriber(Protocol):
    def transcribe(self, samples, sample_rate: int = 16_000) -> str: ...


class FasterWhisperTranscriber:
    def __init__(self, config: STTConfig) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("faster-whisper is not installed.") from exc
        model_path = Path(config.model)
        if not model_path.is_dir():
            raise RuntimeError(f"Local faster-whisper model not found: {model_path}")
        self._model = WhisperModel(str(model_path), device=config.device, compute_type=config.compute_type, local_files_only=True)

    def transcribe(self, samples, sample_rate: int = 16_000) -> str:
        segments, _ = self._model.transcribe(
            samples,
            language="en",
            beam_size=1,
            initial_prompt=VOCABULARY_PROMPT,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return normalize_complexity(text)

