"""Typed configuration and TOML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
from urllib.parse import urlparse


@dataclass(slots=True)
class SessionConfig:
    minutes: int = 35
    bar: str = "intern"


@dataclass(slots=True)
class AudioConfig:
    input_device: str | int = "default"
    sample_rate: int = 16_000
    frame_ms: int = 32
    speech_threshold: float = 0.5
    speech_start_ms: int = 250
    pre_roll_ms: int = 300
    end_silence_ms: int = 700
    end_silence_coding_ms: int = 1_500
    barge_in_min_ms: int = 300


@dataclass(slots=True)
class STTConfig:
    model: str = "models/faster-whisper-small.en"
    device: str = "cpu"
    compute_type: str = "int8"


@dataclass(slots=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local-model"
    timeout_seconds: float = 30.0
    max_tokens: int = 90
    temperature: float = 0.6
    model_path: str = ""
    mmproj_path: str = ""


@dataclass(slots=True)
class TTSConfig:
    backend: str = "kokoro"
    voice: str = "af_heart"
    device: str = "cpu"


@dataclass(slots=True)
class ScreenConfig:
    monitor: int = 1
    editor_region: tuple[float, float, float, float] = (0.5, 0.1, 1.0, 0.9)
    max_width: int = 1280
    settle_seconds: float = 2.0
    change_threshold: float = 7.0


@dataclass(slots=True)
class AppConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a config file; omitted fields retain safe local defaults."""
    if path is None:
        return AppConfig()
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    screen = _section(data, "screen").copy()
    if "editor_region" in screen:
        value = screen["editor_region"]
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("screen.editor_region must contain four fractions")
        screen["editor_region"] = tuple(float(item) for item in value)
    config = AppConfig(
        session=SessionConfig(**_section(data, "session")),
        audio=AudioConfig(**_section(data, "audio")),
        stt=STTConfig(**_section(data, "stt")),
        llm=LLMConfig(**_section(data, "llm")),
        tts=TTSConfig(**_section(data, "tts")),
        screen=ScreenConfig(**screen),
    )
    host = urlparse(config.llm.base_url).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("llm.base_url must point to a local llama-server")
    if config.tts.backend != "kokoro":
        raise ValueError("tts.backend must be 'kokoro' in v1")
    return config


def config_as_dict(config: AppConfig) -> dict[str, object]:
    """Return JSON/TOML-friendly data for diagnostics and test assertions."""
    return asdict(config)

