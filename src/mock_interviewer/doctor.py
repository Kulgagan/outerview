"""Local-only dependency checks and latency smoke benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Callable

from .config import AppConfig
from .llm import OpenAICompatLLM
from .screen import MSSCapture
from .stt import FasterWhisperTranscriber
from .tts import KokoroTTS


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DoctorReport:
    checks: list[Check] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    profile: str = "C"

    @property
    def healthy(self) -> bool:
        return all(check.ok for check in self.checks)


def run_doctor(config: AppConfig, *, benchmark: bool = False) -> DoctorReport:
    report = DoctorReport()
    report.checks.append(Check("python", sys.version_info >= (3, 12), platform.python_version()))
    report.checks.append(Check("espeak-ng", shutil.which("espeak-ng") is not None, "found on PATH" if shutil.which("espeak-ng") else "install espeak-ng and add it to PATH"))
    for label, raw_path in (("LLM model", config.llm.model_path), ("vision projector", config.llm.mmproj_path)):
        if raw_path:
            report.checks.append(Check(label, Path(raw_path).is_file(), raw_path))
    report.checks.append(Check("STT model", Path(config.stt.model).is_dir(), config.stt.model))
    report.checks.append(_llm_health(config))
    report.checks.append(_audio_check())
    cuda = _cuda_available()
    report.checks.append(Check("cuda", cuda, "available" if cuda else "not available; CPU profile remains supported"))
    report.profile = "A" if cuda else "C"
    if benchmark:
        _benchmark_llm(config, report)
        _benchmark_stt(config, report)
        _benchmark_tts(config, report)
        _benchmark_screen(config, report)
        _benchmark_vision(config, report)
    return report


def _llm_health(config: AppConfig) -> Check:
    try:
        import httpx

        response = httpx.get(f"{config.llm.base_url.rstrip('/')}/health", timeout=2.0)
        return Check("llama-server", response.is_success, f"HTTP {response.status_code}")
    except Exception as exc:
        return Check("llama-server", False, str(exc))


def _audio_check() -> Check:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        return Check("audio input", bool(devices), f"{len(devices)} audio devices")
    except Exception as exc:
        return Check("audio input", False, str(exc))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _benchmark_llm(config: AppConfig, report: DoctorReport) -> None:
    client = OpenAICompatLLM(config)
    started = time.perf_counter()
    try:
        first = next(
            client.stream_chat(
                [{"role": "user", "content": "Reply with [ACK] okay."}], max_tokens=8, temperature=0.0
            )
        )
        if first:
            report.timings_ms["llm_first_token"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        report.checks.append(Check("LLM benchmark", False, str(exc)))


def _benchmark_screen(config: AppConfig, report: DoctorReport) -> None:
    started = time.perf_counter()
    try:
        image = MSSCapture(config).capture()
        if image:
            report.timings_ms["screen_capture"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        report.checks.append(Check("screen capture", False, str(exc)))


def _benchmark_stt(config: AppConfig, report: DoctorReport) -> None:
    started = time.perf_counter()
    try:
        import numpy as np

        transcriber = FasterWhisperTranscriber(config.stt)
        transcriber.transcribe(np.zeros(config.audio.sample_rate * 5, dtype="float32"), config.audio.sample_rate)
        report.timings_ms["stt_5_second_clip"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        report.checks.append(Check("STT benchmark", False, str(exc)))


def _benchmark_tts(config: AppConfig, report: DoctorReport) -> None:
    started = time.perf_counter()
    try:
        backend = KokoroTTS(config.tts.voice, config.tts.device)
        backend.synth("Okay.")
        report.timings_ms["tts_first_audio"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        report.checks.append(Check("TTS benchmark", False, str(exc)))


def _benchmark_vision(config: AppConfig, report: DoctorReport) -> None:
    started = time.perf_counter()
    try:
        capture = MSSCapture(config.screen)
        encoded = base64.b64encode(capture.png_bytes(capture.capture())).decode("ascii")
        client = OpenAICompatLLM(config.llm)
        next(
            client.stream_chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Reply only with [ACK] okay."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        ],
                    }
                ],
                max_tokens=8,
                temperature=0.0,
            )
        )
        report.timings_ms["vision_first_token"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        report.checks.append(Check("vision benchmark", False, str(exc)))


def print_doctor(report: DoctorReport) -> None:
    for check in report.checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    for name, elapsed in report.timings_ms.items():
        print(f"{name}: {elapsed:.0f} ms")
    print(f"Recommended profile: {report.profile}")


def write_recommendation(path: str | Path, config: AppConfig, report: DoctorReport) -> None:
    """Write a deliberately small overlay rather than overwrite the user's config."""
    profile = report.profile
    stt_device = "cuda" if profile == "A" else "cpu"
    stt_precision = "int8_float16" if profile == "A" else "int8"
    lines = [
        "# Generated by mock-interviewer --benchmark; copy reviewed values into config.toml.",
        f"# Measured profile: {profile}",
        "[stt]",
        f'device = "{stt_device}"',
        f'compute_type = "{stt_precision}"',
        "",
        "[tts]",
        'device = "cpu"',
    ]
    if report.timings_ms:
        lines += ["", "# Timings (ms)"] + [f"# {name} = {value:.0f}" for name, value in report.timings_ms.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

