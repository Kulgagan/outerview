"""Microphone capture, frame VAD, endpointing, and barge-in detection."""

from __future__ import annotations

from collections import deque
from queue import Queue
from threading import Event, Thread
from typing import Callable, Protocol
import time

from .config import AudioConfig
from .events import BargeIn, Utterance
from .session import Phase


class FrameVAD(Protocol):
    def probability(self, samples) -> float: ...


class SileroFrameVAD:
    """Silero's 16 kHz ONNX/PyTorch model adapted for 512-sample frames."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        if sample_rate != 16_000:
            raise ValueError("Silero live VAD is configured for 16 kHz input")
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("silero-vad is not installed.") from exc
        self._model = load_silero_vad()
        self._sample_rate = sample_rate

    def probability(self, samples) -> float:
        import torch

        frame = torch.as_tensor(samples, dtype=torch.float32)
        return float(self._model(frame, self._sample_rate).item())


class AudioInput(Thread):
    """Turn frame-level speech probabilities into complete utterance events."""

    def __init__(
        self,
        config: AudioConfig,
        output: Queue,
        phase_provider: Callable[[], Phase],
        speaking_provider: Callable[[], bool],
        on_barge_in: Callable[[], None],
        vad: FrameVAD | None = None,
        barge_output: Queue | None = None,
    ) -> None:
        super().__init__(name="audio-input", daemon=True)
        self.config = config
        self.output = output
        self.barge_output = barge_output or output
        self.phase_provider = phase_provider
        self.speaking_provider = speaking_provider
        self.on_barge_in = on_barge_in
        self.vad = vad or SileroFrameVAD(config.sample_rate)
        self._stop_event = Event()
        self._frames: Queue = Queue()
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._barge_emitted = False
        self._pre_roll: deque = deque(maxlen=max(1, config.pre_roll_ms // config.frame_ms))
        self._utterance: list = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - needs microphone
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed.") from exc

        def callback(indata, frames, _time, status) -> None:
            if status:
                # The worker stays alive; an overrun is visible in host logs.
                return
            self._frames.put(indata[:, 0].copy())

        blocksize = int(self.config.sample_rate * self.config.frame_ms / 1000)
        device = None if self.config.input_device == "default" else self.config.input_device
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            callback=callback,
        ):
            while not self._stop_event.is_set():
                try:
                    frame = self._frames.get(timeout=0.1)
                except Exception:
                    continue
                self.process_frame(frame)

    def process_frame(self, frame) -> None:
        """Public for deterministic endpointing tests without audio hardware."""
        probability = self.vad.probability(frame)
        speech = probability >= self.config.speech_threshold
        self._pre_roll.append(frame)
        if speech:
            self._speech_ms += self.config.frame_ms
            self._silence_ms = 0
            if not self._in_speech and self._speech_ms >= self.config.speech_start_ms:
                self._in_speech = True
                self._utterance = list(self._pre_roll)
            if self._in_speech:
                self._utterance.append(frame)
            if self.speaking_provider() and not self._barge_emitted and self._speech_ms >= self.config.barge_in_min_ms:
                self._barge_emitted = True
                self.on_barge_in()
                self.barge_output.put(BargeIn())
            return

        self._speech_ms = 0
        if not self._in_speech:
            return
        self._utterance.append(frame)
        self._silence_ms += self.config.frame_ms
        coding = self.phase_provider() in {Phase.CODING, Phase.TESTING}
        limit = self.config.end_silence_coding_ms if coding else self.config.end_silence_ms
        if self._silence_ms >= limit:
            self._emit_utterance()

    def _emit_utterance(self) -> None:
        import numpy as np

        samples = np.concatenate(self._utterance) if self._utterance else np.empty(0, dtype="float32")
        self.output.put(Utterance(samples=samples, sample_rate=self.config.sample_rate, at=time.monotonic()))
        self._in_speech = False
        self._silence_ms = 0
        self._barge_emitted = False
        self._utterance = []

