"""TTS backends and a stop-safe single-speaker worker."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from typing import Protocol
import os


class TTSBackend(Protocol):
    sample_rate: int

    def synth(self, text: str): ...


class KokoroTTS:
    sample_rate = 24_000

    def __init__(self, voice: str = "af_heart", device: str = "cpu") -> None:
        self.voice = voice
        self.device = device
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("Kokoro is not installed. Run pip install -r requirements.txt.") from exc
        # Startup must consume cached local weights rather than fetch a model mid-interview.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        # Kokoro chooses its available accelerator; CPU is the safe default.
        self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def synth(self, text: str):
        import numpy as np

        parts = [audio for _, _, audio in self._pipeline(text, voice=self.voice)]
        if not parts:
            return np.empty(0, dtype="float32")
        return np.concatenate(parts).astype("float32", copy=False)


class PrintSpeaker:
    """Text-mode speaker with the same submit/stop interface as the live one."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def submit(self, text: str) -> None:
        self.spoken.append(text)
        print(f"Interviewer: {text}")

    def stop(self) -> None:
        return

    @property
    def is_speaking(self) -> bool:
        return False

    def close(self) -> None:
        return


class Speaker(Thread):
    """Serialize synthesis/playback and permit barge-in to drain both queues."""

    def __init__(self, backend: TTSBackend) -> None:
        super().__init__(name="speaker", daemon=True)
        self.backend = backend
        self._queue: Queue[str | None] = Queue()
        self._stopped = Event()
        self._closed = Event()
        self._playing = Event()

    def submit(self, text: str) -> None:
        if text.strip() and not self._closed.is_set():
            self._queue.put(text)

    def stop(self) -> None:
        self._stopped.set()
        self._playing.clear()
        self._drain()
        try:
            import sounddevice as sd

            sd.stop()
        except ImportError:
            pass

    def close(self) -> None:
        self._closed.set()
        self.stop()
        self._queue.put(None)
        self.join(timeout=2)

    @property
    def is_speaking(self) -> bool:
        return self._playing.is_set()

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def run(self) -> None:  # pragma: no cover - needs audio hardware
        try:
            import sounddevice as sd
        except ImportError:
            return
        while not self._closed.is_set():
            text = self._queue.get()
            if text is None:
                return
            self._stopped.clear()
            audio = self.backend.synth(text)
            if len(audio) and not self._stopped.is_set():
                self._playing.set()
                try:
                    sd.play(audio, self.backend.sample_rate, blocking=True)
                finally:
                    self._playing.clear()

    def pre_render(self) -> dict[str, object]:
        """Warm the backend. Cached clip playback is intentionally v2 work."""
        return {clip: self.backend.synth(clip) for clip in ("Mm-hm.", "Okay.", "Right.", "Go ahead.", "Take your time.")}

