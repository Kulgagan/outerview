"""Low-priority screenshot observation with cheap change detection."""

from __future__ import annotations

from io import BytesIO
from queue import Queue
from threading import Event, Thread
from typing import Callable, Protocol
import time

from .config import ScreenConfig
from .events import ScreenUpdate


class ScreenAnalyzer(Protocol):
    def observe_code(self, image_bytes: bytes, mime_type: str = "image/png") -> dict[str, str]: ...


class MSSCapture:
    def __init__(self, config: ScreenConfig) -> None:
        self.config = config

    def _monitor(self) -> dict[str, int]:
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("mss is not installed.") from exc
        with mss.mss() as grabber:
            monitors = grabber.monitors
            if not 0 < self.config.monitor < len(monitors):
                raise ValueError(f"Monitor {self.config.monitor} is unavailable")
            return dict(monitors[self.config.monitor])

    def capture(self, *, editor_only: bool = False):
        try:
            import mss
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("Install mss and Pillow to capture the screen.") from exc
        monitor = self._monitor()
        if editor_only:
            left, top, right, bottom = self.config.editor_region
            width, height = monitor["width"], monitor["height"]
            monitor = {
                "left": monitor["left"] + int(width * left),
                "top": monitor["top"] + int(height * top),
                "width": int(width * (right - left)),
                "height": int(height * (bottom - top)),
            }
        with mss.mss() as grabber:
            raw = grabber.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.rgb)
        if image.width > self.config.max_width:
            image.thumbnail((self.config.max_width, 10_000))
        return image

    @staticmethod
    def png_bytes(image) -> bytes:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def save_calibration(self, directory: str = "calibration") -> tuple[str, str]:
        from pathlib import Path

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        full, editor = target / "full-screen.png", target / "editor-crop.png"
        self.capture(editor_only=False).save(full)
        self.capture(editor_only=True).save(editor)
        return str(full), str(editor)


def frame_difference(before, after) -> float:
    """Mean grayscale delta on 64x36 thumbnails; no model work required."""
    import numpy as np

    left = np.asarray(before.convert("L").resize((64, 36)), dtype="float32")
    right = np.asarray(after.convert("L").resize((64, 36)), dtype="float32")
    return float(np.abs(left - right).mean())


class ScreenObserver(Thread):
    def __init__(
        self,
        capture: MSSCapture,
        analyzer: ScreenAnalyzer,
        output: Queue,
        llm_busy: Callable[[], bool],
        interval_seconds: float = 0.75,
    ) -> None:
        super().__init__(name="screen-observer", daemon=True)
        self.capture = capture
        self.analyzer = analyzer
        self.output = output
        self.llm_busy = llm_busy
        self.interval_seconds = interval_seconds
        self._stop_event = Event()
        self._last_frame = None
        self._changed_at: float | None = None
        self._candidate_code = ""

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - needs display and VLM
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                # Keep the last good snapshot; transient OCR/VLM failure is non-fatal.
                pass
            self._stop_event.wait(self.interval_seconds)

    def poll_once(self) -> None:
        now = time.monotonic()
        frame = self.capture.capture(editor_only=True)
        if self._last_frame is None or frame_difference(self._last_frame, frame) >= self.capture.config.change_threshold:
            self._last_frame = frame
            self._changed_at = now
            return
        if self._changed_at is None or now - self._changed_at < self.capture.config.settle_seconds or self.llm_busy():
            return
        image_bytes = self.capture.png_bytes(frame)
        try:
            data = self.analyzer.observe_code(image_bytes)
        except Exception:
            # One retry handles malformed structured JSON without promoting a bad snapshot.
            data = self.analyzer.observe_code(image_bytes)
        code = data.get("code", "")
        # A code-specific assertion requires two matching VLM reads.
        stable = bool(code and code == self._candidate_code)
        self._candidate_code = code
        self.output.put(ScreenUpdate(code=code, run_status=data.get("run_status", "none"), failing_case=data.get("failing_case", ""), stable=stable))
        self._changed_at = None

