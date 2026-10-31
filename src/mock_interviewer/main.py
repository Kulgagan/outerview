"""Command-line entry point for text and live interview modes."""

from __future__ import annotations

import argparse
from pathlib import Path
from queue import Queue
from threading import Event
import time

from .audio import AudioInput
from .brain import Brain
from .config import AppConfig, load_config
from .doctor import print_doctor, run_doctor, write_recommendation
from .events import Command, CommandEvent, UserTurn
from .llm import OpenAICompatLLM, RuleBasedLLM
from .report import print_report, save_report
from .runtime import Clock, STTWorker, TerminalCommands
from .scoring import LLMJudge
from .screen import MSSCapture, ScreenObserver
from .session import Problem
from .stt import FasterWhisperTranscriber
from .tts import KokoroTTS, PrintSpeaker, Speaker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local voice mock interviewer")
    parser.add_argument("--config", default="config.toml", help="TOML configuration path")
    parser.add_argument("--text-mode", action="store_true", help="Use terminal input/output instead of hardware")
    parser.add_argument("--offline", action="store_true", help="Use deterministic responses in text mode; no local LLM needed")
    parser.add_argument("--calibrate", action="store_true", help="Save full-screen and editor crops, then exit")
    parser.add_argument("--doctor", action="store_true", help="Validate local dependencies and services, then exit")
    parser.add_argument("--benchmark", action="store_true", help="Measure local LLM and screen latency, then exit")
    parser.add_argument("--write-recommendation", metavar="PATH", help="Write a benchmark configuration overlay")
    parser.add_argument("--minutes", type=int, help="Override session duration")
    parser.add_argument("--bar", choices=("intern", "new_grad", "senior"), help="Override scoring bar")
    parser.add_argument("--save-report", metavar="PATH", help="Write the final Markdown scorecard")
    return parser


def _load(path: str, minutes: int | None, bar: str | None) -> AppConfig:
    config = load_config(path) if Path(path).is_file() else AppConfig()
    if minutes is not None:
        if minutes <= 0:
            raise ValueError("--minutes must be positive")
        config.session.minutes = minutes
    if bar is not None:
        config.session.bar = bar
    return config


def _screen_problem_loader(capture: MSSCapture, llm: OpenAICompatLLM):
    def load() -> Problem:
        image = capture.png_bytes(capture.capture(editor_only=False))
        last_error: Exception | None = None
        for _ in range(2):
            try:
                problem = llm.extract_problem(image)
                problem.notes = llm.generate_notes(problem)
                return problem
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Problem extraction failed after one retry: {last_error}")

    return load


def _complete(brain: Brain, save_path: str | None) -> int:
    if brain.result.report is None:
        return 0
    print_report(brain.session, brain.result.report)
    if save_path:
        save_report(save_path, brain.session, brain.result.report)
        print(f"Saved report to {save_path}")
    return 0


def run_text_mode(config: AppConfig, *, offline: bool, save_path: str | None) -> int:
    speaker = PrintSpeaker()
    llm = RuleBasedLLM() if offline else OpenAICompatLLM(config.llm)
    brain = Brain(config, llm, speaker, judge=None if offline else LLMJudge(llm))
    print("Text mode. Type 'start' to begin, 'end interview' to finish, or 'quit' while idle.")
    while not brain.result.quit_requested and brain.result.report is None:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            if brain.phase.value not in {"idle", "report"}:
                brain.process(CommandEvent(command=Command.END))
            break
        now = time.monotonic()
        brain.process(UserTurn(text=text, started_at=now, ended_at=now, at=now))
    return _complete(brain, save_path)


def run_live(config: AppConfig, *, save_path: str | None) -> int:
    events: Queue = Queue()
    utterances: Queue = Queue()
    shutdown = Event()
    llm = OpenAICompatLLM(config.llm)
    capture = MSSCapture(config.screen)
    backend = KokoroTTS(config.tts.voice, config.tts.device)
    speaker = Speaker(backend)
    brain = Brain(config, llm, speaker, problem_loader=_screen_problem_loader(capture, llm), judge=LLMJudge(llm))
    transcriber = FasterWhisperTranscriber(config.stt)
    stt = STTWorker(utterances, events, transcriber)
    clock = Clock(events)
    observer = ScreenObserver(capture, llm, events, lambda: brain.llm_busy)
    audio = AudioInput(
        config.audio,
        utterances,
        lambda: brain.phase,
        lambda: speaker.is_speaking,
        lambda: (brain.cancel_llm.set(), speaker.stop()),
        barge_output=events,
    )
    terminal = TerminalCommands(events)
    try:
        print("Live mode ready. Put LeetCode on screen and say 'start' or press Enter.")
        speaker.pre_render()
        speaker.start()
        stt.start()
        clock.start()
        observer.start()
        audio.start()
        terminal.start()
        brain.run(events, shutdown)
    except KeyboardInterrupt:
        if brain.phase.value not in {"idle", "report"}:
            brain.process(CommandEvent(command=Command.END))
    finally:
        shutdown.set()
        audio.stop()
        observer.stop()
        stt.stop()
        clock.stop()
        speaker.close()
        audio.join(timeout=2)
        observer.join(timeout=2)
        stt.join(timeout=2)
        clock.join(timeout=2)
    return _complete(brain, save_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load(args.config, args.minutes, args.bar)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2
    if args.doctor or args.benchmark:
        report = run_doctor(config, benchmark=args.benchmark)
        print_doctor(report)
        if args.write_recommendation:
            write_recommendation(args.write_recommendation, config, report)
        return 0 if report.healthy else 1
    if args.calibrate:
        try:
            full, editor = MSSCapture(config.screen).save_calibration()
            print(f"Saved {full} and {editor}")
            return 0
        except Exception as exc:
            print(f"Calibration failed: {exc}")
            return 1
    if args.offline and not args.text_mode:
        print("--offline is only available in --text-mode.")
        return 2
    if args.text_mode:
        return run_text_mode(config, offline=args.offline, save_path=args.save_report)
    return run_live(config, save_path=args.save_report)


if __name__ == "__main__":
    raise SystemExit(main())
