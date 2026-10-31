from queue import Queue

import numpy as np

from mock_interviewer.audio import AudioInput
from mock_interviewer.config import AudioConfig
from mock_interviewer.events import BargeIn, Utterance
from mock_interviewer.session import Phase


class FakeVAD:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def probability(self, samples):
        return next(self.probabilities)


def test_endpointing_emits_barge_in_then_utterance():
    utterances, events = Queue(), Queue()
    interrupted = []
    config = AudioConfig(frame_ms=32, speech_start_ms=32, pre_roll_ms=32, end_silence_ms=32, barge_in_min_ms=32)
    worker = AudioInput(
        config,
        utterances,
        lambda: Phase.CLARIFY,
        lambda: True,
        lambda: interrupted.append(True),
        vad=FakeVAD([0.9, 0.0]),
        barge_output=events,
    )
    frame = np.zeros(512, dtype="float32")
    worker.process_frame(frame)
    worker.process_frame(frame)
    assert interrupted == [True]
    assert isinstance(events.get_nowait(), BargeIn)
    utterance = utterances.get_nowait()
    assert isinstance(utterance, Utterance)
    assert len(utterance.samples) >= len(frame)

