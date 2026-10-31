from mock_interviewer.brain import Brain
from mock_interviewer.config import AppConfig
from mock_interviewer.events import BargeIn, Command, CommandEvent, Trigger, UserTurn
from mock_interviewer.session import Phase


class FakeSpeaker:
    def __init__(self):
        self.messages = []
        self.stopped = False

    def submit(self, text):
        self.messages.append(text)

    def stop(self):
        self.stopped = True


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def stream_chat(self, *args, **kwargs):
        yield from self.response.split(" ")

    def json_chat(self, *args, **kwargs):
        return {}


def test_brain_owns_phases_and_blocks_skipped_hints():
    speaker = FakeSpeaker()
    brain = Brain(AppConfig(), FakeLLM("[HINT3] Here is too much detail."), speaker, clock=lambda: 10.0)
    brain.process(CommandEvent(command=Command.START, at=1.0))
    assert brain.phase is Phase.CLARIFY
    brain.process(UserTurn(text="My brute force approach is to compare each pair.", at=2.0, started_at=2.0, ended_at=2.0))
    assert brain.phase is Phase.APPROACH
    brain.process(Trigger(kind="hint_requested", at=40.0))
    assert brain.session.interviewer_events[-1].tag == "PROBE"
    assert all("too much detail" not in message for message in speaker.messages)


def test_barge_in_cancels_generation_and_stops_audio():
    speaker = FakeSpeaker()
    brain = Brain(AppConfig(), FakeLLM("[ACK] okay."), speaker)
    brain.process(BargeIn())
    assert brain.cancel_llm.is_set()
    assert speaker.stopped


def test_followup_is_answered_before_wrapup():
    speaker = FakeSpeaker()
    brain = Brain(AppConfig(), FakeLLM("[PROBE] What changes at scale?"), speaker, clock=lambda: 20.0)
    brain.process(CommandEvent(command=Command.START, at=1.0))
    brain.session.set_phase(Phase.FOLLOWUP, 2.0)
    brain.process(UserTurn(text="I would shard the input by key.", at=3.0, started_at=3.0, ended_at=3.0))
    assert brain.result.report is not None
    assert brain.phase is Phase.REPORT

