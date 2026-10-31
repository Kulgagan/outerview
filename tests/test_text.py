from mock_interviewer.events import Command
from mock_interviewer.llm import SentenceSplitter, TaggedReply, sanitize_spoken_text
from mock_interviewer.text import match_command, normalize_complexity


def test_commands_are_deterministic_and_quit_is_idle_only():
    assert match_command("repeat the question") is Command.REPEAT
    assert match_command("end interview") is Command.END
    assert match_command("quit", idle=False) is None
    assert match_command("quit", idle=True) is Command.QUIT


def test_complexity_normalization_and_tag_sanitization():
    assert normalize_complexity("That is big oh of n squared") == "That is O(n^2)"
    reply = TaggedReply.parse("[HINT1] <think>hidden</think> Consider a map.")
    assert reply.tag == "HINT1"
    assert reply.text == "Consider a map."
    assert TaggedReply.parse("missing tag").tag == "PROBE"
    assert sanitize_spoken_text("**hello** 🙂") == "hello"


def test_sentence_splitter_waits_for_a_spoken_boundary():
    splitter = SentenceSplitter()
    assert splitter.feed("This is a complete sentence.") == ["This is a complete sentence."]
    assert splitter.feed("This is another full sentence. And this is a third sentence.") == ["This is another full sentence.", "And this is a third sentence."]
    assert splitter.feed("A short") == []
    assert splitter.finish() == ["A short"]

