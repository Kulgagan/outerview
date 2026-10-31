from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from mock_interviewer.config import load_config
from mock_interviewer.llm import parse_json_object, validate_problem


def test_config_accepts_editor_region():
    payload = b"[screen]\neditor_region = [0.5, 0.1, 1.0, 0.9]\n"
    with patch.object(Path, "open", return_value=BytesIO(payload)):
        config = load_config("config.toml")
    assert config.screen.editor_region == (0.5, 0.1, 1.0, 0.9)


def test_problem_json_validation_rejects_missing_statement():
    data = parse_json_object('```json\n{"title":"Two Sum","statement":"Find two values","examples":[]}\n```')
    assert validate_problem(data).title == "Two Sum"
    try:
        validate_problem({"title": "missing statement"})
    except ValueError as exc:
        assert "statement" in str(exc)
    else:
        raise AssertionError("missing statement was accepted")

