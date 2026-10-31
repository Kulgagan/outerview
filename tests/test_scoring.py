from mock_interviewer.scoring import score_session, verdict
from mock_interviewer.session import InterviewSession, Phase


class EvidenceJudge:
    def judge(self, session, metrics):
        return {
            "checklist": {
                "communication": {
                    "clarified_before_coding": {"yes": True, "evidence": "What are the constraints?"},
                    "restated_problem_or_example": {"yes": True, "evidence": "So we return the indices."},
                },
                "testing": {
                    "proposed_two_test_cases": {"yes": True, "evidence": "I would test empty and duplicate inputs."},
                    "named_two_real_edge_cases": {"yes": True, "evidence": "Empty and duplicate inputs."},
                    "walked_through_an_example": {"yes": True, "evidence": "For [2, 7], the map finds 2."},
                    "did_not_declare_done_without_testing": {"yes": True, "evidence": "I will test it now."},
                },
            }
        }


def test_missing_evidence_is_not_credited():
    session = InterviewSession(started_at=0.0)
    session.set_phase(Phase.CLARIFY, 0.0)
    card = score_session(session, "intern", EvidenceJudge())
    assert card.checklist["communication"]["clarified_before_coding"]
    assert not card.checklist["communication"]["explained_approach_before_coding"]


def test_done_without_testing_caps_testing_score():
    session = InterviewSession(started_at=0.0)
    session.set_phase(Phase.CODING, 0.0)
    session.add_turn("candidate", "I am done.", 1.0, 2.0)
    session.set_phase(Phase.WRAPUP, 3.0)
    card = score_session(session, "intern", EvidenceJudge())
    assert card.scores["testing"] == 1


def test_verdict_thresholds_match_design():
    assert verdict({"a": 1, "b": 3, "c": 3, "d": 3}) == "Leaning no hire"
    assert verdict({"a": 4, "b": 4, "c": 4, "d": 4}) == "Strong hire"

