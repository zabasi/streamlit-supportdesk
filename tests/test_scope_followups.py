"""The scope classifiers read the user's follow-up messages, so a clarified report is classified again."""
from supportdesk.pipelines.stages.scope_classifier import (
    LLMScopeClassifier,
    RuleScopeClassifier,
    scope_text,
)
from supportdesk.types import CaseInput

REPORT = CaseInput(case_id="c", title="Value disappears", body="My value is lost when I click.")
CLARIFIED = CaseInput(case_id="c", title="Value disappears", body="My value is lost when I click.", history=[
    {"role": "assistant", "body": "Please send a Streamlit question."},
    {"role": "user", "body": "It is a Streamlit app: st.number_input with a key."}])


def test_without_follow_ups_the_text_is_the_title_and_body_as_before():
    assert scope_text(REPORT) == "Value disappears\nMy value is lost when I click."


def test_the_users_follow_ups_are_added_but_not_the_assistants_messages():
    text = scope_text(CLARIFIED)
    assert text.endswith("It is a Streamlit app: st.number_input with a key.")
    assert "Please send a Streamlit question" not in text


def test_the_rule_classifier_accepts_a_report_clarified_with_a_streamlit_api():
    assert RuleScopeClassifier()(CLARIFIED).label == "IN_SCOPE"


def test_the_llm_classifier_sends_the_follow_up_and_an_unchanged_first_request(tmp_path):
    prompt = tmp_path / "scope.txt"
    prompt.write_text("classify", encoding="utf-8")
    sent = []

    class Chat:
        def complete(self, messages, **kw):
            sent.append(messages[1]["content"])
            return '{"label": "IN_SCOPE", "reason": "streamlit"}', {}

    class Clients:
        chat = Chat()
    classifier = LLMScopeClassifier(Clients(), prompt)
    classifier(REPORT)
    classifier(CLARIFIED)
    assert sent[0] == "Value disappears\nMy value is lost when I click."
    assert sent[1].endswith("st.number_input with a key.")
