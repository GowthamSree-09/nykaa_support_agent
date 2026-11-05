"""Regression tests for answer selection on duration questions ("how long ...").

Bug: "How long will a refund take if I paid cash on delivery?" returned the general COD sentence
("Refunds ... are paid to the customer's bank account or as store credit.") instead of the policy sentence that
states the duration ("A bank refund takes 5 to 7 working days ..."). The picker ranked sentences by similarity
and shared words and had no idea that "how long" asks for a duration.
"""
import json
import os

import pytest

from agents.service import handle_request
from kb.documents import KB_DOCUMENTS
from rag import store
from rag.generate import CATEGORY_CLARIFICATION, _score_sentences, answer_question
from rag.relevance import has_duration, is_duration_question
from tests.snapshot_questions import SNAPSHOT_QUESTIONS

HERE = os.path.dirname(__file__)
COD_QUESTION = "How long will a refund take if I paid cash on delivery?"
KB_COD = next(d["text"] for d in KB_DOCUMENTS if d["doc_id"] == "cod_refund")
DURATION_SENTENCE = "A bank refund takes 5 to 7 working days after the returned item passes quality checks."


@pytest.fixture(scope="module", autouse=True)
def _index():
    store.index_documents()


def test_the_duration_sentence_really_is_in_the_knowledge_base():
    assert DURATION_SENTENCE in KB_COD  # the expected answer comes from the written policy, not from the code


def test_cod_refund_time_answer_states_the_duration():
    r = answer_question(COD_QUESTION)
    assert r["answer"].startswith(DURATION_SENTENCE)
    assert r["sources"] == ["cod_refund"] and r["grounded"]
    assert all(s in KB_COD for s in r["used"])  # every sentence is copied from the policy document


def test_cod_refund_time_through_the_full_pipeline():
    r = handle_request(COD_QUESTION, "dur-cod")
    assert r["answer"].startswith(DURATION_SENTENCE) and r["response"]["sources"] == ["cod_refund"]
    assert r["review"]["approved"] and not r.get("guardrail")


def test_hair_dryer_expectation_is_checked_against_the_written_policy():
    """The KB never mentions hair dryers; it defines Electronics (12 month) and Home appliances (6 month) only.
    The answer must be copied from the policy, and the judge must accept either supported warranty."""
    kb_text = " ".join(d["text"].lower() for d in KB_DOCUMENTS)
    assert "hair" not in kb_text and "dryer" not in kb_text
    warranty = next(d["text"] for d in KB_DOCUMENTS if d["doc_id"] == "warranty_terms")
    assert "12 month" in warranty and "6 month" in warranty
    r = answer_question("Is there a warranty on my hair dryer?")
    # Round 4: the policy does not say which category a hair dryer is, so the answer gives BOTH written rules
    # and asks the customer to confirm (see tests/test_round4_defects.py); every other sentence is from the KB.
    assert r["grounded"] and r["sources"] == ["warranty_terms"] and CATEGORY_CLARIFICATION in r["used"]
    assert all(s in warranty for s in r["used"] if s != CATEGORY_CLARIFICATION)
    from eval.judge_eval import HELDOUT_SET, MockJudge
    assert any("12 month|6 month" in facts for _, facts, _, _ in HELDOUT_SET)
    # alternatives work: an answer with only the 6-month sentence satisfies "12 month|6 month"
    prompt = ("KIND: kb\nQUESTION: q\nCONTEXT: []\nEXPECTED_FACTS: [\"12 month|6 month\"]\nEXPECTED_DOCS: [\"warranty_terms\"]\n"
              "SOURCES: [\"warranty_terms\"]\nANSWER: Home appliances carry a 6 month warranty.")
    assert MockJudge().judge(prompt)["accuracy"] == 5.0
    wrong = prompt.replace("Home appliances carry a 6 month warranty.", "Level 1 is the support agent.")
    assert MockJudge().judge(wrong)["accuracy"] == 1.0  # the alternative list is not a free pass


def test_duration_detection():
    for q in ("How long will a refund take?", "How many days do I have to return it?", "How soon must I report it?",
              "How quickly do I get store credit?", "How many working days does delivery take?"):
        assert is_duration_question(q), q
    # Round 4: "when will X reach/arrive/be credited" IS a duration question; "when can I cancel" still is not.
    assert is_duration_question("When will my refund arrive?")
    for q in ("When can I cancel my order?", "What is the return policy?", "How do I return my shoes?",
              "How much of my order can I pay with points?", "Can I cancel after it ships?"):
        assert not is_duration_question(q), q
    for s in ("A bank refund takes 5 to 7 working days after the check.", "Store credit is added within 24 hours.",
              "Points expire 12 months after they are earned.", "Electronics have a 10 day return window."):
        assert has_duration(s), s
    for s in ("Refunds are paid to the customer's bank account.", "Level 1 is the support agent.",
              "The first exchange is free, and each later exchange has a 99 INR pickup fee."):
        assert not has_duration(s), s


def test_duration_bonus_is_a_preference_not_a_filter():
    """With no duration sentence available, every sentence is still scored (nothing is dropped)."""
    chunks = [{"doc_id": "x", "text": "Refunds are paid to the bank account. Store credit is an option."}]
    scored = _score_sentences("How long will a refund take?", chunks)
    assert len(scored) == 2
    plain = {s: x for s, _, x in _score_sentences("Tell me about a refund", chunks)}
    assert all(abs(x - plain[s]) < 0.2 for s, _, x in scored)  # no duration boost was applied to either


def test_other_duration_answers_improved_without_collateral_changes():
    """Compared with the answers recorded BEFORE the duration fix: exactly two answers changed, both for the better.
    (Round 4 later changed two more, checked separately in test_round4_answers_changed_only_as_intended.)"""
    before = json.load(open(os.path.join(HERE, "answer_snapshot_before_duration_fix.json"), encoding="utf-8"))
    changed = {q for q in SNAPSHOT_QUESTIONS if answer_question(q)["answer"] != before[q]}
    assert changed == {COD_QUESTION, "How long are loyalty points valid?", *ROUND4_CHANGED}
    assert answer_question("How long are loyalty points valid?")["answer"] == "Points expire 12 months after they are earned."


ROUND4_CHANGED = {"When will my cash on delivery refund reach my bank account?",
                  "How long does delivery take to remote areas?"}


def test_round4_answers_changed_only_as_intended():
    """Round 4 fixes: of the 32 reviewed answers exactly the two documented known-bad ones changed."""
    before = json.load(open(os.path.join(HERE, "answer_snapshot_before_round4.json"), encoding="utf-8"))
    changed = {q for q in SNAPSHOT_QUESTIONS if answer_question(q)["answer"] != before[q]}
    assert changed == ROUND4_CHANGED
    assert answer_question("How long does delivery take to remote areas?")["answer"] ==         "Remote pin codes may need up to 10 working days."
    assert "5 to 7 working days" in answer_question("When will my cash on delivery refund reach my bank account?")["answer"]


@pytest.mark.parametrize("question", SNAPSHOT_QUESTIONS)
def test_reviewed_answer_snapshot(question):
    """All 32 refund / return / delivery / warranty / exchange / escalation answers stay exactly as reviewed."""
    snapshot = json.load(open(os.path.join(HERE, "answer_snapshot.json"), encoding="utf-8"))
    assert answer_question(question)["answer"] == snapshot[question]
