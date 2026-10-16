"""Regression tests for the two stress-evaluation defects:
  1. legitimate product-noun questions ("How do I return my earphones?") were falsely rejected;
  2. "Who do I contact if it fails?" received an unrelated sentence (dangling "it", no answer-type check).
"""
import re

import pytest

from agents.memory import reset_all_sessions, reset_session
from agents.routing import classify_intent
from agents.schemas import CLARIFY_QUESTION, PARTIAL_ANSWER
from agents.service import handle_request
from agents.tools import TOOL_LOG
from kb.documents import KB_DOCUMENTS
from rag import store
from rag.calibrate import glossary_evidence
from rag.generate import FALLBACK, SIMILARITY_THRESHOLD, answer_question
from rag.queries import NEAR_DOMAIN_OUT_OF_SCOPE, OUT_OF_SCOPE, PRODUCT_NOUN_QUESTIONS
from rag.relevance import (PRODUCT_NOUNS, is_who_question, names_a_role, normalize_products, resolve_reference,
                           topic_hint)
from review.autogen_review import review_draft


@pytest.fixture(scope="module", autouse=True)
def _index():
    store.index_documents()


# ---------------------------------------------------------------- defect 1: false rejection
EXPECTED_ANSWER = {
    "earphones": "Electronics have a 10 day return window",
    "headphones": "Electronics have a 10 day return window",
    "trimmer": "Electronics have a 10 day return window",
    "kurta": "Apparel and footwear can be returned within 15 days",
    "sneakers": "Apparel and footwear can be returned within 15 days",
    "cushion": "Home products can be returned within 14 days",
    "shampoo": "Beauty products can be returned within 7 days",
    "mascara": "Beauty products can be returned within 7 days",
}


@pytest.mark.parametrize("question", PRODUCT_NOUN_QUESTIONS)
def test_product_noun_questions_are_answered_with_the_right_category(question):
    r = answer_question(question)
    noun = next(n for n in EXPECTED_ANSWER if n in question)
    assert r["grounded"], f"falsely rejected: {question} (top-1 {r['top_similarity']})"
    assert r["answer"].startswith(EXPECTED_ANSWER[noun]) and r["sources"] == ["return_window"]


def test_the_reported_question_through_the_full_pipeline():
    r = handle_request("How do I return my earphones?", "d1-earphones")
    assert r["answer"].startswith("Electronics have a 10 day return window")
    assert r["response"]["sources"] == ["return_window"] and r["review"]["approved"]
    assert not r.get("guardrail")


def test_glossary_raises_similarity_with_evidence_and_threshold_is_not_lowered():
    assert SIMILARITY_THRESHOLD == 0.44  # the fix is NOT a lower threshold
    rows = glossary_evidence()
    assert len(rows) == len(PRODUCT_NOUN_QUESTIONS)
    for q, before, after, doc in rows:
        assert after >= before and after >= SIMILARITY_THRESHOLD and doc == "return_window", q
    assert sum(before < SIMILARITY_THRESHOLD for _, before, _, _ in rows) >= 5  # most were rejected before


def test_glossary_does_not_touch_out_of_scope_questions_and_they_stay_rejected():
    for q in OUT_OF_SCOPE + NEAR_DOMAIN_OUT_OF_SCOPE + ["How do I contact the pizza place?",
                                                       "How do I return my library book?"]:
        assert normalize_products(q) == q
        r = answer_question(q)
        assert not r["grounded"] and r["answer"] == FALLBACK, q


def test_glossary_words_never_occur_in_the_knowledge_base():
    kb_words = set(re.findall(r"[a-z]+", " ".join(d["text"].lower() for d in KB_DOCUMENTS)))
    kb_words |= {w[:-1] for w in kb_words if w.endswith("s")}
    assert not [n for nouns in PRODUCT_NOUNS.values() for n in nouns if n in kb_words]


def test_normalization_only_changes_the_product_noun():
    assert normalize_products("How do I return my earphones?") == "How do I return my electronics?"
    assert normalize_products("Can I ship nail polish to another country?").endswith("another country?")


def test_junk_sub_question_cannot_ride_along_with_a_valid_part():
    r = answer_question("How long is the warranty on electronics and how do I order a pizza online?")
    assert r["sources"] == ["warranty_terms"] and r["unanswered"] == ["how do I order a pizza online"]
    full = handle_request("How long is the warranty on electronics and how do I order a pizza online?", "d1-mix")
    assert PARTIAL_ANSWER in full["answer"] and "12 month" in full["answer"]


# ---------------------------------------------------------------- defect 2: contact / follow-up
def test_standalone_vague_question_asks_for_clarification():
    TOOL_LOG.clear()
    r = handle_request("Who do I contact if it fails?", "d2-solo")
    assert classify_intent("Who do I contact if it fails?") == "clarify"
    assert r["answer"] == CLARIFY_QUESTION and r["response"]["sources"] == [] and TOOL_LOG == []
    assert "If both attempts fail" not in r["answer"]


def test_same_message_antecedent_resolves_it_and_answers_both_parts():
    r = answer_question("How long is the warranty on electronics and who do I contact if it fails?")
    assert r["sources"] == ["escalation_matrix", "warranty_terms"]
    assert "12 month" in r["answer"] and "support agent" in r["answer"]
    assert "If both attempts fail" not in r["answer"] and not r["unanswered"]


def test_session_topic_resolves_it_from_the_previous_question():
    reset_all_sessions()
    handle_request("How long is the warranty on my earphones?", "d2-chat")
    r = handle_request("Who do I contact if it fails?", "d2-chat")
    assert r["resolved_question"] == "Who do I contact if my earphones fails?"
    assert r["response"]["sources"] == ["escalation_matrix"] and "support agent" in r["answer"]
    assert "If both attempts fail" not in r["answer"]


def test_new_or_reset_session_does_not_inherit_the_topic():
    reset_all_sessions()
    handle_request("How long is the warranty on my earphones?", "d2-a")
    assert handle_request("Who do I contact if it fails?", "d2-b")["answer"] == CLARIFY_QUESTION  # other session
    assert reset_session("d2-a")
    assert handle_request("Who do I contact if it fails?", "d2-a")["answer"] == CLARIFY_QUESTION  # after reset


def test_order_context_wins_when_the_session_has_an_order():
    reset_all_sessions()
    handle_request("Where is my order ORD-1023?", "d2-ord")
    r = handle_request("Who do I contact if it fails?", "d2-ord")
    assert "(order ORD-1023)" in r["resolved_question"] and r["response"]["order_id"] == "ORD-1023"
    assert "support agent" in r["answer"]


def test_who_question_answer_must_name_a_role():
    assert is_who_question("Who do I contact?") and not is_who_question("How do I contact support?")
    assert names_a_role("Level 1 is the support agent, who must resolve a case within 24 hours.")
    assert not names_a_role("If both attempts fail, the return request is closed.")
    for q in ("Who do I contact for a complaint?", "Who do I contact if my earphones fail?"):
        r = answer_question(q)
        assert r["grounded"] and r["sources"] == ["escalation_matrix"] and all(names_a_role(s) for s in r["used"])


def test_topic_hint_and_reference_helpers():
    assert topic_hint("who do I contact if it fails") == "escalation_matrix"
    assert topic_hint("how long is the warranty") is None
    assert resolve_reference("who do I contact if it fails", "electronics") == "who do I contact if my electronics fails"
    assert resolve_reference("who do I contact if it fails", None) == "who do I contact if it fails"
    assert resolve_reference("how do I return my shoes", "electronics") == "how do I return my shoes"  # no pronoun


def test_reviewer_removes_an_answer_that_names_no_role_for_a_who_question():
    q = "Who do I contact if my electronics fails?"
    wrong = "If both attempts fail, the return request is closed."
    v = review_draft(wrong, [wrong], q)
    assert not v.approved and v.final_answer == FALLBACK and "irrelevant" in v.reason
    good = "Level 1 is the support agent, who must resolve a case within 24 hours."
    v2 = review_draft(good, [good], q)
    assert v2.approved and v2.final_answer == good


def test_order_part_of_a_mixed_question_does_not_search_the_policy_kb():
    """Found by the held-out set: 'status of order ... and who do I contact' pulled in a cancellation sentence."""
    q = "What is the status of order ORD-1047 and who do I contact if it is late?"
    r = handle_request(q, "d2-heldout")
    assert r["answer"].startswith("Order ORD-1047 is currently")
    assert "cancelled for free" not in r["answer"] and r["response"]["sources"] == ["escalation_matrix"]
    assert "support" in r["answer"]


def test_relative_that_is_not_a_dangling_reference():
    """Regression found while fixing defect 2: 'a case THAT is older than 72 hours' wrongly asked to clarify."""
    q = "Who handles a case that is older than 72 hours?"
    assert classify_intent(q) == "policy"
    r = handle_request(q, "d2-relative")
    assert r["answer"].startswith("Level 3 is the support manager") and r["response"]["sources"] == ["escalation_matrix"]
    assert classify_intent("Can I cancel that?") == "policy"            # anchor 'cancel' -> still answerable
    assert classify_intent("Who do I contact about that?") == "clarify"  # a real dangling 'that'
    assert resolve_reference("who do I contact about that", "electronics") == "who do I contact about my electronics"


def test_ambiguous_it_with_a_clear_topic_is_not_over_clarified():
    assert classify_intent("Can I cancel it?") == "policy"
    assert classify_intent("How do I return it?") == "policy"
    assert classify_intent("Where is it?") == "order_id_missing"
    assert classify_intent("Who do I contact for a complaint?") == "policy"
