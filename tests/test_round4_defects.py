"""Regression tests for the round-4 audit defects (each was reproduced on the unfixed code first).

1. "How long does delivery take to remote areas?" returned the metro / other-town sentences, not the
   remote-pin-code rule (customer wording "remote areas" vs policy wording "remote pin codes").
2. "When will my cash on delivery refund reach my bank account?" returned only the generic COD sentence
   ("when will" was not treated as a duration question) and "...my COD return be credited..." matched the
   payment-failure policy (the abbreviation COD is unknown to the embedding model).
3. Hair dryer / kettle / microwave warranty answers committed to one category (or fell back) although the
   policy never says whether such an item is Electronics or Home appliances.
4. In a multi-topic question the vague part "how long will it/that take?" was searched alone and dragged in
   an unrelated return-window sentence.
"""
import pytest

from agents.service import core_pipeline
from rag import store
from rag.generate import CATEGORY_CLARIFICATION, FALLBACK, answer_question
from rag.relevance import ambiguous_appliance_doc, is_duration_question, normalize_phrases

REMOTE = "Remote pin codes may need up to 10 working days."
BANK_DURATION = "A bank refund takes 5 to 7 working days"


@pytest.fixture(scope="module", autouse=True)
def _index():
    store.index_documents()


# ---- 1. remote areas -------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["How long does delivery take to remote areas?",
                               "What is the delivery time for a remote village?",
                               "How many days to deliver to a rural area?"])
def test_remote_area_wording_finds_remote_pin_code_rule(q):
    r = answer_question(q)
    assert REMOTE in r["used"] and r["sources"] == ["delivery_sla"]


def test_remote_rewrite_only_touches_remote_phrases():
    assert normalize_phrases("remote areas") == "remote pin codes"
    assert normalize_phrases("Do metro cities get faster delivery?") == "Do metro cities get faster delivery?"


# ---- 2. COD refund timing --------------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["When will my cash on delivery refund reach my bank account?",
                               "When will the money for my COD return be credited to my bank?"])
def test_when_will_cod_refund_arrive_returns_the_timeline(q):
    r = answer_question(q)
    assert any(BANK_DURATION in s for s in r["used"]) or any(BANK_DURATION in s for s in [r["answer"]])
    assert r["sources"] == ["cod_refund"]


def test_when_question_detection_is_narrow():
    assert is_duration_question("When will my refund reach my bank account?")
    assert not is_duration_question("When can I cancel my order?")
    assert not is_duration_question("When is an order called delayed?")


def test_cod_abbreviation_is_expanded_for_retrieval_only():
    assert normalize_phrases("my COD order") == "my cash on delivery order"
    assert normalize_phrases("Barcode") == "Barcode"


# ---- 3. ambiguous appliances -----------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["What is the warranty on a hair dryer?",
                               "My hair dryer stopped working, is it under warranty?",
                               "Is there a warranty on a microwave oven?",
                               "What is the warranty on a mixer grinder?"])
def test_ambiguous_appliance_warranty_states_both_rules_and_asks(q):
    r = answer_question(q)
    assert "12 month" in r["answer"] and "6 month" in r["answer"]
    assert CATEGORY_CLARIFICATION in r["used"] and r["sources"] == ["warranty_terms"]
    assert "no warranty" not in r["answer"]  # the apparel/footwear/beauty rule is not relevant


def test_ambiguous_appliance_survives_the_full_pipeline():
    out = core_pipeline("What is the warranty on a hair dryer?")
    assert CATEGORY_CLARIFICATION in out["answer"] and "12 month" in out["answer"]


def test_stated_category_is_not_ambiguous():
    assert ambiguous_appliance_doc("Warranty on a hair dryer from home appliances?") is None
    assert ambiguous_appliance_doc("What is the warranty on electronics?") is None
    assert ambiguous_appliance_doc("How do I clean my hair dryer?") is None  # not a category-dependent policy
    r = answer_question("What is the warranty on earphones?")  # unambiguous product -> glossary, no clarification
    assert CATEGORY_CLARIFICATION not in r["used"] and "12 month" in r["answer"]


# ---- 4. unrelated sentence in a multi-topic answer -------------------------------------------------------
@pytest.mark.parametrize("q", ["Do you deliver to remote pin codes and how long will it take?",
                               "Do you ship to remote villages and how many days will that take?"])
def test_vague_second_part_does_not_add_unrelated_sentence(q):
    r = answer_question(q)
    assert r["used"] == [REMOTE] and r["sources"] == ["delivery_sla"]


def test_genuine_two_topic_question_still_gets_both_topics():
    r = answer_question("My parcel is very late, how long should delivery take and who do I escalate to?")
    assert {"delivery_sla", "escalation_matrix"} <= set(r["sources"])


def test_out_of_scope_still_falls_back():
    assert answer_question("Who is the prime minister of India?")["answer"] == FALLBACK
    assert answer_question("How do I order a pizza online?")["answer"] == FALLBACK


def test_judge_counts_the_fixed_category_sentence_as_on_topic():
    from eval.judge_eval import MockJudge
    answer = ("Electronics carry a 12 month brand warranty that covers manufacturing defects only. "
              "Home appliances carry a 6 month warranty. " + CATEGORY_CLARIFICATION)
    prompt = ("KIND: kb\nQUESTION: q\nCONTEXT: [\"" + answer.replace('"', "'") + "\"]\nEXPECTED_FACTS: [\"12 month|6 month\"]\n"
              "EXPECTED_DOCS: [\"warranty_terms\"]\nSOURCES: [\"warranty_terms\"]\nANSWER: " + answer)
    assert MockJudge().judge(prompt)["accuracy"] == 5.0
    unrelated = prompt.replace("ANSWER: " + answer, "ANSWER: " + answer + " Home products can be returned within 14 days of delivery.")
    assert MockJudge().judge(unrelated)["accuracy"] < 5.0  # a genuinely unrelated sentence is still penalised
