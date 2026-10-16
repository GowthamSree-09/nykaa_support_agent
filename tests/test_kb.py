import re

from kb.documents import KB_DOCUMENTS

REQUIRED_TOPICS = {
    "return_window", "cod_refund", "delivery_sla", "reverse_pickup", "warranty_terms",
    "order_cancellation", "loyalty_points", "payment_failure", "size_exchange",
    "damaged_item_claim", "international_shipping", "escalation_matrix",
}


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def test_kb_covers_all_topics():
    assert len(KB_DOCUMENTS) >= 12
    assert REQUIRED_TOPICS <= {d["doc_id"] for d in KB_DOCUMENTS}


def test_each_doc_has_2_to_5_sentences():
    for d in KB_DOCUMENTS:
        assert 2 <= len(sentences(d["text"])) <= 5, d["doc_id"]
