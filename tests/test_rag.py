import pytest

from kb.documents import KB_DOCUMENTS
from rag import evaluate, store
from rag.generate import FALLBACK, SIMILARITY_THRESHOLD, answer_question
from rag.queries import IN_SCOPE, OUT_OF_SCOPE


@pytest.fixture(scope="module", autouse=True)
def indexed():
    return store.index_documents()


def test_two_separate_collections_indexed(indexed):
    assert store.get_collection("fixed").name != store.get_collection("sentence").name
    assert store.get_collection("fixed").count() == indexed["fixed"] > 0
    assert store.get_collection("sentence").count() == indexed["sentence"] > len(KB_DOCUMENTS)


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_sensible_retrieval(strategy):
    top = store.search("How many days do I have to return a lipstick?", strategy, 3)
    assert top[0]["doc_id"] == "return_window"
    assert set(top[0]) >= {"chunk_id", "doc_id", "text", "similarity"}


def test_threshold_separates_clusters():
    for strategy in ("fixed", "sentence"):
        low_in = min(store.search(q, strategy, 1)[0]["similarity"] for q, _ in IN_SCOPE)
        high_out = max(store.search(q, strategy, 1)[0]["similarity"] for q in OUT_OF_SCOPE)
        assert high_out < SIMILARITY_THRESHOLD < low_in


def test_in_scope_grounded_and_out_of_scope_fallback():
    for q, rel in IN_SCOPE:
        r = answer_question(q)
        assert r["grounded"] and r["answer"] != FALLBACK
    for q in OUT_OF_SCOPE:
        r = answer_question(q)
        assert not r["grounded"] and r["answer"] == FALLBACK and r["sources"] == []


def test_metrics_both_collections():
    res = evaluate.run()
    assert res["sentence"]["avg_precision"] > res["fixed"]["avg_precision"]
    for data in res.values():
        assert 0 <= data["avg_precision"] <= 1 and 0 <= data["avg_recall"] <= 1
