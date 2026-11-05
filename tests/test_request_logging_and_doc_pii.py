"""Regression tests for the final PDF verification (round 5).

1. Task 12 says every request must produce one JSON-Lines entry with a trace id and timing. Before the fix,
   GET /health, DELETE /sessions/{id} and Pydantic validation errors (e.g. an empty /ask question) wrote no
   entry. A middleware now logs whatever an endpoint has not already logged, so each request has exactly one.
2. Task 10 says PII is masked on the input side. POST /add-document used to store a phone number or card
   last-4 verbatim in the persistent ChromaDB index (its sentences are copied into customer answers).
   The text is now masked before indexing.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api import logging_utils
from api.main import app
from rag import store

PHONE = "9876543210"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOG_PATH", str(tmp_path / "requests.jsonl"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def entries():
    try:
        return [json.loads(line) for line in open(logging_utils.LOG_PATH, encoding="utf-8")]
    except FileNotFoundError:
        return []


def assert_logged_once(before: int, endpoint: str, status):
    new = entries()[before:]
    assert len(new) == 1, f"expected exactly one log entry, got {new}"
    e = new[0]
    assert e["endpoint"] == endpoint and e["status"] == status
    assert len(e["trace_id"]) == 32 and isinstance(e["duration_ms"], (int, float)) and e["timestamp"]
    return e


# ---- Task 12: one entry per request, for every operation -------------------------------------------------
def test_health_is_logged_once(client):
    n = len(entries())
    assert client.get("/health").status_code == 200
    e = assert_logged_once(n, "/health", 200)
    assert e["method"] == "GET"


def test_delete_session_is_logged_once(client):
    n = len(entries())
    assert client.delete("/sessions/some-session").status_code == 200
    e = assert_logged_once(n, "/sessions/{session_id}", 200)
    assert e["method"] == "DELETE"


def test_validation_error_is_logged_once_without_the_body(client):
    n = len(entries())
    r = client.post("/ask", json={"question": "", "session_id": f"card ending 4242 {PHONE}"})
    assert r.status_code == 422
    e = assert_logged_once(n, "/ask", 422)
    assert e["request_text"] == ""                       # a rejected body is never written to disk
    assert PHONE not in open(logging_utils.LOG_PATH, encoding="utf-8").read()


def test_unknown_path_is_logged_once(client):
    n = len(entries())
    assert client.get("/no-such-endpoint").status_code == 404
    assert_logged_once(n, "/no-such-endpoint", 404)


def test_ask_is_still_logged_once_with_its_rich_fields(client):
    n = len(entries())
    assert client.post("/ask", json={"question": "How many days do I have to return a lipstick?",
                                     "session_id": "log1"}).status_code == 200
    e = assert_logged_once(n, "/ask", 200)               # no second, middleware entry
    assert e["session_id"] == "log1" and e["grounded"] is True


def test_ask_oversized_413_is_logged_once(client):
    n = len(entries())
    assert client.post("/ask", json={"question": "word " * 2000}).status_code == 413
    assert_logged_once(n, "/ask", 413)


def test_add_document_rejected_422_is_logged_once(client):
    n = len(entries())
    assert client.post("/add-document", json={"doc_id": "x", "title": "t", "text": "One sentence."}).status_code == 422
    assert_logged_once(n, "/add-document", 422)


def test_internal_error_is_logged_once(client, monkeypatch):
    from agents import service

    def boom(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(service, "handle_request", boom)
    n = len(entries())
    r = client.post("/ask", json={"question": f"phone {PHONE}, where is my order?", "session_id": "err1"})
    assert r.status_code == 500 and "simulated" not in r.text
    e = assert_logged_once(n, "/ask", 500)
    assert e["error_type"] == "RuntimeError" and PHONE not in json.dumps(e)


def test_websocket_messages_and_disconnect_are_each_logged_once(client):
    n = len(entries())
    with client.websocket_connect("/ws/chat?session_id=wslog") as ws:
        ws.send_text(f"My phone is {PHONE}. Where is my order ORD-1023?")
        assert ws.receive_json()["order_id"] == "ORD-1023"
        ws.send_text("Can I cancel it?")
        ws.receive_json()
    new = entries()[n:]
    assert [e["endpoint"] for e in new] == ["/ws/chat"] * 3
    assert [e["status"] for e in new] == [200, 200, "client_disconnected"]
    assert all(len(e["trace_id"]) == 32 and "duration_ms" in e for e in new)
    assert PHONE not in open(logging_utils.LOG_PATH, encoding="utf-8").read()


# ---- Task 10: documents are masked before they reach the persistent index ------------------------------
def indexed_text(doc_id: str) -> str:
    chunks = []
    for strategy in ("sentence", "fixed"):
        chunks += store.search("phone card manager gift", strategy, 10, doc_id=doc_id)
    return " ".join(c["text"] for c in chunks)


def test_add_document_masks_phone_and_card_before_indexing(client):
    text = f"Call our manager on {PHONE} for gift wrapping. Card ending 4242 is kept on file."
    try:
        r = client.post("/add-document", json={"doc_id": "pii_doc", "title": "Contact", "text": text})
        assert r.status_code == 200 and r.json()["pii_masked"] == ["card", "phone"]
        stored = indexed_text("pii_doc")
        assert PHONE not in stored and "4242" not in stored          # never reaches ChromaDB, either collection
        assert "[PHONE_MASKED]" in stored and "[CARD_LAST4_MASKED]" in stored
        assert "gift wrapping" in stored                             # the policy wording is preserved
        raw_log = open(logging_utils.LOG_PATH, encoding="utf-8").read()
        assert PHONE not in raw_log and "4242" not in raw_log
    finally:
        store.delete_document("pii_doc")


def test_add_document_without_pii_is_unchanged_and_still_retrievable(client):
    text = "Gift wrapping costs 50 INR per order. Gift wrap is available for all categories."
    try:
        r = client.post("/add-document", json={"doc_id": "gift_wrap2", "title": "Gift wrapping", "text": text})
        assert r.status_code == 200 and r.json()["pii_masked"] == [] and r.json()["chunks_indexed"]["sentence"] == 2
        assert "Gift wrapping costs 50 INR per order." in indexed_text("gift_wrap2")
        ans = client.post("/ask", json={"question": "How much does gift wrapping cost?", "session_id": "pd2"}).json()
        assert "50 INR" in ans["answer"]
    finally:
        store.delete_document("gift_wrap2")
        client.post("/ask", json={"question": "warm up cache", "session_id": "pd3"})   # cache was cleared by add-document


def test_retrieved_answer_never_repeats_a_masked_number(client):
    text = f"For gift wrapping contact the store lead on {PHONE}. Gift wrapping costs 75 INR per order."
    try:
        client.post("/add-document", json={"doc_id": "pii_doc2", "title": "Gift", "text": text})
        ans = client.post("/ask", json={"question": "Who do I contact for gift wrapping?", "session_id": "pd4"}).json()
        assert PHONE not in ans["answer"]
    finally:
        store.delete_document("pii_doc2")
