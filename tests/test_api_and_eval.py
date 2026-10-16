import json
import os

import pytest
from fastapi.testclient import TestClient

from api import logging_utils
from api.main import app
from eval import judge_eval
from rag import store


@pytest.fixture(scope="module")
def client():
    if os.path.exists(logging_utils.LOG_PATH):
        os.remove(logging_utils.LOG_PATH)
    with TestClient(app) as c:
        yield c
    store.delete_document("gift_wrap")  # leave the persistent index as we found it


def test_ask_endpoint(client):
    r = client.post("/ask", json={"question": "How many days do I have to return a lipstick?", "session_id": "t1"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] and "7 days" in body["answer"] and body["trace_id"]
    assert client.post("/ask", json={"question": ""}).status_code == 422  # Pydantic validation


def test_ask_oversized_rejected(client):
    assert client.post("/ask", json={"question": "word " * 2000}).status_code == 413


def test_add_document_then_retrievable(client):
    text = "Gift wrapping costs 50 INR per order. Gift wrap is available for all categories."
    r = client.post("/add-document", json={"doc_id": "gift_wrap", "title": "Gift wrapping", "text": text})
    assert r.status_code == 200 and r.json()["chunks_indexed"]["sentence"] == 2
    ans = client.post("/ask", json={"question": "How much does gift wrapping cost?", "session_id": "t2"}).json()
    assert "50 INR" in ans["answer"]
    assert client.post("/add-document", json={"doc_id": "x", "title": "t", "text": "One sentence only."}).status_code == 422


def test_websocket_multi_turn_and_survives_disconnect(client):
    with client.websocket_connect("/ws/chat?session_id=ws1") as ws:
        ws.send_text("Where is my order ORD-1023?")
        assert ws.receive_json()["order_id"] == "ORD-1023"
        ws.send_text("Can I cancel it?")
        assert ws.receive_json()["order_id"] == "ORD-1023"  # memory across turns
    # client has disconnected; the server must still serve new clients
    with client.websocket_connect("/ws/chat?session_id=ws2") as ws:
        ws.send_text("How many days do I have to return a lipstick?")
        assert ws.receive_json()["grounded"]
    assert client.get("/health").json()["status"] == "ok"


def test_jsonl_log_has_trace_ids_and_no_raw_pii(client):
    client.post("/ask", json={"question": "My phone is 9876543210 and card ending 4242, where is ORD-1023?"})
    lines = [json.loads(l) for l in open(logging_utils.LOG_PATH, encoding="utf-8")]
    assert lines and all("trace_id" in e and "duration_ms" in e for e in lines)
    raw = open(logging_utils.LOG_PATH, encoding="utf-8").read()
    assert "9876543210" not in raw and "4242" not in raw and "[PHONE_MASKED]" in raw
    assert any(e["status"] == "client_disconnected" for e in lines)


def test_evaluation_15_queries_four_scores():
    out = judge_eval.run_eval()
    assert len(out["rows"]) == 15 and len(judge_eval.TEST_SET) == 15
    assert sum(r["kind"] in ("out_of_scope", "injection") for r in out["rows"]) >= 2
    for r in out["rows"]:
        assert all(1 <= r[k] <= 5 for k in ("accuracy", "grounding", "completeness", "safety"))
    assert set(out["averages"]) == {"accuracy", "grounding", "completeness", "safety"}
