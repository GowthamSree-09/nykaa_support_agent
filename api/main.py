"""Task 11 - FastAPI backend: POST /ask, POST /add-document, GET /health, WebSocket /ws/chat.

Run:  uvicorn api.main:app --reload      (all LLM work is MOCK_LLM: no API key, no network)
"""
import time

import llm  # noqa: F401  (telemetry off)
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from agents import service
from agents.memory import reset_session
from api.logging_utils import log_request, new_trace_id
from kb.documents import KB_DOCUMENTS  # noqa: F401
from rag import store
from rag.chunking import split_sentences

app = FastAPI(title="Nykaa Support Agent (MOCK_LLM)")


@app.on_event("startup")
def _startup():
    store.index_documents()  # idempotent upsert of the knowledge base


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=20000)
    session_id: str = Field(default="default", min_length=1, max_length=64)


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    grounded: bool
    sources: list[str] = []
    order_id: str | None = None
    order_status: str | None = None
    escalation_score: float | None = None
    escalate: bool = False
    guardrail: str | None = None
    pii_masked: list[str] = []
    review_approved: bool | None = None
    review_reason: str | None = None


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: dict[str, int]


def _to_response(trace_id: str, result: dict) -> AskResponse:
    r, rev = result["response"], result.get("review") or {}
    return AskResponse(trace_id=trace_id, answer=result["answer"], grounded=r["grounded"], sources=r["sources"],
                       order_id=r["order_id"], order_status=r["order_status"],
                       escalation_score=r["escalation_score"], escalate=r["escalate"],
                       guardrail=result.get("guardrail"), pii_masked=result.get("pii_masked", []),
                       review_approved=rev.get("approved"), review_reason=rev.get("reason"))


def _log_ask(trace_id, endpoint, question, started, result, status=200):
    log_request(trace_id, endpoint, question, started, status=status, session_id=result.get("session_id"),
                pii_masked=result.get("pii_masked", []), guardrail=result.get("guardrail"),
                grounded=result["response"]["grounded"], answer_chars=len(result["answer"]))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    trace_id, started = new_trace_id(), time.perf_counter()
    result = service.handle_request(req.question, req.session_id)
    if result.get("budget_rejected"):
        _log_ask(trace_id, "/ask", req.question, started, result, status=413)
        raise HTTPException(status_code=413, detail=result["answer"])
    _log_ask(trace_id, "/ask", req.question, started, result)
    return _to_response(trace_id, result)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    trace_id, started = new_trace_id(), time.perf_counter()
    n = len(split_sentences(req.text))
    if not 2 <= n <= 5:
        log_request(trace_id, "/add-document", req.text, started, status=422, doc_id=req.doc_id)
        raise HTTPException(status_code=422, detail=f"Document must have 2-5 sentences, got {n}")
    out = service.add_document(req.doc_id, req.title, req.text)
    log_request(trace_id, "/add-document", req.text, started, status=200, doc_id=req.doc_id)
    return AddDocumentResponse(trace_id=trace_id, **out)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Never leak internals or the customer's text: log only the trace id and the error type."""
    trace_id = new_trace_id()
    log_request(trace_id, request.url.path, "", time.perf_counter(), status=500, error_type=type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "trace_id": trace_id})


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Reset one conversation's memory (the next question starts with no history)."""
    return {"session_id": session_id, "cleared": reset_session(session_id)}


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": True}


@app.websocket("/ws/chat")
async def chat(ws: WebSocket, session_id: str = "ws-default"):
    await ws.accept()
    try:
        while True:
            question = await ws.receive_text()
            trace_id, started = new_trace_id(), time.perf_counter()
            try:
                result = await run_in_threadpool(service.handle_request, question, session_id)
            except Exception as exc:  # one bad message must not kill the conversation or the server
                log_request(trace_id, "/ws/chat", "", started, status=500, error_type=type(exc).__name__)
                await ws.send_json({"trace_id": trace_id, "error": "Internal server error"})
                continue
            _log_ask(trace_id, "/ws/chat", question, started, result,
                     status=413 if result.get("budget_rejected") else 200)
            await ws.send_json(_to_response(trace_id, result).model_dump())
    except WebSocketDisconnect:
        # A client leaving mid-conversation is normal: log it and let this handler end.
        # Other clients keep their own handlers, so the server stays up.
        log_request(new_trace_id(), "/ws/chat", "", time.perf_counter(), status="client_disconnected",
                    session_id=session_id)
