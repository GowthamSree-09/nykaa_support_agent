"""The one pipeline every channel (script, HTTP, WebSocket) uses:

 input guardrails -> session memory -> budget check -> CrewAI crew (RAG + lookup + composer)
 -> Pydantic validation -> output groundedness guard -> AutoGen review -> groundedness re-check -> answer
"""
import llm  # noqa: F401  (telemetry off before crewai/autogen import)

from agents.crew import run_crew
from agents.guardrails import detect_injection, groundedness_check, mask_pii
from agents.memory import build_chat
from agents.routing import ORDER_ID_RE, needs_policy, policy_query
from agents.schemas import (ASK_ORDER_ID, CLARIFY_QUESTION, PARTIAL_ANSWER, SupportResponse, not_found_sentence,
                            order_sentences)
from governance.audit import audit
from governance.budget import BudgetExceeded, enforce_budget
from governance.cache import CACHE, cached_answer
from rag import store
from rag.generate import FALLBACK, SUBQUERY_FLOOR
from review.autogen_review import review_draft

BLOCKED_ANSWER = "Sorry, I can't help with that request."


def _refusal(answer: str, **extra) -> dict:
    resp = SupportResponse(answer=answer, grounded=False)
    return {"answer": answer, "response": resp.model_dump(), "review": None, **extra}


def support_texts(resp: SupportResponse, question: str) -> list[str]:
    """What the answer is allowed to say: retrieved policy text, looked-up order facts, fixed system sentences."""
    texts: list[str] = []
    if needs_policy(question):
        r = cached_answer(policy_query(question))  # same key the RAG tool used -> cache hit, no second LLM call
        texts += r["used"] + [c["text"] for c in r["chunks"] if c["similarity"] >= SUBQUERY_FLOOR]
    if resp.order_id:
        texts += order_sentences(resp.order_id, resp.order_status, resp.order_value_inr,
                                 resp.escalation_score, resp.escalate)
    texts += [ASK_ORDER_ID, PARTIAL_ANSWER, CLARIFY_QUESTION] +[not_found_sentence(i.upper()) for i in ORDER_ID_RE.findall(question)]
    return texts


def core_pipeline(question: str, review: bool = True) -> dict:
    """Runs after masking/memory: budget -> crew -> schema -> groundedness -> review -> re-check."""
    try:
        budget = enforce_budget(question)
    except BudgetExceeded as e:
        audit("budget_rejected", detail=str(e))
        return _refusal(f"Your request is too large to process. {e}", budget_rejected=True)

    draft = run_crew(question)                                  # CrewAI .kickoff(), validated as SupportResponse
    support = support_texts(draft, question)
    ok, unsupported = groundedness_check(draft.answer, support)
    if not ok or (not draft.grounded and not draft.order_id):   # output-side guardrail: refuse
        audit("groundedness_refusal", stage="draft")
        out = _refusal(FALLBACK, guardrail="groundedness", unsupported=unsupported, budget=budget)
        out["draft"] = draft.model_dump()
        return out

    verdict = review_draft(draft.answer, support, question) if review else None   # AutoGen review stage
    final_answer = verdict.final_answer if verdict else draft.answer
    if verdict and final_answer != draft.answer:
        ok2, bad2 = groundedness_check(final_answer, support)  # the editor's revision must be grounded too
        if not ok2:
            audit("groundedness_refusal", stage="after_review")
            out = _refusal(FALLBACK, guardrail="groundedness_after_review", unsupported=bad2, budget=budget)
            out["draft"], out["review"] = draft.model_dump(), verdict.model_dump()
            return out
    final = draft.model_copy(update={"answer": final_answer})
    SupportResponse.model_validate(final.model_dump())
    return {"answer": final.answer, "response": final.model_dump(), "draft": draft.model_dump(),
            "review": verdict.model_dump() if verdict else None, "budget": budget}


def _pipeline_for_session(review: bool):
    return lambda resolved_question: core_pipeline(resolved_question, review=review)


def handle_request(question: str, session_id: str = "default", review: bool = True) -> dict:
    masked, pii_found = mask_pii(question)                       # input guardrail 1: PII masking
    base = {"masked_question": masked, "pii_masked": pii_found, "session_id": session_id}
    injection = detect_injection(masked)                         # input guardrail 2: prompt injection
    if injection:
        audit("prompt_injection_blocked", session_id=session_id)
        return {**base, **_refusal(BLOCKED_ANSWER, guardrail="prompt_injection", matched=injection)}
    chat = build_chat(_pipeline_for_session(review))
    msg = chat.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
    return {**base, "resolved_question": msg.additional_kwargs["resolved"], **msg.additional_kwargs["result"]}


def add_document(doc_id: str, title: str, text: str) -> dict:
    counts = store.index_documents([{"doc_id": doc_id, "title": title, "text": text}])
    CACHE.clear()  # cached answers may be stale after the knowledge base changes
    audit("cache_cleared", reason="document_added", doc_id=doc_id)
    return {"doc_id": doc_id, "chunks_indexed": counts}
