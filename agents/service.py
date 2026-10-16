"""The one pipeline every channel (script, HTTP, WebSocket) uses:

 input guardrails -> session memory -> budget check -> CrewAI crew (RAG + lookup + composer)
 -> Pydantic validation -> output groundedness guard -> AutoGen review -> final answer
"""
import llm  # noqa: F401  (telemetry off before crewai/autogen import)

from agents.crew import run_crew
from agents.guardrails import detect_injection, groundedness_check, mask_pii
from agents.memory import build_chat
from agents.schemas import SupportResponse, order_sentences
from governance.budget import BudgetExceeded, enforce_budget
from governance.cache import CACHE, cached_answer, normalize
from rag import store
from rag.generate import FALLBACK
from review.autogen_review import review_draft

BLOCKED_ANSWER = "Sorry, I can't help with that request."


def _refusal(answer: str, **extra) -> dict:
    resp = SupportResponse(answer=answer, grounded=False)
    return {"answer": answer, "response": resp.model_dump(), "review": None, **extra}


def support_texts(resp: SupportResponse, question: str) -> list[str]:
    """What the answer is allowed to say: retrieved policy chunks + the looked-up order facts."""
    r = cached_answer(question)  # same key the RAG tool used -> a cache hit, no second LLM call
    texts = [c["text"] for c in r["chunks"] if c["similarity"] >= 0.32]
    if resp.order_id:
        texts += order_sentences(resp.order_id, resp.order_status, resp.order_value_inr,
                                 resp.escalation_score, resp.escalate)
    return texts


def core_pipeline(question: str, review: bool = True) -> dict:
    """Runs after masking/memory: budget -> crew -> schema -> groundedness -> review."""
    try:
        budget = enforce_budget(question)
    except BudgetExceeded as e:
        return _refusal(f"Your request is too large to process. {e}", budget_rejected=True)

    draft = run_crew(question)                                  # CrewAI .kickoff(), validated as SupportResponse
    support = support_texts(draft, question)
    ok, unsupported = groundedness_check(draft.answer, support)
    if not ok or (not draft.grounded and not draft.order_id):   # output-side guardrail: refuse
        out = _refusal(FALLBACK, guardrail="groundedness", unsupported=unsupported, budget=budget)
        out["draft"] = draft.model_dump()
        return out

    verdict = review_draft(draft.answer, support) if review else None   # AutoGen review stage
    final = draft.model_copy(update={"answer": verdict.final_answer if verdict else draft.answer})
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
        return {**base, **_refusal(BLOCKED_ANSWER, guardrail="prompt_injection", matched=injection)}
    chat = build_chat(_pipeline_for_session(review))
    msg = chat.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
    return {**base, "resolved_question": msg.additional_kwargs["resolved"], **msg.additional_kwargs["result"]}


def add_document(doc_id: str, title: str, text: str) -> dict:
    counts = store.index_documents([{"doc_id": doc_id, "title": title, "text": text}])
    CACHE.clear()  # cached answers may be stale after the knowledge base changes
    return {"doc_id": doc_id, "chunks_indexed": counts}
