"""MOCK_LLM for AutoGen: a deterministic ChatCompletionClient (no network, no key).

Two behaviours, chosen by the agent's own system message:
  * Policy-Compliance-Reviewer: reads QUESTION / DRAFT / CONTEXT from the task and checks every draft
    sentence for  - grounding   (supported by the context, incl. numbers / negation / contradictions)
                  - relevance   (answers THIS question, not another product category or another intent)
                  - safety      (no raw phone/card numbers, no system-prompt leak)
                  - completeness (an order id in the question is addressed)
    and replies with a JSON report.
  * Final-Editor: reads that report and returns a Verdict JSON. Nothing wrong -> approve the draft
    unchanged; otherwise remove ONLY the flagged sentences (it never writes new facts).
"""
import json
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence, Union

import llm  # noqa: F401
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, ModelFamily, RequestUsage

from agents.guardrails import FULL_CARD_RE, PHONE_RE, _sentences, groundedness_problems
from agents.routing import classify_intent, extract_order_id
from agents.schemas import ASK_ORDER_ID, CLARIFY_QUESTION, NOT_FOUND_PREFIX, PARTIAL_ANSWER
from rag.generate import FALLBACK
from rag.relevance import category_compatible, is_who_question, names_a_role, split_subqueries

_USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)
ORDER_SENTENCE_STARTS = ("Order ORD", "It has a high escalation")
SYSTEM_SENTENCES = {ASK_ORDER_ID, PARTIAL_ANSWER, FALLBACK, CLARIFY_QUESTION}


def is_system_sentence(sentence: str) -> bool:
    """Fixed sentences written by the system itself (hand-offs, 'order not found', 'please share your id')."""
    return sentence in SYSTEM_SENTENCES or sentence.startswith(NOT_FOUND_PREFIX)


def _text(m) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def split_task(task_text: str) -> tuple[str, str, list[str]]:
    """Returns (question, draft, context_sentences). QUESTION is optional (empty if absent)."""
    question = ""
    if "QUESTION:" in task_text:
        question = task_text.split("QUESTION:", 1)[1].split("DRAFT:", 1)[0].strip()
    draft = task_text.split("DRAFT:", 1)[1].split("CONTEXT:", 1)[0].strip()
    context = [c.strip() for c in task_text.split("CONTEXT:", 1)[1].split("\n") if c.strip()]
    return question, draft, context


def review_report(question: str, draft: str, context: list[str]) -> dict:
    sentences = _sentences(draft)
    unsupported = [s for s, _ in groundedness_problems(draft, context) if not is_system_sentence(s)]
    irrelevant, unsafe = [], []
    intent = classify_intent(question) if question else None
    # Answer-type check: a single-part "who ..." question must be answered by a sentence naming a role.
    who_only = bool(question) and is_who_question(question) and len(split_subqueries(question)) == 1
    for s in sentences:
        if is_system_sentence(s):
            continue
        is_order = s.startswith(ORDER_SENTENCE_STARTS)
        if question and (not category_compatible(question, s)
                         or (intent == "policy" and is_order)
                         or (intent == "order" and not is_order)
                         or (who_only and not is_order and not names_a_role(s))):
            irrelevant.append(s)
        if PHONE_RE.search(s) or FULL_CARD_RE.search(s) or "system prompt" in s.lower():
            unsafe.append(s)
    missing = []
    order_id = extract_order_id(question) if question else None
    if order_id and order_id not in draft and draft != FALLBACK:
        missing.append(f"the draft never mentions order {order_id}")
    return {"unsupported": unsupported, "irrelevant": irrelevant, "unsafe": unsafe, "missing": missing}


class MockAutogenClient(ChatCompletionClient):
    def __init__(self):
        self.calls = 0

    @property
    def model_info(self):
        return {"vision": False, "function_calling": False, "json_output": True,
                "family": ModelFamily.UNKNOWN, "structured_output": True}

    @property
    def capabilities(self):  # deprecated alias
        return self.model_info

    async def create(self, messages: Sequence[Any], *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        self.calls += 1
        system = _text(messages[0])
        body = [_text(m) for m in messages[1:]]
        task = next(t for t in body if "DRAFT:" in t and "CONTEXT:" in t)
        question, draft, context = split_task(task)

        if "Final-Editor" in system:
            reviewer_reply = next((t for t in reversed(body) if "REPORT:" in t), "REPORT: {}")
            report = json.loads(reviewer_reply.split("REPORT:", 1)[1].strip().splitlines()[0])
            remove = set(report.get("unsupported", []) + report.get("irrelevant", []) + report.get("unsafe", []))
            notes = [f"{kind}: {' | '.join(report[kind])}" for kind in ("unsupported", "irrelevant", "unsafe")
                     if report.get(kind)]
            if remove:
                kept = [s for s in _sentences(draft) if s not in remove]
                verdict = {"approved": False, "final_answer": " ".join(kept) if kept else FALLBACK,
                           "reason": "Removed flagged sentence(s) -> " + "; ".join(notes)}
            elif report.get("missing"):
                verdict = {"approved": False, "final_answer": draft,
                           "reason": "Incomplete (nothing could be removed): " + "; ".join(report["missing"])}
            else:
                verdict = {"approved": True, "final_answer": draft,
                           "reason": "Draft is grounded, relevant to the question, safe and complete."}
            content = json.dumps(verdict)
        else:  # Policy-Compliance-Reviewer
            content = "Compliance review of the draft.\nREPORT: " + json.dumps(review_report(question, draft, context))
        return CreateResult(finish_reason="stop", content=content, usage=_USAGE, cached=False)

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                            extra_create_args={}, cancellation_token=None
                            ) -> AsyncGenerator[Union[str, CreateResult], None]:
        yield await self.create(messages, json_output=json_output)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return _USAGE

    def total_usage(self) -> RequestUsage:
        return _USAGE

    def count_tokens(self, messages, *, tools=[]) -> int:
        return sum(len(_text(m)) // 4 for m in messages)

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 8000 - self.count_tokens(messages)
