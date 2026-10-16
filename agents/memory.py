"""Task 8 - session memory with LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory.

(RunnableWithMessageHistory prints a LangChainDeprecationWarning pointing to LangGraph; this is
expected and harmless here.) Memory lives in this Python process only. Each session_id has its
own history, so a new session_id starts completely empty.
"""
import re
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.relevance import DANGLING_RE, find_product_phrase, has_anchor, resolve_reference

_STORES: dict[str, InMemoryChatMessageHistory] = {}
ORDER_ID_RE = re.compile(r"ORD-\d+", re.I)
FOLLOW_UP_RE = re.compile(r"\b(it|that order|this order|my order|the order|same order)\b", re.I)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _STORES.setdefault(session_id, InMemoryChatMessageHistory())


def reset_all_sessions() -> None:
    _STORES.clear()


def reset_session(session_id: str) -> bool:
    """Forget one conversation. Returns True if the session existed."""
    return _STORES.pop(session_id, None) is not None


def resolve_with_history(question: str, history) -> str:
    """Use earlier turns to resolve a vague follow-up.

    1. 'my order' / 'it' without an id  -> reuse the last order id from earlier turns;
    2. 'it' / 'that' with no product, category or policy word ("Who do I contact if it fails?")
       -> point it at the product/category from the most recent earlier question
       ("... if my earphones fails?"). If nothing in the session helps, the question stays vague and the
       pipeline asks the customer to clarify.
    """
    if ORDER_ID_RE.search(question):
        return question
    humans = [m for m in reversed(history) if m.type == "human"]
    if FOLLOW_UP_RE.search(question):
        for msg in humans:
            if m := ORDER_ID_RE.search(msg.content):
                return f"{question} (order {m.group(0).upper()})"
    if DANGLING_RE.search(question) and not has_anchor(question):
        for msg in humans:
            if phrase := find_product_phrase(msg.content):
                return resolve_reference(question, phrase)
    return question


def build_chat(pipeline: Callable[[str], dict]) -> RunnableWithMessageHistory:
    """pipeline(resolved_question) -> dict with at least {'answer': str}. Wrapped with session memory."""
    def step(inputs: dict) -> AIMessage:
        resolved = resolve_with_history(inputs["input"], inputs["history"])
        result = pipeline(resolved)
        return AIMessage(content=result["answer"], additional_kwargs={"result": result, "resolved": resolved})

    return RunnableWithMessageHistory(
        RunnableLambda(step), get_session_history,
        input_messages_key="input", history_messages_key="history",
    )
