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

_STORES: dict[str, InMemoryChatMessageHistory] = {}
ORDER_ID_RE = re.compile(r"ORD-\d+", re.I)
FOLLOW_UP_RE = re.compile(r"\b(it|that order|this order|my order|the order|same order)\b", re.I)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _STORES.setdefault(session_id, InMemoryChatMessageHistory())


def reset_all_sessions() -> None:
    _STORES.clear()


def resolve_with_history(question: str, history) -> str:
    """If the customer says 'my order'/'it' without an id, reuse the last order id from earlier turns."""
    if ORDER_ID_RE.search(question) or not FOLLOW_UP_RE.search(question):
        return question
    for msg in reversed(history):
        if msg.type == "human" and (m := ORDER_ID_RE.search(msg.content)):
            return f"{question} (order {m.group(0).upper()})"
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
