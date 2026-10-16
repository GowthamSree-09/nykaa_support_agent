"""Question routing: what is the customer asking about?

  order            - only an order lookup (has an order id, nothing policy-like left)
  mixed            - an order id AND a policy question ("Can I cancel it? (order ORD-1023)")
  order_id_missing - asks about "my order" but gave no id (and none is in the session memory)
  clarify          - a vague "it/that" with nothing to point at ("Who do I contact if it fails?")
  policy           - a policy question

The three CrewAI agents still exist and each still uses its own tool; routing only decides WHEN a
tool is needed, so an order-only question does not drag unrelated policy text into the answer.
"""
import re

from llm.mock_llm import tokens
from rag.relevance import DANGLING_RE, ORDER_WORDS, has_anchor

ORDER_ID_RE = re.compile(r"ORD-\d+", re.I)
_PII_STATEMENT_RE = re.compile(
    r"(?:and\s+)?(?:my\s+)?(?:phone|mobile|contact|card)(?:\s+number)?(?:\s+(?:is|ending(?:\s+in|\s+with)?))?"
    r"\s*\[[A-Z0-9_]+\]\.?", re.I)
_PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_]+\]")
_ORDER_REF_RE = re.compile(r"\(order\s+ORD-\d+\)", re.I)


def extract_order_id(question: str) -> str | None:
    m = ORDER_ID_RE.search(question)
    return m.group(0).upper() if m else None


def policy_query(question: str) -> str:
    """The question with order ids and masked-PII statements removed (what the RAG tool should search)."""
    q = _PII_STATEMENT_RE.sub(" ", question)
    q = _PLACEHOLDER_RE.sub(" ", _ORDER_REF_RE.sub(" ", q))
    q = ORDER_ID_RE.sub(" ", q)
    return re.sub(r"\s+", " ", q).strip() or question


def classify_intent(question: str) -> str:
    stripped = policy_query(question)
    remaining = tokens(stripped) - ORDER_WORDS
    if extract_order_id(question):
        return "mixed" if remaining else "order"
    if not remaining:
        return "order_id_missing"
    # "Who do I contact if it fails?": a dangling "it" and no product/category/policy word to anchor it.
    if DANGLING_RE.search(stripped) and not has_anchor(stripped):
        return "clarify"
    return "policy"


def needs_policy(question: str) -> bool:
    return classify_intent(question) in ("policy", "mixed")
