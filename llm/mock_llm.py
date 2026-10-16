"""MOCK_LLM - a deterministic stand-in for a real language model (no network, no API key).

A real LLM *writes* new text from its training. This mock only applies fixed rules to the
text it is given (word overlap, keyword checks), so the same input always gives the same
output. That makes demos and tests reproducible. Every call is counted in CALLS so the
cache (Task 16) can prove it avoided a call.
"""
import os
import re

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

CALLS = {"count": 0}

STOPWORDS = set(
    "a an the is are was were be been do does did can i my me we you your it its of to in on for "
    "and or if at by with from how what when where which who why will would should could get have has "
    "had this that these those there their they them as so than then about after before into out up "
    "very much many long".split()
)


def _stem(w: str) -> str:
    """Very crude stemmer so 'returned' matches 'return' and 'refunds' matches 'refund'."""
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def tokens(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


MARGIN = 0.10  # a second sentence must score within this distance of the best one


def select_sentences(question: str, scored: list[tuple[str, str, float]], max_sentences: int = 2,
                     margin: float = MARGIN) -> list[tuple[str, str]]:
    """Pick the sentences that answer ONE question. scored = [(sentence, doc_id, similarity), ...].

    1. drop sentences that are only about product categories the customer did not ask about,
    2. take the highest-similarity sentence (the anchor),
    3. add more sentences ONLY from the anchor's own document and only if they score within
       `margin` of the anchor - so unrelated policies are never appended.
    Returns [(sentence, doc_id)].
    """
    from rag.relevance import category_compatible  # local import: avoids a circular import

    candidates = [t for t in scored if category_compatible(question, t[0])]
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda t: -t[2])
    anchor_sentence, anchor_doc, anchor_score = ranked[0]
    picked = [(anchor_sentence, anchor_doc)]
    for sentence, doc, score in ranked[1:]:
        if len(picked) >= max_sentences:
            break
        if doc == anchor_doc and score >= anchor_score - margin:
            picked.append((sentence, doc))
    return picked


def grounded_answer(parts: list[tuple[str, list[tuple[str, str, float]]]]) -> list[tuple[str, str]]:
    """One mock-LLM call. parts = [(sub_question, scored_sentences), ...] -> [(sentence, doc_id)] used."""
    CALLS["count"] += 1
    used: list[tuple[str, str]] = []
    for question, scored in parts:
        for item in select_sentences(question, scored):
            if item not in used:
                used.append(item)
    return used
