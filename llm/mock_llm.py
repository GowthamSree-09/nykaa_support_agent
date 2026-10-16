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


def grounded_answer(question: str, context_chunks: list[str], max_sentences: int = 2) -> str:
    """Extractive answer: copy the context sentences that best overlap the question words."""
    CALLS["count"] += 1
    q = tokens(question)
    scored = []
    for chunk in context_chunks:
        for s in _sentences(chunk):
            overlap = len(q & tokens(s))
            if overlap:
                scored.append((overlap, s))
    seen, picked = set(), []
    ranked = sorted(scored, key=lambda x: -x[0])
    best = ranked[0][0] if ranked else 0
    for overlap, s in ranked:
        if overlap < max(1, 0.6 * best):  # drop sentences much less relevant than the best one
            break
        if s not in seen:
            seen.add(s)
            picked.append(s)
        if len(picked) == max_sentences:
            break
    return " ".join(picked) if picked else context_chunks[0] if context_chunks else ""
