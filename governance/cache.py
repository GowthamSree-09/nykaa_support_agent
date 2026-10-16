"""Task 16 - in-memory response cache for the grounded-generation step.

Key = normalized query text (lower-case, punctuation removed, spaces collapsed), so
"Can I cancel?" and "  can i CANCEL " share one entry. A hit skips retrieval AND the LLM call.
"""
import re

from rag.generate import DEFAULT_STRATEGY, answer_question


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", query.lower())).strip()


class ResponseCache:
    def __init__(self):
        self.store: dict[tuple[str, str], dict] = {}
        self.hits = 0
        self.misses = 0

    def get_answer(self, question: str, strategy: str = DEFAULT_STRATEGY) -> dict:
        key = (strategy, normalize(question))
        if key in self.store:
            self.hits += 1
            return {**self.store[key], "cache_hit": True}
        self.misses += 1
        result = answer_question(question, strategy)  # the expensive part: embedding + search + LLM
        self.store[key] = result
        return {**result, "cache_hit": False}

    def clear(self):
        self.store.clear()
        self.hits = self.misses = 0


CACHE = ResponseCache()


def cached_answer(question: str, strategy: str = DEFAULT_STRATEGY) -> dict:
    return CACHE.get_answer(question, strategy)
