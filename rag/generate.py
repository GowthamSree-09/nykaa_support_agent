"""Task 4 - grounded generation: retrieve top-k chunks, answer ONLY from them, else say "I don't know"."""
from llm import mock_llm
from rag.store import search

# Measured in rag/calibrate.py: lowest in-scope top-1 = 0.5015 (fixed) / 0.5079 (sentence);
# highest out-of-scope top-1 = 0.1192 (fixed) / 0.1329 (sentence). Midpoint of the wider
# gap (sentence) = 0.3204 -> we use 0.32.
SIMILARITY_THRESHOLD = 0.32
FALLBACK = "I don't know based on our support knowledge base. Let me connect you to a human agent."
DEFAULT_STRATEGY = "sentence"  # confirmed by Task 5 evaluation (see README)


def answer_question(question: str, strategy: str = DEFAULT_STRATEGY, k: int = 3,
                    threshold: float = SIMILARITY_THRESHOLD) -> dict:
    chunks = search(question, strategy, k)
    top = chunks[0]["similarity"] if chunks else 0.0
    if top < threshold:
        return {"answer": FALLBACK, "grounded": False, "top_similarity": top,
                "sources": [], "chunks": chunks}
    answer = mock_llm.grounded_answer(question, [c["text"] for c in chunks if c["similarity"] >= threshold])
    return {"answer": answer, "grounded": True, "top_similarity": top,
            "sources": sorted({c["doc_id"] for c in chunks}), "chunks": chunks}
