"""Task 4 - grounded generation: retrieve top-k chunks, answer ONLY from them, else say "I don't know".

Steps for one question:
 1. product nouns are replaced by their category word for retrieval ("earphones" -> "electronics");
 2. a multi-topic question is split into sub-queries; a dangling "it" in a sub-query is resolved from the
    earlier words of the same question;
 3. the WHOLE question must clear the similarity threshold (0.44), or - for contact/escalation topics -
    a topic hint plus the 0.32 floor; each sub-query must clear 0.44 too (or carry its own topic hint);
 4. candidate sentences are scored (embedding similarity + small word bonus + topic-hint boost); a "who"
    question keeps only sentences that name a role; the (mock) LLM picks the answer sentences;
 5. sub-queries that nothing supports are reported in `unanswered` instead of being answered with
    an unrelated sentence. `sources` lists only the documents whose sentences were actually used.
"""
import numpy as np

from llm import mock_llm
from llm.mock_llm import tokens
from rag.chunking import split_sentences
from rag.relevance import (find_product_phrase, has_duration, is_duration_question, is_order_only, is_who_question,
                           names_a_role, normalize_products, resolve_reference, split_subqueries, topic_hint)
from rag.store import embed, search

# Measured in rag/calibrate.py (see README and transcripts/task4_threshold_calibration.txt):
#   weakest in-scope top-1 = 0.5079, strongest out-of-scope top-1 (incl. near-domain junk such as
#   "How do I order a pizza online?") = 0.3682  ->  midpoint 0.4381 -> question-level threshold 0.44.
SIMILARITY_THRESHOLD = 0.44
# Lower floor (the original clean-cluster midpoint: 0.5079 vs 0.1329 -> 0.32). It is used for: accepting a
# question that has a topic hint (contact / escalate ...), and keeping candidate chunks as answer material.
# Sub-queries without a topic hint need the full threshold.
SUBQUERY_FLOOR = 0.32
HINT_BOOST = 0.15  # added to sentences of the document a topic hint points to
DURATION_BOOST = 0.12  # added to sentences that state a duration when the question asks "how long ..."
FALLBACK = "I don't know based on our support knowledge base. Let me connect you to a human agent."
DEFAULT_STRATEGY = "sentence"  # chosen by the Task 5 evaluation (see README)


def _score_sentences(sub: str, chunks: list[dict], hint_doc: str | None = None) -> list[tuple[str, str, float]]:
    pairs = list({(s, c["doc_id"]) for c in chunks for s in split_sentences(c["text"])})
    if not pairs:
        return []
    vectors = np.array(embed([sub] + [s for s, _ in pairs]))
    sims = vectors[1:] @ vectors[0]  # embeddings are normalized -> dot product = cosine similarity
    q_words = tokens(sub)
    asks_duration = is_duration_question(sub)
    # Hybrid score: cosine similarity + a small bonus per shared content word (capped), so a sentence about
    # "return window" beats a same-category "warranty" sentence for the question "return policy ...";
    # + a topic-hint boost; + a duration bonus when the question asks "how long / how many days" and the
    # sentence actually states a duration.
    return [(s, d, float(x) + min(0.09, 0.03 * len(q_words & tokens(s)))
             + (HINT_BOOST if d == hint_doc else 0.0)
             + (DURATION_BOOST if asks_duration and has_duration(s) else 0.0))
            for (s, d), x in zip(pairs, sims)]


def _merge(chunks: list[dict], extra: list[dict]) -> list[dict]:
    seen = {c["chunk_id"] for c in chunks}
    return chunks + [c for c in extra if c["chunk_id"] not in seen]


def answer_question(question: str, strategy: str = DEFAULT_STRATEGY, k: int = 3,
                    threshold: float = SIMILARITY_THRESHOLD, floor: float = SUBQUERY_FLOOR) -> dict:
    antecedent = find_product_phrase(question)  # what a dangling "it" in a later part can point to
    whole_text = normalize_products(question)
    whole = search(whole_text, strategy, k)
    top = whole[0]["similarity"] if whole else 0.0
    if not (top >= threshold or (topic_hint(question) and top >= floor)):  # question-level decision
        return {"answer": FALLBACK, "grounded": False, "top_similarity": top, "sources": [], "chunks": whole,
                "used": [], "unanswered": []}

    subs = split_subqueries(question)
    if len(subs) > 1:  # an order-only part ("what is the status of order") belongs to the order lookup, not the KB
        subs = [s for s in subs if not is_order_only(s)] or subs
    parts, all_chunks, unanswered = [], list(whole), []
    for sub in subs:
        resolved = normalize_products(resolve_reference(sub, antecedent))
        hint = topic_hint(sub)
        chunks = whole if subs == [question] else search(resolved, strategy, k)
        if chunks is not whole:
            all_chunks = _merge(all_chunks, chunks)
        if hint:  # also look inside the document the topic points to (e.g. the escalation matrix)
            hinted = search(resolved, strategy, k, doc_id=hint)
            all_chunks = _merge(all_chunks, hinted)
            chunks = _merge(list(chunks), hinted)
        supported = [c for c in chunks if c["similarity"] >= floor or (hint and c["doc_id"] == hint)]
        scored = _score_sentences(resolved, supported, hint) if supported else []
        if is_who_question(sub):  # a "who ..." question must be answered by a sentence that names a role
            scored = [t for t in scored if names_a_role(t[0])]
        # A sub-query needs the FULL threshold on its own (so a junk part such as "...and how do I order a
        # pizza online?" cannot ride along with a valid part), unless a topic hint vouches for it.
        if scored and (hint or (chunks and chunks[0]["similarity"] >= threshold)):
            parts.append((resolved, scored))
        else:
            unanswered.append(sub)
    if not parts:
        return {"answer": FALLBACK, "grounded": False, "top_similarity": top, "sources": [], "chunks": all_chunks,
                "used": [], "unanswered": unanswered}

    used = mock_llm.grounded_answer(parts)
    return {"answer": " ".join(s for s, _ in used) if used else FALLBACK, "grounded": bool(used),
            "top_similarity": top, "sources": sorted({d for _, d in used}), "chunks": all_chunks,
            "used": [s for s, _ in used], "unanswered": unanswered}
