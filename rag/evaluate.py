"""Task 5 - document-level precision/recall for both chunking strategies.

Chunks are mapped to their parent documents and de-duplicated BEFORE scoring.
Two retrieval modes are compared:
  single : top-3 chunks for the whole question (the original method)
  multi  : the question is split into sub-questions (rag/relevance.py), top-K chunks for each,
           and the parent documents of all of them are merged (multi-query retrieval)
"""
from rag.queries import IN_SCOPE, MULTI_TOPIC
from rag.relevance import split_subqueries
from rag.store import search

K = 3
K_PER_SUBQUERY = 2


def retrieved_docs(query: str, strategy: str, mode: str) -> tuple[list[str], int]:
    if mode == "single":
        chunks = search(query, strategy, K)
    else:
        chunks = [c for sub in split_subqueries(query) for c in search(sub, strategy, K_PER_SUBQUERY)]
    return sorted({c["doc_id"] for c in chunks}), len(chunks)


def score(query: str, relevant: set[str], strategy: str, mode: str = "single") -> dict:
    docs, n_chunks = retrieved_docs(query, strategy, mode)
    hits = sorted(set(docs) & relevant)
    return {"retrieved": docs, "hits": hits, "precision": len(hits) / len(docs),
            "recall": len(hits) / len(relevant), "n_chunks": n_chunks}


def run(queries=None, mode: str = "single") -> dict:
    queries = queries if queries is not None else IN_SCOPE
    results = {}
    for strategy in ("fixed", "sentence"):
        rows = [(q, rel, score(q, rel, strategy, mode)) for q, rel in queries]
        results[strategy] = {
            "rows": rows,
            "avg_precision": sum(r["precision"] for _, _, r in rows) / len(rows),
            "avg_recall": sum(r["recall"] for _, _, r in rows) / len(rows),
        }
    return results


def _print(res: dict, title: str) -> None:
    for strategy, data in res.items():
        print(f"\n===== {title} | {strategy} collection (chunks deduped to parent documents) =====")
        for q, rel, r in data["rows"]:
            print(f"Q: {q}\n   relevant={sorted(rel)}  retrieved_docs={r['retrieved']}  hits={r['hits']}")
            print(f"   precision = {len(r['hits'])}/{len(r['retrieved'])} = {r['precision']:.3f}"
                  f" | recall = {len(r['hits'])}/{len(rel)} = {r['recall']:.3f}")
        n = len(data["rows"])
        print(f"AVERAGE over {n} queries: precision = {sum(r['precision'] for *_, r in data['rows']):.3f}/{n}"
              f" = {data['avg_precision']:.3f} | recall = {sum(r['recall'] for *_, r in data['rows']):.3f}/{n}"
              f" = {data['avg_recall']:.3f}")


if __name__ == "__main__":
    for title, queries, mode in [("A. SINGLE-QUERY, 7 required queries (original method)", IN_SCOPE, "single"),
                                 ("B. MULTI-QUERY, same 7 queries", IN_SCOPE, "multi"),
                                 ("C. SINGLE-QUERY, 2 extra multi-topic queries", MULTI_TOPIC, "single"),
                                 ("D. MULTI-QUERY, same 2 multi-topic queries", MULTI_TOPIC, "multi")]:
        _print(run(queries, mode), title)
