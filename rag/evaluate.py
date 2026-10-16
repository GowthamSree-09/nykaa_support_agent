"""Task 5 - document-level precision/recall for both chunking strategies (top-k chunks, deduped to docs)."""
from rag.queries import IN_SCOPE
from rag.store import search

K = 3


def score(query: str, relevant: set[str], strategy: str, k: int = K) -> dict:
    chunks = search(query, strategy, k)
    retrieved_docs = sorted({c["doc_id"] for c in chunks})  # map chunk -> parent doc, dedup
    hits = sorted(set(retrieved_docs) & relevant)
    precision = len(hits) / len(retrieved_docs)
    recall = len(hits) / len(relevant)
    return {"retrieved": retrieved_docs, "hits": hits, "precision": precision, "recall": recall,
            "n_chunks": len(chunks)}


def run() -> dict:
    results = {}
    for strategy in ("fixed", "sentence"):
        rows = [(q, rel, score(q, rel, strategy)) for q, rel in IN_SCOPE]
        results[strategy] = {
            "rows": rows,
            "avg_precision": sum(r["precision"] for _, _, r in rows) / len(rows),
            "avg_recall": sum(r["recall"] for _, _, r in rows) / len(rows),
        }
    return results


if __name__ == "__main__":
    res = run()
    for strategy, data in res.items():
        print(f"\n===== {strategy} collection (top-{K} chunks, deduped to parent documents) =====")
        for q, rel, r in data["rows"]:
            print(f"Q: {q}\n   relevant={sorted(rel)}  retrieved_docs={r['retrieved']}  hits={r['hits']}")
            print(f"   precision = {len(r['hits'])}/{len(r['retrieved'])} = {r['precision']:.3f}"
                  f" | recall = {len(r['hits'])}/{len(rel)} = {r['recall']:.3f}")
        n = len(data["rows"])
        print(f"AVERAGE over {n} queries: precision = {sum(r['precision'] for *_, r in data['rows']):.3f}/{n}"
              f" = {data['avg_precision']:.3f} | recall = {sum(r['recall'] for *_, r in data['rows']):.3f}/{n}"
              f" = {data['avg_recall']:.3f}")
