"""Task 4 - measure top-1 cosine similarity for in-scope vs out-of-scope queries."""
from rag.queries import IN_SCOPE, OUT_OF_SCOPE
from rag.store import search


def measure() -> dict:
    out = {}
    for strategy in ("fixed", "sentence"):
        out[strategy] = {
            "in_scope": [(q, search(q, strategy, 1)[0]["similarity"]) for q, _ in IN_SCOPE],
            "out_of_scope": [(q, search(q, strategy, 1)[0]["similarity"]) for q in OUT_OF_SCOPE],
        }
    return out


if __name__ == "__main__":
    for strategy, r in measure().items():
        print(f"\n== {strategy} collection ==")
        for label in ("in_scope", "out_of_scope"):
            for q, s in r[label]:
                print(f"  {label:13s} {s:.4f}  {q}")
        lo = min(s for _, s in r["in_scope"])
        hi = max(s for _, s in r["out_of_scope"])
        print(f"  lowest in-scope = {lo:.4f} | highest out-of-scope = {hi:.4f} | midpoint = {(lo + hi) / 2:.4f}")
