"""Task 4 - measure top-1 cosine similarity for in-scope vs out-of-scope queries."""
from rag.queries import IN_SCOPE, NEAR_DOMAIN_OUT_OF_SCOPE, OUT_OF_SCOPE
from rag.store import search


def measure() -> dict:
    out = {}
    for strategy in ("fixed", "sentence"):
        out[strategy] = {
            "in_scope": [(q, search(q, strategy, 1)[0]["similarity"]) for q, _ in IN_SCOPE],
            "out_of_scope": [(q, search(q, strategy, 1)[0]["similarity"]) for q in OUT_OF_SCOPE],
            "near_domain_oos": [(q, search(q, strategy, 1)[0]["similarity"]) for q in NEAR_DOMAIN_OUT_OF_SCOPE],
        }
    return out


if __name__ == "__main__":
    for strategy, r in measure().items():
        print(f"\n== {strategy} collection ==")
        for label in ("in_scope", "out_of_scope", "near_domain_oos"):
            for q, s in r[label]:
                print(f"  {label:15s} {s:.4f}  {q}")
        lo = min(s for _, s in r["in_scope"])
        hi = max(s for label in ("out_of_scope", "near_domain_oos") for _, s in r[label])
        print(f"  lowest in-scope = {lo:.4f} | highest out-of-scope = {hi:.4f} | midpoint = {(lo + hi) / 2:.4f}")


def glossary_evidence(strategy: str = "sentence") -> list[tuple[str, float, float, str]]:
    """(question, top-1 before, top-1 after replacing the product noun by its category, top document after)."""
    from rag.queries import PRODUCT_NOUN_QUESTIONS
    from rag.relevance import normalize_products

    rows = []
    for q in PRODUCT_NOUN_QUESTIONS:
        before = search(q, strategy, 1)[0]
        after = search(normalize_products(q), strategy, 1)[0]
        rows.append((q, before["similarity"], after["similarity"], after["doc_id"]))
    return rows


if __name__ == "__main__":
    from rag.generate import SIMILARITY_THRESHOLD
    from rag.queries import NEAR_DOMAIN_OUT_OF_SCOPE, OUT_OF_SCOPE
    from rag.relevance import normalize_products

    print(f"\n== product glossary evidence (sentence collection, threshold {SIMILARITY_THRESHOLD}) ==")
    for q, before, after, doc in glossary_evidence():
        print(f"  {before:.4f} -> {after:.4f}  [{doc}]  {'ACCEPTED' if after >= SIMILARITY_THRESHOLD else 'rejected'}"
              f" (before: {'accepted' if before >= SIMILARITY_THRESHOLD else 'rejected'})  {q}")
    junk = OUT_OF_SCOPE + NEAR_DOMAIN_OUT_OF_SCOPE
    changed = [q for q in junk if normalize_products(q) != q]
    top = max(search(normalize_products(q), "sentence", 1)[0]["similarity"] for q in junk)
    print(f"  out-of-scope questions changed by the glossary: {len(changed)} of {len(junk)}; "
          f"highest out-of-scope top-1 after normalization = {top:.4f}")
