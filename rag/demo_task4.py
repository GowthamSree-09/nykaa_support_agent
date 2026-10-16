"""Task 4 demo: 7 in-scope queries + out-of-scope queries that must trigger the fallback."""
from rag.generate import SIMILARITY_THRESHOLD, answer_question
from rag.queries import IN_SCOPE, OUT_OF_SCOPE

print(f"MOCK_LLM grounded generation | threshold = {SIMILARITY_THRESHOLD}\n")
for q in [q for q, _ in IN_SCOPE] + OUT_OF_SCOPE[:2]:
    r = answer_question(q)
    print(f"Q: {q}\n   top1={r['top_similarity']:.4f} grounded={r['grounded']} sources={r['sources']}\n   A: {r['answer']}\n")
