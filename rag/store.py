"""Task 3 - embeddings (SentenceTransformers, local) + two separate ChromaDB collections."""
import os
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from kb.documents import KB_DOCUMENTS
from rag.chunking import STRATEGIES

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAMES = {"fixed": "kb_fixed", "sentence": "kb_sentence"}


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Offline first (graded runs must not touch the network). Only the one-time setup
    # download, when the model is not cached yet, goes online.
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(strategy: str):
    # cosine space -> distance = 1 - cosine similarity
    return _client().get_or_create_collection(
        COLLECTION_NAMES[strategy], metadata={"hnsw:space": "cosine"}
    )


def index_documents(docs: list[dict] | None = None) -> dict[str, int]:
    """Chunk with BOTH strategies and upsert each into its own collection."""
    docs = docs if docs is not None else KB_DOCUMENTS
    counts = {}
    for strategy, chunker in STRATEGIES.items():
        chunks = [c for d in docs for c in chunker(d)]
        get_collection(strategy).upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embed([c["text"] for c in chunks]),
            metadatas=[{"doc_id": c["doc_id"], "strategy": strategy} for c in chunks],
        )
        counts[strategy] = len(chunks)
    return counts


def delete_document(doc_id: str) -> None:
    for strategy in COLLECTION_NAMES:
        get_collection(strategy).delete(where={"doc_id": doc_id})


def search(query: str, strategy: str = "sentence", k: int = 3, doc_id: str | None = None) -> list[dict]:
    """Return top-k chunks: [{chunk_id, doc_id, text, similarity}] (best first).
    doc_id restricts the search to ONE knowledge-base document (used for topic hints)."""
    col = get_collection(strategy)
    where = {"doc_id": doc_id} if doc_id else None
    res = col.query(query_embeddings=embed([query]), n_results=min(k, col.count()), where=where)
    return [
        {"chunk_id": i, "doc_id": m["doc_id"], "text": t, "similarity": round(1 - d, 4)}
        for i, m, t, d in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]
