"""Task 3 - two chunking strategies. Every chunk remembers its parent document id."""
import re

FIXED_SIZE = 250     # characters per chunk
FIXED_OVERLAP = 50   # characters shared between neighbouring chunks


def fixed_chunks(doc: dict, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[dict]:
    text, step, chunks = doc["text"], size - overlap, []
    for n, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append({"chunk_id": f"{doc['doc_id']}::fixed::{n}", "doc_id": doc["doc_id"], "text": piece})
        if start + size >= len(text):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def sentence_chunks(doc: dict) -> list[dict]:
    return [
        {"chunk_id": f"{doc['doc_id']}::sent::{n}", "doc_id": doc["doc_id"], "text": s}
        for n, s in enumerate(split_sentences(doc["text"]))
    ]


STRATEGIES = {"fixed": fixed_chunks, "sentence": sentence_chunks}
