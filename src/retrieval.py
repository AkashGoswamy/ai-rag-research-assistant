from typing import List, Optional

from src.ingestion import Chunk, DocumentStore


def retrieve(
    query: str,
    store: DocumentStore,
    doc_filter: Optional[List[str]] = None,
    k: int = 4,
    similarity_threshold: float = 0.3,
) -> List[Chunk]:
    """Embed the query, search the FAISS index, optionally restrict to a
    subset of doc_ids, drop low-similarity results, and return the top-k
    chunks with full metadata attached.

    doc_filter: list of doc_id values to restrict results to. None = search
    across all ingested documents.
    """
    if store.index is None or store.total_chunks() == 0:
        return []

    query_embedding = store._embed([query])

    # Overfetch when filtering by doc, since FAISS filters post-search here,
    # not pre-search at the index level.
    search_k = k * 4 if doc_filter else k
    search_k = min(search_k, store.total_chunks())

    scores, indices = store.index.search(query_embedding, search_k)

    results: List[Chunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < similarity_threshold:
            continue

        chunk = store.metadata[idx]

        if doc_filter is not None and chunk.doc_id not in doc_filter:
            continue

        results.append(chunk)

        if len(results) >= k:
            break

    return results
